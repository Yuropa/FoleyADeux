"""
app/pipeline.py
───────────────
Core inference pipeline wired to the Gradio UI.

Public surface
--------------
run_inference(video_path, prompt, theme, progress)
    End-to-end handler called by the Generate button.  Runs video
    preprocessing, AudioLDM 2 generation, metrics computation, and plot
    building.  Yields 6-tuples:
        (output_video_path, waveform_fig, spectrogram_fig, rms_fig,
         status_md, metrics_md)

Internal helpers
----------------
_get_config_and_device()
    Lazy-initialise the YACS config and torch device singletons.

_run_audio_generation(...)
    Loop over preprocessed clips: Video2RMS forward → AudioLDM 2 generate
    → amplitude modulation → save audio + muxed video.
    Returns RMS curves and waveforms for downstream visualisation.
"""

import os
import shutil
import subprocess
from glob import glob
from typing import List, Optional, Tuple
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import numpy as np
import torch
import soundfile as sf
import gradio as gr
from yacs.config import CfgNode as CN

# config.py must be imported first — its module-level code sets sys.path
# so that all repo-local libraries below become importable.
from .config import CKPT_DIR, OUTPUT_BASE, THEMES, create_device
from .plots import plot_waveform, plot_spectrogram, plot_rms_envelope
from .preprocess import preprocess_videos
from .metrics import (
    compute_envelope_metrics,
    compute_audio_quality_metrics,
    compute_clap_score,
    aggregate_metrics,
    format_metrics_markdown,
)
# These are repo-local modules; sys.path is set up by .config at import time.
from util import load_config, load_models, save_video_with_audio  # noqa: E402
from data_utils import RMS, pad_or_truncate_feature  # noqa: E402
# Loaded once on first Generate click, then reused for every subsequent run.
_config: Optional[CN] = None
_device: Optional[str] = None
_torch_dtype: Optional[str] = None


def _get_config() -> Tuple[CN, str]:
    global _config, _device
    if _config is None:
        _config = load_config(os.path.join(CKPT_DIR, "opts.yml"))
        _config.defrost()
        _config.data.video_fps    = _config.data.video_samples / _config.data.audio_samples
        _config.data.video_width  = 344
        _config.data.video_height = 256
        _config.freeze()
    if _device is None:
        _device, _torch_dtype = create_device()
    return _config, _device, _torch_dtype


# ── Audio generation loop ────────────────────────────────────────────────────

def _align_audio_to_rms(generated_raw, rms_np, sr, hop_len=160):
    """
    Warp generated_raw so its peaks align with peaks in rms_np.
    """
    n_audio = len(generated_raw)
    
    # --- Find peaks in RMS (video hits) ---
    rms_resampled = np.interp(
        np.linspace(0, 1, n_audio),
        np.linspace(0, 1, len(rms_np)),
        rms_np,
    )
    rms_peaks, _ = find_peaks(rms_resampled, height=rms_resampled.mean(), distance=sr//4)

    # --- Find peaks in generated audio (loudness envelope) ---
    audio_env = np.abs(generated_raw)
    # smooth the audio envelope so we get broad peaks, not individual samples
    audio_env = gaussian_filter1d(audio_env, sigma=sr//20)
    audio_peaks, _ = find_peaks(audio_env, height=audio_env.mean(), distance=sr//4)

    if len(rms_peaks) == 0 or len(audio_peaks) == 0:
        return generated_raw  # can't align, return as-is

    # --- Match audio peaks to nearest RMS peaks ---
    # Use as many pairs as we have (trim to shorter list)
    n_pairs = min(len(rms_peaks), len(audio_peaks))
    src = audio_peaks[:n_pairs].astype(float)
    dst = rms_peaks[:n_pairs].astype(float)

    # Always anchor start and end so the warp covers the full signal
    src = np.concatenate([[0], src, [n_audio - 1]])
    dst = np.concatenate([[0], dst, [n_audio - 1]])

    # --- Build a time-warp map ---
    # For each output sample position, find where to sample in the input
    warp = interp1d(dst, src, kind='linear', bounds_error=False,
                    fill_value=(src[0], src[-1]))
    output_positions = np.arange(n_audio)
    input_positions  = warp(output_positions)

    # --- Resample audio along the warp map ---
    aligned = np.interp(input_positions, np.arange(n_audio), generated_raw)
    return aligned

def _run_audio_generation(
    processed_video_paths: List[str],
    config: CN,
    output_dir: str,
    device: str,
    torch_dtype: str,
    text_prompt: str,
    epoch: int = 500,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Run Video2RMS + AudioLDM 2 generation for each preprocessed clip.

    Audio is generated from the user's text prompt via AudioLDM 2.  The
    Video2RMS predicted RMS envelope is then applied as amplitude modulation
    so the output sound follows the video's rhythm and intensity.

    Parameters
    ----------
    processed_video_paths : list of str
        Paths to the FPS-normalised segment clips from preprocess_videos().
    config : CfgNode
        Pipeline configuration from opts.yml.
    output_dir : str
        Directory where audio/ and video/ subdirs will be written.
    device : str
        Torch device string ("cuda" or "cpu").
    text_prompt : str
        Full prompt (with any theme prefix already prepended).
    epoch : int
        Video2RMS checkpoint epoch to load.

    Returns
    -------
    rms_curves : list of np.ndarray  — Video2RMS predicted RMS (one per segment)
    waveforms  : list of np.ndarray  — amplitude-modulated output waveforms
    audio_paths: list of str         — paths to saved .wav files
    """
    audio_out = os.path.join(output_dir, "audio")
    video_out = os.path.join(output_dir, "video")
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(video_out, exist_ok=True)

    feature_dir = os.path.join(output_dir, "features")
    rgb_dir     = os.path.join(feature_dir, "feature_RGB")
    flow_dir    = os.path.join(feature_dir, "feature_Flow")

    video2rms_model, audio_ldm2 = load_models(
        epoch, CKPT_DIR, config, device, torch_dtype
    )
    video2rms_model.eval()

    mu_bins = RMS.get_mu_bins(
        config.data.rms_mu, config.data.rms_num_bins, config.data.rms_min
    )

    rms_curves:  List[np.ndarray] = []
    waveforms:   List[np.ndarray] = []
    audio_paths: List[str]        = []

    try:
        for clip_path in processed_video_paths:
            seg_id = os.path.splitext(os.path.basename(clip_path))[0]

            # Load and combine RGB + optical-flow BN-Inception features
            rgb_feat  = np.load(os.path.join(rgb_dir,  f"{seg_id}.pkl"), allow_pickle=True)
            flow_feat = np.load(os.path.join(flow_dir, f"{seg_id}.pkl"), allow_pickle=True)
            rgb_feat  = pad_or_truncate_feature(rgb_feat,  config.data.video_samples)
            flow_feat = pad_or_truncate_feature(flow_feat, config.data.video_samples)
            combined  = np.concatenate([rgb_feat, flow_feat], axis=1)
            feat_tensor = (
                torch.from_numpy(combined.astype(np.float32)).unsqueeze(0).to(device)
            )

            # Video2RMS forward pass → predicted RMS envelope
            video2rms_model.parse_batch((
                feat_tensor,
                torch.zeros(1, config.data.rms_samples).to(device),
                None, None,
            ))
            video2rms_model.forward()

            pred_rms_raw      = video2rms_model.pred_rms[0].detach().cpu().numpy()
            pred_bin_tensor   = torch.from_numpy(pred_rms_raw.argmax(axis=0))
            rms_undiscretized = RMS.undiscretize_rms(pred_bin_tensor, mu_bins, ignore_min=True)

            # AudioLDM 2 generation via text prompt
            generated_raw = audio_ldm2(
                prompt=text_prompt,
                negative_prompt="Low quality, average.",
                num_inference_steps=200,
                audio_length_in_s=float(config.data.audio_samples),
                guidance_scale=7.0,
            ).audios[0]

            # Apply Video2RMS amplitude envelope as modulation
            rms_np   = rms_undiscretized.numpy()
            generated_raw = _align_audio_to_rms(generated_raw, rms_np, config.data.audio_sample_rate)

            # Trim the audio
            expected_len = config.data.audio_sample_rate * config.data.audio_samples
            generated_raw = generated_raw[:expected_len]
            if len(generated_raw) < expected_len:
                generated_raw = np.pad(generated_raw, (0, expected_len - len(generated_raw)))

            envelope = np.interp(
                np.linspace(0, 1, len(generated_raw)),
                np.linspace(0, 1, len(rms_np)),
                rms_np,
            )

            # Asymmetric smoothing: fast attack, slow decay
            attack_sigma = config.data.envelope_attack_sigma
            decay_sigma  = config.data.envelope_decay_sigma

            # Smooth rising and falling edges separately
            rising  = gaussian_filter1d(np.maximum(np.diff(envelope, prepend=envelope[0]), 0), sigma=attack_sigma)
            falling = gaussian_filter1d(np.minimum(np.diff(envelope, prepend=envelope[0]), 0), sigma=decay_sigma)

            # Reconstruct envelope from asymmetrically smoothed deltas
            shaped = np.cumsum(rising + falling)
            shaped -= shaped.min()  # ensure non-negative
            shaped /= (shaped.max() + 1e-8)
            generated = generated_raw * shaped

            # Persist audio and mux with the source clip
            audio_path = os.path.join(audio_out, f"{seg_id}_generated.wav")
            sf.write(audio_path, generated, config.data.audio_sample_rate)

            if os.path.exists(clip_path):
                save_video_with_audio(
                    clip_path,
                    generated,
                    os.path.join(video_out, f"{seg_id}_with_audio.mp4"),
                    sr=config.data.audio_sample_rate,
                )

            rms_curves.append(rms_np)
            waveforms.append(generated)
            audio_paths.append(audio_path)

    finally:
        # Free GPU memory
        del video2rms_model, audio_ldm2
        torch.cuda.empty_cache()

    return rms_curves, waveforms, audio_paths


# ── Shared core (no Gradio dependency) ─────────────────────────────────────

def generate(
    video_path: str,
    prompt: str,
    theme: str = "None",
    output_dir: Optional[str] = None,
    log=print,
) -> str:
    """
    Run the full foley pipeline and return the path to the merged output video.

    This function has no Gradio dependency and can be called from the CLI,
    notebooks, or any other Python code.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    prompt : str
        Free-text sound description.
    theme : str
        One of the keys in THEMES (default "None").
    output_dir : str, optional
        Where to write outputs.  Defaults to OUTPUT_BASE/<run_id>/.
    log : callable
        Function used for status messages (default: print).

    Returns
    -------
    str  — absolute path to the merged result video.
    """
    config, device,  torch_dtype = _get_config()

    theme_prefix = THEMES.get(theme, "")
    full_prompt  = (theme_prefix + prompt.strip()).strip(", ").strip()

    run_id    = f"run_{os.getpid()}_{id(video_path)}"
    work_dir  = output_dir or os.path.join(OUTPUT_BASE, run_id)
    input_dir = os.path.join(work_dir, "input")
    out_dir   = os.path.join(work_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(out_dir,   exist_ok=True)

    ext = os.path.splitext(video_path)[1] or ".mp4"
    shutil.copy(video_path, os.path.join(input_dir, f"input{ext}"))

    log("Preprocessing video (optical flow, features)\u2026")
    processed_paths = preprocess_videos(
        video_dir=input_dir,
        config=config,
        output_dir=out_dir,
        device=device,
        dtype=torch_dtype,
        num_workers=1,
        batch_size=1,
    )

    log(f'Generating foley audio \u2014 "{full_prompt}"\u2026')
    rms_curves, waveforms, _ = _run_audio_generation(
        processed_video_paths=processed_paths,
        config=config,
        output_dir=out_dir,
        device=device,
        torch_dtype=torch_dtype,
        text_prompt=full_prompt,
    )

    # Compute and log metrics
    log("Computing metrics\u2026")
    sr = config.data.audio_sample_rate
    full_waveform = np.concatenate(waveforms) if waveforms else np.zeros(256)
    per_seg_metrics = []
    for rms_curve, wav in zip(rms_curves, waveforms):
        seg_m = {}
        seg_m.update(compute_envelope_metrics(rms_curve, wav, sr))
        seg_m.update(compute_audio_quality_metrics(wav, sr))
        per_seg_metrics.append(seg_m)
    clap = compute_clap_score(full_prompt, full_waveform, sr)
    for seg_m in per_seg_metrics:
        seg_m["clap_score"] = clap
    summary = aggregate_metrics(per_seg_metrics)
    log(
        f"  Envelope r={summary.get('envelope_pearson_r', 0):.3f}  "
        f"MAE={summary.get('envelope_mae', 0):.4f}  "
        f"Onset F1={summary.get('onset_f1', 0):.3f}  "
        f"DR={summary.get('dynamic_range_db', 0):.1f}dB"
        + (f"  CLAP={clap:.4f}" if clap is not None else "")
    )

    output_videos = sorted(glob(os.path.join(out_dir, "video", "*.mp4")))
    if not output_videos:
        raise RuntimeError(
            "Audio was generated but no output video was produced. "
            "Check that FFmpeg is installed and the source clip exists."
        )

    result_path = os.path.join(work_dir, "result.mp4")
    if len(output_videos) == 1:
        shutil.copy(output_videos[0], result_path)
    else:
        concat_list = os.path.join(work_dir, "concat.txt")
        with open(concat_list, "w") as fh:
            for vp in output_videos:
                fh.write(f"file '{vp}'\n")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                result_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    log(f"Done \u2014 result saved to {result_path}")
    return result_path


# ── Gradio-facing handler ────────────────────────────────────────────────────

def run_inference(
    video_path: Optional[str],
    prompt: str,
    theme: str,
):
    """
    End-to-end generator wired to the Generate button.

    Yields intermediate status-only tuples so progress text appears solely
    in the status_box component, then yields the final complete tuple.

    Yields
    ------
    Tuple of:
        output_video_path : str | None
        waveform_fig      : matplotlib.figure.Figure | None
        spectrogram_fig   : matplotlib.figure.Figure | None
        rms_fig           : matplotlib.figure.Figure | None
        status_markdown   : str
        metrics_markdown  : str
    """
    _idle = (None, None, None, None, "")  # placeholder for unready outputs

    if video_path is None:
        raise gr.Error("Please upload a video first.")
    if not prompt.strip():
        raise gr.Error("Please enter a sound prompt.")

    config, device, torch_dtype = _get_config()
    full_prompt = (THEMES.get(theme, "") + prompt.strip()).strip(", ").strip()

    run_id    = f"run_{os.getpid()}_{id(video_path)}"
    work_dir  = os.path.join(OUTPUT_BASE, run_id)
    input_dir = os.path.join(work_dir, "input")
    out_dir   = os.path.join(work_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(out_dir,   exist_ok=True)

    # Copy the uploaded video to a predictable filename
    ext = os.path.splitext(video_path)[1] or ".mp4"
    shutil.copy(video_path, os.path.join(input_dir, f"input{ext}"))

    try:
        # Stage 1: preprocess
        yield *_idle, "\u23f3 Preprocessing video (optical flow, features)\u2026"
        processed_paths = preprocess_videos(
            video_dir=input_dir,
            config=config,
            output_dir=out_dir,
            device=device,
            dtype=torch_dtype,
            num_workers=1,
            batch_size=1,
        )

        # Stage 2: generate audio
        yield *_idle, f'\u23f3 Generating foley audio \u2014 "{full_prompt}"\u2026'
        rms_curves, waveforms, _ = _run_audio_generation(
            processed_video_paths=processed_paths,
            config=config,
            output_dir=out_dir,
            device=device,
            torch_dtype=torch_dtype,
            text_prompt=full_prompt,
        )

        # Stage 3: build visualisation figures
        yield *_idle, "\u23f3 Building visualisations\u2026"
        full_waveform = np.concatenate(waveforms) if waveforms else np.zeros(256)
        full_rms      = np.concatenate(rms_curves) if rms_curves else np.zeros(64)
        sr            = config.data.audio_sample_rate

        waveform_fig    = plot_waveform(full_waveform, sr)
        spectrogram_fig = plot_spectrogram(full_waveform, sr)
        rms_fig         = plot_rms_envelope(full_waveform, full_rms, sr)

        # Stage 4: compute metrics (AudioLDM 2 is already freed at this point)
        yield *_idle, "\u23f3 Computing metrics\u2026"
        per_seg_metrics = []
        for rms_curve, wav in zip(rms_curves, waveforms):
            seg_m = {}
            seg_m.update(compute_envelope_metrics(rms_curve, wav, sr))
            seg_m.update(compute_audio_quality_metrics(wav, sr))
            per_seg_metrics.append(seg_m)

        # CLAP score on the full waveform — the prompt applies to the whole piece
        clap = compute_clap_score(full_prompt, full_waveform, sr)
        for seg_m in per_seg_metrics:
            seg_m["clap_score"] = clap

        metrics_md = format_metrics_markdown(
            aggregate_metrics(per_seg_metrics),
            n_segments=len(per_seg_metrics),
        )

        # Merge output segments
        output_videos = sorted(glob(os.path.join(out_dir, "video", "*.mp4")))
        if not output_videos:
            raise gr.Error(
                "Audio was generated but no output video was produced. "
                "Check that FFmpeg is installed and the source clip exists."
            )

        result_path = os.path.join(work_dir, "result.mp4")
        if len(output_videos) == 1:
            shutil.copy(output_videos[0], result_path)
        else:
            concat_list = os.path.join(work_dir, "concat.txt")
            with open(concat_list, "w") as fh:
                for vp in output_videos:
                    fh.write(f"file '{vp}'\n")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_list, "-c", "copy", result_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        status = (
            f"\u2705 Generated successfully &nbsp;|&nbsp; "
            f"Prompt: **{full_prompt}** &nbsp;|&nbsp; "
            f"Duration: {len(full_waveform) / sr:.1f}s"
        )
        yield result_path, waveform_fig, spectrogram_fig, rms_fig, status, metrics_md

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Pipeline error: {exc}") from exc
