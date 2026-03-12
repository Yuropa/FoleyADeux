"""
app/pipeline.py
───────────────
Core inference pipeline wired to the Gradio UI.

Public surface
--------------
run_inference(video_path, prompt, theme, progress)
    End-to-end handler called by the Generate button.  Runs video
    preprocessing, AudioLDM generation, and plot building.  Returns
    (output_video_path, waveform_fig, spectrogram_fig, rms_fig, status_md).

Internal helpers
----------------
_get_config_and_device()
    Lazy-initialise the YACS config and torch device singletons.

_run_audio_generation(...)
    Loop over preprocessed clips: Video2RMS forward → AudioLDM generate
    → amplitude modulation → save audio + muxed video.
    Returns RMS curves and waveforms for downstream visualisation.
"""

import os
import shutil
import subprocess
from glob import glob
from typing import List, Optional, Tuple

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
# These are repo-local modules; sys.path is set up by .config at import time.
from util import load_config, load_models, save_video_with_audio, interpolate_rms_for_rms2sound  # noqa: E402
from data_utils import RMS, pad_or_truncate_feature  # noqa: E402
# Loaded once on first Generate click, then reused for every subsequent run.
_config: Optional[CN] = None
_device: Optional[str] = None


def _get_config_and_device() -> Tuple[CN, str]:
    global _config, _device
    if _config is None:
        _config = load_config(os.path.join(CKPT_DIR, "opts.yml"))
        _config.defrost()
        _config.data.video_fps    = _config.data.video_samples / _config.data.audio_samples
        _config.data.video_width  = 344
        _config.data.video_height = 256
        _config.freeze()
    if _device is None:
        _device = create_device()
    return _config, _device


# ── Audio generation loop ────────────────────────────────────────────────────

def _run_audio_generation(
    processed_video_paths: List[str],
    config: CN,
    output_dir: str,
    device: str,
    text_prompt: str,
    epoch: int = 500,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Run Video2RMS + AudioLDM generation for each preprocessed clip.

    The ControlNet branch is bypassed so the generation is driven purely by
    the CLAP text embedding of the user's prompt.  The Video2RMS predicted
    RMS envelope is then applied as amplitude modulation so the output sound
    follows the video's rhythm and intensity.

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

    video2rms_model, audio_ldm_controlnet = load_models(
        epoch, CKPT_DIR, CKPT_DIR, config, device
    )
    video2rms_model.eval()

    mu_bins = RMS.get_mu_bins(
        config.data.rms_mu, config.data.rms_num_bins, config.data.rms_min
    )

    # Bypass ControlNet: patch apply_model so the RMS branch has zero influence
    _orig_apply = audio_ldm_controlnet.model.apply_model
    audio_ldm_controlnet.model.apply_model = (
        lambda x, t, cond, **kw:
            _orig_apply(x, t, cond, w_control_net_condition=False, **kw)
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

            # Interpolate RMS to the latent frame rate expected by AudioLDM
            rms_for_ldm = interpolate_rms_for_rms2sound(
                rms_undiscretized.unsqueeze(0),
                audio_len=config.data.audio_samples,
                sr=config.data.audio_sample_rate,
                frame_len=1024,
                hop_len=160,
            ).to(device)

            # AudioLDM generation (ControlNet bypassed, CLAP text conditioning)
            generated_raw = audio_ldm_controlnet.generate(
                waveform=None,
                text_prompt=text_prompt,
                rms=rms_for_ldm,
            )

            # Apply Video2RMS amplitude envelope as modulation
            rms_np   = rms_undiscretized.numpy()
            envelope = np.interp(
                np.linspace(0, 1, len(generated_raw)),
                np.linspace(0, 1, len(rms_np)),
                rms_np,
            )
            envelope  /= (envelope.max() + 1e-8)
            generated  = generated_raw * envelope

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
        # Always restore original method and free GPU memory
        audio_ldm_controlnet.model.apply_model = _orig_apply
        del video2rms_model, audio_ldm_controlnet
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
    config, device = _get_config_and_device()

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
        num_workers=1,
        batch_size=1,
    )

    log(f'Generating foley audio \u2014 "{full_prompt}"\u2026')
    _, _, _ = _run_audio_generation(
        processed_video_paths=processed_paths,
        config=config,
        output_dir=out_dir,
        device=device,
        text_prompt=full_prompt,
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
    """
    _idle = (None, None, None, None)  # placeholder for unready outputs

    if video_path is None:
        raise gr.Error("Please upload a video first.")
    if not prompt.strip():
        raise gr.Error("Please enter a sound prompt.")

    config, device = _get_config_and_device()
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
        yield result_path, waveform_fig, spectrogram_fig, rms_fig, status

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Pipeline error: {exc}") from exc
