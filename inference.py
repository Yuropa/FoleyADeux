"""
inference.py
────────────
Full foley pipeline: video folder → generated audio + muxed video.

Any sound can be requested via a text prompt or a reference audio file.
The ControlNet branch is bypassed so the base AudioLDM (general, locked
during ControlNet training) + CLAP text/audio embedding drives generation.
The Video2RMS envelope is then applied as amplitude modulation so timing
and intensity still follow the video motion.

Usage:
    python inference.py -i examples/ -p "a dog barking"
    python inference.py -i examples/ -p "orchestral strings"
    python inference.py -i examples/ -a reference.wav

    # custom output dir
    python inference.py -i examples/ -p "glass shattering" -o ./my_output
"""

import argparse
import os
import sys
import subprocess
from glob import glob
from multiprocessing import Pool
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from yacs.config import CfgNode as CN

# ──────────────────────────────────────────────────────────────────────────────
# Sys-path setup
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR = os.getcwd()
CKPT_DIR = os.path.join(ROOT_DIR, "ckpt")

sys.path.insert(0, os.path.join(ROOT_DIR, "video2rms"))
sys.path.insert(0, os.path.join(ROOT_DIR, "libs", "taming-transformers"))
sys.path.insert(0, os.path.join(ROOT_DIR, "libs", "AudioLDM"))

from util import load_config, load_models, save_video_with_audio, interpolate_rms_for_rms2sound
from data_utils import RMS, pad_or_truncate_feature
from preprocess.extract_audio_and_video import pipline_align, pipline_cut
from preprocess.extract_rgb_flow_raft import cal_for_frames
from preprocess.extract_feature import extract_bn_inception_feature

from utils.utils import create_device
from utils.video_descriptions import VideoDescription


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _add_silent_audio_if_needed(video_path: str) -> None:
    """Add a silent audio track if the video has no audio stream."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0', '-count_packets',
        '-show_entries', 'stream=codec_type,nb_read_packets',
        '-of', 'csv=p=0', video_path,
    ]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    if not (output == '' or output.endswith(',0')):
        return  # already has audio

    print(f"No audio found in {video_path}. Adding silent audio track...")
    temp = os.path.join(os.path.dirname(video_path), f"temp_{os.path.basename(video_path)}")

    dur_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path,
    ]
    duration = float(subprocess.check_output(dur_cmd).decode('utf-8').strip())

    subprocess.call([
        'ffmpeg', '-i', video_path,
        '-f', 'lavfi',
        '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}',
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', temp,
    ])
    os.replace(temp, video_path)
    print(f"Silent audio track added to {video_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Video preprocessing
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def preprocess_videos(
    video_dir: str,
    config: CN,
    output_dir: str,
    device: torch.device,
    torch_dtype: any,
    num_workers: int = 2,
    batch_size: int = 1,
) -> List[str]:
    """
    Run the full video preprocessing pipeline on every .mp4 / .avi in video_dir.

    Returns a list of paths to the FPS-normalised segment clips (one per segment),
    which are used downstream to look up feature files and to mux the final video.
    """
    video_paths = sorted(
        glob(os.path.join(video_dir, '*.mp4')) +
        glob(os.path.join(video_dir, '*.avi'))
    )
    assert video_paths, f"No .mp4 / .avi files found in {video_dir}"

    preproc_dir = os.path.join(output_dir, 'preprocess')
    feature_dir = os.path.join(output_dir, 'features')
    os.makedirs(preproc_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    # ── silent-audio guard ───────────────────────────────────────────────────
    for vp in video_paths:
        _add_silent_audio_if_needed(vp)

    # ── Step 1: Video content analysis ───────────────────────────────────────

    descriptions_dir = os.path.join(feature_dir, "descriptions")
    os.makedirs(descriptions_dir, exist_ok=True)

    video_description = VideoDescription(device=device, torch_dtype=torch_dtype)
    for vp in video_paths:
        video_name = os.path.basename(vp).split('.')[0]
        base = video_name.rsplit('_', 1)[0]
        file_path = os.path.join(descriptions_dir, f"{base}_description.txt")

        if not os.path.exists(file_path):
            video_description_text = video_description.describe_video(vp)
            print(f"Description for {video_name} : {video_description_text}")

            with open(file_path, 'w') as f:
                for line in video_description_text:
                    f.write(f"{line}\n")

    # ── Step 2: align video/audio lengths ────────────────────────────────────
    print("Preprocessing: aligning video/audio lengths...")
    with Pool(num_workers) as p:
        for _ in tqdm(
            p.imap_unordered(partial(pipline_align, output_dir=preproc_dir), video_paths),
            total=len(video_paths),
        ):
            pass

    # ── Step 3: build segment IDs and dummy annotation files ─────────────────
    segment_ids = []
    for vp in video_paths:
        video_name = os.path.basename(vp).split('.')[0]
        aligned_audio = os.path.join(preproc_dir, 'audio_ori', f"{video_name}.wav")
        audio, sr = librosa.load(aligned_audio, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        num_segments = int(np.floor(duration / config.data.audio_samples))

        segment_ids.extend([f"{video_name}_{i}_" for i in range(num_segments)])

        # dummy annotation file (onset times only, no material/action data needed)
        base = video_name.rsplit('_', 1)[0]
        with open(os.path.join(preproc_dir, f"{base}_times.txt"), 'w') as f:
            for i in range(num_segments):
                f.write(f"{i * config.data.audio_samples} \n")

    # ── Step 4: cut segments, resample audio, re-encode at target FPS ────────
    print("Preprocessing: cutting segments...")
    with Pool(num_workers) as p:
        for _ in tqdm(
            p.imap_unordered(
                partial(
                    pipline_cut,
                    metadata_dir=preproc_dir,
                    preproc_dir=preproc_dir,
                    output_dir=feature_dir,
                    sr=config.data.audio_sample_rate,
                    fps=config.data.video_fps,
                    duration_target=config.data.audio_samples,
                ),
                segment_ids,
            ),
            total=len(segment_ids),
        ):
            pass

    segment_ids = [s[:-1] for s in segment_ids]  # drop trailing '_'

    # ── Step 5: RAFT optical flow ─────────────────────────────────────────────
    print("Preprocessing: extracting optical flow...")
    of_dir = os.path.join(feature_dir, f"OF_{config.data.audio_samples}s_{config.data.video_fps}fps")
    clip_dir = os.path.join(feature_dir, f"videos_{config.data.audio_samples}s_{config.data.video_fps}fps")
    os.makedirs(of_dir, exist_ok=True)

    processed_video_paths = []
    for seg_id in tqdm(segment_ids, desc="RAFT optical flow"):
        clip_path = os.path.join(clip_dir, f"{seg_id}.mp4")
        flow_output = os.path.join(of_dir, f"{seg_id}.npy")
        if not os.path.exists(flow_output):
            cal_for_frames(
                video_path=clip_path,
                output_dir=of_dir,
                n_frames=int(config.data.video_fps * config.data.audio_samples),
                width=config.data.video_width,
                height=config.data.video_height,
                batch_size=batch_size,
                device=device,
            )
        processed_video_paths.append(clip_path)

    # ── Step 6: BN-Inception feature extraction ───────────────────────────────
    print("Preprocessing: extracting BN-Inception features...")
    file_list_path = os.path.join(feature_dir, 'temp_file_list.txt')
    with open(file_list_path, 'w') as f:
        for seg_id in segment_ids:
            f.write(f"{seg_id}\n")

    for modality in ['RGB', 'Flow']:
        mod_dir = os.path.join(feature_dir, f"feature_{modality}")
        missing = [s for s in segment_ids if not os.path.exists(os.path.join(mod_dir, f"{s}.pkl"))]
        if missing:
            extract_bn_inception_feature(
                input_dir=of_dir,
                output_dir=os.path.join(feature_dir, f"feature_{modality}"),
                modality=modality,
                test_list=file_list_path,
                workers=num_workers,
                device=device,
            )

    return processed_video_paths


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Any-sound generation via envelope modulation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_audio(
    processed_video_paths: List[str],
    config: CN,
    output_dir: str,
    device: torch.device,
    torch_dtype: any,
    text_prompt: Optional[str] = None,
    epoch: int = 500,
    ckpt_dir: str = CKPT_DIR,
) -> None:
    """
    Generate ANY sound timed to the video's motion.

    The ControlNet branch is bypassed (w_control_net_condition=False) so
    generation is driven solely by the CLAP embedding of the user's text/audio
    prompt — which can encode any sound: explosions, violins, barking dogs, …

    The Video2RMS envelope (timing + dynamics from the video) is then applied
    as amplitude modulation on the generated waveform so the sound follows
    the rhythm and intensity of the on-screen motion.
    """

    audio_out = os.path.join(output_dir, 'audio')
    video_out = os.path.join(output_dir, 'video')
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(video_out, exist_ok=True)

    feature_dir = os.path.join(output_dir, 'features')
    rgb_dir  = os.path.join(feature_dir, 'feature_RGB')
    flow_dir = os.path.join(feature_dir, 'feature_Flow')
    descriptions_dir = os.path.join(feature_dir, "descriptions")

    print("Loading models...")
    video2rms_model, audio_ldm_controlnet = load_models(epoch, ckpt_dir, ckpt_dir, config, device, torch_dtype)
    video2rms_model.eval()

    mu_bins = RMS.get_mu_bins(config.data.rms_mu, config.data.rms_num_bins, config.data.rms_min)

    # ── Bypass ControlNet: patch apply_model to ignore the RMS branch ─────────
    # The ControlNet UNet wrapper passes rms=cond_dict['control_net_condition']
    # only when w_control_net_condition=True (the default).  Force it False so
    # the Greatest-Hits-trained ControlNet branch has zero influence on the
    # denoising, leaving only the general base AudioLDM + CLAP conditioning.
    _orig_apply_model = audio_ldm_controlnet.model.apply_model
    audio_ldm_controlnet.model.apply_model = (
        lambda x, t, cond_dict, **kw:
            _orig_apply_model(x, t, cond_dict, w_control_net_condition=False, **kw)
    )

    try:
        for clip_path in tqdm(processed_video_paths, desc="Generating audio"):
            seg_id = os.path.splitext(os.path.basename(clip_path))[0]

            # ── Load & combine features ───────────────────────────────────────
            rgb_feat  = np.load(os.path.join(rgb_dir,  f"{seg_id}.pkl"), allow_pickle=True)
            flow_feat = np.load(os.path.join(flow_dir, f"{seg_id}.pkl"), allow_pickle=True)

            base_video_name = seg_id.rsplit('_', 1)[0]
            base_video_name = base_video_name.rsplit('_', 1)[0]
            with open(os.path.join(descriptions_dir, f"{base_video_name}_description.txt"), 'r') as f:
                video_descr = f.read().strip()

            rgb_feat  = pad_or_truncate_feature(rgb_feat,  config.data.video_samples)
            flow_feat = pad_or_truncate_feature(flow_feat, config.data.video_samples)
            combined  = np.concatenate([rgb_feat, flow_feat], axis=1)
            feat_tensor = torch.from_numpy(combined.astype(np.float32)).unsqueeze(0).to(device)

            # ── Video2RMS forward ─────────────────────────────────────────────
            video2rms_model.parse_batch((
                feat_tensor,
                torch.zeros(1, config.data.rms_samples).to(device),
                None, None,
            ))
            video2rms_model.forward()

            pred_rms_raw    = video2rms_model.pred_rms[0].detach().cpu().numpy()  # (64, 1250)
            pred_bin_tensor = torch.from_numpy(pred_rms_raw.argmax(axis=0))       # (1250,)
            rms_undiscretized = RMS.undiscretize_rms(pred_bin_tensor, mu_bins, ignore_min=True)  # (1250,)

            # rms_for_ldm is still required as a batch key by generate_sample,
            # but the patched apply_model will never forward it to the UNet.
            rms_for_ldm = interpolate_rms_for_rms2sound(
                rms_undiscretized.unsqueeze(0),
                audio_len=config.data.audio_samples,
                sr=config.data.audio_sample_rate,
                frame_len=1024,
                hop_len=160,
            ).to(device)

            parts = []
            if text_prompt:
                parts.append(text_prompt)
            if video_descr:
                parts.append(video_descr)
            combined_prompt = " ".join(parts)

            # ── Generate with base AudioLDM (ControlNet bypassed) ─────────────
            generated_audio_raw = audio_ldm_controlnet.generate(
                waveform=None,
                text_prompt=combined_prompt,
                rms=rms_for_ldm,
            )  # numpy (samples,)

            # ── Apply Video2RMS amplitude envelope ────────────────────────────
            # Interpolate the RMS curve (1250 frames) to audio sample resolution,
            # then multiply to impose the video's impact timing onto the sound.
            rms_np   = rms_undiscretized.numpy()  # (1250,)
            envelope = np.interp(
                np.linspace(0, 1, len(generated_audio_raw)),
                np.linspace(0, 1, len(rms_np)),
                rms_np,
            )
            envelope /= (envelope.max() + 1e-8)  # normalise to [0, 1]
            generated_audio = generated_audio_raw * envelope

            # ── Save results ──────────────────────────────────────────────────
            audio_path = os.path.join(audio_out, f"{seg_id}_generated.wav")
            sf.write(audio_path, generated_audio, config.data.audio_sample_rate)

            src_video = os.path.join(
                feature_dir,
                f"videos_{config.data.audio_samples}s",
                f"{seg_id}.mp4",
            )
            if os.path.exists(src_video):
                video_path = os.path.join(video_out, f"{seg_id}_with_audio.mp4")
                save_video_with_audio(src_video, generated_audio, video_path,
                                      sr=config.data.audio_sample_rate)
    finally:
        # Always restore the original method
        audio_ldm_controlnet.model.apply_model = _orig_apply_model

    del video2rms_model
    del audio_ldm_controlnet
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="FoleyADeux: generate foley audio for a folder of videos"
    )
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Folder containing input .mp4 / .avi files')
    parser.add_argument('-o', '--output-dir', default='./output',
                        help='Output folder (default: ./output)')

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('-p', '--prompt', type=str,
                               help='Text prompt describing the desired sound')

    parser.add_argument('--epoch',       type=int, default=500,
                        help='Video2RMS checkpoint epoch (default: 500)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Parallel workers for preprocessing (default: 2)')
    parser.add_argument('--batch-size',  type=int, default=1,
                        help='Batch size (default: 1)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(os.path.join(CKPT_DIR, 'opts.yml'))
    config.defrost()
    config.data.video_fps    = config.data.video_samples / config.data.audio_samples
    config.data.video_width  = 344
    config.data.video_height = 256
    config.data.training_files = []
    config.freeze()

    device, torch_dtype = create_device()

    processed_video_paths = preprocess_videos(
        video_dir=args.input_dir,
        config=config,
        output_dir=args.output_dir,
        device=device,
        torch_dtype=torch_dtype,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    generate_audio(
        processed_video_paths=processed_video_paths,
        config=config,
        output_dir=args.output_dir,
        device=device,
        torch_dtype=torch_dtype,
        text_prompt=args.prompt,
        epoch=args.epoch,
        ckpt_dir=CKPT_DIR,
    )

    print(f"\nDone. Results saved to {args.output_dir}/")
    print(f"  audio/ — generated .wav files")
    print(f"  video/ — mixed .mp4 files")
