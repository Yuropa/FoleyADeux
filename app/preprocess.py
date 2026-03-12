"""
app/preprocess.py
─────────────────
Video preprocessing pipeline: silence guard → align → segment → optical flow
→ BN-Inception features.

This module was previously housed in inference.py.  Moving it here lets
inference.py (the standalone CLI script) be removed if desired, since the
app/ package is now fully self-contained.

Public API
----------
preprocess_videos(video_dir, config, output_dir, device, num_workers, batch_size)
    Run the full preprocessing pipeline on every .mp4 / .avi in video_dir.
    Returns a list of paths to the FPS-normalised segment clips.

_add_silent_audio_if_needed(video_path)
    Internal guard: adds a silent audio track via FFmpeg if the video has none.
"""

import os
import subprocess
from functools import partial
from glob import glob
from multiprocessing import Pool
from typing import List

import librosa
import numpy as np
import torch
from tqdm import tqdm
from yacs.config import CfgNode as CN

# config.py sets sys.path, so these repo-local imports are safe
from .config import ROOT_DIR  # noqa: F401 — triggers sys.path bootstrap

from preprocess.extract_audio_and_video import pipline_align, pipline_cut
from preprocess.extract_rgb_flow_raft import cal_for_frames
from preprocess.extract_feature import extract_bn_inception_feature


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_silent_audio_if_needed(video_path: str) -> None:
    """Add a silent stereo audio track if the video has no audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0", "-count_packets",
        "-show_entries", "stream=codec_type,nb_read_packets",
        "-of", "csv=p=0", video_path,
    ]
    output = subprocess.check_output(cmd).decode("utf-8").strip()
    if not (output == "" or output.endswith(",0")):
        return  # already has audio

    print(f"No audio found in {video_path}. Adding silent audio track...")
    temp = os.path.join(
        os.path.dirname(video_path), f"temp_{os.path.basename(video_path)}"
    )

    dur_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    duration = float(subprocess.check_output(dur_cmd).decode("utf-8").strip())

    subprocess.call([
        "ffmpeg", "-i", video_path,
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}",
        "-c:v", "copy", "-c:a", "aac", "-shortest", temp,
    ])
    os.replace(temp, video_path)
    print(f"Silent audio track added to {video_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

@torch.no_grad()
def preprocess_videos(
    video_dir: str,
    config: CN,
    output_dir: str,
    device: str,
    dtype: any,
    num_workers: int = 2,
    batch_size: int = 1,
) -> List[str]:
    """
    Run the full video preprocessing pipeline on every .mp4 / .avi in video_dir.

    Steps
    -----
    1. Silence guard  — ensures every clip has an audio stream.
    2. Align          — trims audio/video to the same duration.
    3. Segment + cut  — chops aligned clips into fixed-length segments,
                        resamples audio, and re-encodes at the target FPS.
    4. RAFT           — computes dense optical-flow frames per segment.
    5. BN-Inception   — extracts RGB and Flow appearance features.

    Returns
    -------
    List of absolute paths to the FPS-normalised segment clips (one per
    segment), used downstream for feature lookup and final video muxing.
    """
    video_paths = sorted(
        glob(os.path.join(video_dir, "*.mp4")) +
        glob(os.path.join(video_dir, "*.avi"))
    )
    assert video_paths, f"No .mp4 / .avi files found in {video_dir}"

    preproc_dir = os.path.join(output_dir, "preprocess")
    feature_dir = os.path.join(output_dir, "features")
    os.makedirs(preproc_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    # Step 0: silent-audio guard
    for vp in video_paths:
        _add_silent_audio_if_needed(vp)

    # Step 1: align video/audio lengths
    print("Preprocessing: aligning video/audio lengths...")
    with Pool(num_workers) as p:
        for _ in tqdm(
            p.imap_unordered(partial(pipline_align, output_dir=preproc_dir), video_paths),
            total=len(video_paths),
        ):
            pass

    # Step 2: build segment IDs and dummy annotation files
    segment_ids = []
    for vp in video_paths:
        video_name    = os.path.basename(vp).split(".")[0]
        aligned_audio = os.path.join(preproc_dir, "audio_ori", f"{video_name}.wav")
        audio, sr     = librosa.load(aligned_audio, sr=None)
        duration      = librosa.get_duration(y=audio, sr=sr)
        num_segments  = int(np.floor(duration / config.data.audio_samples))

        segment_ids.extend([f"{video_name}_{i}_" for i in range(num_segments)])

        # Dummy annotation file with onset times only
        base = video_name.rsplit("_", 1)[0]
        with open(os.path.join(preproc_dir, f"{base}_times.txt"), "w") as f:
            for i in range(num_segments):
                f.write(f"{i * config.data.audio_samples} \n")

    # Step 3: cut segments, resample audio, re-encode at target FPS
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

    segment_ids = [s[:-1] for s in segment_ids]  # drop trailing "_"

    # Step 4: RAFT optical flow
    print("Preprocessing: extracting optical flow...")
    of_dir   = os.path.join(
        feature_dir,
        f"OF_{config.data.audio_samples}s_{config.data.video_fps}fps",
    )
    clip_dir = os.path.join(
        feature_dir,
        f"videos_{config.data.audio_samples}s_{config.data.video_fps}fps",
    )
    os.makedirs(of_dir, exist_ok=True)

    processed_video_paths = []
    for seg_id in tqdm(segment_ids, desc="RAFT optical flow"):
        clip_path = os.path.join(clip_dir, f"{seg_id}.mp4")
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

    # Step 5: BN-Inception feature extraction
    print("Preprocessing: extracting BN-Inception features...")
    file_list_path = os.path.join(feature_dir, "temp_file_list.txt")
    with open(file_list_path, "w") as f:
        for seg_id in segment_ids:
            f.write(f"{seg_id}\n")

    for modality in ["RGB", "Flow"]:
        extract_bn_inception_feature(
            input_dir=of_dir,
            output_dir=os.path.join(feature_dir, f"feature_{modality}"),
            modality=modality,
            test_list=file_list_path,
            workers=num_workers,
            device=device,
            dtype=dtype
        )

    return processed_video_paths
