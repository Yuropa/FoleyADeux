import os
from glob import glob
from typing import Tuple

from yacs.config import CfgNode as CN
import yaml
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import moviepy as mp
import moviepy.editor as mpe
from diffusers import AudioLDM2Pipeline

from config import _C as config
from model import Video2Sound


def load_config(config_path: str) -> CN:
    result_config = config.clone()
    result_config.merge_from_file(config_path)
    return result_config


def load_model(epoch:int, ckpt_dir:str, config:CN, device, torch_dtype) -> Video2Sound:
    '''Returns Video2RMS model with loaded checkpoint'''
    model = Video2Sound(config, device, torch_dtype)
    
    checkpoint_path = glob(os.path.join(ckpt_dir, f'checkpoint_{epoch:06d}_*'))
    if len(checkpoint_path) == 0:
        raise ValueError(f"checkpoint not found: {checkpoint_path}")
    checkpoint_path = '_'.join(checkpoint_path[0].split('_')[:-1]) # format: /dirname/checkpoint_06d_modelname
    model.load_checkpoint(checkpoint_path)
    
    return model


def save_video_with_audio(video_path:str, audio: np.ndarray, output_path:str, sr:int=16000) -> None:
    # load video, and mix given audio to silent video
    video = mpe.VideoFileClip(video_path)
    audio_clip = mp.audio.AudioClip.AudioArrayClip(np.expand_dims(audio, axis=1), fps=sr*2)
    video = video.set_audio(audio_clip)
    
    # save video to output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
def load_models(epoch: int, video2rms_ckpt_dir: str, config: CN,
                device: torch.device, torch_dtype: any) -> Tuple[nn.Module, AudioLDM2Pipeline]:
    '''Returns Video2RMS model and AudioLDM2Pipeline'''
    # Check for checkpoint file
    if epoch > -1:
        checkpoint_path = os.path.join(video2rms_ckpt_dir, f'checkpoint_{epoch:06d}_Video2RMS.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    else:
        checkpoint_files = glob.glob(os.path.join(video2rms_ckpt_dir, 'checkpoint_*_Video2RMS.pt'))
        if len(checkpoint_files) > 1:
            raise ValueError("Multiple checkpoint files found. Please specify --epoch.")
        elif len(checkpoint_files) == 1:
            epoch = int(os.path.basename(checkpoint_files[0]).split('_')[1])
            print(f"Using checkpoint from epoch {epoch}")
        else:
            raise FileNotFoundError("No checkpoint files found in the specified directory.")

    # Load Video2RMS model
    video2rms_model = load_model(epoch, video2rms_ckpt_dir, config, device, torch_dtype).to(device)

    # Load AudioLDM 2 pipeline from HuggingFace
    audio_ldm2 = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2",
        torch_dtype=torch_dtype,
    ).to(device)

    return video2rms_model, audio_ldm2