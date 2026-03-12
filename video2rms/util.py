import os
from glob import glob
from typing import Tuple

from yacs.config import CfgNode as CN
import yaml
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import moviepy as mp
import moviepy.editor as mpe

from config import _C as config
from model import Video2Sound
from AudioLDM.AudioLDMControlNet import AudioLDMControlNet


def load_config(config_path: str) -> CN:
    # make a copy of config
    result_config = config.clone()
    with open(config_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    # if log.loss.gls_num_classes exists, make it also in config
    if yaml_cfg.get('log', {}).get('loss', {}).get('gls_num_classes', None) is not None:
        result_config.log.loss.gls_num_classes = yaml_cfg['log']['loss']['gls_num_classes']
    if yaml_cfg.get('log', {}).get('loss', {}).get('gls_blur_range', None) is not None:
        result_config.log.loss.gls_blur_range = yaml_cfg['log']['loss']['gls_blur_range']
    if yaml_cfg.get('train', {}).get('loss', {}).get('gls_num_classes', None) is not None:
        result_config.train.loss.gls_num_classes = yaml_cfg['train']['loss']['gls_num_classes']
    if yaml_cfg.get('train', {}).get('loss', {}).get('gls_blur_range', None) is not None:
        result_config.train.loss.gls_blur_range = yaml_cfg['train']['loss']['gls_blur_range']
    # if data.onset_supervision doesn't exists, make it False in config
    if yaml_cfg.get('train', {}).get('onset_supervision', None) is None:
        assert yaml_cfg.get('data', {}).get('onset_supervision', None) is None, \
            "Mismatch between train.onset_supervision and data.onset_supervision: "\
            + f"{yaml_cfg.get('train', {}).get('onset_supervision', None)}"\
            + f"{yaml_cfg.get('data', {}).get('onset_supervision', None)}"
        result_config.train.onset_supervision = False
        result_config.data.onset_supervision = False
    elif yaml_cfg.get('train', {}).get('onset_supervision', None) is True:
        assert yaml_cfg.get('data', {}).get('onset_supervision', None) is True, \
            "Mismatch between train.onset_supervision and data.onset_supervision: "\
            + f"{yaml_cfg.get('train', {}).get('onset_supervision', None)}"\
            + f"{yaml_cfg.get('data', {}).get('onset_supervision', None)}"
        result_config.train.onset_supervision = yaml_cfg['train']['onset_supervision']
        result_config.train.onset_loss_lambda = yaml_cfg['train']['onset_loss_lambda']
        result_config.data.onset_supervision = yaml_cfg['train']['onset_supervision']
        result_config.data.onset_annotation_dir = yaml_cfg['data']['onset_annotation_dir']
        result_config.data.onset_tolerance = yaml_cfg['data']['onset_tolerance']
        
    result_config.merge_from_file(config_path)
    return result_config


def load_model(epoch:int, ckpt_dir:str, config:CN, device) -> Video2Sound:
    '''Returns Video2RMS model with loaded checkpoint'''
    model = Video2Sound(config, device)
    
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
    
def interpolate_rms_for_rms2sound(rms:torch.tensor, audio_len:int=10, sr:int=16000,
                                  audioldm_samples:int=int(16000*10.24),
                               frame_len:int=1024, hop_len:int=160) -> torch.tensor:
    # get target length by calculating length with one values
    _dummy_audio = np.pad(np.zeros(audio_len*sr),
                    (int((frame_len - hop_len) / 2), int((frame_len - hop_len) / 2), ), 
                    mode="reflect")
    target_rms_len = int(librosa.feature.rms(y=_dummy_audio, \
                                            frame_length=frame_len, hop_length=hop_len, \
                                            center=False, pad_mode="reflect").shape[1])
    # interpolate rms to target_rms_shape (rms.shape: (frames))
    interpolated_rms = torch.nn.functional.interpolate(rms.unsqueeze(0).unsqueeze(0),
                                                         size=(1, target_rms_len), 
                                                         mode='nearest')
    interpolated_rms = interpolated_rms.squeeze(0).squeeze(0)
    
    # get rms length for AudioLDM
    _dummy_audio_audioldm = np.pad(np.zeros(audioldm_samples),
                    (int((frame_len - hop_len) / 2), int((frame_len - hop_len) / 2), ), 
                    mode="reflect")
    rms_len_audioldm = int(librosa.feature.rms(y=_dummy_audio_audioldm, \
                                            frame_length=frame_len, hop_length=hop_len, \
                                            center=False, pad_mode="reflect").shape[1])
    
    # pad zeros at right
    interpolated_rms = torch.nn.functional.pad(interpolated_rms, (0, rms_len_audioldm - target_rms_len))
    
    return interpolated_rms


def load_models(epoch: int, video2rms_ckpt_dir: str, rms2sound_ckpt_dir: str, config: CN, 
                device: torch.device) -> Tuple[nn.Module, AudioLDMControlNet]:
    '''Returns Video2RMS model and AudioLDMControlNet model with loaded checkpoint'''
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
    
    if not glob(os.path.join(rms2sound_ckpt_dir, 'ControlNetstep*.pth')):
        raise FileNotFoundError(f"No ControlNetstep*.pth file found in {rms2sound_ckpt_dir}")
    
    # Load Video2RMS model
    video2rms_model = load_model(epoch, video2rms_ckpt_dir, config, device).to(device)

    # Load AudioLDMControlNet model
    audio_ldm_controlnet = AudioLDMControlNet(
        control_net_pretrained_path = os.path.join(rms2sound_ckpt_dir, 'ControlNetstep300000.pth'),
        device = device
    )

    return video2rms_model, audio_ldm_controlnet