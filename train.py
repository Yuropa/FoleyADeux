import torch
from utils.utils import create_device
from utils.MOSSSoundEffect import MOSSSoundEffectModel

device = create_device()
sound_effect_model = MOSSSoundEffectModel(device)

sound_effect_model.save_audio('A ball boucning on the ground', './audio_output')