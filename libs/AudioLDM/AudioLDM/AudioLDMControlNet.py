from typing import Optional

import os
import yaml
import torch
import numpy

from AudioLDM.Model.ControlNet.LatentDiffusionControlRMS import LatentDiffusionControlRMS

class AudioLDMControlNet:
    def __init__(self,
                 control_net_pretrained_path:str,
                 device:torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 ) -> None:
        self.device = device
        self.model = self.get_model()
        self.pretrained_load(control_net_pretrained_path)
        
    def get_model(self) -> None:
        config_yaml_path:str = f"{os.path.dirname(os.path.abspath(__file__))}/Config/audioldm_original.yaml"
        vocoder_config_path:str = f"{os.path.dirname(os.path.abspath(__file__))}/Config/hifigan_16k_64bins"

        configs:dict = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)
        configs['model']['params']['first_stage_config']['params']['reload_from_ckpt'] = None
        configs['model']['params']['first_stage_config']['params']['vocoder_path'] = vocoder_config_path
        if "precision" in configs.keys(): torch.set_float32_matmul_precision( configs["precision"] )  # highest, high, medium

        model_args_dict = configs["model"].get("params", dict())
        model_args_dict['unet_config']['target'] = 'AudioLDM.Model.ControlNet.UNetWControlNetRMS.UNetWControlNetRMS'
        latent_diffusion = LatentDiffusionControlRMS(**model_args_dict)
        return latent_diffusion
    
    def pretrained_load(self,pretrain_path:str) -> None:
        if pretrain_path is None:
            return
        pretrained_load:dict = torch.load(pretrain_path,map_location='cpu')
        unknown_keys = [ key for key in self.model.state_dict().keys() if key not in pretrained_load]
        for key in unknown_keys:
            pretrained_load[key] = self.model.state_dict()[key]

        self.model.load_state_dict(pretrained_load)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(self,
                 waveform:Optional[torch.tensor] = None, # [1, time]. for timbre.
                 text_prompt:Optional[str] = None,
                 rms:torch.tensor = None, #[1, time/hop]. hop = 160 (16000 * 10.24) / 160 = 1024
                 ) -> numpy.ndarray: #[time]
        assert waveform is not None or text_prompt is not None, 'waveform or text_prompt is needed for timbre'
        generated_audio = self.model.generate_sample(
            [{
                'waveform': None if waveform is None else waveform.unsqueeze(1).to(self.device),
                'text': [text_prompt],
                'control_net_condition': rms.unsqueeze(1).unsqueeze(-1).to(self.device).float()
            }],
            device = self.device,
            unconditional_guidance_scale=3.5, 
            ddim_steps=200,
            clap_embed_mode = 'text' if text_prompt is not None else 'audio'
        )[0]
        return generated_audio
    