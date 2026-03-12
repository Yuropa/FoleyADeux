import yaml
import torch
import torch.nn as nn
from audioldm_train.utilities.diffusion_util import ( conv_nd, zero_module, timestep_embedding,)
from AudioLDM.Model.ControlNet.ControlNetForAudioLDM import ControlNetForAudioLDM
from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel, timestep_embedding

class UNetWControlNetRMS(UNetModel):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        extra_sa_layer=True,
        num_classes=None,
        extra_film_condition_dim=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
    ) -> None:
        super().__init__(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, dims, extra_sa_layer, num_classes, extra_film_condition_dim, use_checkpoint, use_fp16, num_heads, num_head_channels, num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order, use_spatial_transformer, transformer_depth, context_dim, n_embed, legacy)
        self.control_net = ControlNetForAudioLDM(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, dims, extra_sa_layer, num_classes, extra_film_condition_dim, use_checkpoint, use_fp16, num_heads, num_head_channels, num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order, use_spatial_transformer, transformer_depth, context_dim, n_embed, legacy)
        self.control_net_input_projection_list = nn.ModuleList([
            nn.Conv2d(1, 8, kernel_size=(3,1),stride=(4,1)),
            zero_module(conv_nd(2, 8, 8, 1, padding=0))
        ])
        
    
    def forward(
        self,
        x,
        timesteps=None,
        rms = None,
        y=None, #clap condition
        context_list=None,
        context_attn_mask_list=None,
        **kwargs,
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional. an [N, extra_film_condition_dim] Tensor if film-embed conditional
        :return: an [N x C x ...] Tensor of outputs.
        """
        if rms is not None:
            for control_net_input_projection in self.control_net_input_projection_list:
                rms = control_net_input_projection(rms)
        
            control_net_output_list = self.control_net(x + rms, timesteps = timesteps, y = y, context_list = context_list, context_attn_mask_list = context_attn_mask_list)

        if not self.shape_reported:
            print("The shape of UNet input is", x.size())
            self.shape_reported = True

        assert (y is not None) == (
            self.num_classes is not None or self.extra_film_condition_dim is not None
        ), "must specify y if and only if the model is class-conditional or film embedding conditional"
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

            if self.use_extra_film_by_concat:
                emb = torch.cat([emb, self.film_emb(y)], dim=-1)

            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context_list, context_attn_mask_list)
                hs.append(h)
            h = self.middle_block(h, emb, context_list, context_attn_mask_list)

        if rms is not None:
            h = h + control_net_output_list.pop()

        for module in self.output_blocks:
            concate_tensor = hs.pop() 
            if rms is not None:
                concate_tensor = concate_tensor + control_net_output_list.pop()
            h = torch.cat([h, concate_tensor], dim=1)
            h = module(h, emb, context_list, context_attn_mask_list)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

if __name__ == '__main__':
    config_yaml_path:str = './Model/AudioLdm/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml'
    configs = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)
    control_net = AudioLdmWControlNetRMS(**configs['model']['params']['unet_config']['params'])
    control_net(
        x = torch.rand(4, 8, 256, 16), # mel spec shape is 4, 1, 1024, 64
        rms = torch.rand(4, 1, 1024, 1),
        timesteps = torch.tensor([541,  97, 191, 172]),
        y= torch.rand(4, 512),
        context_list=list(),
        context_attn_mask_list=list(),
    )