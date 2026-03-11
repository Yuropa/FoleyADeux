from typing import Literal
import torch

from audioldm_train.modules.latent_diffusion.ddim import DDIMSampler
from audioldm_train.utilities.diffusion_util import noise_like

class DDIMSamplerForControlNet(DDIMSampler):
    def __init__(self, cfg_of_control_net:Literal['WOCFG', 'CFG'], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg_of_control_net:Literal['WOCFG', 'CFG'] = cfg_of_control_net
      
    def cfg_apply_model(self, x_in, t_in, c, unconditional_conditioning, unconditional_guidance_scale):
        if self.cfg_of_control_net == 'WOCFG':
            model_uncond = self.model.apply_model( x_in, t_in, unconditional_conditioning )
        elif self.cfg_of_control_net == 'CFG':
            model_uncond = self.model.apply_model( x_in, t_in, unconditional_conditioning, w_control_net_condition = False )
        model_t = self.model.apply_model(x_in, t_in, c)
        model_output = model_uncond + unconditional_guidance_scale * ( model_t - model_uncond )
        return model_output
    
    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
    ):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = x
            t_in = t

            assert isinstance(c, dict)
            assert isinstance(unconditional_conditioning, dict)

            model_output = self.cfg_apply_model(x_in, t_in, c, unconditional_conditioning, unconditional_guidance_scale)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", "not implemented"
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs
            )

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0