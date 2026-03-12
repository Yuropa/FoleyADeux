from typing import Literal
from audioldm_train.modules.latent_diffusion.ddpm import LatentDiffusion
from audioldm_train.conditional_models import *
from AudioLDM.Model.ControlNet.DDIMSamplerForControlNet import DDIMSamplerForControlNet

class LatentDiffusionControlRMS(LatentDiffusion):
    def __init__(self, 
                 cfg_of_control_net:Literal['WOCFG', 'CFG'] = 'WOCFG',
                 *args, 
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.cond_stage_models[0].embed_mode = 'audio'
        self.cfg_of_control_net:Literal['WOCFG', 'CFG'] = cfg_of_control_net
    
    def random_clap_condition(self):
        # This function is only used during training, let the CLAP model to use both text and audio as condition
        assert self.training == True

        for key in self.cond_stage_model_metadata.keys():
            metadata = self.cond_stage_model_metadata[key]
            model_idx, cond_stage_key, conditioning_key = ( metadata["model_idx"], metadata["cond_stage_key"], metadata["conditioning_key"],)

            # If we use CLAP as condition, we might use audio for training, but we also must use text for evaluation
            if isinstance( self.cond_stage_models[model_idx], CLAPAudioEmbeddingClassifierFreev2 ):
                self.cond_stage_model_metadata[key][
                    "cond_stage_key_orig"
                ] = self.cond_stage_model_metadata[key]["cond_stage_key"]
                self.cond_stage_model_metadata[key][
                    "embed_mode_orig"
                ] = self.cond_stage_models[model_idx].embed_mode
                self.cond_stage_model_metadata[key]["cond_stage_key"] = "waveform"
                self.cond_stage_models[model_idx].embed_mode = "audio"
    
    def get_input(
        self,
        batch,
        k,
        return_decoding_output=False,
        unconditional_prob_cfg=0.1,
        is_no_text:bool = True
    ):  
        
        ret = {
            'waveform': None if batch["waveform"] is None else batch["waveform"].to(memory_format=torch.contiguous_format).float(),
        }

        if 'text' in batch: ret['text'] = list(batch["text"])

        for key in batch.keys():
            if key not in ret.keys(): ret[key] = batch[key]
            
        z = None

        cond_dict = {}
        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False
            if self.conditional_dry_run_finished and self.make_decision(unconditional_prob_cfg):
                unconditional_cfg = True
            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key]["cond_stage_key"]

                if cond_model_key in cond_dict.keys():
                    continue

                if not self.training:
                    if isinstance( self.cond_stage_models[ self.cond_stage_model_metadata[cond_model_key]["model_idx"]],CLAPAudioEmbeddingClassifierFreev2,):
                        print( "Warning: CLAP model normally should use text for evaluation" )

                # The original data for conditioning
                # If cond_model_key is "all", that means the conditional model need all the information from a batch
                
                if is_no_text and cond_stage_key == "text": cond_stage_key = 'waveform'
                if cond_stage_key != "all":
                    xc = ret[cond_stage_key] # text is not in the batch
                    if type(xc) == torch.Tensor:
                        xc = xc.to(self.device)
                else:
                    xc = batch

                # if cond_stage_key is "all", xc will be a dictionary containing all keys
                # Otherwise xc will be an entry of the dictionary
                c = self.get_learned_conditioning(
                    xc, key=cond_model_key, unconditional_cfg=unconditional_cfg
                )

                # cond_dict will be used to condition the diffusion model
                # If one conditional model return multiple conditioning signal
                if isinstance(c, dict):
                    for k in c.keys():
                        cond_dict[k] = c[k]
                else:
                    cond_dict[cond_model_key] = c

        cond_dict['control_net_condition'] = batch['control_net_condition']

        out = [z, cond_dict]

        if return_decoding_output:
            xrec = self.decode_first_stage(z)
            out += [xrec]

        if not self.conditional_dry_run_finished:
            self.conditional_dry_run_finished = True

        # Output is a dictionary, where the value could only be tensor or tuple
        return out
    
    def filter_useful_cond_dict(self, cond_dict):
        return cond_dict
    
    def apply_model(self, x, t, cond_dict, w_control_net_condition:bool = True, return_ids=False):
        x = x.contiguous()
        t = t.contiguous()

        # x with condition (or maybe not)
        xc = x

        y = None
        context_list, attn_mask_list = [], []

        conditional_keys = cond_dict.keys()

        for key in conditional_keys:
            if "concat" in key:
                xc = torch.cat([x, cond_dict[key].unsqueeze(1)], dim=1)
            elif "film" in key:
                if y is None:
                    y = cond_dict[key].squeeze(1)
                else:
                    y = torch.cat([y, cond_dict[key].squeeze(1)], dim=-1)
            elif "crossattn" in key:
                # assert context is None, "You can only have one context matrix, got %s" % (cond_dict.keys())
                if isinstance(cond_dict[key], dict):
                    for k in cond_dict[key].keys():
                        if "crossattn" in k:
                            context, attn_mask = cond_dict[key][
                                k
                            ]  # crossattn_audiomae_pooled: torch.Size([12, 128, 768])
                else:
                    assert len(cond_dict[key]) == 2, (
                        "The context condition for %s you returned should have two element, one context one mask"
                        % (key)
                    )
                    context, attn_mask = cond_dict[key]

                # The input to the UNet model is a list of context matrix
                context_list.append(context)
                attn_mask_list.append(attn_mask)

            elif (
                "noncond" in key
            ):  # If you use loss function in the conditional module, include the keyword "noncond" in the return dictionary
                continue
            else:
                continue

        out = self.model.diffusion_model( xc, t, rms = cond_dict['control_net_condition'] if w_control_net_condition else None, context_list=context_list, y=y, context_attn_mask_list=attn_mask_list)
        x_recon =  out

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
    
    @torch.no_grad()
    def generate_sample(
        self,
        batchs,
        device,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        limit_num=None,
        clap_embed_mode:Literal['text', 'audio'] = 'text',
    ):
        # Generate n_gen times and select the best
        # Batch: audio, text, fnames
        initial_clap_embed_mode:str = self.cond_stage_models[0].embed_mode
        self.cond_stage_models[0].embed_mode = clap_embed_mode
        assert x_T is None

        if use_plms: assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_list = list()

        #with self.ema_scope("Plotting"):
        for i, batch in enumerate(batchs):
            z, c = self.get_input(
                batch,
                self.first_stage_key,
                unconditional_prob_cfg=0.0,  # Do not output unconditional information in the c
                is_no_text = clap_embed_mode == 'audio'
            )

            if limit_num is not None and i * z.size(0) > limit_num:
                break

            c = self.filter_useful_cond_dict(c)

            text = list(batch["text"]) if 'text' in batch else list()

            # Generate multiple samples
            batch_size = 1 * n_gen

            # Generate multiple samples at a time and filter out the best
            # The condition to the diffusion wrapper can have many format
            for cond_key in c.keys():
                if 'control_net' in cond_key: continue
                if isinstance(c[cond_key], list):
                    for i in range(len(c[cond_key])):
                        c[cond_key][i] = torch.cat([c[cond_key][i]] * n_gen, dim=0)
                elif isinstance(c[cond_key], dict):
                    for k in c[cond_key].keys():
                        c[cond_key][k] = torch.cat([c[cond_key][k]] * n_gen, dim=0)
                else:
                    c[cond_key] = torch.cat([c[cond_key]] * n_gen, dim=0)

            text = text * n_gen

            if unconditional_guidance_scale != 1.0:
                unconditional_conditioning = {}
                for key in self.cond_stage_model_metadata:
                    model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                    unconditional_conditioning[key] = self.cond_stage_models[
                        model_idx
                    ].get_unconditional_condition(batch_size)

            unconditional_conditioning['control_net_condition'] = batch['control_net_condition']

            samples, _ = self.sample_log(
                cond=c,
                batch_size=batch_size,
                x_T=x_T,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                device=device,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                use_plms=use_plms,
            )

            mel = self.decode_first_stage(samples)

            waveform = self.mel_spectrogram_to_waveform( mel, savepath=None, bs=None, name='fnames', save=False)

            if n_gen > 1:
                try:
                    best_index = []
                    similarity = self.clap.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), text
                    )
                    for i in range(z.shape[0]):
                        candidates = similarity[i :: z.shape[0]]
                        max_index = torch.argmax(candidates).item()
                        best_index.append(i + max_index * z.shape[0])

                    waveform = waveform[best_index]

                    print("Similarity between generated audio and text", similarity)
                    print("Choose the following indexes:", best_index)
                except Exception as e:
                    print("Warning: while calculating CLAP score (not fatal), ", e)

            waveform_list = waveform_list + self.get_waveform_list(waveform)
        
        self.cond_stage_models[0].embed_mode = initial_clap_embed_mode
        return waveform_list
    
    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        device,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        mask=None,
        **kwargs,
    ):
        if mask is not None:
            shape = (self.channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)

        intermediate = None
        if ddim:
            print("Use ddim sampler")

            ddim_sampler = DDIMSamplerForControlNet(cfg_of_control_net=self.cfg_of_control_net, model = self, device=device)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                **kwargs,
            )
        else:
            print("Use DDPM sampler")
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        return samples, intermediate