from pathlib import Path
import importlib.util
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
from typing import Union, List

class MOSSSoundEffectModel:
    def __init__(self, device):
        self.device = device
        self.model_name = "OpenMOSS-Team/MOSS-SoundEffect"
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.attn_implementation = MOSSSoundEffectModel.preferred_attn_implementation(self.device, self.dtype)

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(device)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
        ).to(device)

    def _convert_to_list(self, prompts):
        if isinstance(prompts, str):
            return [prompts]
        elif isinstance(prompts, list):
            return [str(p) for p in prompts]
        else:
            raise TypeError(f"Expected str or list of str, got {type(prompts)}")


    def generate_audio(self, prompts):
        self.model.eval()

        prompts = self._convert_to_list(prompts)
        conversations = [[self.processor.build_user_message(ambient_sound=p)] for p in prompts]

        batch_size = 1

        result = []
        with torch.no_grad():
            for start in range(0, len(conversations), batch_size):
                batch_conversations = conversations[start : start + batch_size]
                batch = self.processor(batch_conversations, mode="generation")
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=4096,
                )

                for message in self.processor.decode(outputs):
                    audio = message.audio_codes_list[0]
                    result.append(message)

        return result

    def save_audio(self, prompts, path):
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True, parents=True)

        audio_result = self.generate_audio(prompts)

        for (idx, audio) in enumerate(audio_result):
            out_path = save_dir / f"sample{idx}.wav"
            torchaudio.save(out_path, audio.unsqueeze(0), self.processor.model_config.sampling_rate)

    @classmethod
    def preferred_attn_implementation(cls, device, dtype) -> str:
        # Prefer FlashAttention 2 when package + device conditions are met.
        if (
            device == "cuda"
            and importlib.util.find_spec("flash_attn") is not None
            and dtype in {torch.float16, torch.bfloat16}
        ):
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"

        # CUDA fallback: use PyTorch SDPA kernels.
        if device == "cuda":
            return "sdpa"
        if device == "mps":
            return "sdpa"

        # CPU fallback.
        return "eager"
