from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import TextStreamer
import torch
import os

os.environ["VIDEO_BACKEND"] = "torchvision"

class VideoDescription:
    def __init__(self, device, torch_dtype):
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct", 
            torch_dtype=torch_dtype, 
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        )

    def _create_messages(self, video_path):
        path = Path(video_path).resolve()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "path": str(path),
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        return messages

    def describe_video(self, video_path):
        print(f"Generating descriptions for video {video_path}...")
        messages = self._create_messages(video_path=video_path)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        streamer = TextStreamer(self.processor.tokenizer, skip_special_tokens=True)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128, streamer=streamer)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text