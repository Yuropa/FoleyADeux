"""
app/caption.py
──────────────
Auto-caption using SmolVLM2 (HuggingFaceTB/SmolVLM2-2.2B-Instruct) and
theme-description helper.

The model is loaded lazily on the first Auto-caption click so start-up is
instant and the ~4 GB weights are only pulled when actually needed.
"""

import os
from pathlib import Path
from typing import Optional

import gradio as gr
import torch

from .config import THEME_DESCRIPTIONS, create_device

os.environ.setdefault("VIDEO_BACKEND", "torchvision")

# ── Lazy singletons ───────────────────────────────────────────────────────────────
_model     = None
_processor = None
_device    = None

_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Instruct"


def _load_model():
    global _model, _processor, _device
    if _model is None:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        _device, _torch_dtype     = create_device()
        _model     = AutoModelForImageTextToText.from_pretrained(
            _MODEL_ID, torch_dtype=_torch_dtype, device_map=_device
        )
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
    return _model, _processor, _device


# ── Public API ───────────────────────────────────────────────────────────────────

def update_theme_info(theme: str) -> str:
    """Return the human-readable description for the selected theme dropdown value."""
    return THEME_DESCRIPTIONS.get(theme, "")


def auto_caption(video_path: Optional[str]) -> dict:
    """
    Analyse the uploaded video with SmolVLM2 and populate the prompt field
    with a suggested sound description.
    """
    if video_path is None:
        return gr.update(value="Upload a video first, then click Auto-caption.")

    try:
        model, processor, device = _load_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(Path(video_path).resolve())},
                    {
                        "type": "text",
                        "text": (
                            "Describe only the sounds that would be heard in this video. "
                            "Be concise and specific, suitable as a foley sound prompt. "
                            'Example: "footsteps on gravel, distant thunder, rustling leaves"'
                        ),
                    },
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=64)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        caption = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return gr.update(value=caption)

    except Exception as exc:
        return gr.update(value=f"Auto-caption failed: {exc}")
