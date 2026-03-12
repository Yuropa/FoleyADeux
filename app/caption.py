"""
app/caption.py
──────────────
Auto-caption stub and theme-description helper.

The auto_caption function is currently a placeholder. To enable it, replace
the TODO block with a real VLM call — e.g. BLIP-2, LLaVA, or GPT-4V:

    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    _processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    _model     = Blip2ForConditionalGeneration.from_pretrained(…)

    def auto_caption(video_path):
        frames  = _sample_frames(video_path, n=8)
        inputs  = _processor(images=frames, text="Describe the sounds:", …)
        out_ids = _model.generate(**inputs)
        caption = _processor.decode(out_ids[0], skip_special_tokens=True)
        return gr.update(value=caption)
"""

from typing import Optional

import gradio as gr

from .config import THEME_DESCRIPTIONS


def update_theme_info(theme: str) -> str:
    """Return the human-readable description for the selected theme dropdown value."""
    return THEME_DESCRIPTIONS.get(theme, "")


def auto_caption(video_path: Optional[str]) -> dict:
    """
    Analyse the uploaded video and populate the prompt field with a
    suggested sound description.

    Currently returns a placeholder hint. When a VLM is integrated,
    this function should:
      1. Sample N evenly-spaced frames from ``video_path``.
      2. Pass them (+ an instruction prompt) to the VLM.
      3. Return ``gr.update(value=<generated caption>)`` so Gradio
         injects the result directly into the prompt Textbox.
    """
    if video_path is None:
        return gr.update(value="Upload a video first, then click Auto-caption.")

    # TODO: replace with real VLM inference
    return gr.update(
        value=(
            "[Auto-caption] Describe the sounds you want — e.g. "
            '"footsteps on gravel, distant thunder, rustling leaves"'
        )
    )
