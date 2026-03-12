"""
app/ui.py
─────────
Gradio Blocks layout, custom CSS, and event wiring.

All rendering logic lives here; heavy computation is delegated to
pipeline.py and caption.py so this file stays easy to read and restyle.

To launch the app, call build_ui() and then .launch() on the returned
gr.Blocks instance (see app/__main__.py).
"""

import gradio as gr

import glob as _glob
import os

from .config import THEMES, EXAMPLES_DIR
from .pipeline import run_inference
from .caption import auto_caption, update_theme_info


# ── Custom CSS ────────────────────────────────────────────────────────────────

CSS = """
/* ── Reset / base ────────────────────────────────────────────── */
body, .gradio-container {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Header ───────────────────────────────────────────────────── */
#foley-header {
    text-align: center;
    padding: 28px 0 12px;
}
#foley-header h1 {
    font-size: 2.4em;
    font-weight: 800;
    background: linear-gradient(90deg, #58a6ff 0%, #3fb950 50%, #f78166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
#foley-header p {
    color: #8b949e;
    font-size: 1.0em;
    margin: 0;
}

/* ── Section headings ─────────────────────────────────────────── */
.section-label {
    color: #f0f6fc !important;
    font-size: 1.05em !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-bottom: 4px;
}

/* ── Panel containers ─────────────────────────────────────────── */
.gr-group, .gr-box {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 12px !important;
}

/* ── Text inputs ──────────────────────────────────────────────── */
.gr-input, .gr-textarea, textarea, input[type=text] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-size: 0.95em !important;
}
.gr-input:focus, .gr-textarea:focus, textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
    outline: none !important;
}

/* ── Labels ────────────────────────────────────────────────────── */
label, .gr-label, .label-wrap span {
    color: #8b949e !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Primary (Generate) button ────────────────────────────────── */
.generate-btn {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.0em !important;
    border-radius: 8px !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.generate-btn:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(46,160,67,0.45) !important;
}

/* ── Secondary (Auto-caption) button ──────────────────────────── */
.autocap-btn {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
    font-size: 0.9em !important;
    transition: background 0.15s !important;
}
.autocap-btn:hover { background: #30363d !important; }

/* ── Theme description blurb ──────────────────────────────────── */
#theme-info {
    font-size: 12px !important;
    color: #8b949e !important;
    min-height: 20px;
    padding: 2px 0;
}

/* ── Status / result bar ──────────────────────────────────────── */
#status-box {
    padding: 10px 14px;
    border-radius: 8px;
    background: #161b22;
    border: 1px solid #21262d;
    font-size: 13px;
    min-height: 40px;
}

/* ── Tab navigation ───────────────────────────────────────────── */
.tab-nav button {
    color: #8b949e !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.9em !important;
    font-weight: 500 !important;
    transition: color 0.15s !important;
}
.tab-nav button.selected {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
}

/* ── Plot panels ──────────────────────────────────────────────── */
.gr-plot { border-radius: 8px !important; overflow: hidden !important; }
"""


# ── Layout ────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """Construct and return the fully wired Gradio Blocks application."""

    with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="FoleyADeux") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="foley-header">
          <h1>🎬 FoleyADeux</h1>
          <p>AI-powered foley sound generation — synchronised to on-screen motion.</p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ════ LEFT COLUMN — inputs and controls ═══════════════════════
            with gr.Column(scale=1, min_width=340):

                gr.Markdown("### Input", elem_classes=["section-label"])

                video_input = gr.Video(
                    label="Upload Video (.mp4 / .avi)",
                    sources=["upload"],
                    height=260,
                )

                # ── Example videos ────────────────────────────────────────
                _example_videos = sorted(
                    _glob.glob(os.path.join(EXAMPLES_DIR, "*.mp4"))
                    + _glob.glob(os.path.join(EXAMPLES_DIR, "*.avi"))
                )
                if _example_videos:
                    gr.Examples(
                        examples=[[v] for v in _example_videos],
                        inputs=video_input,
                        label="Examples — click to load",
                        examples_per_page=6,
                    )

                auto_btn = gr.Button(
                    "✨ Auto-caption  (coming soon)",
                    variant="secondary",
                    elem_classes=["autocap-btn"],
                )

                prompt_input = gr.Textbox(
                    label="Sound Prompt",
                    placeholder=(
                        'e.g. "a dog barking", '
                        '"orchestral strings", '
                        '"glass shattering on marble"'
                    ),
                    lines=2,
                )

                theme_dropdown = gr.Dropdown(
                    choices=list(THEMES.keys()),
                    value="None",
                    label="Theme",
                    info="Prepends a style descriptor to your prompt.",
                )

                theme_info = gr.Markdown(
                    "No theme — use your prompt exactly as typed.",
                    elem_id="theme-info",
                )

                generate_btn = gr.Button(
                    "🎵  Generate Foley",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"],
                )

                status_box = gr.Markdown(
                    "Ready. Upload a video and enter a prompt.",
                    elem_id="status-box",
                )

            # ════ RIGHT COLUMN — outputs ═══════════════════════════════════
            with gr.Column(scale=2):

                gr.Markdown("### Output", elem_classes=["section-label"])

                video_output = gr.Video(
                    label="Foley Video",
                    autoplay=True,
                    height=300,
                )

                with gr.Tabs():
                    with gr.Tab("🌊 Waveform"):
                        waveform_plot = gr.Plot(show_label=False)

                    with gr.Tab("🌈 Mel Spectrogram"):
                        spectrogram_plot = gr.Plot(show_label=False)

                    with gr.Tab("📈 RMS Envelope"):
                        gr.Markdown(
                            "_Orange: Video2RMS prediction (motion timing from the video). "
                            "Green: realised audio RMS after amplitude modulation._",
                            elem_id="theme-info",
                        )
                        rms_plot = gr.Plot(show_label=False)

        # ── How it works accordion ─────────────────────────────────────────
        with gr.Accordion("ℹ️  How it works", open=False):
            gr.Markdown("""
**Pipeline overview**

1. **Video preprocessing** — Your video is segmented; optical flow is extracted
   with RAFT, and BN-Inception RGB+Flow features are computed per segment.
2. **Video2RMS** — A learned model predicts an RMS envelope (timing + dynamics)
   from the visual features.
3. **AudioLDM** — The base AudioLDM model + CLAP text embedding generates
   *any* sound you describe in the prompt.
4. **Amplitude modulation** — The Video2RMS envelope is applied to the generated audio
   so rhythm and intensity follow the on-screen motion.
5. **Theme prefix** — Selecting a theme prepends a style descriptor to your prompt,
   biasing the CLAP embedding toward that aesthetic without overriding your text.

**Planned: Auto-caption**
A vision-language model (BLIP-2 / LLaVA / GPT-4V) will analyse key frames from the
uploaded video and suggest a default sound description that you can freely edit before
generating.
            """)

        # ── Event wiring ───────────────────────────────────────────────────
        theme_dropdown.change(
            fn=update_theme_info,
            inputs=theme_dropdown,
            outputs=theme_info,
        )

        auto_btn.click(
            fn=auto_caption,
            inputs=video_input,
            outputs=prompt_input,
        )

        generate_btn.click(
            fn=run_inference,
            inputs=[video_input, prompt_input, theme_dropdown],
            outputs=[video_output, waveform_plot, spectrogram_plot, rms_plot, status_box],
        )

    return demo
