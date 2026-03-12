"""
app/config.py
─────────────
Shared constants: paths, sys.path bootstrap, theme definitions,
and the matplotlib colour palette used across plot modules.

Importing this module (or any module that imports it) is all that is
needed to make the repo-local libraries (video2rms, AudioLDM, …)
importable from anywhere in the package.
"""

import os
import sys

# ── Project root (parent of this file's app/ package directory) ──────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR     = os.path.join(ROOT_DIR, "ckpt")
OUTPUT_BASE  = os.path.join(ROOT_DIR, "gradio_output")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

# ── Inject repo-local lib paths into sys.path (idempotent) ───────────────────
for _p in [
    ROOT_DIR,                                             # utils/
    os.path.join(ROOT_DIR, "video2rms"),                  # util.py, data_utils.py …
    os.path.join(ROOT_DIR, "libs", "AudioLDM"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Sound themes ──────────────────────────────────────────────────────────────
# Keys are display labels; values are prompt prefixes injected before the
# user's text so the CLAP embedding is biased toward the chosen aesthetic.
THEMES = {
    "None":          "",
    "🎬 Cinematic":  "cinematic dramatic orchestral, ",
    "🎨 Cartoon":    "cartoon animation exaggerated sound effects, ",
    "😂 Funny":      "comedic funny over-the-top silly, ",
    "👻 Horror":     "scary haunting eerie unsettling atmospheric, ",
    "🌿 Nature":     "natural organic environmental ambient, ",
    "🚀 Sci-Fi":     "futuristic electronic sci-fi synthesized, ",
    "✨ Fantasy":    "magical mystical whimsical fantasy, ",
    "🎸 Rock":       "electric guitar rock band drums, ",
    "🎺 Jazz":       "jazz piano trumpet upright bass swing, ",
}

THEME_DESCRIPTIONS = {
    "None":          "No theme — use your prompt exactly as typed.",
    "🎬 Cinematic":  "Dramatic, orchestral sounds suited for film scoring.",
    "🎨 Cartoon":    "Exaggerated, bouncy Looney-Tunes-style sound effects.",
    "😂 Funny":      "Comedic and humorous audio cues.",
    "👻 Horror":     "Eerie, haunting and unsettling atmospheric sound design.",
    "🌿 Nature":     "Organic, natural ambient soundscapes.",
    "🚀 Sci-Fi":     "Futuristic electronic tones and synthesized effects.",
    "✨ Fantasy":    "Magical, mystical and otherworldly sounds.",
    "🎸 Rock":       "Electric guitar-driven rock energy.",
    "🎺 Jazz":       "Jazz instruments with a swinging groove.",
}

# ── Matplotlib colour palette (GitHub dark theme) ────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#21262d"
TEXT    = "#c9d1d9"
HEADING = "#f0f6fc"
MUTED   = "#8b949e"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#f78166"


# ── Device selection ─────────────────────────────────────────────────────────

def create_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
