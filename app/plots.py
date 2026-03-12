"""
app/plots.py
────────────
Matplotlib figure builders for the three audio-visualisation tabs:

  • plot_waveform        — amplitude vs. time
  • plot_spectrogram     — mel-frequency spectrogram (magma colourmap)
  • plot_rms_envelope    — Video2RMS predicted envelope vs. actual audio RMS
                          (two-panel figure, shared time axis with the video)

All figures use the GitHub-dark colour palette defined in app/config.py so
they blend into the UI without any extra styling.
"""

import warnings

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import BG, PANEL, BORDER, TEXT, HEADING, MUTED, BLUE, GREEN, ORANGE


# ── Internal helpers ─────────────────────────────────────────────────────────

def _style_axes(ax: plt.Axes, title: str) -> None:
    """Apply the dark-theme style to a single Axes object."""
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=HEADING, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


# ── Public figure builders ────────────────────────────────────────────────────

def plot_waveform(waveform: np.ndarray, sr: int) -> plt.Figure:
    """Return a figure showing the generated waveform with an area fill."""
    t = np.linspace(0, len(waveform) / sr, len(waveform))
    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor=BG)
    ax.plot(t, waveform, color=BLUE, linewidth=0.5, alpha=0.85)
    ax.fill_between(t, waveform, alpha=0.18, color=BLUE)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_xlim(0, t[-1] if len(t) > 0 else 1)
    _style_axes(ax, "Generated Waveform")
    fig.tight_layout(pad=1.2)
    return fig


def plot_spectrogram(waveform: np.ndarray, sr: int) -> plt.Figure:
    """Return a mel-spectrogram figure (magma colourmap, dB scale)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=BG)
    img = librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000,
        cmap="magma", ax=ax,
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BORDER)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    _style_axes(ax, "Mel Spectrogram")
    fig.tight_layout(pad=1.2)
    return fig


def plot_rms_envelope(
    waveform: np.ndarray,
    rms_curve: np.ndarray,
    sr: int,
) -> plt.Figure:
    """
    Two-panel figure that visualises the Video2RMS envelope and how the
    generated audio realises it.

    Top panel
        Waveform (blue) overlaid with the Video2RMS predicted envelope
        (orange, scaled to match the waveform amplitude).

    Bottom panel
        Normalised comparison: Video2RMS prediction (orange) vs. the
        frame-level RMS measured on the generated audio (green).
    """
    duration = len(waveform) / sr
    t_audio  = np.linspace(0, duration, len(waveform))
    t_rms    = np.linspace(0, duration, len(rms_curve))

    # Compute actual frame-level RMS from the generated waveform
    frame_len  = 1024
    hop_len    = 256
    actual_rms = librosa.feature.rms(
        y=waveform, frame_length=frame_len, hop_length=hop_len
    )[0]
    t_actual = librosa.frames_to_time(
        np.arange(len(actual_rms)), sr=sr, hop_length=hop_len
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5.5), facecolor=BG, sharex=False
    )

    # ── Top panel ─────────────────────────────────────────────────────────
    ax1.plot(t_audio, waveform, color=BLUE, linewidth=0.4, alpha=0.65,
             label="Waveform")
    env_interp = np.interp(t_audio, t_rms, rms_curve)
    amp_scale  = np.abs(waveform).max()
    env_scaled = env_interp / (env_interp.max() + 1e-8) * amp_scale
    ax1.plot(t_audio,  env_scaled, color=ORANGE, linewidth=1.4, alpha=0.9,
             label="Video2RMS envelope (scaled)")
    ax1.plot(t_audio, -env_scaled, color=ORANGE, linewidth=1.4, alpha=0.9)
    ax1.fill_between(t_audio,  env_scaled, alpha=0.10, color=ORANGE)
    ax1.fill_between(t_audio, -env_scaled, alpha=0.10, color=ORANGE)
    ax1.set_ylabel("Amplitude", fontsize=9)
    ax1.set_xlim(0, max(duration, 1e-3))
    ax1.legend(fontsize=8, facecolor=BORDER, labelcolor=TEXT, edgecolor=BORDER)
    _style_axes(ax1, "Waveform + Video2RMS Envelope")

    # ── Bottom panel ──────────────────────────────────────────────────────
    pred_norm   = rms_curve  / (rms_curve.max()  + 1e-8)
    actual_norm = actual_rms / (actual_rms.max() + 1e-8)
    ax2.plot(t_rms,    pred_norm,   color=ORANGE, linewidth=1.6,
             label="Predicted RMS (Video2RMS)")
    ax2.plot(t_actual, actual_norm, color=GREEN,  linewidth=1.6, alpha=0.85,
             label="Actual audio RMS")
    ax2.fill_between(t_rms,    pred_norm,   alpha=0.12, color=ORANGE)
    ax2.fill_between(t_actual, actual_norm, alpha=0.12, color=GREEN)
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.set_ylabel("Normalised RMS", fontsize=9)
    ax2.set_xlim(0, max(duration, 1e-3))
    ax2.legend(fontsize=8, facecolor=BORDER, labelcolor=TEXT, edgecolor=BORDER)
    _style_axes(ax2, "Predicted vs Actual RMS Envelope")

    fig.tight_layout(pad=1.6)
    return fig
