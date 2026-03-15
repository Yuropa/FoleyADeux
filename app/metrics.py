"""
app/metrics.py
──────────────
Audio quality and synchronisation metrics for the Foley pipeline.

All functions are pure (no side effects on disk) except compute_clap_score,
which lazily loads and caches the CLAP model in process memory on first call.

Public API
----------
compute_envelope_metrics(predicted_rms, waveform, sr)  → dict
    Pearson r, MAE, and onset alignment F1 between the Video2RMS predicted
    envelope and the actual RMS of the generated waveform.

compute_audio_quality_metrics(waveform, sr)            → dict
    Dynamic range, spectral centroid/bandwidth, and zero-crossing rate.

compute_clap_score(prompt, waveform, sr)               → float
    Cosine similarity between CLAP text and audio embeddings
    (laion/larger_clap_general, computed on CPU, cached after first load).

aggregate_metrics(metrics_list)                        → dict
    Average a list of per-segment metric dicts into one summary.

format_metrics_markdown(metrics, n_segments)           → str
    Render a metrics dict as a Gradio-ready markdown string with
    colour-coded quality badges.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from scipy.stats import pearsonr

# ── CLAP model cache (loaded once, kept in CPU RAM) ──────────────────────────
_clap_model     = None
_clap_processor = None
_CLAP_MODEL_ID  = "laion/larger_clap_general"


# ── Envelope synchronisation ─────────────────────────────────────────────────

def compute_envelope_metrics(
    predicted_rms: np.ndarray,
    waveform: np.ndarray,
    sr: int,
) -> Dict[str, float]:
    """
    Synchronisation metrics between the Video2RMS predicted envelope and
    the actual frame-level RMS of the amplitude-modulated output waveform.

    Parameters
    ----------
    predicted_rms : 1-D array
        Envelope predicted by Video2RMS (arbitrary length).
    waveform : 1-D float32 array
        Generated audio waveform at sample rate `sr`.
    sr : int
        Sample rate of `waveform`.

    Returns
    -------
    dict with keys
        envelope_pearson_r  – Pearson r ∈ [−1, 1] (1 = perfect shape match)
        envelope_mae        – MAE on normalised curves ∈ [0, 1] (0 = perfect)
        onset_precision     – ∈ [0, 1], ±200 ms tolerance window
        onset_recall        – ∈ [0, 1]
        onset_f1            – harmonic mean of precision and recall
    """
    frame_len = 1024
    hop_len   = 256

    # Actual frame-level RMS from the generated waveform
    actual_rms = librosa.feature.rms(
        y=waveform, frame_length=frame_len, hop_length=hop_len
    )[0]

    # Resample predicted curve to match actual_rms length
    pred_resampled = np.interp(
        np.linspace(0, 1, len(actual_rms)),
        np.linspace(0, 1, len(predicted_rms)),
        predicted_rms,
    )

    pred_norm   = pred_resampled / (pred_resampled.max() + 1e-8)
    actual_norm = actual_rms     / (actual_rms.max()     + 1e-8)

    r, _   = pearsonr(pred_norm, actual_norm)
    mae    = float(np.mean(np.abs(pred_norm - actual_norm)))

    # Onset alignment — find peaks in both curves, match within ±200 ms
    duration = len(waveform) / sr
    t_pred   = np.linspace(0, duration, len(pred_norm))
    t_actual = librosa.frames_to_time(
        np.arange(len(actual_rms)), sr=sr, hop_length=hop_len
    )

    pred_peaks,   _ = find_peaks(pred_norm,   prominence=0.15, distance=10)
    actual_peaks, _ = find_peaks(actual_norm, prominence=0.15, distance=10)

    tolerance = 0.20  # seconds
    tp        = 0
    matched   = set()
    for pp in pred_peaks:
        for i, ap in enumerate(actual_peaks):
            if i not in matched and abs(t_pred[pp] - t_actual[ap]) <= tolerance:
                tp += 1
                matched.add(i)
                break

    precision = tp / max(len(pred_peaks),   1)
    recall    = tp / max(len(actual_peaks), 1)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "envelope_pearson_r": float(r),
        "envelope_mae":       mae,
        "onset_precision":    float(precision),
        "onset_recall":       float(recall),
        "onset_f1":           float(f1),
    }


# ── Audio quality descriptors ─────────────────────────────────────────────────

def compute_audio_quality_metrics(
    waveform: np.ndarray,
    sr: int,
) -> Dict[str, float]:
    """
    Signal-level audio quality descriptors derived from the generated waveform.

    Parameters
    ----------
    waveform : 1-D float32 array
    sr : int

    Returns
    -------
    dict with keys
        dynamic_range_db        – Peak-to-RMS ratio in dB.
                                  Higher → more dynamic (expressive).
                                  Near 0 dB → flat / clipped audio.
        spectral_centroid_mean  – Mean spectral centroid in Hz.
                                  Higher → brighter / trebly sound.
        spectral_centroid_std   – Frame-to-frame variation in centroid (Hz).
        spectral_bandwidth_mean – Mean spectral bandwidth in Hz.
                                  Higher → richer / more harmonically dense.
        zcr_mean                – Mean zero-crossing rate.
                                  Higher → noisier / more percussive content.
    """
    rms_val = float(np.sqrt(np.mean(waveform ** 2)) + 1e-8)
    peak    = float(np.abs(waveform).max()           + 1e-8)
    dr_db   = float(20 * np.log10(peak / rms_val))

    centroid  = librosa.feature.spectral_centroid( y=waveform, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
    zcr       = librosa.feature.zero_crossing_rate(y=waveform)[0]

    return {
        "dynamic_range_db":        dr_db,
        "spectral_centroid_mean":  float(np.mean(centroid)),
        "spectral_centroid_std":   float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "zcr_mean":                float(np.mean(zcr)),
    }


# ── CLAP prompt-adherence score ───────────────────────────────────────────────

def compute_clap_score(
    prompt: str,
    waveform: np.ndarray,
    sr: int,
) -> Optional[float]:
    """
    CLAP score: cosine similarity between the CLAP text embedding of `prompt`
    and the CLAP audio embedding of `waveform`.

    Uses laion/larger_clap_general (~900 MB, downloaded from HuggingFace the
    first time).  The model is kept in CPU RAM and reused across calls.

    ClapFeatureExtractor handles resampling to 48 kHz internally.

    Returns
    -------
    float ∈ [−1, 1], or None if loading fails.
    Values ≳ 0.20 indicate a semantically good match.
    """
    global _clap_model, _clap_processor

    try:
        from transformers import ClapModel, ClapProcessor

        if _clap_model is None:
            _clap_model     = ClapModel.from_pretrained(_CLAP_MODEL_ID).eval()
            _clap_processor = ClapProcessor.from_pretrained(_CLAP_MODEL_ID)

        with torch.no_grad():
            text_in  = _clap_processor(
                text=[prompt], return_tensors="pt", padding=True
            )
            text_emb = _clap_model.get_text_features(**text_in)

            # Pass waveform as float32; ClapFeatureExtractor resamples to 48 kHz
            audio_in  = _clap_processor(
                audios=[waveform.astype(np.float32)],
                sampling_rate=sr,
                return_tensors="pt",
            )
            audio_emb = _clap_model.get_audio_features(**audio_in)

            return float(F.cosine_similarity(text_emb, audio_emb).item())

    except Exception:  # noqa: BLE001
        return None


# ── Aggregation and formatting ────────────────────────────────────────────────

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of per-segment metric dicts into a single summary dict."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    result = {}
    for k in keys:
        values = [m[k] for m in metrics_list if k in m and m[k] is not None]
        result[k] = float(np.mean(values)) if values else None
    return result


def format_metrics_markdown(
    metrics: Dict[str, float],
    n_segments: int = 1,
) -> str:
    """
    Render a metrics dict as a Gradio-ready markdown string.

    Colour badges:
        🟢  good     🟡  acceptable     🔴  poor
    """

    def _badge(val: float, good: float, ok: float) -> str:
        if val >= good:
            return "🟢"
        if val >= ok:
            return "🟡"
        return "🔴"

    seg_label = f"{n_segments} segment{'s' if n_segments != 1 else ''}"

    r       = metrics.get("envelope_pearson_r", 0.0)
    mae     = metrics.get("envelope_mae",        0.0)
    f1      = metrics.get("onset_f1",            0.0)
    prec    = metrics.get("onset_precision",     0.0)
    rec     = metrics.get("onset_recall",        0.0)
    dr      = metrics.get("dynamic_range_db",    0.0)
    sc_mu   = metrics.get("spectral_centroid_mean",  0.0)
    sc_sig  = metrics.get("spectral_centroid_std",   0.0)
    bw      = metrics.get("spectral_bandwidth_mean", 0.0)
    zcr     = metrics.get("zcr_mean",            0.0)
    clap    = metrics.get("clap_score",          None)

    # MAE badge is inverted (lower is better)
    mae_badge = "🟢" if mae < 0.10 else ("🟡" if mae < 0.20 else "🔴")

    lines = [
        f"### Metrics  _(averaged over {seg_label})_",
        "",
        "**Synchronisation** — how well audio dynamics follow video motion",
        "",
        "| Metric | Value | |",
        "|---|---|---|",
        f"| Envelope Pearson r | `{r:+.3f}` | {_badge(r, 0.70, 0.40)} |",
        f"| Envelope MAE (normalised) | `{mae:.4f}` | {mae_badge} |",
        f"| Onset F1 (±200 ms) | `{f1:.3f}` | {_badge(f1, 0.60, 0.30)} |",
        f"| Onset precision | `{prec:.3f}` | |",
        f"| Onset recall | `{rec:.3f}` | |",
        "",
        "**Audio Quality** — signal-level descriptors of the generated sound",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Dynamic range | `{dr:.1f} dB` |",
        f"| Spectral centroid | `{sc_mu:.0f} ± {sc_sig:.0f} Hz` |",
        f"| Spectral bandwidth | `{bw:.0f} Hz` |",
        f"| Zero-crossing rate | `{zcr:.4f}` |",
    ]

    if clap is not None:
        lines += [
            "",
            "**Prompt Adherence** — does the audio sound like the prompt?",
            "",
            "| Metric | Value | |",
            "|---|---|---|",
            f"| CLAP score (text–audio cosine sim.) | `{clap:.4f}` "
            f"| {_badge(clap, 0.25, 0.15)} |",
        ]
    else:
        lines += [
            "",
            "_CLAP score unavailable (model could not be loaded)._",
        ]

    return "\n".join(lines)
