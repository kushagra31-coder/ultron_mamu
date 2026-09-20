"""Fast local Ultron-style DSP for a Windows TTS WAV.

This does not clone a film actor or copyrighted character voice. It applies
local signal processing to an ordinary male Windows TTS voice: modest pitch
drop, controlled low-pass, ring modulation, soft saturation, and a short
stereo-like doubling delay.
"""
from __future__ import annotations

import os
from typing import Final

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt


# Tunable from environment without editing code.
PITCH_STEPS: Final[float] = float(os.getenv("ULTRON_PITCH", "-3.0"))
LOWPASS_CUTOFF: Final[float] = float(os.getenv("ULTRON_LOWPASS", "5200"))
RING_FREQ: Final[float] = float(os.getenv("ULTRON_RING_FREQ", "32"))
RING_DEPTH: Final[float] = float(os.getenv("ULTRON_RING_DEPTH", "0.10"))
DISTORTION: Final[float] = float(os.getenv("ULTRON_DISTORTION", "0.045"))
DELAY_MS: Final[float] = float(os.getenv("ULTRON_DELAY_MS", "16"))
DELAY_MIX: Final[float] = float(os.getenv("ULTRON_DELAY_MIX", "0.16"))
OUTPUT_PEAK: Final[float] = 0.94


def _pitch_down(y: np.ndarray, semitones: float) -> np.ndarray:
    """Fast approximate pitch drop via high-quality polyphase resampling.

    This intentionally trades exact duration preservation for low latency and
    zero phase-vocoder dependency. Small shifts (-2 to -4 semitones) sound
    substantially deeper while keeping the response reasonably short.
    """
    if abs(semitones) < 0.01:
        return y

    ratio = 2.0 ** (semitones / 12.0)
    # For a negative semitone value, up/down > 1 -> longer waveform -> lower pitch.
    scale = 1.0 / ratio
    denominator = 1000
    numerator = max(1, int(round(scale * denominator)))
    return resample_poly(y, numerator, denominator).astype(np.float32, copy=False)


def _lowpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    cutoff = max(200.0, min(cutoff, sr * 0.45))
    sos = butter(4, cutoff / (sr / 2.0), btype="low", output="sos")
    return sosfilt(sos, y).astype(np.float32, copy=False)


def _ring_modulate(y: np.ndarray, sr: int, freq: float, depth: float) -> np.ndarray:
    depth = float(np.clip(depth, 0.0, 1.0))
    if depth <= 0:
        return y
    t = np.arange(y.size, dtype=np.float64) / sr
    carrier = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    wet = y * carrier
    return (y * (1.0 - depth) + wet * depth).astype(np.float32, copy=False)


def _soft_clip(y: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    drive = 1.0 + 5.0 * amount
    denominator = np.tanh(drive)
    return (np.tanh(y * drive) / denominator).astype(np.float32, copy=False)


def _doubling_delay(y: np.ndarray, sr: int, delay_ms: float, mix: float) -> np.ndarray:
    delay_samples = int(sr * delay_ms / 1000.0)
    if delay_samples <= 0 or delay_samples >= y.size:
        return y

    mix = float(np.clip(mix, 0.0, 1.0))
    delayed = np.zeros_like(y)
    delayed[delay_samples:] = y[:-delay_samples]
    return (y * (1.0 - mix) + delayed * mix).astype(np.float32, copy=False)


def ultronize(input_path: str, output_path: str) -> None:
    """Transform a mono Windows-TTS WAV into the Ultron-style voice."""
    y, sr = sf.read(input_path, dtype="float32", always_2d=False)

    if y.ndim > 1:
        y = np.mean(y, axis=1, dtype=np.float32)

    if y.size == 0:
        raise ValueError("TTS input WAV is empty.")

    y = _pitch_down(y, PITCH_STEPS)
    y = _lowpass(y, sr, LOWPASS_CUTOFF)
    y = _ring_modulate(y, sr, RING_FREQ, RING_DEPTH)
    y = _soft_clip(y, DISTORTION)
    y = _doubling_delay(y, sr, DELAY_MS, DELAY_MIX)

    peak = float(np.max(np.abs(y)))
    if peak > 1e-7:
        y = y * (OUTPUT_PEAK / peak)

    sf.write(output_path, y.astype(np.float32, copy=False), sr, subtype="PCM_16")
