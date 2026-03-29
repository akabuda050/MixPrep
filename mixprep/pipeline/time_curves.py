"""
F3 — Time curves computation via librosa.

All curves operate on mono audio at the sample rate provided by the caller
(16000 Hz from Essentia's MonoLoader). Frame/hop sizes are fixed constants
so curve lengths are deterministic for a given audio length.

Frame parameters:
    n_fft    = 1024  — FFT window
    hop_length = 512  — hop between frames

Band boundaries (Hz):
    low_band:  0     – 500
    mid_band:  500   – 4000
    high_band: 4000  – Nyquist
"""

from __future__ import annotations

import numpy as np

from mixprep.pipeline.schemas import TimeCurves

# Fixed parameters — changing these changes curve lengths and breaks idempotency
_N_FFT = 1024
_HOP = 512

# Band edges in Hz
_LOW_MAX_HZ = 500
_MID_MAX_HZ = 4000


def compute_time_curves(audio: np.ndarray, sample_rate: int) -> TimeCurves:
    """
    Compute all time curves for *audio* sampled at *sample_rate* Hz.

    audio — 1-D float32 numpy array (mono).
    sample_rate — integer sample rate (should be 16000).

    Returns TimeCurves with all curves having the same length.
    """
    import librosa

    # RMS energy per frame
    rms_frames = librosa.feature.rms(y=audio, frame_length=_N_FFT, hop_length=_HOP)[0]

    # Onset strength (spectral flux-based novelty)
    onset_frames = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=_HOP)

    # STFT magnitude for band energies
    stft = np.abs(librosa.stft(audio, n_fft=_N_FFT, hop_length=_HOP))  # (freq_bins, frames)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=_N_FFT)

    low_mask = freqs <= _LOW_MAX_HZ
    mid_mask = (freqs > _LOW_MAX_HZ) & (freqs <= _MID_MAX_HZ)
    high_mask = freqs > _MID_MAX_HZ

    low_frames = stft[low_mask].mean(axis=0) if low_mask.any() else np.zeros(stft.shape[1])
    mid_frames = stft[mid_mask].mean(axis=0) if mid_mask.any() else np.zeros(stft.shape[1])
    high_frames = stft[high_mask].mean(axis=0) if high_mask.any() else np.zeros(stft.shape[1])

    # Novelty = onset strength (same signal, stored separately for semantic clarity)
    novelty_frames = onset_frames.copy()

    # Align all curves to the minimum length (STFT may be 1 frame longer than onset)
    n = min(
        len(rms_frames),
        len(onset_frames),
        stft.shape[1],
        len(novelty_frames),
    )

    return TimeCurves(
        rms=rms_frames[:n].tolist(),
        onset_strength=onset_frames[:n].tolist(),
        low_band=low_frames[:n].tolist(),
        mid_band=mid_frames[:n].tolist(),
        high_band=high_frames[:n].tolist(),
        novelty=novelty_frames[:n].tolist(),
    )
