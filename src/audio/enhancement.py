"""Audio Noise Filtering and Speech Enhancement Module.

Provides deterministic digital signal processing (DSP) for clinical audio:
1. Speech Bandpass Filtering: Attenuates low-frequency HVAC rumble (<80 Hz) and high-frequency hiss (>7500 Hz).
2. Spectral Subtraction Noise Reduction: Attenuates stationary background room noise using Berouti spectral subtraction.
3. Speech Automatic Gain Control (AGC) & Normalization: Targets optimal -20 dBFS RMS for Whisper ASR.
4. Soft Peak Limiting: Prevents digital clipping while preserving speech dynamics.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import soundfile as sf


def speech_bandpass_filter(
    audio: np.ndarray,
    sample_rate: int = 16000,
    low_cutoff_hz: float = 80.0,
    high_cutoff_hz: float = 7500.0,
    transition_width_hz: float = 30.0,
) -> np.ndarray:
    """Applies a zero-phase frequency-domain bandpass filter with smooth Hann skirts.
    
    Args:
        audio: 1D numpy array of audio samples (float32).
        sample_rate: Audio sampling frequency in Hz.
        low_cutoff_hz: High-pass cutoff frequency in Hz (removes HVAC rumble/thuds).
        high_cutoff_hz: Low-pass cutoff frequency in Hz (removes high-frequency hiss).
        transition_width_hz: Frequency transition band width in Hz.
        
    Returns:
        Filtered 1D numpy array.
    """
    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    n = len(audio)
    # Compute real Fast Fourier Transform
    fft_vals = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Construct smooth transfer function H(f)
    h = np.ones_like(freqs, dtype=np.float32)

    # 1. Low-frequency roll-off (High-pass skirt)
    hp_mask_zero = freqs < (low_cutoff_hz - transition_width_hz)
    hp_mask_trans = (freqs >= (low_cutoff_hz - transition_width_hz)) & (freqs < low_cutoff_hz)
    h[hp_mask_zero] = 0.0
    if np.any(hp_mask_trans):
        norm_trans = (freqs[hp_mask_trans] - (low_cutoff_hz - transition_width_hz)) / transition_width_hz
        h[hp_mask_trans] = 0.5 * (1.0 - np.cos(np.pi * norm_trans))

    # 2. High-frequency roll-off (Low-pass skirt)
    effective_high = min(high_cutoff_hz, (sample_rate / 2.0) - transition_width_hz)
    if effective_high > low_cutoff_hz:
        lp_mask_trans = (freqs > effective_high) & (freqs <= (effective_high + transition_width_hz))
        lp_mask_zero = freqs > (effective_high + transition_width_hz)
        h[lp_mask_zero] = 0.0
        if np.any(lp_mask_trans):
            norm_trans = (freqs[lp_mask_trans] - effective_high) / transition_width_hz
            h[lp_mask_trans] = 0.5 * (1.0 + np.cos(np.pi * norm_trans))

    # Apply filter in frequency domain and inverse transform
    filtered_fft = fft_vals * h
    filtered = np.fft.irfft(filtered_fft, n=n)
    return filtered.astype(np.float32)


def spectral_subtraction_noise_reduction(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: float = 32.0,
    hop_ms: float = 16.0,
    over_subtraction: float = 1.6,
    spectral_floor: float = 0.08,
) -> np.ndarray:
    """Attenuates stationary background room noise via windowed spectral subtraction.
    
    Args:
        audio: 1D numpy array of audio samples (float32).
        sample_rate: Sampling frequency in Hz.
        frame_ms: Frame analysis duration in milliseconds.
        hop_ms: Frame step duration in milliseconds.
        over_subtraction: Alpha factor for aggressive noise suppression without musical artifacts.
        spectral_floor: Beta factor minimum floor to prevent zero-power hole artifacts.
        
    Returns:
        Noise-attenuated 1D numpy array.
    """
    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    frame_len = int(sample_rate * (frame_ms / 1000.0))
    hop_len = int(sample_rate * (hop_ms / 1000.0))

    if len(audio) < frame_len:
        return audio.copy().astype(np.float32)

    window = np.hanning(frame_len).astype(np.float32)
    n_frames = (len(audio) - frame_len) // hop_len + 1

    # Extract STFT frames
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        frames[i] = audio[start : start + frame_len] * window

    # Compute magnitude and phase
    stft = np.fft.rfft(frames, axis=1)
    mag = np.abs(stft)
    phase = np.angle(stft)

    # Estimate stationary noise floor from the quietest 20% of frames
    frame_energies = np.mean(mag ** 2, axis=1)
    n_noise_frames = max(2, int(0.20 * n_frames))
    quiet_indices = np.argsort(frame_energies)[:n_noise_frames]
    noise_mag_estimate = np.mean(mag[quiet_indices], axis=0)

    # Spectral subtraction with floor
    subtracted_mag = np.zeros_like(mag)
    for i in range(n_frames):
        sub = mag[i] - (over_subtraction * noise_mag_estimate)
        floor = spectral_floor * mag[i]
        subtracted_mag[i] = np.maximum(sub, floor)

    # Inverse STFT and Overlap-Add (OLA)
    recon_stft = subtracted_mag * np.exp(1j * phase)
    recon_frames = np.fft.irfft(recon_stft, n=frame_len, axis=1) * window

    out_len = (n_frames - 1) * hop_len + frame_len
    out_audio = np.zeros(out_len, dtype=np.float32)
    norm_window = np.zeros(out_len, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_len
        out_audio[start : start + frame_len] += recon_frames[i]
        norm_window[start : start + frame_len] += window ** 2

    # Normalize window overlap
    valid_mask = norm_window > 1e-4
    out_audio[valid_mask] /= norm_window[valid_mask]

    # Pad or slice to match original audio length exactly
    if len(out_audio) < len(audio):
        pad_width = len(audio) - len(out_audio)
        out_audio = np.pad(out_audio, (0, pad_width), mode="constant")
    else:
        out_audio = out_audio[: len(audio)]

    return out_audio.astype(np.float32)


def normalize_speech_gain(
    audio: np.ndarray,
    target_rms_dbfs: float = -20.0,
    max_gain_db: float = 18.0,
    peak_ceiling_dbfs: float = -1.0,
) -> np.ndarray:
    """Normalizes speech volume towards target RMS dBFS with soft limiting to prevent clipping.
    
    Args:
        audio: 1D numpy array of audio samples.
        target_rms_dbfs: Target RMS level in dBFS (default: -20 dBFS).
        max_gain_db: Maximum boost in dB to avoid over-amplifying pure silence.
        peak_ceiling_dbfs: Maximum allowed peak amplitude level in dBFS.
        
    Returns:
        Gain-normalized 1D numpy array.
    """
    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 1e-5:
        # Near silence, do not amplify noise
        return audio.copy().astype(np.float32)

    current_rms_dbfs = 20.0 * math.log10(rms)
    gain_db = np.clip(target_rms_dbfs - current_rms_dbfs, -12.0, max_gain_db)
    gain_linear = 10.0 ** (gain_db / 20.0)

    amplified = audio * gain_linear

    # Soft tanh limiter for peaks exceeding ceiling
    ceiling_linear = 10.0 ** (peak_ceiling_dbfs / 20.0)  # ~0.891 for -1 dBFS
    peak = float(np.max(np.abs(amplified)))
    if peak > ceiling_linear:
        # Smooth soft clipping compression
        scale = ceiling_linear / peak
        # Only compress samples that exceed ceiling
        over_mask = np.abs(amplified) > ceiling_linear
        amplified[over_mask] = np.sign(amplified[over_mask]) * (
            ceiling_linear + (1.0 - ceiling_linear) * np.tanh((np.abs(amplified[over_mask]) - ceiling_linear) * scale)
        )

    return amplified.astype(np.float32)


def enhance_audio_data(
    audio: np.ndarray,
    sample_rate: int = 16000,
    bandpass: bool = True,
    noise_reduction: bool = True,
    gain_normalize: bool = True,
) -> np.ndarray:
    """Complete end-to-end speech enhancement pipeline for clinical consultations.
    
    Args:
        audio: Input 1D audio sample array.
        sample_rate: Sampling frequency (default 16000 Hz).
        bandpass: Apply speech frequency bandpass filter.
        noise_reduction: Apply spectral subtraction noise reduction.
        gain_normalize: Apply speech AGC gain normalization.
        
    Returns:
        Enhanced float32 1D audio array.
    """
    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    processed = audio.copy().astype(np.float32)
    if processed.ndim > 1:
        processed = processed.flatten()

    if bandpass:
        processed = speech_bandpass_filter(processed, sample_rate=sample_rate)

    if noise_reduction:
        processed = spectral_subtraction_noise_reduction(processed, sample_rate=sample_rate)

    if gain_normalize:
        processed = normalize_speech_gain(processed)

    return processed


def enhance_audio_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    bandpass: bool = True,
    noise_reduction: bool = True,
    gain_normalize: bool = True,
) -> Path:
    """Reads audio file from disk, applies enhancement pipeline, and saves enhanced WAV.
    
    Args:
        input_path: Path to source audio file.
        output_path: Destination path for enhanced WAV. If None, appends '_enhanced.wav'.
        
    Returns:
        Path to the saved enhanced audio file.
    """
    src = Path(input_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source audio file not found: {src}")

    data, sr = sf.read(str(src), dtype="float32")
    enhanced = enhance_audio_data(
        data,
        sample_rate=sr,
        bandpass=bandpass,
        noise_reduction=noise_reduction,
        gain_normalize=gain_normalize,
    )

    dest = Path(output_path) if output_path else src.parent / f"{src.stem}_enhanced.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), enhanced, samplerate=sr, subtype="PCM_16")
    return dest
