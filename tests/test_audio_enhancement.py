"""Unit tests for the speech enhancement and noise filtering module."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.audio.enhancement import (
    speech_bandpass_filter,
    spectral_subtraction_noise_reduction,
    normalize_speech_gain,
    enhance_audio_data,
    enhance_audio_file,
)
from src.utils.paths import get_synthetic_wav_path


def test_bandpass_filter_attenuates_low_and_high_frequencies():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 30 Hz sub-bass rumble + 1000 Hz human voice tone + 7800 Hz high hiss
    sig_low = 0.5 * np.sin(2 * np.pi * 30 * t)
    sig_mid = 0.5 * np.sin(2 * np.pi * 1000 * t)
    sig_high = 0.5 * np.sin(2 * np.pi * 7800 * t)
    composite = (sig_low + sig_mid + sig_high).astype(np.float32)

    filtered = speech_bandpass_filter(composite, sample_rate=sr, low_cutoff_hz=80.0, high_cutoff_hz=7500.0)

    assert len(filtered) == len(composite)
    # Mid-frequency tone energy should be preserved
    fft_orig = np.abs(np.fft.rfft(composite))
    fft_filt = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(composite), 1.0 / sr)

    idx_30 = np.argmin(np.abs(freqs - 30))
    idx_1000 = np.argmin(np.abs(freqs - 1000))
    idx_7800 = np.argmin(np.abs(freqs - 7800))

    # 30 Hz and 7800 Hz should be significantly attenuated
    assert fft_filt[idx_30] < 0.15 * fft_orig[idx_30]
    assert fft_filt[idx_7800] < 0.20 * fft_orig[idx_7800]
    # 1000 Hz tone should be virtually unattenuated
    assert fft_filt[idx_1000] > 0.90 * fft_orig[idx_1000]


def test_spectral_subtraction_reduces_noise_energy():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Speech tone burst during 0.4s to 0.7s
    speech = np.zeros(sr, dtype=np.float32)
    speech[6400:11200] = 0.5 * np.sin(2 * np.pi * 800 * t[6400:11200])

    # Background stationary noise throughout entire signal
    noise = np.random.normal(0, 0.05, sr).astype(np.float32)
    noisy = speech + noise

    denoised = spectral_subtraction_noise_reduction(noisy, sample_rate=sr)
    assert len(denoised) == len(noisy)

    # Noise region energy should be reduced
    noise_energy_orig = np.mean(noisy[:3200] ** 2)
    noise_energy_clean = np.mean(denoised[:3200] ** 2)
    assert noise_energy_clean < noise_energy_orig


def test_normalize_speech_gain_boosts_quiet_audio():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Very quiet speech tone at -35 dBFS
    quiet = (0.01 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    orig_rms = float(np.sqrt(np.mean(quiet ** 2)))

    normalized = normalize_speech_gain(quiet, target_rms_dbfs=-20.0)
    norm_rms = float(np.sqrt(np.mean(normalized ** 2)))

    assert norm_rms > orig_rms
    # Verify no clipping occurred
    assert np.max(np.abs(normalized)) <= 1.0


def test_enhance_audio_data_empty_input():
    empty = np.array([], dtype=np.float32)
    out = enhance_audio_data(empty)
    assert len(out) == 0


def test_enhance_audio_file(tmp_path):
    synth_path = get_synthetic_wav_path()
    assert synth_path.is_file()

    out_file = tmp_path / "enhanced_consultation.wav"
    result = enhance_audio_file(synth_path, output_path=out_file)

    assert result.is_file()
    assert result.stat().st_size > 1000

    # Verify output audio file can be read and matches input sample rate
    data, sr = sf.read(str(result))
    orig_info = sf.info(str(synth_path))
    assert sr == orig_info.samplerate
    assert len(data) > 0
