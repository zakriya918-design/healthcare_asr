"""Unit tests for the audio pre-flight diagnostic module."""

import numpy as np
import pytest
from unittest.mock import patch

from src.audio.diagnostics import (
    AudioDiagnosticResult,
    analyze_audio_data,
    run_audio_diagnostic,
    _amplitude_to_db,
)


def test_amplitude_to_db():
    assert _amplitude_to_db(0.0) == -96.0
    assert abs(_amplitude_to_db(1.0) - 0.0) < 0.01
    assert abs(_amplitude_to_db(0.1) - (-20.0)) < 0.01


def test_analyze_empty_audio():
    empty = np.array([], dtype=np.float32)
    res = analyze_audio_data(empty)
    assert res.is_silent is True
    assert res.quality_rating == "CRITICAL"
    assert res.duration_seconds == 0.0
    assert len(res.recommendations) > 0


def test_analyze_silent_audio():
    # 2 seconds of total silence (zeros)
    silent = np.zeros(32000, dtype=np.float32)
    res = analyze_audio_data(silent, sample_rate=16000)
    assert res.is_silent is True
    assert res.quality_rating == "CRITICAL"
    assert res.peak_db <= -90.0
    assert any("silent" in r.lower() for r in res.recommendations)


def test_analyze_clipped_audio():
    # Audio with deliberate hard clipping (values exceeding 0.99)
    samples = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
    # Clip 1000 samples (over 6% of total samples)
    samples[:1000] = 1.0
    res = analyze_audio_data(samples, sample_rate=16000)
    assert res.has_clipping is True
    assert res.quality_rating == "POOR"
    assert any("clipping" in r.lower() for r in res.recommendations)


def test_analyze_low_snr_audio():
    # Audio where noise floor is almost as loud as the signal
    noise = np.random.uniform(-0.2, 0.2, 16000).astype(np.float32)
    res = analyze_audio_data(noise, sample_rate=16000)
    # With pure uniform noise, SNR is low (< 12 dB)
    assert res.snr_db < 15.0
    assert res.quality_rating in ("FAIR", "POOR")
    assert any("noise" in r.lower() or "ratio" in r.lower() for r in res.recommendations)


def test_analyze_optimal_speech_audio():
    # Synthesize clean 440 Hz tone burst at -18 dBFS amplitude (~0.125)
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    # 1 second silence, 1 second tone
    tone = np.zeros(32000, dtype=np.float32)
    tone[16000:] = 0.125 * np.sin(2 * np.pi * 440 * t[16000:])
    
    res = analyze_audio_data(tone, sample_rate=16000)
    assert res.is_silent is False
    assert res.has_clipping is False
    assert res.snr_db > 20.0
    assert res.quality_rating in ("EXCELLENT", "GOOD")


def test_run_audio_diagnostic_mocked_success():
    fake_audio = (0.1 * np.sin(np.linspace(0, 10, 16000))).astype(np.float32).reshape(-1, 1)

    with patch("sounddevice.rec", return_value=fake_audio):
        with patch("sounddevice.wait"):
            with patch("sounddevice.query_devices", return_value={"name": "Studio USB Mic"}):
                res = run_audio_diagnostic(device_index=1, duration=1.0)
                assert res.device_name == "Studio USB Mic"
                assert res.is_silent is False
                assert res.clipping_ratio < 0.001


def test_run_audio_diagnostic_device_failure():
    with patch("sounddevice.rec", side_effect=RuntimeError("Device unavailable")):
        res = run_audio_diagnostic(device_index=99, duration=1.0)
        assert res.quality_rating == "CRITICAL"
        assert res.is_silent is True
        assert any("Hardware capture failed" in r for r in res.recommendations)


def test_diagnostic_result_to_dict():
    res = AudioDiagnosticResult(
        device_name="Test Mic",
        device_index=2,
        sample_rate=16000,
        duration_seconds=3.0,
        rms_db=-20.5,
        peak_db=-10.2,
        noise_floor_db=-55.0,
        snr_db=44.8,
        clipping_ratio=0.0,
        is_silent=False,
        has_clipping=False,
        quality_rating="EXCELLENT",
        recommendations=["Hardware is optimal"]
    )
    d = res.to_dict()
    assert d["device_name"] == "Test Mic"
    assert d["rms_db"] == -20.5
    assert d["snr_db"] == 44.8
    assert d["quality_rating"] == "EXCELLENT"
