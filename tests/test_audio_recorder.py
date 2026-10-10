"""Unit tests for audio recording controls, device handling, and WAV validation."""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.audio.recorder import (
    AudioRecorder,
    AudioRecorderError,
    AudioDeviceError,
    AudioValidationError,
    validate_wav_file
)


def test_validate_wav_file_valid(tmp_path):
    wav_path = tmp_path / "test_valid.wav"
    sample_rate = 16000
    duration = 1.0  # 1 second
    data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))).astype(np.float32)
    sf.write(str(wav_path), data, sample_rate)

    is_valid, msg, dur = validate_wav_file(wav_path)
    assert is_valid is True
    assert "Valid" in msg
    assert abs(dur - 1.0) < 0.05


def test_validate_wav_file_empty(tmp_path):
    wav_path = tmp_path / "test_empty.wav"
    sf.write(str(wav_path), np.array([], dtype=np.float32), 16000)

    is_valid, msg, dur = validate_wav_file(wav_path)
    assert is_valid is False
    assert "empty" in msg.lower()


def test_validate_wav_file_nonexistent(tmp_path):
    nonexistent = tmp_path / "does_not_exist.wav"
    is_valid, msg, dur = validate_wav_file(nonexistent)
    assert is_valid is False
    assert "does not exist" in msg.lower()


def test_recorder_initial_state(tmp_path):
    recorder = AudioRecorder(recordings_dir=tmp_path)
    assert recorder.is_recording is False


def test_recorder_stop_without_start_raises_error(tmp_path):
    recorder = AudioRecorder(recordings_dir=tmp_path)
    with pytest.raises(AudioRecorderError):
        recorder.stop_recording()


def test_recorder_no_devices_raises_error(tmp_path):
    recorder = AudioRecorder(recordings_dir=tmp_path)
    with patch.object(AudioRecorder, "get_available_input_devices", return_value=[]):
        with pytest.raises(AudioDeviceError) as exc_info:
            recorder.start_recording()
        assert "No audio input device" in str(exc_info.value)


def test_recorder_empty_audio_raises_validation_error(tmp_path):
    recorder = AudioRecorder(recordings_dir=tmp_path)
    mock_dev = [{"name": "Mock Mic", "max_input_channels": 1}]
    with patch.object(AudioRecorder, "get_available_input_devices", return_value=mock_dev):
        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            recorder.start_recording()
            assert recorder.is_recording is True

            # Stop without adding data to queue
            with pytest.raises(AudioValidationError):
                recorder.stop_recording()
            assert recorder.is_recording is False


def test_recorder_with_explicit_device(tmp_path):
    recorder = AudioRecorder(recordings_dir=tmp_path, device=2)
    assert recorder.device == 2
    mock_dev = [{"index": 2, "name": "External Mic", "max_input_channels": 1}]
    with patch.object(AudioRecorder, "get_available_input_devices", return_value=mock_dev):
        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            recorder.start_recording()
            assert recorder.is_recording is True
            mock_stream_cls.assert_called_once()
            _, kwargs = mock_stream_cls.call_args
            assert kwargs.get("device") == 2
            recorder.cancel_recording()
            assert recorder.is_recording is False


def test_recorder_level_callback_and_tracking(tmp_path):
    received_levels = []

    def on_level(peak, rms):
        received_levels.append((peak, rms))

    recorder = AudioRecorder(recordings_dir=tmp_path, level_callback=on_level)
    assert recorder.last_peak == 0.0
    assert recorder.last_rms == 0.0

    mock_dev = [{"index": 0, "name": "Default Mic", "max_input_channels": 1}]
    with patch.object(AudioRecorder, "get_available_input_devices", return_value=mock_dev):
        with patch("sounddevice.InputStream") as mock_stream_cls:
            recorder.start_recording()
            assert recorder.is_recording is True

            # Simulate audio callback with 0.5 amplitude sine block
            sample_block = np.full((160, 1), 0.5, dtype=np.float32)
            recorder._audio_callback(sample_block, 160, None, None)

            assert recorder.last_peak == pytest.approx(0.5, abs=0.01)
            assert recorder.last_rms == pytest.approx(0.5, abs=0.01)
            assert len(received_levels) >= 1
            assert received_levels[-1][0] == pytest.approx(0.5, abs=0.01)

            recorder.cancel_recording()
            assert recorder.is_recording is False
            assert recorder.last_peak == 0.0
            assert recorder.last_rms == 0.0
            assert received_levels[-1] == (0.0, 0.0)


