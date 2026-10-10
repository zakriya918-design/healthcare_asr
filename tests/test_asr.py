"""Unit and integration tests for WhisperTranscriber."""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.audio.asr import (
    WhisperTranscriber,
    ASRError,
    ModelLoadError,
    TranscriptionError
)


def test_asr_initial_state():
    transcriber = WhisperTranscriber(model_size="tiny.en")
    assert transcriber.is_model_loaded is False


def test_transcribe_missing_file():
    transcriber = WhisperTranscriber(model_size="tiny.en")
    with pytest.raises(TranscriptionError) as exc_info:
        transcriber.transcribe_file(Path("nonexistent_recording_999.wav"))
    assert "not found" in str(exc_info.value).lower()


def test_transcribe_empty_file(tmp_path):
    empty_wav = tmp_path / "empty.wav"
    sf.write(str(empty_wav), np.array([], dtype=np.float32), 16000)

    transcriber = WhisperTranscriber(model_size="tiny.en")
    with pytest.raises(TranscriptionError) as exc_info:
        transcriber.transcribe_file(empty_wav)
    assert "empty or too short" in str(exc_info.value).lower()


def test_transcribe_corrupted_file(tmp_path):
    bad_wav = tmp_path / "corrupted.wav"
    bad_wav.write_text("This is not a real audio file.", encoding="utf-8")

    transcriber = WhisperTranscriber(model_size="tiny.en")
    with pytest.raises(TranscriptionError) as exc_info:
        transcriber.transcribe_file(bad_wav)
    assert "cannot read" in str(exc_info.value).lower()


def test_transcribe_mocked_model(tmp_path):
    # Test pipeline flow with mocked Faster-Whisper model
    sample_wav = tmp_path / "sample.wav"
    data = (np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000))).astype(np.float32)
    sf.write(str(sample_wav), data, 16000)

    transcriber = WhisperTranscriber(model_size="tiny.en")

    # Mock the internal model and its transcribe output
    mock_model = MagicMock()
    mock_seg = MagicMock()
    mock_seg.text = " Patient reports cough for two days. "
    mock_seg.start = 0.0
    mock_seg.end = 1.8
    mock_model.transcribe.return_value = ([mock_seg], None)

    transcriber._model = mock_model
    transcript, segments = transcriber.transcribe_file(sample_wav)

    assert transcript == "Patient reports cough for two days."
    assert len(segments) == 1
    assert segments[0].text == "Patient reports cough for two days."
    assert segments[0].start_time == 0.0
    assert segments[0].end_time == 1.8


def test_load_model_from_offline_dir(tmp_path):
    # Setup dummy offline model directory with model.bin
    offline_model_dir = tmp_path / "tiny.en"
    offline_model_dir.mkdir(parents=True)
    (offline_model_dir / "model.bin").write_bytes(b"dummy_weights")

    with patch("src.utils.paths.get_models_dir", return_value=tmp_path):
        with patch("faster_whisper.WhisperModel") as mock_wm_cls:
            transcriber = WhisperTranscriber(model_size="tiny.en")
            transcriber.load_model()
            assert transcriber.is_model_loaded is True
            mock_wm_cls.assert_called_once()
            call_kwargs = mock_wm_cls.call_args[1]
            assert call_kwargs["model_size_or_path"] == str(offline_model_dir)

