"""Unit tests for bulk audio batch processing module."""

from pathlib import Path
import json
import numpy as np
import pytest
import soundfile as sf
from unittest.mock import MagicMock

from src.processing.batch_processor import process_audio_folder, process_audio_file
from src.audio.asr import WhisperTranscriber
from src.models.patient_record import TranscriptSegment


def create_synthetic_wav(path: Path, duration: float = 1.0, sample_rate: int = 16000):
    t = np.linspace(0, duration, int(sample_rate * duration))
    data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(path), data, sample_rate)


def test_process_empty_folder(tmp_path: Path):
    empty_dir = tmp_path / "empty_input"
    empty_dir.mkdir()
    out_dir = tmp_path / "output"

    summary = process_audio_folder(empty_dir, out_dir)
    assert summary.total_files == 0
    assert summary.successful_files == 0


def test_process_audio_folder_with_valid_and_invalid_files(tmp_path: Path):
    in_dir = tmp_path / "input_audio"
    in_dir.mkdir()
    out_dir = tmp_path / "batch_out"

    # File 1: Valid WAV
    valid_wav = in_dir / "consultation_01.wav"
    create_synthetic_wav(valid_wav, duration=1.0)

    # File 2: Empty/corrupted WAV
    empty_wav = in_dir / "consultation_empty.wav"
    sf.write(str(empty_wav), np.array([], dtype=np.float32), 16000)

    # Mock WhisperTranscriber
    mock_transcriber = MagicMock(spec=WhisperTranscriber)
    mock_transcriber.model_size = "tiny.en"
    mock_transcriber.transcribe_file.return_value = (
        "Doctor: How are you feeling? Patient: I have a mild fever.",
        [
            TranscriptSegment(id=1, text="How are you feeling?", start_time=0.0, end_time=1.5),
            TranscriptSegment(id=2, text="I have a mild fever.", start_time=1.6, end_time=3.0)
        ]
    )

    progress_log = []
    def progress_cb(current, total, filename):
        progress_log.append((current, total, filename))

    summary = process_audio_folder(
        input_dir=in_dir,
        output_dir=out_dir,
        transcriber=mock_transcriber,
        export_pdf=True,
        export_html=True,
        progress_callback=progress_cb
    )

    assert summary.total_files == 2
    assert summary.successful_files == 1
    assert summary.failed_files == 1
    assert len(progress_log) == 2

    # Check generated files
    summary_file = out_dir / "batch_summary.json"
    assert summary_file.exists()
    summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary_data["successful_files"] == 1
    assert summary_data["failed_files"] == 1

    # Check that successful record exported files exist
    success_result = next(r for r in summary.results if r.success)
    assert success_result.json_path is not None and success_result.json_path.exists()
    assert success_result.pdf_path is not None and success_result.pdf_path.exists()
    assert success_result.html_path is not None and success_result.html_path.exists()
