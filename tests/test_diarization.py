"""Unit tests for acoustic speaker diarization and voice clustering."""

import numpy as np
import pytest
from src.audio.diarization import (
    extract_acoustic_features,
    cluster_segments_by_voice,
    diarize_consultation_audio,
)
from src.models.patient_record import SpeakerRole, TranscriptSegment
from src.utils.paths import get_synthetic_wav_path


def test_extract_acoustic_features_valid_and_empty():
    sr = 16000
    # Valid sine wave
    t = np.linspace(0, 0.5, int(sr * 0.5))
    sig = np.sin(2 * np.pi * 180 * t).astype(np.float32)
    feats = extract_acoustic_features(sig, sr=sr)
    assert len(feats) == 6
    assert 150 < feats[0] < 210  # Pitch around 180 Hz
    assert feats[1] > 0  # Spectral centroid
    assert feats[5] > 0  # RMS

    # Empty signal returns zeros
    empty_feats = extract_acoustic_features(np.array([], dtype=np.float32), sr=sr)
    assert len(empty_feats) == 6
    assert np.all(empty_feats == 0)


def test_cluster_segments_by_voice_two_clusters():
    # 4 segments: 2 low pitch (e.g. 100Hz), 2 high pitch (e.g. 240Hz)
    features = np.array([
        [100.0, 1200.0, 500.0, 2000.0, 0.05, 0.3],
        [240.0, 2500.0, 900.0, 3500.0, 0.12, 0.4],
        [105.0, 1250.0, 510.0, 2050.0, 0.05, 0.3],
        [235.0, 2450.0, 880.0, 3450.0, 0.11, 0.4],
    ])
    labels = cluster_segments_by_voice(features)
    assert len(labels) == 4
    # Samples 0 & 2 must share a cluster, and 1 & 3 must share the other
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_diarize_consultation_audio_with_synthetic_audio():
    wav_path = get_synthetic_wav_path()
    assert wav_path.is_file()

    segments = [
        TranscriptSegment(id=1, text="Doctor, what brings you in today?", start_time=0.0, end_time=2.0),
        TranscriptSegment(id=2, text="Patient, I have had a severe sore throat.", start_time=3.0, end_time=6.7),
        TranscriptSegment(id=3, text="Doctor, any chest pain?", start_time=7.7, end_time=9.5),
        TranscriptSegment(id=4, text="Patient, no chest pain.", start_time=10.6, end_time=12.6),
    ]

    annotated = diarize_consultation_audio(segments, audio_path=wav_path)
    assert len(annotated) == 4
    assert annotated[0].speaker_role == SpeakerRole.DOCTOR
    assert annotated[1].speaker_role == SpeakerRole.PATIENT
    assert annotated[2].speaker_role == SpeakerRole.DOCTOR
    assert annotated[3].speaker_role == SpeakerRole.PATIENT


def test_diarize_consultation_audio_fallback_without_audio():
    segments = [
        TranscriptSegment(id=1, text="Doctor: Good morning Mr. Davis, how are you?"),
        TranscriptSegment(id=2, text="I've been feeling very dizzy and nauseous since yesterday."),
        TranscriptSegment(id=3, text="Let me check your blood pressure."),
    ]
    annotated = diarize_consultation_audio(segments, audio_path=None)
    assert len(annotated) == 3
    assert annotated[0].speaker_role == SpeakerRole.DOCTOR
    assert annotated[1].speaker_role == SpeakerRole.PATIENT
    assert annotated[2].speaker_role == SpeakerRole.DOCTOR
