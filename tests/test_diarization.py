"""Unit tests for multi-speaker acoustic diarization and voice clustering (2 to 4 persons)."""

import numpy as np
import pytest
from src.audio.diarization import (
    extract_acoustic_features,
    cluster_segments_by_voice,
    diarize_consultation_audio,
)
from src.models.patient_record import SpeakerRole, TranscriptSegment
from src.processing.speaker_processor import estimate_speaker_role
from src.utils.paths import get_synthetic_wav_path


def test_extract_acoustic_features_valid_and_empty():
    sr = 16000
    # Valid sine wave
    t = np.linspace(0, 0.5, int(sr * 0.5))
    sig = np.sin(2 * np.pi * 180 * t).astype(np.float32)
    feats = extract_acoustic_features(sig, sr=sr)
    assert len(feats) == 14
    assert 150 < feats[0] < 210  # Pitch around 180 Hz
    assert feats[1] > 0  # Spectral centroid
    assert feats[5] > 0  # RMS

    # Empty signal returns zeros
    empty_feats = extract_acoustic_features(np.array([], dtype=np.float32), sr=sr)
    assert len(empty_feats) == 14
    assert np.all(empty_feats == 0)


def test_cluster_segments_by_voice_two_clusters():
    # 4 segments: 2 low pitch (e.g. 100Hz), 2 high pitch (e.g. 240Hz)
    features = np.array([
        [100.0, 1200.0, 500.0, 2000.0, 0.05, 0.3, 1000.0, 0.2, 0.3, 0.3, 0.2, 5.0, 100.0, 0.8],
        [240.0, 2500.0, 900.0, 3500.0, 0.12, 0.4, 1800.0, 0.1, 0.2, 0.4, 0.3, 8.0, 240.0, 0.9],
        [105.0, 1250.0, 510.0, 2050.0, 0.05, 0.3, 1050.0, 0.2, 0.3, 0.3, 0.2, 5.0, 105.0, 0.8],
        [235.0, 2450.0, 880.0, 3450.0, 0.11, 0.4, 1750.0, 0.1, 0.2, 0.4, 0.3, 7.0, 235.0, 0.9],
    ])
    labels = cluster_segments_by_voice(features, num_speakers=2)
    assert len(labels) == 4
    # Samples 0 & 2 must share a cluster, and 1 & 3 must share the other
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_cluster_segments_by_voice_three_and_four_persons():
    rng = np.random.default_rng(42)
    # 3 distinct speakers (e.g. Doctor, Patient, Caregiver)
    s1 = rng.normal(loc=[100.0, 1200.0] + [0.0]*12, scale=1.0, size=(4, 14))
    s2 = rng.normal(loc=[200.0, 2200.0] + [0.0]*12, scale=1.0, size=(4, 14))
    s3 = rng.normal(loc=[300.0, 3200.0] + [0.0]*12, scale=1.0, size=(4, 14))
    X3 = np.vstack([s1, s2, s3])

    labels_3 = cluster_segments_by_voice(X3, num_speakers=3)
    assert len(np.unique(labels_3)) == 3

    # 4 distinct speakers (e.g. Doctor, Patient, Caregiver, Nurse)
    s4 = rng.normal(loc=[400.0, 4200.0] + [0.0]*12, scale=1.0, size=(4, 14))
    X4 = np.vstack([s1, s2, s3, s4])
    labels_4 = cluster_segments_by_voice(X4, num_speakers=4)
    assert len(np.unique(labels_4)) == 4


def test_caregiver_and_nurse_linguistic_recognition():
    # Caregiver speaking about patient
    role_cg, conf_cg, _ = estimate_speaker_role("My mother hasn't been eating well and she complained of dizziness.")
    assert role_cg == SpeakerRole.CAREGIVER
    assert conf_cg >= 0.60

    # Nurse taking vitals
    role_nurse, conf_nurse, _ = estimate_speaker_role("Let's check your blood pressure and vitals before the doctor comes in.")
    assert role_nurse == SpeakerRole.NURSE
    assert conf_nurse >= 0.60


def test_multi_speaker_consultation_3_person_room():
    # Room with Doctor, Patient, and Family Member (Caregiver)
    segments = [
        TranscriptSegment(id=1, text="Doctor: Good morning Mrs. Williams, what brings you in today?"),
        TranscriptSegment(id=2, text="Patient: I have had bad headaches and trouble sleeping."),
        TranscriptSegment(id=3, text="Caregiver: My mother was confused yesterday and she almost fell in the kitchen."),
        TranscriptSegment(id=4, text="Doctor: I see. Let me examine her pupils and reflexes."),
    ]
    annotated = diarize_consultation_audio(segments, audio_path=None)
    assert len(annotated) == 4
    assert annotated[0].speaker_role == SpeakerRole.DOCTOR
    assert annotated[1].speaker_role == SpeakerRole.PATIENT
    assert annotated[2].speaker_role == SpeakerRole.CAREGIVER
    assert annotated[3].speaker_role == SpeakerRole.DOCTOR


def test_diarize_consultation_audio_with_synthetic_audio():
    wav_path = get_synthetic_wav_path()
    assert wav_path.is_file()

    segments = [
        TranscriptSegment(id=1, text="Doctor, what brings you in today?", start_time=0.0, end_time=2.0),
        TranscriptSegment(id=2, text="Patient, I have had a severe sore throat.", start_time=3.0, end_time=6.7),
        TranscriptSegment(id=3, text="Doctor, any chest pain?", start_time=7.7, end_time=9.5),
        TranscriptSegment(id=4, text="Patient, no chest pain.", start_time=10.6, end_time=12.6),
    ]

    annotated = diarize_consultation_audio(segments, audio_path=wav_path, num_speakers=2)
    assert len(annotated) == 4
    assert annotated[0].speaker_role == SpeakerRole.DOCTOR
    assert annotated[1].speaker_role == SpeakerRole.PATIENT
    assert annotated[2].speaker_role == SpeakerRole.DOCTOR
    assert annotated[3].speaker_role == SpeakerRole.PATIENT


def test_doctor_patient_vocative_and_cluster_invariance():
    """Verify that addressing as 'Doctor' and addressing as 'Mr. Johnson' properly identifies roles without flip-flopping."""
    # Test vocative addressing in linguistic classifier
    role1, conf1, _ = estimate_speaker_role("Good morning, Mr. Johnson. How are you feeling today?")
    assert role1 == SpeakerRole.DOCTOR

    role2, conf2, _ = estimate_speaker_role("Good morning, Doctor. I've been experiencing some pain in my chest.")
    assert role2 == SpeakerRole.PATIENT

    role3, conf3, _ = estimate_speaker_role("I understand, Doctor, what tests do you recommend?")
    assert role3 == SpeakerRole.PATIENT

    role4, conf4, _ = estimate_speaker_role("Thank you for your trust, Mr. Johnson, in the meantime.")
    assert role4 == SpeakerRole.DOCTOR

    role5, conf5, _ = estimate_speaker_role("I trust your judgment, doctor. Please go ahead with the necessary tests.")
    assert role5 == SpeakerRole.PATIENT

    role6, conf6, _ = estimate_speaker_role("I'd like to order an electrocardiogram also called ECG.")
    assert role6 == SpeakerRole.DOCTOR
