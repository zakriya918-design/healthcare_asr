"""Unit tests for rule-based speaker role estimation."""

import pytest
from src.processing.speaker_processor import estimate_speaker_role, process_transcript_segments
from src.models.patient_record import SpeakerRole, TranscriptSegment


def test_explicit_prefixes():
    role_doc, conf_doc, reason_doc = estimate_speaker_role("Doctor: Good morning, how are you feeling?")
    assert role_doc == SpeakerRole.DOCTOR
    assert conf_doc >= 0.90
    assert "Explicit" in reason_doc

    role_pt, conf_pt, reason_pt = estimate_speaker_role("Patient: I have had a bad cough for three days.")
    assert role_pt == SpeakerRole.PATIENT
    assert conf_pt >= 0.90
    assert "Explicit" in reason_pt


def test_clinician_patterns():
    # Clinician questions / examinations
    role, conf, _ = estimate_speaker_role("What brings you in today? Let me examine your chest.")
    assert role == SpeakerRole.DOCTOR
    assert conf >= 0.60

    role_rx, _, _ = estimate_speaker_role("I am going to prescribe amoxicillin 500 mg twice daily.")
    assert role_rx == SpeakerRole.DOCTOR


def test_patient_patterns():
    # Patient complaints
    role, conf, _ = estimate_speaker_role("My throat hurts and I have been having chills since yesterday.")
    assert role == SpeakerRole.PATIENT
    assert conf >= 0.60

    role_pt, _, _ = estimate_speaker_role("I took some ibuprofen at home but it did not help.")
    assert role_pt == SpeakerRole.PATIENT


def test_ambiguous_utterance_defaults_to_unknown():
    role, conf, _ = estimate_speaker_role("Good morning.")
    assert role == SpeakerRole.UNKNOWN
    assert conf <= 0.30

    role_random, conf_random, _ = estimate_speaker_role("The quick brown fox jumps.")
    assert role_random == SpeakerRole.UNKNOWN
    assert conf_random == 0.0


def test_process_transcript_segments_sequence():
    raw_segments = [
        TranscriptSegment(id=1, text="Doctor: What brings you in today?"),
        TranscriptSegment(id=2, text="I have had a sharp pain in my knee for 2 days."),
        TranscriptSegment(id=3, text="Yes."),
        TranscriptSegment(id=4, text="Let's examine the joint and order an x-ray.")
    ]

    processed = process_transcript_segments(raw_segments)
    assert processed[0].speaker_role == SpeakerRole.DOCTOR
    assert processed[1].speaker_role == SpeakerRole.PATIENT
    assert processed[3].speaker_role == SpeakerRole.DOCTOR
