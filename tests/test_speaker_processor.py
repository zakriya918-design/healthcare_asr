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


def test_caregiver_proxy_discrimination():
    # Caregiver speaking about patient using 3rd person proxy
    role_cg1, conf1, _ = estimate_speaker_role("Doctor, my mother was very confused yesterday and she fell in the hallway.")
    assert role_cg1 == SpeakerRole.CAREGIVER
    assert conf1 >= 0.85

    role_cg2, conf2, _ = estimate_speaker_role("I drove my father in because he couldn't stand on his own.")
    assert role_cg2 == SpeakerRole.CAREGIVER
    assert conf2 >= 0.70

    # Patient speaking for themselves
    role_pt, conf_pt, _ = estimate_speaker_role("Doctor, my lower back has been hurting since yesterday.")
    assert role_pt == SpeakerRole.PATIENT
    assert conf_pt >= 0.85

    # Doctor inquiry starting with 'Doctor,'
    role_doc, conf_doc, _ = estimate_speaker_role("Doctor, how long have you had this fever?")
    assert role_doc == SpeakerRole.DOCTOR
    assert conf_doc >= 0.85


def test_three_way_dialogue_segregation():
    segments = [
        TranscriptSegment(id=1, text="Good morning Mr. Davis, how are you feeling today?", start_time=0.0, end_time=2.5),
        TranscriptSegment(id=2, text="I feel very weak and my chest hurts when I breathe.", start_time=3.1, end_time=6.4),
        TranscriptSegment(id=3, text="My father hasn't eaten since yesterday and we are very worried about him.", start_time=7.0, end_time=11.2),
        TranscriptSegment(id=4, text="Let me listen to your heart and lungs, and order a chest x-ray.", start_time=12.0, end_time=16.1),
    ]
    processed = process_transcript_segments(segments)
    assert processed[0].speaker_role == SpeakerRole.DOCTOR
    assert processed[1].speaker_role == SpeakerRole.PATIENT
    assert processed[2].speaker_role == SpeakerRole.CAREGIVER
    assert processed[3].speaker_role == SpeakerRole.DOCTOR
