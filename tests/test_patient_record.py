"""Unit tests for PatientRecord dataclasses and clinical data models."""

import pytest
from src.models.patient_record import (
    ConsultationRecord,
    PatientDemographics,
    TranscriptSegment,
    ExtractedFinding,
    MedicationFinding,
    VitalSigns,
    ExtractedClinicalInformation,
    SOAPNote,
    SpeakerRole,
    FindingStatus,
    ReviewStatus
)


def test_demographics_initialization():
    demo = PatientDemographics(
        patient_id="PT-9821",
        age=54,
        gender="Female",
        consultation_datetime="2026-10-10T10:00:00"
    )
    assert demo.patient_id == "PT-9821"
    assert demo.age == 54
    assert demo.gender == "Female"
    assert demo.consultation_datetime == "2026-10-10T10:00:00"


def test_missing_demographics_remain_none():
    demo = PatientDemographics()
    assert demo.patient_id is None
    assert demo.age is None
    assert demo.gender is None
    assert demo.consultation_datetime == ""


def test_vital_signs_missing_values_not_zero():
    vitals = VitalSigns()
    # Missing vitals MUST be None, never 0 or normal placeholders
    assert vitals.blood_pressure_systolic is None
    assert vitals.blood_pressure_diastolic is None
    assert vitals.heart_rate_bpm is None
    assert vitals.temperature_f is None
    assert vitals.temperature_c is None
    assert vitals.spo2_percent is None
    assert vitals.blood_glucose_mg_dl is None


def test_vital_signs_explicit_assignment():
    vitals = VitalSigns(
        blood_pressure_systolic=135,
        blood_pressure_diastolic=85,
        blood_pressure_str="135/85 mmHg",
        heart_rate_bpm=76,
        temperature_f=99.2,
        spo2_percent=98,
        blood_glucose_mg_dl=110,
        source_phrase="BP 135/85, pulse 76, temp 99.2F, O2 98%"
    )
    assert vitals.blood_pressure_systolic == 135
    assert vitals.blood_pressure_diastolic == 85
    assert vitals.heart_rate_bpm == 76
    assert vitals.temperature_f == 99.2
    assert vitals.spo2_percent == 98
    assert vitals.blood_glucose_mg_dl == 110


def test_transcript_segment_creation():
    seg = TranscriptSegment(
        id=1,
        text="Good morning, I have had a severe dry cough for three days.",
        start_time=0.5,
        end_time=3.8,
        speaker_role=SpeakerRole.PATIENT,
        role_confidence=0.85,
        speaker_reason="Patient symptom description pattern"
    )
    assert seg.id == 1
    assert seg.speaker_role == SpeakerRole.PATIENT
    assert seg.role_confidence == 0.85
    assert seg.start_time == 0.5
    assert seg.end_time == 3.8


def test_finding_status_negation_and_uncertainty():
    finding_positive = ExtractedFinding(
        category="symptom",
        name="cough",
        status=FindingStatus.PRESENT,
        duration="3 days",
        source_phrase="cough for 3 days"
    )
    finding_negative = ExtractedFinding(
        category="symptom",
        name="chest pain",
        status=FindingStatus.EXPLICITLY_ABSENT,
        source_phrase="patient denies chest pain"
    )
    finding_uncertain = ExtractedFinding(
        category="diagnosis",
        name="pneumonia",
        status=FindingStatus.UNCERTAIN,
        source_phrase="rule out pneumonia"
    )

    assert finding_positive.status == FindingStatus.PRESENT
    assert finding_negative.status == FindingStatus.EXPLICITLY_ABSENT
    assert finding_uncertain.status == FindingStatus.UNCERTAIN


def test_medication_finding_attributes():
    med = MedicationFinding(
        name="Amoxicillin",
        dosage="500 mg",
        frequency="three times daily",
        status=FindingStatus.PRESENT,
        is_current_plan=True,
        source_phrase="Take amoxicillin 500mg three times a day"
    )
    assert med.name == "Amoxicillin"
    assert med.dosage == "500 mg"
    assert med.frequency == "three times daily"
    assert med.is_current_plan is True


def test_consultation_record_default_state():
    record = ConsultationRecord()
    assert record.consultation_id is not None
    assert len(record.consultation_id) > 10
    assert record.review_status == ReviewStatus.DRAFT_PENDING_REVIEW
    assert "CLINICAL SAFETY NOTICE" in record.soap_note.review_warning
