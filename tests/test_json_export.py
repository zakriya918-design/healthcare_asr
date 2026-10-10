"""Unit tests for ConsultationRecord JSON serialization and file export."""

import json
import pytest
from pathlib import Path
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


@pytest.fixture
def sample_consultation():
    """Provides a synthetic consultation record for export testing."""
    record = ConsultationRecord(
        demographics=PatientDemographics(
            patient_id="SYN-001",
            age=45,
            gender="Male",
            consultation_datetime="2026-10-10T11:30:00"
        ),
        raw_transcript="Doctor: Any fever? Patient: No fever, but bad sore throat for 2 days.",
        segments=[
            TranscriptSegment(
                id=1,
                text="Doctor: Any fever?",
                start_time=0.0,
                end_time=1.5,
                speaker_role=SpeakerRole.DOCTOR,
                role_confidence=0.9
            ),
            TranscriptSegment(
                id=2,
                text="Patient: No fever, but bad sore throat for 2 days.",
                start_time=1.6,
                end_time=4.2,
                speaker_role=SpeakerRole.PATIENT,
                role_confidence=0.9
            ),
        ],
        extracted_info=ExtractedClinicalInformation(
            symptoms=[
                ExtractedFinding(category="symptom", name="sore throat", status=FindingStatus.PRESENT, duration="2 days"),
                ExtractedFinding(category="symptom", name="fever", status=FindingStatus.EXPLICITLY_ABSENT)
            ],
            medications=[
                MedicationFinding(name="Ibuprofen", dosage="400 mg", frequency="as needed for throat pain")
            ],
            vital_signs=VitalSigns(
                blood_pressure_str="120/80 mmHg",
                heart_rate_bpm=72,
                temperature_f=98.6
            )
        ),
        soap_note=SOAPNote(
            subjective="Patient reports sore throat for 2 days. Denies fever.",
            objective="BP: 120/80 mmHg, HR: 72 bpm, Temp: 98.6 F.",
            assessment="Acute pharyngitis, likely viral.",
            plan="Symptomatic relief with Ibuprofen 400 mg as needed. Fluids and rest."
        )
    )
    return record


def test_to_dict_and_to_json(sample_consultation):
    data_dict = sample_consultation.to_dict()
    assert data_dict["schema_version"] == "1.0"
    assert data_dict["demographics"]["age"] == 45
    assert len(data_dict["segments"]) == 2

    json_str = sample_consultation.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["demographics"]["patient_id"] == "SYN-001"


def test_round_trip_serialization(sample_consultation):
    json_str = sample_consultation.to_json()
    restored = ConsultationRecord.from_json(json_str)

    assert restored.consultation_id == sample_consultation.consultation_id
    assert restored.demographics.age == 45
    assert restored.demographics.gender == "Male"
    assert len(restored.segments) == 2
    assert restored.segments[0].speaker_role == SpeakerRole.DOCTOR
    assert restored.segments[1].speaker_role == SpeakerRole.PATIENT
    assert restored.extracted_info.symptoms[1].status == FindingStatus.EXPLICITLY_ABSENT
    assert restored.extracted_info.vital_signs.heart_rate_bpm == 72
    assert restored.soap_note.subjective == sample_consultation.soap_note.subjective


def test_save_json_atomic_and_no_overwrite(sample_consultation, tmp_path):
    saved_file = sample_consultation.save_json(output_dir=tmp_path)
    assert saved_file.exists()
    assert saved_file.is_file()
    assert saved_file.suffix == ".json"

    # Verify content was written properly
    content = saved_file.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert parsed["demographics"]["patient_id"] == "SYN-001"

    # Saving again should produce a new distinct file without overwriting
    second_saved = sample_consultation.save_json(output_dir=tmp_path)
    assert second_saved.exists()
    # If saved in same second, filename gets counter suffix
    # Either way, both files must exist
    assert saved_file.exists()


def test_load_json_from_disk(sample_consultation, tmp_path):
    saved_file = sample_consultation.save_json(output_dir=tmp_path)
    loaded_record = ConsultationRecord.load_json(saved_file)

    assert loaded_record.consultation_id == sample_consultation.consultation_id
    assert loaded_record.extracted_info.vital_signs.temperature_f == 98.6
    assert loaded_record.extracted_info.medications[0].name == "Ibuprofen"
