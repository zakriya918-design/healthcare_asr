"""Unit tests for SOAP clinical note generation rules and formatting."""

import pytest
from src.processing.soap_note import (
    generate_soap_note,
    format_subjective_section,
    format_objective_section,
    format_assessment_section,
    format_plan_section,
    identify_missing_or_uncertain_information
)
from src.models.patient_record import (
    ExtractedClinicalInformation,
    ExtractedFinding,
    FindingStatus,
    MedicationFinding,
    ReviewStatus,
    VitalSigns
)


@pytest.fixture
def sample_clinical_data():
    info = ExtractedClinicalInformation(
        symptoms=[
            ExtractedFinding(category="symptom", name="dry cough", status=FindingStatus.PRESENT, duration="3 days"),
            ExtractedFinding(category="symptom", name="fever", status=FindingStatus.EXPLICITLY_ABSENT),
            ExtractedFinding(category="symptom", name="chest pain", status=FindingStatus.EXPLICITLY_ABSENT)
        ],
        existing_conditions=[
            ExtractedFinding(category="condition", name="hypertension", status=FindingStatus.PRESENT)
        ],
        allergies=[
            ExtractedFinding(category="allergy", name="Penicillin", status=FindingStatus.PRESENT)
        ],
        vital_signs=VitalSigns(
            blood_pressure_str="130/80 mmHg",
            heart_rate_bpm=76
            # Temperature and SpO2 missing
        ),
        diagnoses=[
            ExtractedFinding(category="diagnosis", name="Acute Bronchitis", status=FindingStatus.PRESENT),
            ExtractedFinding(category="diagnosis", name="Pneumonia", status=FindingStatus.UNCERTAIN)
        ],
        medications=[
            MedicationFinding(name="Albuterol", dosage="2 puffs", frequency="every 4 hours as needed", is_current_plan=True)
        ],
        investigations=[
            ExtractedFinding(category="investigation", name="Chest X-Ray", status=FindingStatus.PRESENT)
        ],
        treatment_plans=[
            ExtractedFinding(category="treatment_plan", name="Rest And Fluids", status=FindingStatus.PRESENT)
        ],
        advice_followup=[
            ExtractedFinding(category="advice_followup", name="Follow-up instructions", status=FindingStatus.PRESENT, details="Return if shortness of breath develops")
        ]
    )
    return info


def test_soap_subjective_formatting(sample_clinical_data):
    subj = format_subjective_section(sample_clinical_data, demographics_summary="45-year-old female")
    assert "45-year-old female" in subj
    assert "dry cough (3 days)" in subj
    assert "Denies fever, chest pain" in subj
    assert "Existing Conditions: hypertension" in subj
    assert "Allergies: Penicillin" in subj


def test_soap_objective_missing_vitals_not_zero(sample_clinical_data):
    obj = format_objective_section(sample_clinical_data.vital_signs)
    assert "Blood Pressure: 130/80 mmHg" in obj
    assert "Heart Rate: 76 bpm" in obj
    assert "Body Temperature: Not recorded" in obj
    assert "Oxygen Saturation (SpO2): Not recorded" in obj
    assert "Blood Glucose: Not recorded" in obj


def test_soap_assessment_preserves_uncertainty(sample_clinical_data):
    assessment = format_assessment_section(sample_clinical_data)
    assert "Acute Bronchitis" in assessment
    assert "Pneumonia (Tentative / To rule out)" in assessment


def test_soap_plan_formatting(sample_clinical_data):
    plan = format_plan_section(sample_clinical_data)
    assert "Albuterol (2 puffs, every 4 hours as needed)" in plan
    assert "Chest X-Ray" in plan
    assert "Rest And Fluids" in plan
    assert "Return if shortness of breath develops" in plan


def test_soap_missing_information_detection(sample_clinical_data):
    missing = identify_missing_or_uncertain_information(sample_clinical_data)
    assert any("temperature" in m.lower() for m in missing)
    assert any("oxygen saturation" in m.lower() for m in missing)
    assert any("pneumonia" in m.lower() and "tentative" in m.lower() for m in missing)


def test_generate_soap_note_complete(sample_clinical_data):
    note = generate_soap_note(sample_clinical_data)
    assert note.review_status == ReviewStatus.DRAFT_PENDING_REVIEW
    assert "CLINICAL SAFETY NOTICE" in note.review_warning
    assert "dry cough" in note.subjective
    assert "130/80 mmHg" in note.objective
    assert "Acute Bronchitis" in note.assessment
    assert "Albuterol" in note.plan
    assert len(note.missing_or_uncertain_info) >= 2
