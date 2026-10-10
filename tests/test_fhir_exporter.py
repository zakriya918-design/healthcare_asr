"""Tests for HL7 FHIR R4 clinical record exporter and validator."""

import json
from pathlib import Path
import pytest

from src.models.patient_record import (
    ConsultationRecord,
    PatientDemographics,
    ExtractedClinicalInformation,
    ExtractedFinding,
    MedicationFinding,
    VitalSigns,
    FindingStatus,
    ReviewStatus,
    SOAPNote
)
from src.processing.fhir_exporter import (
    build_fhir_bundle,
    export_fhir_json,
    validate_fhir_bundle,
    _sanitize_id
)


@pytest.fixture
def complex_consultation_record():
    return ConsultationRecord(
        demographics=PatientDemographics(
            patient_id="FHIR-PT-409",
            age=58,
            gender="Male",
            consultation_datetime="2026-10-10T14:30:00Z"
        ),
        raw_transcript="Doctor: Checking symptoms. Patient: I have asthma, no chest pain. Taking lisinopril 10mg daily. Allergic to penicillin.",
        extracted_info=ExtractedClinicalInformation(
            symptoms=[
                ExtractedFinding(category="symptom", name="asthma", status=FindingStatus.PRESENT),
                ExtractedFinding(category="symptom", name="chest pain", status=FindingStatus.EXPLICITLY_ABSENT)
            ],
            existing_conditions=[
                ExtractedFinding(category="condition", name="hypertension", status=FindingStatus.PRESENT)
            ],
            diagnoses=[
                ExtractedFinding(category="diagnosis", name="essential hypertension", status=FindingStatus.PRESENT)
            ],
            medications=[
                MedicationFinding(name="lisinopril", dosage="10mg", frequency="daily", is_current_plan=True)
            ],
            allergies=[
                ExtractedFinding(category="allergy", name="penicillin", status=FindingStatus.PRESENT)
            ],
            vital_signs=VitalSigns(
                blood_pressure_systolic=135,
                blood_pressure_diastolic=85,
                heart_rate_bpm=74,
                temperature_f=98.6,
                spo2_percent=98,
                blood_glucose_mg_dl=105
            )
        ),
        soap_note=SOAPNote(
            subjective="Patient reports asthma symptoms. Explicitly denies chest pain.",
            objective="BP 135/85 mmHg, HR 74 bpm, Temp 98.6 deg F, SpO2 98%.",
            assessment="Essential hypertension well controlled.",
            plan="Continue lisinopril 10mg daily.",
            missing_or_uncertain_info=["Respiratory rate not stated."],
            review_status=ReviewStatus.DRAFT_PENDING_REVIEW
        ),
        review_status=ReviewStatus.DRAFT_PENDING_REVIEW
    )


def test_sanitize_id():
    assert _sanitize_id("PT-1234") == "PT-1234"
    assert _sanitize_id("PT 1234 @#$") == "PT-1234"
    assert len(_sanitize_id("")) > 0


def test_build_fhir_bundle_structure(complex_consultation_record):
    bundle = build_fhir_bundle(complex_consultation_record)

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "document"
    assert len(bundle["entry"]) >= 6

    # 1. First entry must be Composition
    first_res = bundle["entry"][0]["resource"]
    assert first_res["resourceType"] == "Composition"
    assert first_res["status"] == "preliminary"  # Draft pending review
    assert len(first_res["section"]) >= 4

    # 2. Patient resource
    patient_entries = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Patient"]
    assert len(patient_entries) == 1
    pat = patient_entries[0]
    assert pat["gender"] == "male"
    assert pat["extension"][0]["valueQuantity"]["value"] == 58


def test_fhir_review_status_approved(complex_consultation_record):
    complex_consultation_record.review_status = ReviewStatus.REVIEWED_APPROVED
    bundle = build_fhir_bundle(complex_consultation_record)
    comp = bundle["entry"][0]["resource"]
    assert comp["status"] == "final"


def test_fhir_condition_negation_and_coding(complex_consultation_record):
    bundle = build_fhir_bundle(complex_consultation_record)
    conditions = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Condition"]
    assert len(conditions) >= 3

    # Negated finding (chest pain)
    neg_chest_pain = next((c for c in conditions if c["code"]["text"] == "chest pain"), None)
    assert neg_chest_pain is not None
    assert neg_chest_pain["verificationStatus"]["coding"][0]["code"] == "refuted"
    assert neg_chest_pain["clinicalStatus"]["coding"][0]["code"] == "resolved"

    # Positive finding (asthma)
    pos_asthma = next((c for c in conditions if c["code"]["text"] == "asthma"), None)
    assert pos_asthma is not None
    assert pos_asthma["verificationStatus"]["coding"][0]["code"] == "confirmed"
    assert pos_asthma["clinicalStatus"]["coding"][0]["code"] == "active"
    # Verify ICD-10 coding attached
    assert any(cd["system"] == "http://hl7.org/fhir/sid/icd-10-cm" for cd in pos_asthma["code"]["coding"])


def test_fhir_observations_vitals(complex_consultation_record):
    bundle = build_fhir_bundle(complex_consultation_record)
    observations = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"]
    assert len(observations) >= 5

    # Check Blood Pressure Observation with components
    bp_obs = next((o for o in observations if o["code"]["coding"][0]["code"] == "85354-9"), None)
    assert bp_obs is not None
    assert len(bp_obs["component"]) == 2
    assert bp_obs["component"][0]["valueQuantity"]["value"] == 135
    assert bp_obs["component"][1]["valueQuantity"]["value"] == 85

    # Check Heart Rate
    hr_obs = next((o for o in observations if o["code"]["coding"][0]["code"] == "8867-4"), None)
    assert hr_obs is not None
    assert hr_obs["valueQuantity"]["value"] == 74


def test_fhir_medication_and_allergy(complex_consultation_record):
    bundle = build_fhir_bundle(complex_consultation_record)
    meds = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "MedicationStatement"]
    assert len(meds) == 1
    assert meds[0]["medicationCodeableConcept"]["text"] == "lisinopril"
    assert "10mg" in meds[0]["dosage"][0]["text"]

    allergies = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "AllergyIntolerance"]
    assert len(allergies) == 1
    assert allergies[0]["code"]["text"] == "penicillin"


def test_export_fhir_json_and_validation(complex_consultation_record, tmp_path):
    dest = tmp_path / "fhir_bundle.json"
    exported_path = export_fhir_json(complex_consultation_record, dest)

    assert exported_path.is_file()
    data = json.loads(exported_path.read_text(encoding="utf-8"))

    is_valid, errors = validate_fhir_bundle(data)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_fhir_bundle_rejects_malformed():
    is_valid, errors = validate_fhir_bundle({"resourceType": "NotBundle"})
    assert is_valid is False
    assert any("Bundle" in e for e in errors)

    is_valid, errors = validate_fhir_bundle({
        "resourceType": "Bundle",
        "type": "document",
        "entry": [{"resource": {"resourceType": "Observation", "id": "123"}}]
    })
    assert is_valid is False
    assert any("Composition" in e for e in errors)
