"""Test suite for Streamlit Web Application (streamlit_app.py)."""

import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from src.models.patient_record import (
    ConsultationRecord,
    PatientDemographics,
    ExtractedClinicalInformation,
    ExtractedFinding,
    MedicationFinding,
    VitalSigns,
    SOAPNote,
    ReviewStatus,
    FindingStatus,
    SpeakerRole,
    TranscriptSegment,
)
from src.processing.exporter import ClinicBranding
from src.utils.paths import get_output_dir


def create_sample_record() -> ConsultationRecord:
    """Helper to create a deterministic sample ConsultationRecord for testing."""
    return ConsultationRecord(
        consultation_id="test-consult-12345",
        demographics=PatientDemographics(patient_id="PT-9876", age=45, gender="Female"),
        raw_transcript="Doctor: Hello. Patient: I have a sore throat and fever. Denies chest pain. Doctor: Taking Amoxicillin.",
        segments=[
            TranscriptSegment(id=0, text="Doctor: Hello.", start_time=0.0, end_time=2.0, speaker_role=SpeakerRole.DOCTOR),
            TranscriptSegment(id=1, text="Patient: I have a sore throat and fever.", start_time=2.0, end_time=6.0, speaker_role=SpeakerRole.PATIENT),
            TranscriptSegment(id=2, text="Doctor: Taking Amoxicillin.", start_time=6.0, end_time=9.0, speaker_role=SpeakerRole.DOCTOR),
        ],
        extracted_info=ExtractedClinicalInformation(
            symptoms=[
                ExtractedFinding(category="symptom", name="sore throat", status=FindingStatus.PRESENT),
                ExtractedFinding(category="symptom", name="chest pain", status=FindingStatus.EXPLICITLY_ABSENT),
            ],
            medications=[
                MedicationFinding(name="Amoxicillin", dosage="500mg", frequency="TID", status=FindingStatus.PRESENT),
            ],
            vital_signs=VitalSigns(
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                heart_rate_bpm=72,
                temperature_f=101.2,
                spo2_percent=99,
            ),
            diagnoses=[
                ExtractedFinding(category="diagnosis", name="acute pharyngitis", status=FindingStatus.PRESENT),
            ],
        ),
        soap_note=SOAPNote(
            subjective="Sore throat x 3 days. Denies chest pain.",
            objective="BP 120/80, HR 72, Temp 101.2 F, SpO2 99%. Pharyngeal erythema.",
            assessment="Acute pharyngitis.",
            plan="Amoxicillin 500mg PO TID x 10 days.",
            review_status=ReviewStatus.DRAFT_PENDING_REVIEW,
        ),
        review_status=ReviewStatus.DRAFT_PENDING_REVIEW,
    )


APP_PATH = Path(__file__).parent.parent / "streamlit_app.py"


def test_streamlit_app_initial_state():
    """Verify Streamlit app initializes cleanly without unhandled exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert len(at.exception) == 0, f"App execution raised unexpected exceptions: {at.exception}"
    assert len(at.sidebar.selectbox) >= 1
    assert len(at.sidebar.text_input) >= 4  # clinic branding inputs
    assert len(at.sidebar.radio) >= 1  # ingestion mode


def test_streamlit_app_renders_active_encounter():
    """Verify Streamlit app renders all clinical tabs and components when a record is loaded."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    # Pre-populate session state with sample record
    at.session_state["record"] = create_sample_record()
    at.session_state["audio_path"] = None
    at.session_state["branding"] = ClinicBranding()
    at.run()

    assert len(at.exception) == 0, f"Encounter view raised exceptions: {at.exception}"
    # Tabs exist
    assert len(at.tabs) == 6
    # Safety banner or draft notice rendered
    assert len(at.text_area) >= 5  # raw transcript + 4 SOAP note sections
    # Download buttons in export tab exist
    assert len(at.download_button) >= 3


def test_streamlit_app_sign_off_workflow():
    """Verify clinician sign-off action transitions draft to REVIEWED_APPROVED."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sample_rec = create_sample_record()
    at.session_state["record"] = sample_rec
    at.session_state["branding"] = ClinicBranding()
    at.run()

    assert at.session_state["record"].review_status == ReviewStatus.DRAFT_PENDING_REVIEW

    # Check attestation checkbox
    attest_cb = at.checkbox(key=None)  # finds the attestation checkbox
    for cb in at.checkbox:
        if "attest" in cb.label.lower():
            cb.check()
            break

    # Click the sign button
    sign_btn = None
    for btn in at.button:
        if "sign & finalize" in btn.label.lower():
            sign_btn = btn
            break

    assert sign_btn is not None
    sign_btn.click().run()

    assert len(at.exception) == 0
    updated_rec = at.session_state["record"]
    assert updated_rec.review_status == ReviewStatus.REVIEWED_APPROVED
    assert updated_rec.signed_by is not None
