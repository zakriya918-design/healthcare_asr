"""Test suite for FastEHR REST API and Clinical Interoperability Server."""

import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.utils.paths import get_synthetic_wav_path


@pytest.fixture(scope="module")
def client():
    """Shared FastAPI TestClient fixture."""
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "version" in data
    assert "disclaimer" in data


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "speech_engine" in data
    assert "storage" in data


def test_process_text_consultation_pipeline(client):
    payload = {
        "transcript": (
            "Doctor: Good morning John, what brings you in today?\n"
            "Patient: Hello doctor, I have a painful sore throat and fever for 3 days.\n"
            "Doctor: Do you have chest pain or breathing difficulty?\n"
            "Patient: No chest pain at all. Denies difficulty breathing.\n"
            "Doctor: Blood pressure is 120/80, pulse 75, temperature 101.4 F.\n"
            "Doctor: You have acute pharyngitis. I will prescribe Amoxicillin 500mg three times daily."
        ),
        "model_size": "tiny.en",
        "clinic_name": "Apex Test Clinic",
        "physician_name": "Dr. Sarah Jenkins, MD",
    }
    response = client.post("/api/v1/consultations/text", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "consultation_id" in data
    consult_id = data["consultation_id"]
    assert data["review_status"] == "draft_pending_review"
    assert "soap_note" in data
    assert "sore throat" in data["soap_note"]["subjective"].lower()
    assert "denies" in data["soap_note"]["subjective"].lower()
    assert "icd10_recommendations" in data
    assert len(data["icd10_recommendations"]) > 0

    # Verify encounter retrieval
    get_res = client.get(f"/api/v1/consultations/{consult_id}")
    assert get_res.status_code == 200
    assert get_res.json()["consultation_id"] == consult_id

    # Verify FHIR R4 Bundle endpoint
    fhir_res = client.get(f"/api/v1/consultations/{consult_id}/fhir")
    assert fhir_res.status_code == 200
    fhir_bundle = fhir_res.json()
    assert fhir_bundle["resourceType"] == "Bundle"
    assert fhir_bundle["type"] == "document"
    assert len(fhir_bundle["entry"]) >= 3

    # Verify PDF export endpoint
    pdf_res = client.get(f"/api/v1/consultations/{consult_id}/pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000

    # Verify ZIP Archive export endpoint
    zip_res = client.get(f"/api/v1/consultations/{consult_id}/archive")
    assert zip_res.status_code == 200
    assert zip_res.headers["content-type"] == "application/zip"
    assert len(zip_res.content) > 1000

    # Verify clinician attestation sign-off endpoint
    sign_payload = {
        "signed_by": "Dr. Sarah Jenkins, MD",
        "license_npi": "NPI-98471203",
        "attestation_confirmed": True,
        "edited_plan": "Amoxicillin 500mg PO TID x 10 days. Increased fluids and rest.",
    }
    sign_res = client.post(f"/api/v1/consultations/{consult_id}/sign-off", json=sign_payload)
    assert sign_res.status_code == 200
    signed_data = sign_res.json()
    assert signed_data["review_status"] == "reviewed_approved"
    assert "Dr. Sarah Jenkins" in signed_data["signed_by"]


def test_list_and_search_consultations(client):
    response = client.get("/api/v1/consultations")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) > 0


def test_audio_diagnostics_endpoint(client):
    synth_path = get_synthetic_wav_path()
    assert synth_path.is_file()

    with open(synth_path, "rb") as f:
        response = client.post(
            "/api/v1/audio/diagnostics",
            files={"audio": ("synthetic.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "peak_db" in data
    assert "rms_db" in data
    assert "snr_db" in data
    assert "quality_rating" in data


def test_audio_enhance_endpoint(client):
    synth_path = get_synthetic_wav_path()
    assert synth_path.is_file()

    with open(synth_path, "rb") as f:
        response = client.post(
            "/api/v1/audio/enhance",
            files={"audio": ("noisy_input.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 1000
