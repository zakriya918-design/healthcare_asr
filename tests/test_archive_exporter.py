"""Tests for consultation bundle archive exporter and integrity verification."""

import io
import json
from pathlib import Path
import zipfile
import pytest

from src.models.patient_record import (
    ConsultationRecord,
    PatientDemographics,
    ExtractedClinicalInformation,
    ExtractedFinding,
    FindingStatus,
    SOAPNote
)
from src.processing.archive_exporter import (
    compute_sha256_bytes,
    compute_sha256_file,
    export_consultation_zip,
    verify_consultation_zip
)
from src.processing.exporter import ClinicBranding


@pytest.fixture
def sample_record():
    return ConsultationRecord(
        demographics=PatientDemographics(patient_id="ZIP-TEST-001", age=45, gender="Female"),
        raw_transcript="Doctor: How are you? Patient: I have a mild cough.",
        extracted_info=ExtractedClinicalInformation(
            symptoms=[ExtractedFinding(category="symptom", name="cough", status=FindingStatus.PRESENT)]
        ),
        soap_note=SOAPNote(
            subjective="Patient reports mild cough.",
            objective="Vitals unrecorded.",
            assessment="Upper respiratory irritation.",
            plan="Hydration and rest."
        )
    )


def test_sha256_helpers(tmp_path):
    data = b"Hello, Medical Documentation!"
    h1 = compute_sha256_bytes(data)
    assert len(h1) == 64

    test_file = tmp_path / "test.txt"
    test_file.write_bytes(data)
    h2 = compute_sha256_file(test_file)
    assert h1 == h2


def test_export_consultation_zip_without_audio(sample_record, tmp_path):
    branding = ClinicBranding(clinic_name="Metro Clinic", physician_name="Dr. Roberts")
    zip_path = export_consultation_zip(
        record=sample_record,
        output_dir=tmp_path,
        branding=branding
    )

    assert zip_path.is_file()
    assert zip_path.suffix == ".zip"

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = zf.namelist()
        assert "record.json" in names
        assert "report.html" in names
        assert "report.pdf" in names
        assert "manifest.json" in names
        assert not any(n.endswith(".wav") for n in names)

        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["patient_id"] == "ZIP-TEST-001"
        assert manifest["verification_algorithm"] == "SHA-256"
        assert "record.json" in manifest["files"]
        assert "report.html" in manifest["files"]
        assert "report.pdf" in manifest["files"]

    is_valid, msg, manifest = verify_consultation_zip(zip_path)
    assert is_valid is True
    assert "verified" in msg.lower()


def test_export_consultation_zip_with_audio(sample_record, tmp_path):
    dummy_audio = tmp_path / "encounter.wav"
    dummy_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00data\x00\x00\x00\x00")

    zip_path = export_consultation_zip(
        record=sample_record,
        output_dir=tmp_path,
        audio_path=dummy_audio,
        zip_filename="custom_encounter.zip"
    )

    assert zip_path.name == "custom_encounter.zip"

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = zf.namelist()
        assert "audio_encounter.wav" in names

    is_valid, msg, _ = verify_consultation_zip(zip_path)
    assert is_valid is True


def test_verify_consultation_zip_tamper_detection(sample_record, tmp_path):
    zip_path = export_consultation_zip(sample_record, output_dir=tmp_path)

    # Corrupt one file in the zip by recreating archive with modified report.html
    tampered_zip = tmp_path / "tampered.zip"
    with zipfile.ZipFile(str(zip_path), "r") as src_zf:
        with zipfile.ZipFile(str(tampered_zip), "w") as dst_zf:
            for item in src_zf.infolist():
                content = src_zf.read(item.filename)
                if item.filename == "report.html":
                    content = content + b"<!-- Tampered Content Injection -->"
                dst_zf.writestr(item, content)

    is_valid, msg, _ = verify_consultation_zip(tampered_zip)
    assert is_valid is False
    assert "Checksum mismatch" in msg


def test_verify_consultation_zip_missing_manifest(tmp_path):
    bad_zip = tmp_path / "no_manifest.zip"
    with zipfile.ZipFile(str(bad_zip), "w") as zf:
        zf.writestr("test.txt", "data")

    is_valid, msg, _ = verify_consultation_zip(bad_zip)
    assert is_valid is False
    assert "missing manifest.json" in msg


def test_verify_consultation_zip_nonexistent_file(tmp_path):
    nonexistent = tmp_path / "does_not_exist.zip"
    is_valid, msg, _ = verify_consultation_zip(nonexistent)
    assert is_valid is False
    assert "not found" in msg
