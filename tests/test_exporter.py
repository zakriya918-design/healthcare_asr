"""Unit tests for HTML and PDF report export module."""

from pathlib import Path
from src.models.patient_record import (
    ConsultationRecord,
    ExtractedClinicalInformation,
    ExtractedFinding,
    FindingStatus,
    MedicationFinding,
    PatientDemographics,
    SOAPNote,
    VitalSigns
)
from src.processing.exporter import export_html_report, export_pdf_report, generate_html_report


def create_sample_consultation_record() -> ConsultationRecord:
    info = ExtractedClinicalInformation(
        vital_signs=VitalSigns(blood_pressure_str="120/80 mmHg", heart_rate_bpm=72, temperature_f=98.6),
        symptoms=[ExtractedFinding(category="symptoms", name="fever", status=FindingStatus.PRESENT)],
        medications=[MedicationFinding(name="Amoxicillin", dosage="500 mg", frequency="twice daily")]
    )
    soap = SOAPNote(
        subjective="Patient reports fever.",
        objective="Vitals normal.",
        assessment="Possible pharyngitis.",
        plan="Start Amoxicillin 500mg twice daily."
    )
    return ConsultationRecord(
        demographics=PatientDemographics(age=30, gender="Female"),
        raw_transcript="Doctor, I have a fever. Take Amoxicillin 500mg twice daily.",
        extracted_info=info,
        soap_note=soap
    )


def test_generate_html_report_contains_required_fields():
    record = create_sample_consultation_record()
    html_content = generate_html_report(record)

    assert "<!DOCTYPE html>" in html_content
    assert record.consultation_id in html_content
    assert "Patient reports fever." in html_content
    assert "CLINICAL SAFETY NOTICE" in html_content
    assert "Amoxicillin" in html_content


def test_export_html_report_file_creation(tmp_path: Path):
    record = create_sample_consultation_record()
    out_file = tmp_path / "test_report.html"

    result_path = export_html_report(record, out_file)
    assert result_path.exists()
    assert result_path.stat().st_size > 500
    assert "SOAP Note" in result_path.read_text(encoding="utf-8")


def test_export_pdf_report_file_creation(tmp_path: Path):
    record = create_sample_consultation_record()
    out_file = tmp_path / "test_report.pdf"

    result_path = export_pdf_report(record, out_file)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000
    assert result_path.read_bytes().startswith(b"%PDF")


def test_export_reports_with_custom_clinic_branding(tmp_path: Path):
    from src.processing.exporter import ClinicBranding
    record = create_sample_consultation_record()
    custom_brand = ClinicBranding(
        clinic_name="St. Jude Memorial Health Center",
        physician_name="Dr. Gregory House, MD",
        department="Diagnostic Medicine",
        facility_address="221B Baker St Medical Plaza",
        contact_info="help@stjudehealth.org"
    )

    html_file = tmp_path / "branded.html"
    export_html_report(record, html_file, branding=custom_brand)
    content = html_file.read_text(encoding="utf-8")
    assert "St. Jude Memorial Health Center" in content
    assert "Dr. Gregory House, MD" in content
    assert "Diagnostic Medicine" in content

    pdf_file = tmp_path / "branded.pdf"
    export_pdf_report(record, pdf_file, branding=custom_brand)
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 1000

