"""Automated tests for HealthcareASRApp Tkinter GUI interface and state transitions."""

import queue
import tkinter as tk
from unittest.mock import MagicMock, patch
import pytest

from app import HealthcareASRApp
from src.models.patient_record import (
    ConsultationRecord,
    ExtractedClinicalInformation,
    ExtractedFinding,
    FindingStatus,
    MedicationFinding,
    PatientDemographics,
    ReviewStatus,
    SOAPNote,
    SpeakerRole,
    TranscriptSegment,
    VitalSigns
)


@pytest.fixture(scope="module")
def shared_tk_root():
    """Module-level Tk root to avoid multiple Tk instances in the same process."""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tkinter display not available in headless environment.")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def tk_app(shared_tk_root):
    """Instantiates a clean HealthcareASRApp on the shared root for each test."""
    # Clear existing child widgets from shared root
    for child in shared_tk_root.winfo_children():
        child.destroy()

    app = HealthcareASRApp(shared_tk_root)
    shared_tk_root.update_idletasks()
    return app


def test_gui_initialization_state(tk_app):
    assert tk_app.is_busy is False
    assert tk_app.current_record is None
    assert str(tk_app.btn_start["state"]) == str(tk.NORMAL)
    assert str(tk_app.btn_stop["state"]) == str(tk.DISABLED)
    assert str(tk_app.btn_export["state"]) == str(tk.DISABLED)
    assert "Idle" in tk_app.lbl_record_status["text"]


def test_gui_render_consultation_record(tk_app):
    # Assemble mock consultation record
    record = ConsultationRecord(
        demographics=PatientDemographics(age=52, gender="Male"),
        raw_transcript="Doctor: How are you? Patient: I have had a severe cough for three days.",
        segments=[
            TranscriptSegment(id=1, text="How are you?", speaker_role=SpeakerRole.DOCTOR, role_confidence=0.85, start_time=0.0, end_time=1.5),
            TranscriptSegment(id=2, text="I have had a severe cough for three days.", speaker_role=SpeakerRole.PATIENT, role_confidence=0.90, start_time=1.8, end_time=4.2),
        ],
        extracted_info=ExtractedClinicalInformation(
            symptoms=[ExtractedFinding(category="symptom", name="cough", status=FindingStatus.PRESENT, duration="3 days")],
            vital_signs=VitalSigns(blood_pressure_str="125/80 mmHg", heart_rate_bpm=72),
            medications=[MedicationFinding(name="Dextromethorphan", dosage="10 mg", frequency="daily", is_current_plan=True)]
        ),
        soap_note=SOAPNote(
            subjective="Patient reports cough (3 days).",
            objective="Blood Pressure: 125/80 mmHg.",
            assessment="Upper respiratory cough.",
            plan="Dextromethorphan 10 mg daily.",
            review_status=ReviewStatus.DRAFT_PENDING_REVIEW
        )
    )

    tk_app._handle_transcription_complete(record)

    # Check transcript tab populated
    assert "severe cough" in tk_app.txt_transcript.get("1.0", tk.END)
    tree_items = tk_app.tree_segments.get_children()
    assert len(tree_items) == 2

    # Check medical findings tab populated
    medical_text = tk_app.txt_medical.get("1.0", tk.END)
    assert "cough" in medical_text
    assert "125/80 mmHg" in medical_text

    # Check SOAP tab populated
    soap_text = tk_app.txt_soap.get("1.0", tk.END)
    assert "Patient reports cough" in soap_text
    assert "DRAFT FOR REVIEW" in soap_text
    assert "CLINICAL SAFETY NOTICE" in soap_text

    # Check JSON tab populated
    json_text = tk_app.txt_json.get("1.0", tk.END)
    assert "consultation_id" in json_text

    # Export button is enabled
    assert str(tk_app.btn_export["state"]) == str(tk.NORMAL)


def test_gui_clear_consultation(tk_app):
    tk_app.txt_transcript.insert("1.0", "Temporary transcript")
    tk_app.txt_soap.insert("1.0", "Temporary SOAP")
    tk_app.btn_export.config(state=tk.NORMAL)

    tk_app.clear_consultation()

    assert tk_app.current_record is None
    assert str(tk_app.btn_export["state"]) == str(tk.DISABLED)
    assert tk_app.txt_transcript.get("1.0", tk.END).strip() == ""
    assert tk_app.txt_soap.get("1.0", tk.END).strip() == ""


def test_gui_worker_error_handling(tk_app):
    with patch("tkinter.messagebox.showerror") as mock_err:
        tk_app.worker_queue.put({"type": "error", "error_msg": "Test error occurred"})
        tk_app._poll_worker_queue()
        mock_err.assert_called_once_with("Processing Error", "Test error occurred")
        assert "Test error occurred" in tk_app.lbl_feedback["text"]
        assert tk_app.is_busy is False


def test_gui_model_selector_switch(tk_app):
    assert tk_app.selected_model_size == "tiny.en"
    assert "tiny.en" in tk_app.combo_model.get()

    tk_app.combo_model.set("base.en")
    tk_app._on_model_changed()

    assert tk_app.selected_model_size == "base.en"
    assert "base.en" in tk_app.lbl_asr_status["text"]
    assert "base.en" in tk_app.lbl_feedback["text"]
    assert tk_app.transcriber is None


def test_gui_sign_off_consultation(tk_app):
    record = ConsultationRecord(
        demographics=PatientDemographics(age=40, gender="Female"),
        raw_transcript="Doctor: Hello. Patient: My shoulder aches.",
        soap_note=SOAPNote(subjective="Shoulder ache.", assessment="Tendinitis.", plan="Rest.")
    )
    tk_app._handle_transcription_complete(record)

    with patch("tkinter.simpledialog.askstring", return_value="Sarah Connor, MD"):
        with patch("tkinter.messagebox.showinfo") as mock_info:
            tk_app.sign_off_consultation()
            mock_info.assert_called_once()
            assert tk_app.current_record.review_status == ReviewStatus.REVIEWED_APPROVED
            assert "Dr. Sarah Connor" in tk_app.txt_soap.get("1.0", tk.END)
            assert "REVIEWED & APPROVED" in tk_app.lbl_soap_warning["text"]


def test_gui_open_saved_record(tk_app, tmp_path):
    record = ConsultationRecord(
        demographics=PatientDemographics(age=65, gender="Male"),
        raw_transcript="Doctor: Follow-up visit. Patient: Feeling better.",
        soap_note=SOAPNote(subjective="Feeling better.", assessment="Stable.", plan="Continue meds.")
    )
    saved_file = record.save_json(output_dir=tmp_path)

    with patch("tkinter.filedialog.askopenfilename", return_value=str(saved_file)):
        tk_app.open_saved_record()
        assert tk_app.current_record is not None
        assert tk_app.current_record.demographics.age == 65
        assert "Feeling better" in tk_app.txt_transcript.get("1.0", tk.END)


def test_gui_mic_selector_switch(tk_app):
    assert tk_app.selected_mic_index is None
    assert "Default System Mic" in tk_app.combo_mic.get()

    tk_app.combo_mic.set("[3] Realtek USB Audio")
    tk_app._on_mic_changed()
    assert tk_app.selected_mic_index == 3
    assert "Realtek USB Audio" in tk_app.lbl_feedback["text"]

    tk_app.combo_mic.set("Default System Mic")
    tk_app._on_mic_changed()
    assert tk_app.selected_mic_index is None


def test_gui_record_browser_window(tk_app, tmp_path):
    record = ConsultationRecord(
        demographics=PatientDemographics(patient_id="BROWSER-PT-1", age=52, gender="Female"),
        raw_transcript="Doctor: Checking blood pressure. Patient: Feeling fine.",
        soap_note=SOAPNote(subjective="Feeling fine.", assessment="Hypertension stable.", plan="Continue lisinopril.")
    )
    record.save_json(output_dir=tmp_path)

    with patch("src.processing.record_search.OUTPUT_DIR", tmp_path, create=True):
        win = tk_app.open_record_browser()
        assert win is not None
        assert "Past Consultation Records Browser" in win.title()
        # Cleanly destroy window after test
        win.destroy()


def test_gui_audio_diagnostic_dialog(tk_app):
    win = tk_app.open_audio_diagnostic_dialog()
    assert win is not None
    assert "Microphone Pre-Flight Diagnostic" in win.title()
    win.destroy()


def test_gui_icd10_tab_render(tk_app):
    record = ConsultationRecord(
        demographics=PatientDemographics(age=45, gender="Male"),
        raw_transcript="Doctor: Diagnosis is hypertension and asthma.",
        extracted_info=ExtractedClinicalInformation(
            diagnoses=[ExtractedFinding(category="diagnosis", name="asthma", status=FindingStatus.PRESENT)],
            existing_conditions=[ExtractedFinding(category="condition", name="hypertension", status=FindingStatus.PRESENT)]
        )
    )
    tk_app._handle_transcription_complete(record)

    items = tk_app.tree_icd10.get_children()
    assert len(items) >= 2

    codes = [tk_app.tree_icd10.item(i, "values")[0] for i in items]
    assert "J45.909" in codes
    assert "I10" in codes

    # Test reset clears the table
    tk_app.clear_consultation()
    assert len(tk_app.tree_icd10.get_children()) == 0


def test_gui_vu_meter(tk_app):
    assert tk_app.canvas_vu is not None
    assert tk_app.vu_bar_id is not None
    assert tk_app.lbl_vu_text["text"] == "-- dB"

    # Simulate level callback
    tk_app._on_audio_level(0.75, 0.5)
    assert tk_app.latest_peak == 0.75
    assert tk_app.latest_rms == 0.5

    # Mock recorder active
    mock_recorder = MagicMock()
    mock_recorder.is_recording = True
    tk_app.recorder = mock_recorder

    # Trigger update
    tk_app._update_vu_meter()
    coords = tk_app.canvas_vu.coords(tk_app.vu_bar_id)
    assert coords[2] > 0  # Width is greater than 0
    assert "dB" in tk_app.lbl_vu_text["text"]

    # When recorder stops, meter returns to idle
    mock_recorder.is_recording = False
    tk_app._update_vu_meter()
    assert tk_app.lbl_vu_text["text"] == "-- dB"


def test_gui_export_zip_and_fhir(tk_app, tmp_path):
    assert tk_app.btn_export_zip["state"] == tk.DISABLED
    assert tk_app.btn_export_fhir["state"] == tk.DISABLED

    record = ConsultationRecord(
        demographics=PatientDemographics(patient_id="GUI-EXP-1", age=40, gender="Male"),
        raw_transcript="Doctor: All clear.",
        soap_note=SOAPNote(subjective="Clear.", assessment="Healthy.", plan="Annual checkup.")
    )
    tk_app._handle_transcription_complete(record)

    assert tk_app.btn_export_zip["state"] == tk.NORMAL
    assert tk_app.btn_export_fhir["state"] == tk.NORMAL

    # Test FHIR export triggered via GUI
    with patch("src.processing.fhir_exporter.OUTPUT_DIR", tmp_path, create=True), \
         patch("app.OUTPUT_DIR", tmp_path), \
         patch("tkinter.messagebox.showinfo") as mock_info:
        tk_app.export_fhir_record()
        mock_info.assert_called_once()
        fhir_files = list(tmp_path.glob("fhir_bundle_*.json"))
        assert len(fhir_files) == 1

    # Test ZIP export triggered via GUI
    target_zip = tmp_path / "gui_bundle.zip"
    with patch("tkinter.filedialog.asksaveasfilename", return_value=str(target_zip)), \
         patch("tkinter.messagebox.showinfo") as mock_info:
        tk_app.export_zip_bundle()
        assert target_zip.is_file()
        mock_info.assert_called_once()

    # Test clear disables both
    tk_app.clear_consultation()
    assert tk_app.btn_export_zip["state"] == tk.DISABLED
    assert tk_app.btn_export_fhir["state"] == tk.DISABLED






