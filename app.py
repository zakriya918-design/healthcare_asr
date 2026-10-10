"""Healthcare AI Clinical Documentation Assistant - Desktop GUI Application.

Built with Tkinter and ttk.
Features:
- Thread-safe non-blocking audio recording and Faster-Whisper transcription.
- Contextual rule-based speaker role estimation (Doctor / Patient / Unknown).
- Deterministic clinical entity extraction with explicit negation and uncertainty preservation.
- Automated SOAP note generation with mandatory clinician review disclaimers.
- Atomic JSON consultation record persistence and re-loading.
- Audio file import and built-in synthetic consultation demo for offline / no-mic testing.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from src.models.patient_record import (
    ConsultationRecord,
    PatientDemographics,
    TranscriptSegment,
    ExtractedClinicalInformation,
    SOAPNote,
    FindingStatus,
    ReviewStatus,
    SpeakerRole
)
from src.audio.recorder import AudioRecorder, AudioRecorderError, AudioDeviceError
from src.audio.asr import WhisperTranscriber, ASRError
from src.processing.speaker_processor import process_transcript_segments
from src.processing.medical_processor import process_clinical_information, extract_demographics
from src.processing.soap_note import generate_soap_note
from src.processing.exporter import export_html_report, export_pdf_report, ClinicBranding
from src.processing.record_search import index_consultation_records, search_records, RecordSummary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HealthcareASR")

APP_TITLE = "Healthcare AI Clinical Documentation Assistant"
SYNTHETIC_WAV_PATH = Path(r"C:\Users\Aziza\HEALTHCARE_ASR\recordings\synthetic_consultation_test.wav")
OUTPUT_DIR = Path(r"C:\Users\Aziza\HEALTHCARE_ASR\output")
RECORDINGS_DIR = Path(r"C:\Users\Aziza\HEALTHCARE_ASR\recordings")


class HealthcareASRApp:
    """Main Tkinter Application Controller for Healthcare ASR Assistant."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x780")
        self.root.minsize(900, 650)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        self.recorder: Optional[AudioRecorder] = None
        self.transcriber: Optional[WhisperTranscriber] = None
        self.selected_model_size: str = "tiny.en"
        self.selected_mic_index: Optional[int] = None
        self.available_mics = AudioRecorder.get_available_input_devices()
        self.branding: ClinicBranding = self._load_clinic_branding()
        self.current_record: Optional[ConsultationRecord] = None
        self.worker_queue: queue.Queue = queue.Queue()
        self.recording_start_time: Optional[float] = None
        self.is_busy = False

        self._setup_styles()
        self._build_header()
        self._build_controls()
        self._build_status_bar()
        self._build_main_notebook()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll_worker_queue)
        self.root.after(500, self._update_recording_timer)

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.bg_color = "#F4F6F9"
        self.primary_color = "#1E3A8A"
        self.root.configure(bg=self.bg_color)
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabelframe", background=self.bg_color)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), background=self.bg_color, foreground=self.primary_color)
        style.configure("TLabel", background=self.bg_color, font=("Segoe UI", 9))

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg="#1E293B", padx=16, pady=12)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_frame,
            text="🏥 Healthcare AI Clinical Documentation Assistant",
            font=("Segoe UI", 15, "bold"),
            fg="#FFFFFF",
            bg="#1E293B"
        )
        title_lbl.pack(anchor=tk.W)

        disclaimer_lbl = tk.Label(
            header_frame,
            text="⚠️ CLINICAL SAFETY NOTICE: AI-generated draft documentation for licensed clinician review only. Not for autonomous clinical decisions.",
            font=("Segoe UI", 9),
            fg="#FDE68A",
            bg="#1E293B"
        )
        disclaimer_lbl.pack(anchor=tk.W, pady=(2, 0))

    def _build_controls(self):
        ctrl_frame = ttk.Frame(self.root, padding=(12, 10))
        ctrl_frame.pack(fill=tk.X, side=tk.TOP)

        self.btn_start = tk.Button(
            ctrl_frame,
            text="🎙️ Start Recording",
            font=("Segoe UI", 9, "bold"),
            bg="#15803D",
            fg="white",
            padx=12,
            pady=6,
            relief=tk.FLAT,
            command=self.start_recording
        )
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_stop = tk.Button(
            ctrl_frame,
            text="⏹️ Stop & Transcribe",
            font=("Segoe UI", 9, "bold"),
            bg="#B91C1C",
            fg="white",
            padx=12,
            pady=6,
            relief=tk.FLAT,
            state=tk.DISABLED,
            command=self.stop_and_transcribe
        )
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        self.btn_import = tk.Button(
            ctrl_frame,
            text="📁 Import WAV File",
            font=("Segoe UI", 9),
            bg="#0284C7",
            fg="white",
            padx=10,
            pady=6,
            relief=tk.FLAT,
            command=self.import_audio_file
        )
        self.btn_import.pack(side=tk.LEFT, padx=4)

        self.btn_demo = tk.Button(
            ctrl_frame,
            text="✨ Run Synthetic Demo",
            font=("Segoe UI", 9),
            bg="#4F46E5",
            fg="white",
            padx=10,
            pady=6,
            relief=tk.FLAT,
            command=self.load_synthetic_demo
        )
        self.btn_demo.pack(side=tk.LEFT, padx=4)

        self.btn_open = tk.Button(
            ctrl_frame,
            text="📂 Open JSON",
            font=("Segoe UI", 9),
            bg="#0369A1",
            fg="white",
            padx=10,
            pady=6,
            relief=tk.FLAT,
            command=self.open_saved_record
        )
        self.btn_open.pack(side=tk.LEFT, padx=4)

        self.btn_browser = tk.Button(
            ctrl_frame,
            text="🔍 Browse Past",
            font=("Segoe UI", 9),
            bg="#4338CA",
            fg="white",
            padx=8,
            pady=6,
            relief=tk.FLAT,
            command=self.open_record_browser
        )
        self.btn_browser.pack(side=tk.LEFT, padx=4)

        self.btn_branding = tk.Button(
            ctrl_frame,
            text="🏥 Clinic Header",
            font=("Segoe UI", 9),
            bg="#0E7490",
            fg="white",
            padx=8,
            pady=6,
            relief=tk.FLAT,
            command=self.configure_clinic_branding
        )
        self.btn_branding.pack(side=tk.LEFT, padx=4)

        self.btn_batch = tk.Button(
            ctrl_frame,
            text="📦 Batch Ingest",
            font=("Segoe UI", 9),
            bg="#9333EA",
            fg="white",
            padx=8,
            pady=6,
            relief=tk.FLAT,
            command=self.start_batch_ingest
        )
        self.btn_batch.pack(side=tk.LEFT, padx=4)

        self.btn_clear = tk.Button(
            ctrl_frame,
            text="🔄 New / Clear",
            font=("Segoe UI", 9),
            bg="#6B7280",
            fg="white",
            padx=10,
            pady=6,
            relief=tk.FLAT,
            command=self.clear_consultation
        )
        self.btn_clear.pack(side=tk.RIGHT, padx=4)

        self.btn_export = tk.Button(
            ctrl_frame,
            text="💾 Save JSON",
            font=("Segoe UI", 9, "bold"),
            bg="#0D9488",
            fg="white",
            padx=10,
            pady=6,
            relief=tk.FLAT,
            state=tk.DISABLED,
            command=self.export_json_record
        )
        self.btn_export.pack(side=tk.RIGHT, padx=2)

        self.btn_export_html = tk.Button(
            ctrl_frame,
            text="🌐 Export HTML",
            font=("Segoe UI", 9, "bold"),
            bg="#0284C7",
            fg="white",
            padx=10,
            pady=6,
            relief=tk.FLAT,
            state=tk.DISABLED,
            command=self.export_html_record
        )
        self.btn_export_html.pack(side=tk.RIGHT, padx=2)

        self.btn_export_pdf = tk.Button(
            ctrl_frame,
            text="📄 Export PDF",
            font=("Segoe UI", 9, "bold"),
            bg="#7C3AED",
            fg="white",
            padx=10,
            pady=6,
            relief=tk.FLAT,
            state=tk.DISABLED,
            command=self.export_pdf_record
        )
        self.btn_export_pdf.pack(side=tk.RIGHT, padx=2)

        self.btn_sign_off = tk.Button(
            ctrl_frame,
            text="✍️ Sign Off Note",
            font=("Segoe UI", 9, "bold"),
            bg="#059669",
            fg="white",
            padx=10,
            pady=6,
            relief=tk.FLAT,
            state=tk.DISABLED,
            command=self.sign_off_consultation
        )
        self.btn_sign_off.pack(side=tk.RIGHT, padx=2)

    def _build_status_bar(self):
        status_frame = tk.Frame(self.root, bg="#E2E8F0", padx=12, pady=6)
        status_frame.pack(fill=tk.X, side=tk.TOP)

        self.lbl_record_status = tk.Label(
            status_frame,
            text="● Audio: Idle",
            font=("Segoe UI", 9, "bold"),
            fg="#475569",
            bg="#E2E8F0"
        )
        self.lbl_record_status.pack(side=tk.LEFT, padx=(0, 16))

        self.lbl_asr_status = tk.Label(
            status_frame,
            text="Model: Ready (tiny.en CPU)",
            font=("Segoe UI", 9),
            fg="#334155",
            bg="#E2E8F0"
        )
        self.lbl_asr_status.pack(side=tk.LEFT, padx=(0, 16))

        self.lbl_feedback = tk.Label(
            status_frame,
            text='Ready for consultation. Click "Start Recording" or "Run Synthetic Demo".',
            font=("Segoe UI", 9),
            fg="#1E3A8A",
            bg="#E2E8F0"
        )
        self.lbl_feedback.pack(side=tk.LEFT, fill=tk.X, expand=True)

        model_frame = tk.Frame(status_frame, bg="#E2E8F0")
        model_frame.pack(side=tk.RIGHT, padx=(8, 0))

        tk.Label(
            model_frame,
            text="ASR Model:",
            font=("Segoe UI", 9, "bold"),
            fg="#334155",
            bg="#E2E8F0"
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.combo_model = ttk.Combobox(
            model_frame,
            values=["tiny.en", "base.en", "small.en"],
            state="readonly",
            width=10,
            font=("Segoe UI", 9)
        )
        self.combo_model.set("tiny.en")
        self.combo_model.bind("<<ComboboxSelected>>", self._on_model_changed)
        self.combo_model.pack(side=tk.LEFT)

        mic_frame = tk.Frame(status_frame, bg="#E2E8F0")
        mic_frame.pack(side=tk.RIGHT, padx=(8, 0))

        tk.Label(
            mic_frame,
            text="🎙️ Mic:",
            font=("Segoe UI", 9, "bold"),
            fg="#334155",
            bg="#E2E8F0"
        ).pack(side=tk.LEFT, padx=(0, 4))

        mic_options = ["Default System Mic"]
        for d in self.available_mics:
            clean_name = d["name"][:24]
            mic_options.append(f"[{d['index']}] {clean_name}")

        self.combo_mic = ttk.Combobox(
            mic_frame,
            values=mic_options,
            state="readonly",
            width=20,
            font=("Segoe UI", 8)
        )
        self.combo_mic.set("Default System Mic")
        self.combo_mic.bind("<<ComboboxSelected>>", self._on_mic_changed)
        self.combo_mic.pack(side=tk.LEFT)

    def _on_mic_changed(self, event=None):
        selection = self.combo_mic.get()
        if selection == "Default System Mic" or not selection:
            self.selected_mic_index = None
            self.lbl_feedback.config(text="Microphone set to: Default System Mic.")
            logger.info("Microphone set to default input device.")
        else:
            try:
                idx_str = selection.split("]")[0].replace("[", "").strip()
                self.selected_mic_index = int(idx_str)
                self.lbl_feedback.config(text=f"Selected microphone: {selection}")
                logger.info("User selected microphone index %d: %s", self.selected_mic_index, selection)
            except Exception:
                self.selected_mic_index = None

    def _on_model_changed(self, event=None):
        new_model = self.combo_model.get()
        if new_model != self.selected_model_size:
            self.selected_model_size = new_model
            self.transcriber = None
            self.lbl_asr_status.config(text=f"Model: Ready ({self.selected_model_size} CPU)")
            self.lbl_feedback.config(text=f"Switched model to '{self.selected_model_size}'. Will initialize on next transcription.")
            logger.info("User switched ASR model size to %s", self.selected_model_size)

    def _build_main_notebook(self):
        notebook_frame = ttk.Frame(self.root, padding=8)
        notebook_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Transcript & Speaker Roles
        tab_transcript = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_transcript, text=" 📝 Transcript & Speakers ")
        self._build_transcript_tab(tab_transcript)

        # Tab 2: Extracted Medical Findings
        tab_medical = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_medical, text=" 🩺 Extracted Clinical Data ")
        self._build_medical_tab(tab_medical)

        # Tab 3: Formatted SOAP Note
        tab_soap = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_soap, text=" 📋 SOAP Clinical Note ")
        self._build_soap_tab(tab_soap)

        # Tab 4: JSON Record View
        tab_json = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_json, text=" 📦 Structured JSON Record ")
        self._build_json_tab(tab_json)

    def _build_transcript_tab(self, parent: ttk.Frame):
        lbl_info = ttk.Label(
            parent,
            text="Speech-to-Text output with contextual speaker-role estimation (Doctor / Patient / Unknown):"
        )
        lbl_info.pack(anchor=tk.W, pady=(0, 4))

        self.txt_transcript = tk.Text(parent, wrap=tk.WORD, font=("Segoe UI", 10), height=7, bg="#FFFFFF", relief=tk.SOLID, bd=1)
        self.txt_transcript.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        columns = ("id", "time", "speaker", "confidence", "text")
        self.tree_segments = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse")
        self.tree_segments.heading("id", text="#")
        self.tree_segments.heading("time", text="Time")
        self.tree_segments.heading("speaker", text="Estimated Role")
        self.tree_segments.heading("confidence", text="Confidence")
        self.tree_segments.heading("text", text="Utterance Text")

        self.tree_segments.column("id", width=40, anchor=tk.CENTER)
        self.tree_segments.column("time", width=90, anchor=tk.CENTER)
        self.tree_segments.column("speaker", width=110, anchor=tk.CENTER)
        self.tree_segments.column("confidence", width=90, anchor=tk.CENTER)
        self.tree_segments.column("text", width=600, anchor=tk.W)

        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree_segments.yview)
        self.tree_segments.configure(yscrollcommand=scroll.set)

        self.tree_segments.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_medical_tab(self, parent: ttk.Frame):
        lbl_info = ttk.Label(
            parent,
            text="Deterministic entity extraction with explicit negation handling and uncertainty preservation:"
        )
        lbl_info.pack(anchor=tk.W, pady=(0, 4))

        self.txt_medical = tk.Text(parent, wrap=tk.WORD, font=("Consolas", 10), bg="#FFFFFF", relief=tk.SOLID, bd=1)
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.txt_medical.yview)
        self.txt_medical.configure(yscrollcommand=scroll.set)

        self.txt_medical.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_soap_tab(self, parent: ttk.Frame):
        self.frame_soap_banner = tk.Frame(parent, bg="#FEF3C7", padx=10, pady=6, relief=tk.SOLID, bd=1)
        self.frame_soap_banner.pack(fill=tk.X, pady=(0, 8))

        self.lbl_soap_warning = tk.Label(
            self.frame_soap_banner,
            text="⚠️ STATUS: DRAFT PENDING REVIEW — A licensed clinician must review, edit if necessary, and sign before clinical use.",
            font=("Segoe UI", 9, "bold"),
            fg="#92400E",
            bg="#FEF3C7"
        )
        self.lbl_soap_warning.pack(side=tk.LEFT, anchor=tk.W)

        btn_tab_sign = tk.Button(
            self.frame_soap_banner,
            text="✍️ Sign Note",
            font=("Segoe UI", 8, "bold"),
            bg="#059669",
            fg="white",
            padx=8,
            pady=2,
            relief=tk.FLAT,
            command=self.sign_off_consultation
        )
        btn_tab_sign.pack(side=tk.RIGHT)

        self.txt_soap = tk.Text(parent, wrap=tk.WORD, font=("Segoe UI", 10), bg="#FFFFFF", relief=tk.SOLID, bd=1)
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.txt_soap.yview)
        self.txt_soap.configure(yscrollcommand=scroll.set)

        self.txt_soap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_json_tab(self, parent: ttk.Frame):
        lbl_info = ttk.Label(parent, text="Structured Consultation Record (Schema v1.0):")
        lbl_info.pack(anchor=tk.W, pady=(0, 4))

        self.txt_json = tk.Text(parent, wrap=tk.WORD, font=("Consolas", 9), bg="#F8FAFC", relief=tk.SOLID, bd=1)
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.txt_json.yview)
        self.txt_json.configure(yscrollcommand=scroll.set)

        self.txt_json.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _poll_worker_queue(self):
        try:
            while True:
                msg = self.worker_queue.get_nowait()
                msg_type = msg.get("type")

                if msg_type == "status":
                    self.lbl_feedback.config(text=msg.get("text", ""))
                    if "asr_text" in msg:
                        self.lbl_asr_status.config(text=msg["asr_text"])

                elif msg_type == "transcription_complete":
                    self._handle_transcription_complete(msg["record"])

                elif msg_type == "batch_complete":
                    summary = msg["summary"]
                    self._set_busy_state(False)
                    self.lbl_record_status.config(text="● Batch: Done", fg="#15803D")
                    self.lbl_asr_status.config(text="Model: Ready")
                    self.lbl_feedback.config(
                        text=f"✅ Batch completed: {summary.successful_files}/{summary.total_files} succeeded in {summary.total_time_seconds:.1f}s."
                    )
                    messagebox.showinfo(
                        "Batch Ingest Complete",
                        f"Batch Processing Completed!\n\n"
                        f"• Total Files: {summary.total_files}\n"
                        f"• Successfully Processed: {summary.successful_files}\n"
                        f"• Failed: {summary.failed_files}\n"
                        f"• Elapsed Time: {summary.total_time_seconds:.1f} seconds\n\n"
                        f"Generated SOAP notes, JSON records, HTML, and PDF files are saved in:\n{OUTPUT_DIR}"
                    )

                elif msg_type == "error":
                    self._handle_error(msg["error_msg"])

                self.worker_queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_worker_queue)

    def _update_recording_timer(self):
        if self.recorder and self.recorder.is_recording and self.recording_start_time:
            elapsed = int(time.time() - self.recording_start_time)
            mins, secs = divmod(elapsed, 60)
            self.lbl_record_status.config(
                text=f"🔴 Recording: {mins:02d}:{secs:02d}",
                fg="#DC2626"
            )
        self.root.after(500, self._update_recording_timer)

    def start_recording(self):
        if self.is_busy:
            return

        try:
            self.recorder = AudioRecorder(
                recordings_dir=RECORDINGS_DIR,
                device=self.selected_mic_index
            )
            self.recorder.start_recording()
            self.recording_start_time = time.time()

            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_import.config(state=tk.DISABLED)
            self.btn_demo.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED)

            self.lbl_record_status.config(text="🔴 Recording: 00:00", fg="#DC2626")
            self.lbl_feedback.config(text='Recording consultation audio... Click "Stop & Transcribe" when finished.')
            logger.info("Started audio recording from microphone.")
        except AudioDeviceError as e:
            messagebox.showerror("Microphone Error", f"Cannot access microphone device:\n\n{e}\n\nYou can still import existing WAV files or click \"Run Synthetic Demo\".")
            self.lbl_feedback.config(text=f"Microphone error: {e}")
        except Exception as e:
            messagebox.showerror("Recording Error", f"Unexpected recording error: {e}")
            self.lbl_feedback.config(text=f"Error: {e}")

    def stop_and_transcribe(self):
        if not self.recorder or not self.recorder.is_recording:
            return

        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_record_status.config(text="● Audio: Processing...", fg="#2563EB")
        self.lbl_feedback.config(text="Stopping recording and validating WAV file...")

        def worker_stop_and_transcribe():
            try:
                wav_path, duration = self.recorder.stop_recording()
                self.worker_queue.put({
                    "type": "status",
                    "text": f"Recorded {duration:.1f}s of audio. Launching Whisper ASR...",
                    "asr_text": "Model: Loading & Transcribing..."
                })
                self._run_transcription_pipeline(wav_path)
            except AudioRecorderError as e:
                self.worker_queue.put({"type": "error", "error_msg": f"Audio recording failed: {e}"})
            except Exception as e:
                self.worker_queue.put({"type": "error", "error_msg": f"Unexpected error: {e}"})

        threading.Thread(target=worker_stop_and_transcribe, daemon=True).start()

    def import_audio_file(self):
        if self.is_busy:
            return

        file_selected = filedialog.askopenfilename(
            title="Select Consultation Audio File",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        if not file_selected:
            return

        wav_path = Path(file_selected)
        self._start_pipeline_on_file(wav_path)

    def load_synthetic_demo(self):
        if self.is_busy:
            return

        if not SYNTHETIC_WAV_PATH.is_file():
            messagebox.showerror("Demo Missing", f"Synthetic audio fixture not found at {SYNTHETIC_WAV_PATH}")
            return

        self._start_pipeline_on_file(SYNTHETIC_WAV_PATH)

    def _start_pipeline_on_file(self, wav_path: Path):
        self._set_busy_state(True)
        self.lbl_record_status.config(text=f"● Audio: {wav_path.name}", fg="#059669")
        self.lbl_feedback.config(text=f"Loading audio: {wav_path.name}... Starting transcription.")

        def worker_transcribe():
            try:
                self._run_transcription_pipeline(wav_path)
            except Exception as e:
                self.worker_queue.put({"type": "error", "error_msg": f"Processing error: {e}"})

        threading.Thread(target=worker_transcribe, daemon=True).start()

    def _run_transcription_pipeline(self, wav_path: Path):
        try:
            self.worker_queue.put({
                "type": "status",
                "text": f"Transcribing {wav_path.name} with Faster-Whisper ({self.selected_model_size} CPU)...",
                "asr_text": f"Model: Loading ({self.selected_model_size})..."
            })

            if self.transcriber is None or self.transcriber.model_size != self.selected_model_size:
                self.transcriber = WhisperTranscriber(
                    model_size=self.selected_model_size,
                    device="cpu",
                    compute_type="int8"
                )

            raw_transcript, raw_segments = self.transcriber.transcribe_file(wav_path)

            self.worker_queue.put({"type": "status", "text": "Estimating speaker roles..."})
            annotated_segments = process_transcript_segments(raw_segments)

            self.worker_queue.put({"type": "status", "text": "Extracting clinical findings & vital signs..."})
            extracted_info = process_clinical_information(raw_transcript)
            demographics = extract_demographics(raw_transcript)

            self.worker_queue.put({"type": "status", "text": "Compiling draft SOAP clinical note..."})
            soap_note = generate_soap_note(extracted_info, demographics_summary=self._get_demo_summary(demographics))

            record = ConsultationRecord(
                demographics=demographics,
                raw_transcript=raw_transcript,
                segments=annotated_segments,
                extracted_info=extracted_info,
                soap_note=soap_note,
                review_status=ReviewStatus.DRAFT_PENDING_REVIEW
            )

            self.worker_queue.put({"type": "transcription_complete", "record": record})

        except ASRError as e:
            self.worker_queue.put({"type": "error", "error_msg": f"Speech recognition error: {e}"})
        except Exception as e:
            logger.exception("Pipeline failure")
            self.worker_queue.put({"type": "error", "error_msg": f"Pipeline failure: {e}"})

    def _get_demo_summary(self, demo: PatientDemographics) -> str:
        parts = []
        if demo.age is not None:
            parts.append(f"{demo.age}-year-old")
        if demo.gender:
            parts.append(demo.gender.lower())
        return " ".join(parts)

    def _handle_transcription_complete(self, record: ConsultationRecord):
        self.current_record = record
        self._set_busy_state(False)

        self.lbl_record_status.config(text="● Audio: Completed", fg="#15803D")
        self.lbl_asr_status.config(text="Model: Ready")
        self.lbl_feedback.config(text="✅ Consultation transcribed and analyzed successfully! Review findings below.")

        self._render_transcript_tab(record)
        self._render_medical_tab(record.extracted_info)
        self._render_soap_tab(record.soap_note)
        self._render_json_tab(record)

        self.btn_export.config(state=tk.NORMAL)
        self.btn_export_html.config(state=tk.NORMAL)
        self.btn_export_pdf.config(state=tk.NORMAL)
        self.btn_sign_off.config(state=tk.NORMAL)

    def _handle_error(self, error_msg: str):
        self._set_busy_state(False)
        self.lbl_record_status.config(text="● Audio: Idle", fg="#475569")
        self.lbl_asr_status.config(text="Model: Ready")
        self.lbl_feedback.config(text=f"❌ Error: {error_msg}")
        messagebox.showerror("Processing Error", error_msg)

    def _set_busy_state(self, busy: bool):
        self.is_busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_start.config(state=state)
        self.btn_import.config(state=state)
        self.btn_demo.config(state=state)
        self.btn_open.config(state=state)
        self.btn_branding.config(state=state)
        self.btn_batch.config(state=state)
        self.btn_clear.config(state=state)
        self.combo_model.config(state=tk.DISABLED if busy else "readonly")
        self.combo_mic.config(state=tk.DISABLED if busy else "readonly")

    def _render_transcript_tab(self, record: ConsultationRecord):
        self.txt_transcript.delete("1.0", tk.END)
        self.txt_transcript.insert(tk.END, record.raw_transcript)

        for item in self.tree_segments.get_children():
            self.tree_segments.delete(item)

        for seg in record.segments:
            time_str = f"{seg.start_time:.1f}s - {seg.end_time:.1f}s" if seg.start_time is not None and seg.end_time is not None else "--"
            conf_str = f"{int(seg.role_confidence * 100)}%" if seg.role_confidence > 0 else "--"
            role_val = seg.speaker_role.value if hasattr(seg.speaker_role, "value") else str(seg.speaker_role)
            self.tree_segments.insert(
                "",
                tk.END,
                values=(seg.id, time_str, role_val, conf_str, seg.text)
            )

    def _render_medical_tab(self, info: ExtractedClinicalInformation):
        self.txt_medical.delete("1.0", tk.END)

        lines = ["=================================================================="]
        lines.append("               EXTRACTED CLINICAL FINDINGS (RULE-BASED)            ")
        lines.append("==================================================================\n")

        lines.append("[SYMPTOMS & COMPLAINTS]")
        if info.symptoms:
            for s in info.symptoms:
                dur = f" | Duration: {s.duration}" if s.duration else ""
                status_icon = "✓" if s.status == FindingStatus.PRESENT else "✗ (NEGATED)"
                lines.append(f"  {status_icon} {s.name:<24} [{s.status.value}]{dur}")
        else:
            lines.append("  None detected.")

        lines.append("\n[VITAL SIGNS (EXPLICIT RECORDING ONLY)]")
        v = info.vital_signs
        bp = v.blood_pressure_str or (f"{v.blood_pressure_systolic}/{v.blood_pressure_diastolic} mmHg" if v.blood_pressure_systolic else "Not recorded")
        hr = f"{v.heart_rate_bpm} bpm" if v.heart_rate_bpm else "Not recorded"
        temp = v.temperature_str or (f"{v.temperature_f}°F" if v.temperature_f else (f"{v.temperature_c}°C" if v.temperature_c else "Not recorded"))
        spo2 = f"{v.spo2_percent}%" if v.spo2_percent else "Not recorded"
        bg = v.blood_glucose_str or (f"{v.blood_glucose_mg_dl} mg/dL" if v.blood_glucose_mg_dl else "Not recorded")

        lines.append(f"  • Blood Pressure:   {bp}")
        lines.append(f"  • Heart Rate:       {hr}")
        lines.append(f"  • Temperature:      {temp}")
        lines.append(f"  • SpO2:             {spo2}")
        lines.append(f"  • Blood Glucose:    {bg}")

        lines.append("\n[ALLERGIES]")
        if info.allergies:
            for a in info.allergies:
                lines.append(f"  • {a.name} [{a.status.value}]")
        else:
            lines.append("  Not mentioned.")

        lines.append("\n[MEDICATIONS]")
        if info.medications:
            for m in info.medications:
                kind = "Plan" if m.is_current_plan else "Prior/Home"
                dos = m.dosage or "dosage unstated"
                freq = m.frequency or "frequency unstated"
                lines.append(f"  • {m.name} ({dos}, {freq}) [{kind}]")
        else:
            lines.append("  None detected.")

        lines.append("\n[ASSESSMENT & DIAGNOSES]")
        if info.diagnoses:
            for d in info.diagnoses:
                lines.append(f"  • {d.name} [{d.status.value}]")
        else:
            lines.append("  None explicitly stated.")

        lines.append("\n[INVESTIGATIONS & FOLLOW-UP]")
        if info.investigations:
            for inv in info.investigations:
                lines.append(f"  • Investigation: {inv.name}")
        if info.treatment_plans:
            for tp in info.treatment_plans:
                lines.append(f"  • Plan: {tp.name}")
        if info.advice_followup:
            for af in info.advice_followup:
                detail = f": {af.details}" if af.details else ""
                lines.append(f"  • Follow-up: {af.name}{detail}")
        if not info.investigations and not info.treatment_plans and not info.advice_followup:
            lines.append("  None explicitly stated.")

        if info.unclassified_statements:
            lines.append("\n[UNCLASSIFIED CONSULTATION STATEMENTS]")
            for st in info.unclassified_statements[:10]:
                lines.append(f'  - "{st}"')

        self.txt_medical.insert(tk.END, "\n".join(lines))

    def _render_soap_tab(self, soap: SOAPNote):
        self.txt_soap.delete("1.0", tk.END)

        content = []
        content.append("==================================================================")
        content.append("              CLINICAL SOAP NOTE (DRAFT FOR REVIEW)               ")
        content.append("==================================================================\n")
        content.append(soap.review_warning + "\n")
        content.append("------------------------------------------------------------------")
        content.append("S — SUBJECTIVE:")
        content.append("------------------------------------------------------------------")
        content.append(soap.subjective or "No subjective history recorded.")
        content.append("\n------------------------------------------------------------------")
        content.append("O — OBJECTIVE:")
        content.append("------------------------------------------------------------------")
        content.append(soap.objective or "No objective findings recorded.")
        content.append("\n------------------------------------------------------------------")
        content.append("A — ASSESSMENT:")
        content.append("------------------------------------------------------------------")
        content.append(soap.assessment or "Assessment not documented.")
        content.append("\n------------------------------------------------------------------")
        content.append("P — PLAN:")
        content.append("------------------------------------------------------------------")
        content.append(soap.plan or "Plan not documented.")

        if soap.missing_or_uncertain_info:
            content.append("\n------------------------------------------------------------------")
            content.append("MISSING OR UNCERTAIN INFORMATION REQUIRING ATTENTION:")
            content.append("------------------------------------------------------------------")
            for item in soap.missing_or_uncertain_info:
                content.append(f"• {item}")

        self.txt_soap.insert(tk.END, "\n".join(content))

    def _render_json_tab(self, record: ConsultationRecord):
        self.txt_json.delete("1.0", tk.END)
        self.txt_json.insert(tk.END, record.to_json(indent=2))

    def export_json_record(self):
        if not self.current_record:
            messagebox.showwarning("No Data", "No completed consultation record available to export.")
            return

        try:
            saved_path = self.current_record.save_json(output_dir=OUTPUT_DIR)
            messagebox.showinfo(
                "Consultation Saved",
                f"Consultation record successfully exported to:\n\n{saved_path}"
            )
            self.lbl_feedback.config(text=f"Exported JSON to: {saved_path.name}")
            logger.info("Saved consultation record to %s", saved_path)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export consultation record:\n\n{e}")
            self.lbl_feedback.config(text=f"Export error: {e}")

    def export_html_record(self):
        if not self.current_record:
            messagebox.showwarning("No Data", "No completed consultation record available to export.")
            return

        try:
            out_file = OUTPUT_DIR / f"consultation_{self.current_record.consultation_id[:8]}.html"
            saved_path = export_html_report(self.current_record, out_file, branding=self.branding)
            messagebox.showinfo(
                "HTML Report Exported",
                f"Clinical HTML report successfully saved to:\n\n{saved_path}"
            )
            self.lbl_feedback.config(text=f"Exported HTML report to: {saved_path.name}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export HTML report:\n\n{e}")
            self.lbl_feedback.config(text=f"Export error: {e}")

    def export_pdf_record(self):
        if not self.current_record:
            messagebox.showwarning("No Data", "No completed consultation record available to export.")
            return

        try:
            out_file = OUTPUT_DIR / f"consultation_{self.current_record.consultation_id[:8]}.pdf"
            saved_path = export_pdf_report(self.current_record, out_file, branding=self.branding)
            messagebox.showinfo(
                "PDF Report Exported",
                f"Clinical PDF report successfully saved to:\n\n{saved_path}"
            )
            self.lbl_feedback.config(text=f"Exported PDF report to: {saved_path.name}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export PDF report:\n\n{e}")
            self.lbl_feedback.config(text=f"Export error: {e}")

    def _load_clinic_branding(self) -> ClinicBranding:
        settings_file = Path("clinic_settings.json")
        if settings_file.is_file():
            try:
                data = json.loads(settings_file.read_text(encoding="utf-8"))
                return ClinicBranding(
                    clinic_name=data.get("clinic_name", "Apex Healthcare Clinic"),
                    department=data.get("department", "Ambulatory Care & Family Medicine"),
                    physician_name=data.get("physician_name", "Dr. Attending Physician, MD"),
                    facility_address=data.get("facility_address", "100 Medical Center Blvd, Suite 400"),
                    contact_info=data.get("contact_info", "Tel: (555) 019-2834 | records@apexclinic.health")
                )
            except Exception as e:
                logger.warning("Could not read clinic_settings.json: %s", e)
        return ClinicBranding()

    def configure_clinic_branding(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Configure Clinic Letterhead & Branding")
        dlg.geometry("500x380")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        frame = ttk.Frame(dlg, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Customize Clinic Report Header", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        fields = [
            ("Clinic Name:", "clinic_name", self.branding.clinic_name),
            ("Department / Unit:", "department", self.branding.department),
            ("Physician Name:", "physician_name", self.branding.physician_name),
            ("Facility Address:", "facility_address", self.branding.facility_address),
            ("Contact / Phone:", "contact_info", self.branding.contact_info),
        ]

        entries = {}
        for label_text, key, val in fields:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=label_text, width=18, anchor=tk.W).pack(side=tk.LEFT)
            ent = ttk.Entry(row)
            ent.insert(0, val)
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            entries[key] = ent

        def save_and_close():
            self.branding = ClinicBranding(
                clinic_name=entries["clinic_name"].get().strip() or "Healthcare Clinic",
                department=entries["department"].get().strip(),
                physician_name=entries["physician_name"].get().strip(),
                facility_address=entries["facility_address"].get().strip(),
                contact_info=entries["contact_info"].get().strip()
            )
            # Persist to JSON
            try:
                data = {
                    "clinic_name": self.branding.clinic_name,
                    "department": self.branding.department,
                    "physician_name": self.branding.physician_name,
                    "facility_address": self.branding.facility_address,
                    "contact_info": self.branding.contact_info,
                }
                Path("clinic_settings.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to save clinic_settings.json: %s", e)

            self.lbl_feedback.config(text=f"Updated clinic branding: {self.branding.clinic_name}")
            dlg.destroy()

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(16, 0))
        ttk.Button(btn_row, text="Save Header", command=save_and_close).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=4)

    def start_batch_ingest(self):
        if self.is_busy:
            return

        selected_folder = filedialog.askdirectory(
            title="Select Folder Containing WAV Audio Consultations"
        )
        if not selected_folder:
            return

        in_dir = Path(selected_folder)
        wav_files = sorted(list(in_dir.glob("*.wav")))
        if not wav_files:
            messagebox.showinfo("No Audio Files", f"No .wav files found in:\n\n{in_dir}")
            return

        if not messagebox.askyesno(
            "Batch Ingest Confirmation",
            f"Found {len(wav_files)} WAV audio file(s) in:\n{in_dir.name}\n\n"
            f"Process all files using Faster-Whisper ({self.selected_model_size}) "
            f"and auto-generate SOAP notes, JSON, HTML, and PDF reports in output folder?"
        ):
            return

        self._set_busy_state(True)
        self.lbl_record_status.config(text="● Batch: Processing...", fg="#9333EA")
        self.lbl_feedback.config(text=f"Batch Ingestion started: 0 of {len(wav_files)} completed...")

        def worker_batch():
            try:
                from src.processing.batch_processor import process_audio_folder

                def progress_cb(current, total, filename):
                    self.worker_queue.put({
                        "type": "status",
                        "text": f"Batch [{current}/{total}]: Transcribing {filename}...",
                        "asr_text": f"Batch: {current}/{total}"
                    })

                summary = process_audio_folder(
                    input_dir=in_dir,
                    output_dir=OUTPUT_DIR,
                    model_size=self.selected_model_size,
                    export_pdf=True,
                    export_html=True,
                    branding=self.branding,
                    progress_callback=progress_cb
                )

                self.worker_queue.put({
                    "type": "batch_complete",
                    "summary": summary
                })
            except Exception as e:
                logger.exception("Batch processing failed")
                self.worker_queue.put({"type": "error", "error_msg": f"Batch processing failed: {e}"})

        threading.Thread(target=worker_batch, daemon=True).start()

    def open_saved_record(self):
        if self.is_busy:
            return

        file_selected = filedialog.askopenfilename(
            title="Open Saved Consultation JSON",
            initialdir=str(OUTPUT_DIR),
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not file_selected:
            return

        try:
            record = ConsultationRecord.load_json(file_selected)
            self._handle_transcription_complete(record)
            self.lbl_feedback.config(text=f"Loaded saved record from: {Path(file_selected).name}")
            logger.info("Loaded saved record: %s", file_selected)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load consultation JSON:\n\n{e}")

    def open_record_browser(self):
        """Opens interactive search and filter browser for past consultation records."""
        browser_win = tk.Toplevel(self.root)
        browser_win.title("🔍 Past Consultation Records Browser")
        browser_win.geometry("980x620")
        browser_win.minsize(800, 500)

        records_dir_var = tk.StringVar(value=str(OUTPUT_DIR))
        all_summaries: List[RecordSummary] = []
        path_by_item_id: Dict[str, str] = {}

        # Top Directory & Filter Frame
        top_frame = tk.Frame(browser_win, bg="#F1F5F9", padx=12, pady=10)
        top_frame.pack(fill=tk.X)

        # Folder selector row
        folder_row = tk.Frame(top_frame, bg="#F1F5F9")
        folder_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(folder_row, text="📁 Directory:", font=("Segoe UI", 9, "bold"), bg="#F1F5F9").pack(side=tk.LEFT, padx=(0, 4))
        lbl_dir = tk.Label(folder_row, textvariable=records_dir_var, font=("Segoe UI", 9), bg="#F1F5F9", fg="#1E293B")
        lbl_dir.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        def change_folder():
            chosen = filedialog.askdirectory(title="Select Consultation Directory", initialdir=records_dir_var.get())
            if chosen:
                records_dir_var.set(chosen)
                refresh_records()

        btn_ch_dir = tk.Button(
            folder_row,
            text="Change Folder",
            font=("Segoe UI", 8),
            bg="#E2E8F0",
            relief=tk.FLAT,
            command=change_folder
        )
        btn_ch_dir.pack(side=tk.RIGHT, padx=4)

        # Filter criteria row
        filter_row = tk.Frame(top_frame, bg="#F1F5F9")
        filter_row.pack(fill=tk.X)

        tk.Label(filter_row, text="Query:", font=("Segoe UI", 9), bg="#F1F5F9").pack(side=tk.LEFT, padx=(0, 2))
        entry_query = tk.Entry(filter_row, font=("Segoe UI", 9), width=20)
        entry_query.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(filter_row, text="Status:", font=("Segoe UI", 9), bg="#F1F5F9").pack(side=tk.LEFT, padx=(0, 2))
        combo_status = ttk.Combobox(
            filter_row,
            values=["All", "draft_pending_review", "reviewed_approved"],
            state="readonly",
            width=16,
            font=("Segoe UI", 9)
        )
        combo_status.set("All")
        combo_status.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(filter_row, text="Diagnosis:", font=("Segoe UI", 9), bg="#F1F5F9").pack(side=tk.LEFT, padx=(0, 2))
        entry_diag = tk.Entry(filter_row, font=("Segoe UI", 9), width=14)
        entry_diag.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(filter_row, text="Medication:", font=("Segoe UI", 9), bg="#F1F5F9").pack(side=tk.LEFT, padx=(0, 2))
        entry_med = tk.Entry(filter_row, font=("Segoe UI", 9), width=14)
        entry_med.pack(side=tk.LEFT, padx=(0, 8))

        lbl_count = tk.Label(filter_row, text="0 records", font=("Segoe UI", 9, "italic"), bg="#F1F5F9", fg="#64748B")
        lbl_count.pack(side=tk.RIGHT, padx=4)

        btn_reset = tk.Button(
            filter_row,
            text="↺ Reset",
            font=("Segoe UI", 8),
            bg="#94A3B8",
            fg="white",
            relief=tk.FLAT,
            padx=6,
            command=lambda: reset_filters()
        )
        btn_reset.pack(side=tk.RIGHT, padx=4)

        btn_do_search = tk.Button(
            filter_row,
            text="🔍 Search",
            font=("Segoe UI", 8, "bold"),
            bg="#4338CA",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            command=lambda: apply_search()
        )
        btn_do_search.pack(side=tk.RIGHT, padx=4)

        # Center Treeview table
        table_frame = tk.Frame(browser_win)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        columns = ("datetime", "patient_id", "demographics", "status", "diagnoses", "medications", "snippet")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")

        tree.heading("datetime", text="Date & Time", anchor=tk.W)
        tree.heading("patient_id", text="Patient ID", anchor=tk.W)
        tree.heading("demographics", text="Age / Sex", anchor=tk.W)
        tree.heading("status", text="Status", anchor=tk.W)
        tree.heading("diagnoses", text="Diagnoses", anchor=tk.W)
        tree.heading("medications", text="Medications", anchor=tk.W)
        tree.heading("snippet", text="Transcript Snippet", anchor=tk.W)

        tree.column("datetime", width=130, minwidth=110)
        tree.column("patient_id", width=120, minwidth=90)
        tree.column("demographics", width=80, minwidth=70)
        tree.column("status", width=130, minwidth=100)
        tree.column("diagnoses", width=150, minwidth=100)
        tree.column("medications", width=150, minwidth=100)
        tree.column("snippet", width=220, minwidth=150)

        scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        # Bottom Action Bar
        bottom_frame = tk.Frame(browser_win, bg="#F8FAFC", padx=12, pady=10)
        bottom_frame.pack(fill=tk.X)

        def get_selected_path() -> Optional[str]:
            sel = tree.selection()
            if not sel:
                return None
            return path_by_item_id.get(sel[0])

        def on_load_selected():
            filepath = get_selected_path()
            if not filepath:
                messagebox.showwarning("Select Record", "Please select a consultation record from the table first.", parent=browser_win)
                return
            try:
                record = ConsultationRecord.load_json(filepath)
                self._handle_transcription_complete(record)
                self.lbl_feedback.config(text=f"Loaded from browser: {Path(filepath).name}")
                messagebox.showinfo("Loaded", f"Consultation record {record.consultation_id[:8]} loaded into main window!", parent=browser_win)
                browser_win.lift()
            except Exception as ex:
                messagebox.showerror("Error", f"Failed to load record:\n\n{ex}", parent=browser_win)

        def on_export_pdf_selected():
            filepath = get_selected_path()
            if not filepath:
                messagebox.showwarning("Select Record", "Please select a consultation record first.", parent=browser_win)
                return
            try:
                record = ConsultationRecord.load_json(filepath)
                pdf_path = export_pdf_report(record, Path(records_dir_var.get()), self.branding)
                messagebox.showinfo("PDF Exported", f"Exported PDF report successfully:\n\n{pdf_path.name}", parent=browser_win)
            except Exception as ex:
                messagebox.showerror("Export Error", f"Failed to generate PDF:\n\n{ex}", parent=browser_win)

        def on_export_html_selected():
            filepath = get_selected_path()
            if not filepath:
                messagebox.showwarning("Select Record", "Please select a consultation record first.", parent=browser_win)
                return
            try:
                record = ConsultationRecord.load_json(filepath)
                html_path = export_html_report(record, Path(records_dir_var.get()), self.branding)
                messagebox.showinfo("HTML Exported", f"Exported HTML report successfully:\n\n{html_path.name}", parent=browser_win)
            except Exception as ex:
                messagebox.showerror("Export Error", f"Failed to generate HTML:\n\n{ex}", parent=browser_win)

        def on_open_in_explorer():
            folder = Path(records_dir_var.get())
            if folder.exists():
                import os
                os.startfile(str(folder.resolve()))
            else:
                messagebox.showwarning("Folder Missing", f"Folder does not exist:\n{folder}", parent=browser_win)

        btn_load = tk.Button(
            bottom_frame,
            text="📂 Load into Main App",
            font=("Segoe UI", 9, "bold"),
            bg="#0369A1",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            command=on_load_selected
        )
        btn_load.pack(side=tk.LEFT, padx=4)

        btn_pdf = tk.Button(
            bottom_frame,
            text="📄 Export PDF",
            font=("Segoe UI", 9),
            bg="#D97706",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            pady=4,
            command=on_export_pdf_selected
        )
        btn_pdf.pack(side=tk.LEFT, padx=4)

        btn_html = tk.Button(
            bottom_frame,
            text="🌐 Export HTML",
            font=("Segoe UI", 9),
            bg="#0284C7",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            pady=4,
            command=on_export_html_selected
        )
        btn_html.pack(side=tk.LEFT, padx=4)

        btn_exp = tk.Button(
            bottom_frame,
            text="📁 Show in Explorer",
            font=("Segoe UI", 9),
            bg="#64748B",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            pady=4,
            command=on_open_in_explorer
        )
        btn_exp.pack(side=tk.LEFT, padx=4)

        btn_close = tk.Button(
            bottom_frame,
            text="Close",
            font=("Segoe UI", 9),
            bg="#E2E8F0",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            command=browser_win.destroy
        )
        btn_close.pack(side=tk.RIGHT, padx=4)

        # Helper methods to populate table
        def render_treeview(summaries: List[RecordSummary]):
            tree.delete(*tree.get_children())
            path_by_item_id.clear()

            for s in summaries:
                # Format datetime
                dt_str = s.created_at[:16].replace("T", " ") if len(s.created_at) >= 16 else s.created_at
                
                # Format demographics
                demo_str = f"{s.patient_age or '?'}, {s.patient_gender or '?'}"
                
                # Format status badge
                status_display = "✅ Approved" if "approved" in s.review_status.lower() else "⚠️ Draft"

                diag_str = ", ".join(s.diagnoses[:3]) + ("..." if len(s.diagnoses) > 3 else "") if s.diagnoses else "-"
                med_str = ", ".join(s.medications[:3]) + ("..." if len(s.medications) > 3 else "") if s.medications else "-"

                item_id = tree.insert(
                    "",
                    tk.END,
                    values=(
                        dt_str,
                        s.patient_id,
                        demo_str,
                        status_display,
                        diag_str,
                        med_str,
                        s.raw_snippet
                    )
                )
                path_by_item_id[item_id] = s.file_path

            lbl_count.config(text=f"{len(summaries)} record(s) found")

        def refresh_records():
            nonlocal all_summaries
            folder = Path(records_dir_var.get())
            all_summaries = index_consultation_records(folder)
            apply_search()

        def apply_search():
            q = entry_query.get()
            stat = combo_status.get()
            diag = entry_diag.get()
            med = entry_med.get()

            filtered = search_records(
                all_summaries,
                query=q if q.strip() else None,
                status_filter=stat if stat != "All" else None,
                diagnosis_filter=diag if diag.strip() else None,
                medication_filter=med if med.strip() else None,
            )
            render_treeview(filtered)

        def reset_filters():
            entry_query.delete(0, tk.END)
            combo_status.set("All")
            entry_diag.delete(0, tk.END)
            entry_med.delete(0, tk.END)
            render_treeview(all_summaries)

        # Double-click to load immediately
        tree.bind("<Double-1>", lambda event: on_load_selected())

        # Enter key triggers search
        entry_query.bind("<Return>", lambda event: apply_search())
        entry_diag.bind("<Return>", lambda event: apply_search())
        entry_med.bind("<Return>", lambda event: apply_search())

        # Initial load
        refresh_records()
        return browser_win

    def sign_off_consultation(self):
        if not self.current_record:
            messagebox.showwarning("No Data", "No consultation record available to review.")
            return

        clinician_name = simpledialog.askstring("Clinician Sign-Off", "Enter clinician name or license ID for approval stamp:")
        if not clinician_name or not clinician_name.strip():
            return

        # Update review statuses
        self.current_record.review_status = ReviewStatus.REVIEWED_APPROVED
        self.current_record.soap_note.review_status = ReviewStatus.REVIEWED_APPROVED

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        stamp = (
            f"\n\n==================================================================\n"
            f"[CLINICIAN APPROVAL STAMP]\n"
            f"Review Status: REVIEWED & APPROVED\n"
            f"Signed by: Dr. {clinician_name.strip()}\n"
            f"Timestamp: {timestamp}\n"
            f"=================================================================="
        )

        current_soap_text = self.txt_soap.get("1.0", tk.END).strip()
        if "[CLINICIAN APPROVAL STAMP]" not in current_soap_text:
            current_soap_text += stamp
            self.txt_soap.delete("1.0", tk.END)
            self.txt_soap.insert(tk.END, current_soap_text)

        self.lbl_soap_warning.config(
            text=f"✅ STATUS: REVIEWED & APPROVED — Signed by Dr. {clinician_name.strip()} ({timestamp})",
            fg="#065F46",
            bg="#D1FAE5"
        )
        self.frame_soap_banner.config(bg="#D1FAE5")

        # Update JSON tab
        self._render_json_tab(self.current_record)

        self.lbl_feedback.config(text=f"✅ Note approved and signed by Dr. {clinician_name.strip()}.")
        messagebox.showinfo(
            "Note Approved",
            f"Clinical SOAP note for consultation {self.current_record.consultation_id[:8]} has been signed and approved!"
        )

    def clear_consultation(self):
        if self.recorder and self.recorder.is_recording:
            if not messagebox.askyesno("Active Recording", "A recording is currently in progress. Cancel it?"):
                return
            try:
                self.recorder.cancel_recording()
            except Exception:
                pass

        self.current_record = None
        self.recorder = None
        self.recording_start_time = None
        self.is_busy = False

        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_import.config(state=tk.NORMAL)
        self.btn_demo.config(state=tk.NORMAL)
        self.btn_export.config(state=tk.DISABLED)
        self.btn_export_html.config(state=tk.DISABLED)
        self.btn_export_pdf.config(state=tk.DISABLED)

        self.lbl_record_status.config(text="● Audio: Idle", fg="#475569")
        self.lbl_asr_status.config(text="Model: Ready")
        self.lbl_feedback.config(text="Consultation reset. Ready for new recording.")

        self.txt_transcript.delete("1.0", tk.END)
        for item in self.tree_segments.get_children():
            self.tree_segments.delete(item)
        self.txt_medical.delete("1.0", tk.END)
        self.txt_soap.delete("1.0", tk.END)
        self.txt_json.delete("1.0", tk.END)

    def on_close(self):
        if self.recorder and self.recorder.is_recording:
            if messagebox.askyesno("Exit Confirmation", "An audio recording is active. Stop recording and exit?"):
                try:
                    self.recorder.cancel_recording()
                except Exception:
                    pass
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = HealthcareASRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
