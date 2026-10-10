"""Healthcare AI Clinical Documentation Assistant — Streamlit Web Deployment.

DISCLAIMER:
All generated documentation is marked DRAFT PENDING REVIEW.
No output constitutes final medical records until verified, edited,
and signed by a licensed healthcare professional.
"""

from __future__ import annotations
import base64
from datetime import datetime
import io
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Optional, List, Dict, Any

import streamlit as st

# Core domain & processing modules
from src.models.patient_record import (
    ConsultationRecord,
    PatientDemographics,
    ExtractedClinicalInformation,
    SOAPNote,
    ReviewStatus,
    FindingStatus,
    SpeakerRole,
    TranscriptSegment,
)
from src.audio.asr import WhisperTranscriber
from src.audio.diagnostics import AudioDiagnosticResult, analyze_audio_data
from src.processing.speaker_processor import process_transcript_segments
from src.processing.medical_processor import process_clinical_information, extract_demographics
from src.processing.soap_note import generate_soap_note
from src.processing.exporter import ClinicBranding, generate_html_report, export_pdf_report
from src.processing.archive_exporter import export_consultation_zip
from src.processing.fhir_exporter import build_fhir_bundle
from src.processing.icd10_coding import suggest_icd10_codes, ICD10CodeMatch
from src.processing.record_search import index_consultation_records, search_records
from src.utils.paths import (
    get_output_dir,
    get_recordings_dir,
    get_models_dir,
    get_synthetic_wav_path,
)

# Configure Streamlit page
st.set_page_config(
    page_title="Healthcare AI Clinical Documentation Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling & Aesthetics
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, .clinic-brand-title {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.02em;
}

/* Clinical Safety Alert Banner */
.safety-banner-draft {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(245, 158, 11, 0.15) 100%);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-left: 6px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
    color: #b91c1c;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.08);
}

.safety-banner-approved {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.15) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-left: 6px solid #10b981;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
    color: #047857;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
}

/* Glassmorphism Cards */
.medical-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.medical-card:hover {
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.12);
}

/* Role Dialogue Cards */
.bubble-doctor {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(2, 132, 199, 0.12) 100%);
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-left: 4px solid #0284c7;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

.bubble-patient {
    background: linear-gradient(135deg, rgba(20, 184, 166, 0.08) 0%, rgba(13, 148, 136, 0.12) 100%);
    border: 1px solid rgba(20, 184, 166, 0.3);
    border-left: 4px solid #0d9488;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

.bubble-unknown {
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-left: 4px solid #94a3b8;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

/* Entity Badges */
.badge-present {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    margin: 3px;
}

.badge-negated {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    margin: 3px;
}

.badge-uncertain {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
    margin: 3px;
}

/* Metric Display Cards */
.metric-box {
    text-align: center;
    padding: 12px;
    border-radius: 10px;
    background: rgba(14, 165, 233, 0.06);
    border: 1px solid rgba(14, 165, 233, 0.2);
}
.metric-box .metric-num {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0284c7;
    font-family: 'Outfit', sans-serif;
}
.metric-box .metric-lbl {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "record" not in st.session_state:
    st.session_state.record: Optional[ConsultationRecord] = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path: Optional[str] = None
if "branding" not in st.session_state:
    st.session_state.branding = ClinicBranding()
if "transcriber" not in st.session_state:
    # We will initialize lazily or with cached loader
    st.session_state.transcriber_model = "tiny.en"


@st.cache_resource
def get_cached_transcriber(model_size: str = "tiny.en") -> WhisperTranscriber:
    """Instantiate and cache Faster-Whisper transcriber on CPU int8."""
    return WhisperTranscriber(model_size=model_size, device="cpu", compute_type="int8")


# ---------------------------------------------------------------------------
# Sidebar: Configuration & Clinical Ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=300&auto=format&fit=crop&q=80",
        caption="Apex Health Informatics Suite",
        use_container_width=True,
    )
    st.title("🩺 Healthcare ASR")
    st.caption("AI Clinical Documentation & Interoperability Platform")

    st.markdown("---")
    st.subheader("⚙️ Engine Configuration")

    model_choice = st.selectbox(
        "Whisper Speech Model",
        ["tiny.en", "base.en", "small.en"],
        index=0,
        help="tiny.en is bundled locally for offline CPU inference.",
    )
    if model_choice != st.session_state.transcriber_model:
        st.session_state.transcriber_model = model_choice

    device_info = "⚡ CPU (int8 Quantized, 100% Offline)"
    st.info(f"**Inference:** {device_info}")
    enhance_audio_toggle = st.checkbox(
        "🔊 Speech Enhancement & Noise Gate",
        value=True,
        help="Applies digital bandpass filtering and spectral subtraction noise reduction to improve ASR accuracy on noisy audio.",
    )

    st.markdown("---")
    st.subheader("🏥 Clinic Header & Letterhead")
    clinic_name = st.text_input("Clinic / Hospital Name", value=st.session_state.branding.clinic_name)
    physician_name = st.text_input("Attending Physician", value=st.session_state.branding.physician_name)
    department = st.text_input("Department", value=st.session_state.branding.department)
    contact_info = st.text_input("Contact Info", value=st.session_state.branding.contact_info)

    st.session_state.branding = ClinicBranding(
        clinic_name=clinic_name,
        physician_name=physician_name,
        department=department,
        contact_info=contact_info,
    )

    st.markdown("---")
    st.subheader("📥 Consultation Audio Ingestion")

    input_mode = st.radio(
        "Ingestion Method",
        ["✨ Run Synthetic Demo Consultation", "📁 Upload Audio File (WAV/MP3)", "📝 Direct Clinical Transcript"],
        index=0,
    )

    uploaded_audio = None
    direct_transcript_text = ""

    if input_mode == "✨ Run Synthetic Demo Consultation":
        st.caption("Runs pre-recorded clinical doctor-patient encounter audio.")
        demo_audio_file = get_synthetic_wav_path()
        if not demo_audio_file.exists():
            st.warning("Synthetic consultation WAV will be generated automatically on run.")
    elif input_mode == "📁 Upload Audio File (WAV/MP3)":
        uploaded_audio = st.file_uploader("Upload Medical Audio", type=["wav", "mp3", "m4a", "flac"])
    else:
        direct_transcript_text = st.text_area(
            "Paste Consultation Dialogue",
            value="Doctor: Hello John, how can I help you today?\n"
                  "Patient: Hi Doctor, I've had a bad sore throat and fever of 101.4 degrees for three days.\n"
                  "Doctor: Any chest pain or shortness of breath?\n"
                  "Patient: No chest pain at all, and no breathing trouble. But swallowing hurts.\n"
                  "Doctor: Let's check vitals. Blood pressure is 120/80, pulse 76, oxygen 98%. "
                  "You have acute pharyngitis. I will prescribe Amoxicillin 500mg three times daily.",
            height=160,
        )

    col_btn1, col_btn2 = st.columns(2)
    process_btn = col_btn1.button("🚀 Process", type="primary", use_container_width=True)
    reset_btn = col_btn2.button("🔄 Reset", use_container_width=True)

    if reset_btn:
        st.session_state.record = None
        st.session_state.audio_path = None
        st.rerun()


# ---------------------------------------------------------------------------
# Helper: Processing Routine
# ---------------------------------------------------------------------------
def execute_clinical_pipeline(
    audio_path: Optional[Path],
    raw_text_input: Optional[str] = None,
    enhance_audio: bool = False,
):
    with st.spinner("Transcribing audio & executing medical NLP engines..."):
        progress_bar = st.progress(10)

        # 1. Transcription or Direct Text
        if audio_path and audio_path.exists():
            transcriber = get_cached_transcriber(st.session_state.transcriber_model)
            progress_bar.progress(30)
            try:
                raw_transcript, raw_segments = transcriber.transcribe_file(
                    audio_path,
                    enhance_audio=enhance_audio,
                )
            except TypeError:
                # Defensive fallback if cached memory model does not take enhance_audio
                raw_transcript, raw_segments = transcriber.transcribe_file(audio_path)
            st.session_state.audio_path = str(audio_path)
        else:
            raw_transcript = raw_text_input or ""
            raw_segments = []
            if raw_transcript:
                # Synthesize segments from lines
                lines = [l.strip() for l in raw_transcript.splitlines() if l.strip()]
                for idx, line in enumerate(lines):
                    raw_segments.append(
                        TranscriptSegment(
                            id=idx,
                            text=line,
                            start_time=float(idx * 4),
                            end_time=float((idx + 1) * 4),
                            speaker_role=SpeakerRole.UNKNOWN,
                        )
                    )
            st.session_state.audio_path = None

        progress_bar.progress(55)

        # 2. Speaker-role Estimation
        annotated_segments = process_transcript_segments(raw_segments)

        # 3. Clinical Entity & Demographics Extraction
        extracted_info = process_clinical_information(raw_transcript)
        demographics = extract_demographics(raw_transcript)

        progress_bar.progress(80)

        # 4. SOAP Note Generation
        soap_note = generate_soap_note(extracted_info)

        # 5. Master Record Construction
        record = ConsultationRecord(
            demographics=demographics,
            raw_transcript=raw_transcript,
            segments=annotated_segments,
            extracted_info=extracted_info,
            soap_note=soap_note,
            review_status=ReviewStatus.DRAFT_PENDING_REVIEW,
        )

        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()

        # Save record atomically to output directory
        out_dir = get_output_dir()
        record.save_json(output_dir=out_dir)

        st.session_state.record = record


# Execute process button click
if process_btn:
    if input_mode == "✨ Run Synthetic Demo Consultation":
        synth_path = get_synthetic_wav_path()
        if not synth_path.exists():
            from src.audio.diagnostics import SyntheticAudioGenerator
            SyntheticAudioGenerator.generate_synthetic_consultation_wav(synth_path)
        execute_clinical_pipeline(audio_path=synth_path, enhance_audio=enhance_audio_toggle)
        st.success("Synthetic consultation transcribed and processed successfully!")
        st.rerun()

    elif input_mode == "📁 Upload Audio File (WAV/MP3)":
        if uploaded_audio is None:
            st.error("Please upload an audio file before clicking Process.")
        else:
            # Save uploaded audio to temp file
            suffix = Path(uploaded_audio.name).suffix or ".wav"
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_audio.write(uploaded_audio.read())
            temp_audio.close()
            execute_clinical_pipeline(audio_path=Path(temp_audio.name), enhance_audio=enhance_audio_toggle)
            st.success(f"Uploaded audio '{uploaded_audio.name}' processed successfully!")
            st.rerun()

    else:
        if not direct_transcript_text.strip():
            st.error("Please enter consultation transcript text.")
        else:
            execute_clinical_pipeline(audio_path=None, raw_text_input=direct_transcript_text)
            st.success("Consultation transcript parsed and processed successfully!")
            st.rerun()


# ---------------------------------------------------------------------------
# Main View Dashboard
# ---------------------------------------------------------------------------
rec: Optional[ConsultationRecord] = st.session_state.record

# Top Banner: Clinician Safety & Status
if rec:
    is_approved = rec.review_status == ReviewStatus.REVIEWED_APPROVED
    if is_approved:
        st.markdown(
            """
            <div class="safety-banner-approved">
                <strong>✅ CLINICALLY VERIFIED & APPROVED:</strong>
                This clinical consultation note has been reviewed, attested, and signed by an authorized physician.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="safety-banner-draft">
                <strong>⚠️ DRAFT PENDING REVIEW — CLINICAL SAFETY NOTICE:</strong>
                This document is generated by an artificial intelligence assistant.
                It is <strong>NOT</strong> a certified medical record until reviewed, verified, edited, and signed by a licensed healthcare provider.
            </div>
            """,
            unsafe_allow_html=True,
        )

# Page Header
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown(f"## 🏥 {st.session_state.branding.clinic_name}")
    st.markdown(
        f"**Department:** {st.session_state.branding.department} &nbsp;|&nbsp; "
        f"**Physician:** {st.session_state.branding.physician_name}"
    )
with header_col2:
    if rec:
        status_color = "green" if rec.review_status == ReviewStatus.REVIEWED_APPROVED else "orange"
        st.markdown(f"**Encounter ID:** `{rec.consultation_id[:12]}`")
        st.markdown(f"**Status:** :{status_color}[{rec.review_status.value}]")
    else:
        st.markdown("**Status:** :gray[Awaiting Consultation Input]")

st.markdown("---")

if not rec:
    st.info("👈 **Welcome to Healthcare AI Documentation Assistant.** Please select an input option in the sidebar (such as **Run Synthetic Demo Consultation**) and click **🚀 Process** to transcribe audio, extract clinical entities, generate SOAP notes, and view ICD-10 suggestions.")
    
    # Quick demo preview section
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("### 🎙️ Faster-Whisper ASR")
        st.caption("100% Offline local speech recognition with automatic speaker-role categorization (Doctor vs Patient).")
    with col_b:
        st.markdown("### 🧠 Clinical NLP Engine")
        st.caption("Extracts symptoms with negation sensitivity ('denies chest pain'), vital signs, medications, and allergies.")
    with col_c:
        st.markdown("### 📦 Interoperability Suite")
        st.caption("One-click exports for HL7 FHIR R4 JSON, ReportLab PDF, print-ready HTML, and ZIP archives with SHA-256 manifests.")
    st.stop()


# ---------------------------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎙️ Audio & Transcript",
    "🔬 Clinical Extraction",
    "📝 SOAP Note & Sign-Off",
    "🏷️ ICD-10-CM Coding",
    "📦 Export & Interoperability",
    "🗂️ Encounter Archive",
])


# ===========================================================================
# TAB 1: Audio & Transcription
# ===========================================================================
with tab1:
    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        st.subheader("🔊 Audio Signal & Diagnostics")
        if st.session_state.audio_path and Path(st.session_state.audio_path).exists():
            st.audio(st.session_state.audio_path)
            
            # Audio Diagnostic Check
            try:
                import soundfile as sf
                audio_data, sr = sf.read(st.session_state.audio_path, dtype="float32")
                diag = analyze_audio_data(audio_data, sample_rate=sr, device_name=Path(st.session_state.audio_path).name)
                st.markdown(
                    f"""
                    <div class="medical-card">
                        <strong>Pre-Flight Acoustic Metrics ({diag.quality_rating}):</strong><br>
                        • <strong>Peak Amplitude:</strong> {diag.peak_db:.1f} dBFS<br>
                        • <strong>RMS Energy:</strong> {diag.rms_db:.1f} dBFS<br>
                        • <strong>Estimated SNR:</strong> {diag.snr_db:.1f} dB<br>
                        • <strong>Clipping Detected:</strong> {"⚠️ Yes" if diag.has_clipping else "✅ No"}<br>
                        • <strong>Signal Quality:</strong> {"❌ Silent" if diag.is_silent else "✅ Active Audio"}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.caption(f"Audio diagnostic info unavailable: {e}")
        else:
            st.info("No audio recording file attached (direct transcript ingestion).")

        st.subheader("📜 Raw Consultation Transcript")
        st.text_area(
            "Verbatim Speech Output",
            value=rec.raw_transcript,
            height=260,
            disabled=True,
        )

    with col_t2:
        st.subheader("👥 Conversational Dialogue Timeline")
        st.caption("Speaker role estimated via clinical contextual heuristics.")

        if rec.segments:
            for seg in rec.segments:
                role_val = seg.speaker_role.value if isinstance(seg.speaker_role, SpeakerRole) else str(seg.speaker_role)
                time_tag = f"[{seg.start_time:.1f}s - {seg.end_time:.1f}s]"

                if role_val == SpeakerRole.DOCTOR.value:
                    bubble_class = "bubble-doctor"
                    icon = "🩺"
                    title = f"DOCTOR {time_tag}"
                elif role_val == SpeakerRole.PATIENT.value:
                    bubble_class = "bubble-patient"
                    icon = "👤"
                    title = f"PATIENT {time_tag}"
                else:
                    bubble_class = "bubble-unknown"
                    icon = "❓"
                    title = f"UNKNOWN {time_tag}"

                st.markdown(
                    f"""
                    <div class="{bubble_class}">
                        <div style="font-size: 0.8rem; font-weight: 700; opacity: 0.8; margin-bottom: 4px;">
                            {icon} {title}
                        </div>
                        <div style="font-size: 0.95rem;">
                            {seg.text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write(rec.raw_transcript)


# ===========================================================================
# TAB 2: Clinical Entity Extraction
# ===========================================================================
with tab2:
    info = rec.extracted_info
    vitals = info.vital_signs

    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    present_symptoms = [s for s in info.symptoms if s.status == FindingStatus.PRESENT]
    negated_symptoms = [s for s in info.symptoms if s.status == FindingStatus.EXPLICITLY_ABSENT]
    
    with m1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-num">{len(present_symptoms)}</div>
                <div class="metric-lbl">Active Symptoms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-num">{len(negated_symptoms)}</div>
                <div class="metric-lbl">Negated Findings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-num">{len(info.medications)}</div>
                <div class="metric-lbl">Medications</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-num">{len(info.diagnoses)}</div>
                <div class="metric-lbl">Diagnoses</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        # Symptoms & Negation
        st.subheader("🌡️ Symptoms & Clinical Findings")
        if info.symptoms:
            badges_html = ""
            for sym in info.symptoms:
                if sym.status == FindingStatus.PRESENT:
                    badges_html += f'<span class="badge-present">✓ {sym.name}</span>'
                elif sym.status == FindingStatus.EXPLICITLY_ABSENT:
                    badges_html += f'<span class="badge-negated">✗ Denies {sym.name} (NEGATED)</span>'
                else:
                    badges_html += f'<span class="badge-uncertain">? {sym.name} (UNCERTAIN)</span>'
            st.markdown(f'<div class="medical-card">{badges_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No specific symptoms identified.")

        # Vitals Card
        st.subheader("📊 Vital Signs Matrix")
        bp_val = vitals.blood_pressure_str or (
            f"{vitals.blood_pressure_systolic}/{vitals.blood_pressure_diastolic} mmHg"
            if vitals.blood_pressure_systolic and vitals.blood_pressure_diastolic
            else "Not recorded"
        )
        hr_val = f"{vitals.heart_rate_bpm} bpm" if vitals.heart_rate_bpm else "Not recorded"
        temp_val = (
            f"{vitals.temperature_f:.1f} °F"
            if vitals.temperature_f
            else (f"{vitals.temperature_c:.1f} °C" if vitals.temperature_c else "Not recorded")
        )
        spo2_val = f"{vitals.spo2_percent}%" if vitals.spo2_percent else "Not recorded"
        glucose_val = vitals.blood_glucose_str or (f"{vitals.blood_glucose_mg_dl} mg/dL" if vitals.blood_glucose_mg_dl else "Not recorded")

        st.markdown(
            f"""
            <div class="medical-card">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(148,163,184,0.2);">
                        <td style="padding: 8px 4px; font-weight: 600;">Blood Pressure</td>
                        <td style="padding: 8px 4px; text-align: right; color: #0284c7;">{bp_val}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(148,163,184,0.2);">
                        <td style="padding: 8px 4px; font-weight: 600;">Heart Rate</td>
                        <td style="padding: 8px 4px; text-align: right; color: #0284c7;">{hr_val}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(148,163,184,0.2);">
                        <td style="padding: 8px 4px; font-weight: 600;">Temperature</td>
                        <td style="padding: 8px 4px; text-align: right; color: #0284c7;">{temp_val}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(148,163,184,0.2);">
                        <td style="padding: 8px 4px; font-weight: 600;">Oxygen Saturation (SpO2)</td>
                        <td style="padding: 8px 4px; text-align: right; color: #0284c7;">{spo2_val}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 4px; font-weight: 600;">Blood Glucose</td>
                        <td style="padding: 8px 4px; text-align: right; color: #0284c7;">{glucose_val}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c2:
        # Medications
        st.subheader("💊 Medications & Dosages")
        if info.medications:
            for med in info.medications:
                dose = f" • Dosage: {med.dosage}" if med.dosage else ""
                freq = f" • Frequency: {med.frequency}" if med.frequency else ""
                st.markdown(
                    f"""
                    <div class="medical-card" style="padding: 12px 16px;">
                        <strong>💊 {med.name}</strong><br>
                        <span style="font-size: 0.88rem; color: #64748b;">{dose}{freq}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No medications discussed.")

        # Diagnoses & Conditions
        st.subheader("🩺 Diagnoses & Medical History")
        if info.diagnoses:
            for diag in info.diagnoses:
                st.markdown(
                    f"""
                    <div class="medical-card" style="padding: 12px 16px;">
                        <strong>🩺 {diag.name}</strong> ({diag.status.value})
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No explicit diagnoses recorded.")

        # Allergies
        st.subheader("⚠️ Allergies")
        if info.allergies:
            for alg in info.allergies:
                st.markdown(f'<span class="badge-negated">Allergy: {alg.name}</span>', unsafe_allow_html=True)
        else:
            st.caption("No known drug allergies reported.")


# ===========================================================================
# TAB 3: SOAP Note & Clinician Sign-Off
# ===========================================================================
with tab3:
    st.subheader("📋 Structured Clinical SOAP Note")
    st.caption("Subjective, Objective, Assessment, Plan documentation.")

    soap = rec.soap_note

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### **S — Subjective**")
        edited_s = st.text_area(
            "Chief complaint & history of present illness",
            value=soap.subjective,
            height=160,
            key="soap_subj",
        )

        st.markdown("#### **A — Assessment**")
        edited_a = st.text_area(
            "Clinical evaluation, differential & confirmed diagnoses",
            value=soap.assessment,
            height=140,
            key="soap_assess",
        )

    with col_s2:
        st.markdown("#### **O — Objective**")
        edited_o = st.text_area(
            "Vitals, physical examination & objective findings",
            value=soap.objective,
            height=160,
            key="soap_obj",
        )

        st.markdown("#### **P — Plan**")
        edited_p = st.text_area(
            "Prescriptions, patient counseling, diagnostics & follow-up",
            value=soap.plan,
            height=140,
            key="soap_plan",
        )

    st.markdown("---")
    st.subheader("✍️ Clinician Verification & Attestation")

    sign_col1, sign_col2 = st.columns([2, 1])
    with sign_col1:
        phys_name_input = st.text_input("Signing Clinician Name", value=st.session_state.branding.physician_name)
        phys_npi_input = st.text_input("License / NPI Number", value="NPI-9182348123")
        attest_check = st.checkbox(
            "I attest that I have reviewed, verified, and edited this clinical note. "
            "I take full clinical responsibility for its medical accuracy.",
            value=rec.review_status == ReviewStatus.REVIEWED_APPROVED,
        )

    with sign_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📝 Sign & Finalize Note", type="primary", use_container_width=True):
            if not attest_check:
                st.error("You must check the attestation box to verify clinical responsibility.")
            else:
                # Update SOAP note content
                soap.subjective = edited_s
                soap.objective = edited_o
                soap.assessment = edited_a
                soap.plan = edited_p
                soap.review_status = ReviewStatus.REVIEWED_APPROVED

                # Update Record sign-off metadata
                rec.review_status = ReviewStatus.REVIEWED_APPROVED
                rec.signed_by = phys_name_input
                rec.signed_at = datetime.now().isoformat()

                # Re-save updated record
                rec.save_json(output_dir=get_output_dir())
                st.success("Note officially signed and finalized by clinician!")
                st.rerun()


# ===========================================================================
# TAB 4: ICD-10-CM Coding Assistant
# ===========================================================================
with tab4:
    st.subheader("🏷️ ICD-10-CM Diagnostic Coding Recommendations")
    st.caption("Automated code matching from clinical diagnoses and symptom mentions.")

    icd_matches = suggest_icd10_codes(rec.extracted_info)

    if icd_matches:
        for match in icd_matches:
            col_m1, col_m2, col_m3 = st.columns([1, 3, 1])
            with col_m1:
                st.markdown(f"### `{match.code}`")
            with col_m2:
                st.markdown(f"**{match.description}**")
                st.caption(f"Category: {match.category} | Matched from: *\"{match.matched_term}\"*")
            with col_m3:
                conf_pct = int(match.confidence * 100)
                st.metric("Confidence", f"{conf_pct}%")
                st.progress(match.confidence)
            st.markdown("---")
    else:
        st.info("No ICD-10 codes matched for current findings.")


# ===========================================================================
# TAB 5: Export & Interoperability Suite
# ===========================================================================
with tab5:
    st.subheader("📦 Clinical Report & Interoperability Exports")
    st.caption("Generate multi-standard outputs for EHR integration and clinical archives.")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("#### 📄 Document Exports")

        # 1. HTML Report
        html_str = generate_html_report(rec, branding=st.session_state.branding)
        st.download_button(
            label="🌐 Download Printable HTML Report",
            data=html_str,
            file_name=f"Consultation_{rec.consultation_id[:8]}.html",
            mime="text/html",
            use_container_width=True,
        )

        # 2. PDF Report
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                tmp_pdf_path = Path(tmp_pdf.name)
            export_pdf_report(rec, tmp_pdf_path, branding=st.session_state.branding)
            with open(tmp_pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📄 Download ReportLab PDF Document",
                data=pdf_bytes,
                file_name=f"Consultation_{rec.consultation_id[:8]}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF export error: {e}")

        # 3. Consultation Record JSON
        json_data = json.dumps(rec.to_dict(), indent=2, ensure_ascii=False)
        st.download_button(
            label="📊 Download Clinical Record JSON (v1.0)",
            data=json_data,
            file_name=f"Consultation_{rec.consultation_id[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_exp2:
        st.markdown("#### 🏥 Healthcare Interoperability")

        # 4. HL7 FHIR R4 Bundle
        try:
            fhir_bundle = build_fhir_bundle(rec)
            fhir_str = json.dumps(fhir_bundle, indent=2, ensure_ascii=False)
            st.download_button(
                label="🏥 Download HL7 FHIR R4 Document Bundle (JSON)",
                data=fhir_str,
                file_name=f"FHIR_Bundle_{rec.consultation_id[:8]}.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"FHIR export error: {e}")

        # 5. ZIP Archive with SHA-256 Manifest
        try:
            audio_p = Path(st.session_state.audio_path) if st.session_state.audio_path else None
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = export_consultation_zip(
                    rec,
                    output_dir=Path(tmp_dir),
                    audio_path=audio_p,
                    branding=st.session_state.branding,
                )
                with open(zip_path, "rb") as f:
                    zip_bytes = f.read()

            st.download_button(
                label="🔒 Download ZIP Archive (with SHA-256 Manifest)",
                data=zip_bytes,
                file_name=f"Archive_{rec.consultation_id[:8]}.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"ZIP export error: {e}")


# ===========================================================================
# TAB 6: Historical Encounter Archive
# ===========================================================================
with tab6:
    st.subheader("🗂️ Historical Encounters Browser")
    st.caption("Search, filter, and inspect previously recorded patient consultations.")

    out_dir = get_output_dir()
    summaries = index_consultation_records(out_dir)

    search_q = st.text_input("🔍 Search encounters by patient, diagnosis, symptom, or ID", value="")
    status_filter = st.selectbox("Filter Status", ["All", "draft_pending_review", "reviewed_approved"])

    filtered = search_records(
        summaries,
        query=search_q if search_q else None,
        status_filter=status_filter if status_filter != "All" else None,
    )

    st.write(f"Showing **{len(filtered)}** of **{len(summaries)}** encounters.")

    for idx, item in enumerate(filtered):
        with st.expander(f"🩺 Encounter {item.consultation_id[:8]} — Patient: {item.patient_id} ({item.created_at[:10]})"):
            st.markdown(f"**Status:** `{item.review_status}`")
            st.markdown(f"**Diagnoses:** {', '.join(item.diagnoses) if item.diagnoses else 'None'}")
            st.markdown(f"**Symptoms:** {', '.join(item.symptoms) if item.symptoms else 'None'}")
            st.markdown(f"**Transcript Snippet:** *\"{item.raw_snippet}\"*")

            if st.button(f"Load Encounter {item.consultation_id[:8]}", key=f"load_{item.consultation_id}_{idx}"):
                with open(item.file_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                st.session_state.record = ConsultationRecord.from_dict(loaded_data)
                st.success(f"Encounter {item.consultation_id[:8]} loaded!")
                st.rerun()
