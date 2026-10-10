"""FastEHR REST API and Clinical Interoperability Service.

Provides standard HTTP/REST endpoints for Hospital Electronic Medical Record (EMR/EHR) systems:
- Audio consultation upload and transcription (Faster-Whisper on CPU int8)
- Clinical entity extraction with negation and uncertainty preservation
- SOAP note compilation and clinician attestation sign-off
- ICD-10 diagnostic coding recommendations
- Multi-format exports: HL7 FHIR R4 Document Bundle JSON, ReportLab PDF, and ZIP Archives with SHA-256 manifests
- Pre-flight acoustic diagnostics and digital speech enhancement
"""

from __future__ import annotations
from datetime import datetime
import json
import logging
from pathlib import Path
import tempfile
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Core domain & processing engines
from src.models.patient_record import (
    ConsultationRecord,
    ReviewStatus,
    FindingStatus,
    TranscriptSegment,
    SpeakerRole,
)
from src.audio.asr import WhisperTranscriber
from src.audio.diagnostics import analyze_audio_data
from src.audio.enhancement import enhance_audio_file, enhance_audio_data
from src.processing.speaker_processor import process_transcript_segments
from src.processing.medical_processor import process_clinical_information, extract_demographics
from src.processing.soap_note import generate_soap_note
from src.processing.exporter import ClinicBranding, generate_html_report, export_pdf_report
from src.processing.archive_exporter import export_consultation_zip
from src.processing.fhir_exporter import build_fhir_bundle
from src.processing.icd10_coding import suggest_icd10_codes
from src.processing.record_search import index_consultation_records, search_records
from src.utils.paths import get_output_dir, get_models_dir, get_synthetic_wav_path

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Healthcare AI Clinical Documentation & FHIR Interoperability API",
    description=(
        "Production-ready local REST API enabling Electronic Health Record (EHR/EMR) systems "
        "to transcribe medical consultations, generate structured clinical SOAP notes, suggest "
        "ICD-10-CM codes, and export standard HL7 FHIR R4 document bundles."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for hospital intranet integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class DirectTranscriptRequest(BaseModel):
    """Payload for submitting raw consultation text without audio hardware."""
    transcript: str = Field(..., description="Verbatim consultation dialogue transcript")
    model_size: str = Field(default="tiny.en", description="Whisper model tier")
    clinic_name: Optional[str] = Field(default="Apex Healthcare Clinic")
    physician_name: Optional[str] = Field(default="Dr. Attending Physician, MD")
    department: Optional[str] = Field(default="Ambulatory Care")


class ClinicianSignOffRequest(BaseModel):
    """Payload for clinician review and digital attestation."""
    signed_by: str = Field(..., description="Licensed clinician full name")
    license_npi: str = Field(..., description="State medical license or NPI number")
    attestation_confirmed: bool = Field(..., description="Clinician affirmation of clinical accuracy")
    edited_subjective: Optional[str] = None
    edited_objective: Optional[str] = None
    edited_assessment: Optional[str] = None
    edited_plan: Optional[str] = None


# ---------------------------------------------------------------------------
# Root & Health Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["System"])
def root_info() -> Dict[str, Any]:
    """Returns service metadata and available endpoint links."""
    return {
        "service": "Healthcare AI Clinical Documentation & FHIR Interoperability API",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "disclaimer": "All generated records are DRAFT PENDING REVIEW until attested by a licensed clinician.",
    }


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Comprehensive health check including storage paths and offline models."""
    out_dir = get_output_dir()
    models_dir = get_models_dir()
    tiny_model = models_dir / "tiny.en" / "model.bin"

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "storage": {
            "output_dir": str(out_dir),
            "output_dir_exists": out_dir.is_dir(),
        },
        "speech_engine": {
            "backend": "Faster-Whisper (CPU int8)",
            "models_dir": str(models_dir),
            "offline_tiny_bundled": tiny_model.is_file(),
        },
    }


# ---------------------------------------------------------------------------
# Consultation Processing Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/consultations/transcribe", tags=["Consultations"])
async def transcribe_consultation(
    audio: Optional[UploadFile] = File(None, description="Consultation audio file (WAV/MP3/M4A)"),
    transcript_text: Optional[str] = Form(None, description="Direct text transcript if no audio file"),
    model_size: str = Form("tiny.en"),
    enhance_audio: bool = Form(False, description="Apply digital bandpass & spectral noise reduction"),
    clinic_name: str = Form("Apex Healthcare Clinic"),
    physician_name: str = Form("Dr. Attending Physician, MD"),
    department: str = Form("Ambulatory Care"),
) -> Dict[str, Any]:
    """
    Transcribes audio or processes dialogue into a complete ConsultationRecord.
    Performs speech recognition, speaker role estimation, clinical NLP entity extraction,
    SOAP note synthesis, and ICD-10 diagnostic recommendations.
    """
    if not audio and not transcript_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either an audio file upload or direct 'transcript_text' must be provided.",
        )

    # 1. Obtain transcript & segments
    if audio and audio.filename:
        suffix = Path(audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
            content = await audio.read()
            tmp_audio.write(content)
            tmp_audio_path = Path(tmp_audio.name)

        try:
            transcriber = WhisperTranscriber(model_size=model_size, device="cpu", compute_type="int8")
            raw_transcript, raw_segments = transcriber.transcribe_file(
                tmp_audio_path,
                enhance_audio=enhance_audio,
            )
            from src.audio.diarization import diarize_consultation_audio
            annotated_segments = diarize_consultation_audio(raw_segments, audio_path=tmp_audio_path)
        finally:
            if tmp_audio_path.exists():
                tmp_audio_path.unlink()
    else:
        raw_transcript = transcript_text or ""
        raw_segments = []
        lines = [l.strip() for l in raw_transcript.splitlines() if l.strip()]
        for idx, line in enumerate(lines):
            raw_segments.append(
                TranscriptSegment(
                    id=idx + 1,
                    text=line,
                    start_time=float(idx * 4),
                    end_time=float((idx + 1) * 4),
                    speaker_role=SpeakerRole.UNKNOWN,
                )
            )
        from src.audio.diarization import diarize_consultation_audio
        annotated_segments = diarize_consultation_audio(raw_segments)

    # 3. Clinical Entity & Demographics Extraction
    extracted_info = process_clinical_information(raw_transcript)
    demographics = extract_demographics(raw_transcript)

    # 4. SOAP Note Generation
    soap_note = generate_soap_note(extracted_info)

    # 5. Assemble Master Record
    record = ConsultationRecord(
        demographics=demographics,
        raw_transcript=raw_transcript,
        segments=annotated_segments,
        extracted_info=extracted_info,
        soap_note=soap_note,
        review_status=ReviewStatus.DRAFT_PENDING_REVIEW,
    )

    # 6. Save atomically to output storage
    out_dir = get_output_dir()
    record.save_json(output_dir=out_dir)

    # 7. Compute ICD-10 matches
    icd_matches = [m.to_dict() for m in suggest_icd10_codes(extracted_info)]

    res_dict = record.to_dict()
    res_dict["icd10_recommendations"] = icd_matches
    return res_dict


@app.post("/api/v1/consultations/text", tags=["Consultations"])
def process_text_consultation(request: DirectTranscriptRequest) -> Dict[str, Any]:
    """Process consultation from structured text payload."""
    raw_transcript = request.transcript
    raw_segments = [
        TranscriptSegment(
            id=idx + 1,
            text=line.strip(),
            start_time=float(idx * 4),
            end_time=float((idx + 1) * 4),
            speaker_role=SpeakerRole.UNKNOWN,
        )
        for idx, line in enumerate(raw_transcript.splitlines())
        if line.strip()
    ]

    annotated_segments = process_transcript_segments(raw_segments)
    extracted_info = process_clinical_information(raw_transcript)
    demographics = extract_demographics(raw_transcript)
    soap_note = generate_soap_note(extracted_info)

    record = ConsultationRecord(
        demographics=demographics,
        raw_transcript=raw_transcript,
        segments=annotated_segments,
        extracted_info=extracted_info,
        soap_note=soap_note,
        review_status=ReviewStatus.DRAFT_PENDING_REVIEW,
    )

    out_dir = get_output_dir()
    record.save_json(output_dir=out_dir)

    icd_matches = [m.to_dict() for m in suggest_icd10_codes(extracted_info)]
    res = record.to_dict()
    res["icd10_recommendations"] = icd_matches
    return res


# ---------------------------------------------------------------------------
# Consultation Management & Query Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/consultations", tags=["Consultations"])
def list_consultations(
    q: Optional[str] = Query(None, description="Search query across patient ID, diagnoses, symptoms"),
    status: Optional[str] = Query(None, description="Review status filter ('draft_pending_review', 'reviewed_approved')"),
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """Lists and searches past clinical encounters."""
    summaries = index_consultation_records(get_output_dir())
    filtered = search_records(summaries, query=q, status_filter=status)
    return [s.__dict__ for s in filtered[:limit]]


def _find_record_by_id(consultation_id: str) -> Tuple[ConsultationRecord, Path]:
    """Helper to locate and deserialize a ConsultationRecord on disk."""
    out_dir = get_output_dir()
    for json_file in out_dir.glob("*.json"):
        if json_file.name == "batch_summary.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("consultation_id") == consultation_id:
                return ConsultationRecord.from_dict(data), json_file
        except Exception:
            continue
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Consultation with ID '{consultation_id}' not found.",
    )


@app.get("/api/v1/consultations/{consultation_id}", tags=["Consultations"])
def get_consultation(consultation_id: str) -> Dict[str, Any]:
    """Retrieves full consultation record JSON by ID."""
    record, _ = _find_record_by_id(consultation_id)
    res = record.to_dict()
    res["icd10_recommendations"] = [m.to_dict() for m in suggest_icd10_codes(record.extracted_info)]
    return res


@app.post("/api/v1/consultations/{consultation_id}/sign-off", tags=["Consultations"])
def sign_off_consultation(
    consultation_id: str,
    sign_off: ClinicianSignOffRequest,
) -> Dict[str, Any]:
    """Clinician attestation endpoint to verify, approve, and sign clinical documentation."""
    if not sign_off.attestation_confirmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Clinician must affirm the attestation statement to approve medical documentation.",
        )

    record, file_path = _find_record_by_id(consultation_id)

    # Apply clinician edits if supplied
    if sign_off.edited_subjective:
        record.soap_note.subjective = sign_off.edited_subjective
    if sign_off.edited_objective:
        record.soap_note.objective = sign_off.edited_objective
    if sign_off.edited_assessment:
        record.soap_note.assessment = sign_off.edited_assessment
    if sign_off.edited_plan:
        record.soap_note.plan = sign_off.edited_plan

    # Apply attestation
    record.review_status = ReviewStatus.REVIEWED_APPROVED
    record.soap_note.review_status = ReviewStatus.REVIEWED_APPROVED
    record.signed_by = f"{sign_off.signed_by} ({sign_off.license_npi})"
    record.signed_at = datetime.now().isoformat()

    # Re-save atomically
    record.save_json(output_dir=get_output_dir())
    return record.to_dict()


# ---------------------------------------------------------------------------
# Interoperability & Export Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/consultations/{consultation_id}/fhir", tags=["Interoperability"])
def get_fhir_bundle(consultation_id: str) -> Dict[str, Any]:
    """Returns standardized HL7 FHIR R4 Document Bundle JSON for hospital EMR integration."""
    record, _ = _find_record_by_id(consultation_id)
    return build_fhir_bundle(record)


@app.get("/api/v1/consultations/{consultation_id}/pdf", tags=["Exports"])
def get_pdf_report(
    consultation_id: str,
    clinic_name: Optional[str] = "Apex Healthcare Clinic",
    physician_name: Optional[str] = "Dr. Attending Physician, MD",
) -> FileResponse:
    """Generates and downloads a print-ready ReportLab clinical PDF document."""
    record, _ = _find_record_by_id(consultation_id)
    branding = ClinicBranding(clinic_name=clinic_name, physician_name=physician_name)

    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_pdf_path = Path(temp_pdf.name)
    export_pdf_report(record, temp_pdf_path, branding=branding)

    return FileResponse(
        path=str(temp_pdf_path),
        media_type="application/pdf",
        filename=f"Clinical_Report_{consultation_id[:8]}.pdf",
    )


@app.get("/api/v1/consultations/{consultation_id}/archive", tags=["Exports"])
def get_zip_archive(consultation_id: str) -> FileResponse:
    """Generates a complete consultation ZIP archive with SHA-256 cryptographic integrity manifest."""
    record, _ = _find_record_by_id(consultation_id)
    temp_dir = tempfile.mkdtemp()
    zip_path = export_consultation_zip(record, output_dir=Path(temp_dir))

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"Archive_{consultation_id[:8]}.zip",
    )


# ---------------------------------------------------------------------------
# Acoustic Pre-Flight & Enhancement Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/audio/diagnostics", tags=["Audio & DSP"])
async def audio_diagnostics_endpoint(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyzes audio recording for RMS dBFS, peak amplitude, SNR, and digital clipping."""
    import soundfile as sf

    suffix = Path(audio.filename or "recording.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
        content = await audio.read()
        tmp_f.write(content)
        tmp_f_path = Path(tmp_f.name)

    try:
        data, sr = sf.read(str(tmp_f_path), dtype="float32")
        result = analyze_audio_data(data, sample_rate=sr, device_name=audio.filename or "Uploaded Audio")
        return result.to_dict()
    finally:
        if tmp_f_path.exists():
            tmp_f_path.unlink()


@app.post("/api/v1/audio/enhance", tags=["Audio & DSP"])
async def audio_enhance_endpoint(audio: UploadFile = File(...)) -> FileResponse:
    """Applies digital bandpass and spectral subtraction noise reduction to audio file."""
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        content = await audio.read()
        tmp_in.write(content)
        tmp_in_path = Path(tmp_in.name)

    enhanced_out = tmp_in_path.parent / f"enhanced_{tmp_in_path.stem}.wav"
    try:
        enhance_audio_file(tmp_in_path, output_path=enhanced_out)
        return FileResponse(
            path=str(enhanced_out),
            media_type="audio/wav",
            filename=f"enhanced_{audio.filename or 'audio.wav'}",
        )
    finally:
        if tmp_in_path.exists():
            tmp_in_path.unlink()
