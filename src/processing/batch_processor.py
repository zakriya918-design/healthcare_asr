"""Batch processing module for bulk clinical audio transcription and report generation.

Processes directories of WAV files, transcribing each consultation,
extracting medical entities, compiling SOAP notes, and saving JSON/PDF/HTML documentation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Callable, List, Optional, Union

from src.audio.asr import WhisperTranscriber, ASRError
from src.audio.recorder import validate_wav_file
from src.models.patient_record import ConsultationRecord, PatientDemographics, ReviewStatus
from src.processing.speaker_processor import process_transcript_segments
from src.processing.medical_processor import process_clinical_information, extract_demographics
from src.processing.soap_note import generate_soap_note
from src.processing.exporter import export_html_report, export_pdf_report, ClinicBranding

logger = logging.getLogger(__name__)


@dataclass
class BatchItemResult:
    """Individual result for a single processed audio file."""
    audio_path: Path
    success: bool
    record: Optional[ConsultationRecord] = None
    json_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    html_path: Optional[Path] = None
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "audio_file": self.audio_path.name,
            "success": self.success,
            "consultation_id": self.record.consultation_id if self.record else None,
            "json_path": str(self.json_path) if self.json_path else None,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "html_path": str(self.html_path) if self.html_path else None,
            "error_message": self.error_message,
            "processing_time_seconds": round(self.processing_time_seconds, 2)
        }


@dataclass
class BatchSummary:
    """Aggregated statistics for a batch processing run."""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_time_seconds: float = 0.0
    results: List[BatchItemResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "results": [r.to_dict() for r in self.results]
        }


def process_audio_file(
    audio_file: Path,
    output_dir: Path,
    transcriber: WhisperTranscriber,
    export_pdf: bool = True,
    export_html: bool = True,
    branding: Optional[ClinicBranding] = None
) -> BatchItemResult:
    """Processes a single audio file through the full ASR and documentation pipeline."""
    start_time = time.time()

    is_valid, err_msg, _ = validate_wav_file(audio_file)
    if not is_valid:
        return BatchItemResult(
            audio_path=audio_file,
            success=False,
            error_message=f"Invalid WAV file: {err_msg}",
            processing_time_seconds=time.time() - start_time
        )

    try:
        raw_transcript, raw_segments = transcriber.transcribe_file(audio_file)
        annotated_segments = process_transcript_segments(raw_segments)
        clinical_info = process_clinical_information(raw_transcript)
        demographics = extract_demographics(raw_transcript)

        demo_str = ""
        if demographics.age:
            demo_str += f"{demographics.age}yo "
        if demographics.gender:
            demo_str += f"{demographics.gender}"

        soap_note = generate_soap_note(clinical_info, demographics_summary=demo_str.strip())

        record = ConsultationRecord(
            demographics=demographics,
            raw_transcript=raw_transcript,
            segments=annotated_segments,
            extracted_info=clinical_info,
            soap_note=soap_note,
            review_status=ReviewStatus.DRAFT_PENDING_REVIEW
        )

        # Export JSON
        json_path = record.save_json(output_dir=output_dir)

        # Export HTML
        html_path = None
        if export_html:
            html_out = output_dir / f"consultation_{record.consultation_id[:8]}.html"
            html_path = export_html_report(record, html_out, branding=branding)

        # Export PDF
        pdf_path = None
        if export_pdf:
            pdf_out = output_dir / f"consultation_{record.consultation_id[:8]}.pdf"
            pdf_path = export_pdf_report(record, pdf_out, branding=branding)

        elapsed = time.time() - start_time
        return BatchItemResult(
            audio_path=audio_file,
            success=True,
            record=record,
            json_path=json_path,
            pdf_path=pdf_path,
            html_path=html_path,
            processing_time_seconds=elapsed
        )

    except Exception as e:
        logger.exception("Failed processing audio file: %s", audio_file.name)
        elapsed = time.time() - start_time
        return BatchItemResult(
            audio_path=audio_file,
            success=False,
            error_message=str(e),
            processing_time_seconds=elapsed
        )


def process_audio_folder(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    transcriber: Optional[WhisperTranscriber] = None,
    model_size: str = "tiny.en",
    export_pdf: bool = True,
    export_html: bool = True,
    branding: Optional[ClinicBranding] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> BatchSummary:
    """Processes all WAV files within a folder in sequence."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(in_path.glob("*.wav")))
    if not wav_files:
        return BatchSummary(total_files=0)

    if transcriber is None or transcriber.model_size != model_size:
        transcriber = WhisperTranscriber(model_size=model_size, device="cpu", compute_type="int8")

    batch_start = time.time()
    summary = BatchSummary(total_files=len(wav_files))

    for idx, wav_file in enumerate(wav_files, start=1):
        if progress_callback:
            progress_callback(idx, len(wav_files), wav_file.name)

        result = process_audio_file(
            audio_file=wav_file,
            output_dir=out_path,
            transcriber=transcriber,
            export_pdf=export_pdf,
            export_html=export_html,
            branding=branding
        )

        summary.results.append(result)
        if result.success:
            summary.successful_files += 1
        else:
            summary.failed_files += 1

    summary.total_time_seconds = time.time() - batch_start

    # Save summary report to output directory
    summary_file = out_path / "batch_summary.json"
    summary_file.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    return summary
