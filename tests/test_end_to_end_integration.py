"""End-to-End integration test covering the entire clinical pipeline with real Faster-Whisper ASR."""

import json
from pathlib import Path
import pytest

from src.audio.asr import WhisperTranscriber
from src.processing.speaker_processor import process_transcript_segments
from src.processing.medical_processor import process_clinical_information, extract_demographics
from src.processing.soap_note import generate_soap_note
from src.models.patient_record import (
    ConsultationRecord,
    FindingStatus,
    ReviewStatus,
    SpeakerRole
)


def test_full_pipeline_end_to_end(tmp_path):
    # 1. Synthetic Consultation Audio
    audio_file = Path(r"C:\Users\Aziza\HEALTHCARE_ASR\recordings\synthetic_consultation_test.wav")
    assert audio_file.is_file(), f"Synthetic audio fixture missing at {audio_file}"

    # 2. Actual Faster-Whisper transcription (tiny.en, CPU, int8)
    transcriber = WhisperTranscriber(model_size="tiny.en", device="cpu", compute_type="int8")
    raw_transcript, segments = transcriber.transcribe_file(audio_file)

    assert len(raw_transcript) > 20
    assert len(segments) >= 1
    assert "throat" in raw_transcript.lower()
    assert "fever" in raw_transcript.lower()

    # 3. Speaker-role estimation
    annotated_segments = process_transcript_segments(segments)
    assert len(annotated_segments) == len(segments)

    # 4. Medical information extraction
    extracted_info = process_clinical_information(raw_transcript)
    demographics = extract_demographics(raw_transcript)

    # Verify clinical extraction safeguards
    symptom_map = {s.name: s for s in extracted_info.symptoms}
    assert "sore throat" in symptom_map
    assert symptom_map["sore throat"].status == FindingStatus.PRESENT
    assert "chest pain" in symptom_map
    assert symptom_map["chest pain"].status == FindingStatus.EXPLICITLY_ABSENT

    # 5. SOAP note generation
    soap_note = generate_soap_note(extracted_info)
    assert soap_note.review_status == ReviewStatus.DRAFT_PENDING_REVIEW
    assert "CLINICAL SAFETY NOTICE" in soap_note.review_warning
    assert "sore throat" in soap_note.subjective
    assert "Denies chest pain" in soap_note.subjective

    # 6. Master Consultation Record assembly
    record = ConsultationRecord(
        demographics=demographics,
        raw_transcript=raw_transcript,
        segments=annotated_segments,
        extracted_info=extracted_info,
        soap_note=soap_note,
        review_status=ReviewStatus.DRAFT_PENDING_REVIEW
    )

    # 7. Atomic JSON export
    exported_file = record.save_json(output_dir=tmp_path)
    assert exported_file.is_file()
    assert exported_file.suffix == ".json"

    # 8. Reload and verify exported record
    reloaded_record = ConsultationRecord.load_json(exported_file)
    assert reloaded_record.consultation_id == record.consultation_id
    assert reloaded_record.raw_transcript == record.raw_transcript
    assert len(reloaded_record.segments) == len(record.segments)
    assert reloaded_record.soap_note.subjective == record.soap_note.subjective
    assert reloaded_record.review_status == ReviewStatus.DRAFT_PENDING_REVIEW
