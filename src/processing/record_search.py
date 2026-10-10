"""Record search and indexing utility for past consultation records."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.models.patient_record import ConsultationRecord, ReviewStatus, FindingStatus


@dataclass
class RecordSummary:
    """Lightweight summary of a consultation record for indexing and display."""
    file_path: str
    consultation_id: str
    created_at: str
    patient_id: str
    patient_age: Optional[int]
    patient_gender: Optional[str]
    review_status: str
    diagnoses: List[str]
    medications: List[str]
    symptoms: List[str]
    raw_snippet: str

    @classmethod
    def from_record(cls, record: ConsultationRecord, file_path: str) -> "RecordSummary":
        diagnoses = [d.name for d in record.extracted_info.diagnoses if d.name]
        medications = [m.name for m in record.extracted_info.medications if m.name]
        symptoms = [
            s.name for s in record.extracted_info.symptoms
            if s.name and s.status in (FindingStatus.PRESENT, "present")
        ]
        
        status_val = (
            record.review_status.value
            if isinstance(record.review_status, ReviewStatus)
            else str(record.review_status)
        )

        raw_snippet = (record.raw_transcript[:120] + "...") if len(record.raw_transcript) > 120 else record.raw_transcript

        return cls(
            file_path=file_path,
            consultation_id=record.consultation_id,
            created_at=record.created_at,
            patient_id=record.demographics.patient_id or "Unspecified Patient",
            patient_age=record.demographics.age,
            patient_gender=record.demographics.gender,
            review_status=status_val,
            diagnoses=diagnoses,
            medications=medications,
            symptoms=symptoms,
            raw_snippet=raw_snippet or "No transcript available",
        )


def index_consultation_records(records_dir: Path) -> List[RecordSummary]:
    """Scan directory for JSON consultation records and return summaries.
    
    Args:
        records_dir: Path to directory containing saved consultation JSON files.
        
    Returns:
        List of RecordSummary instances sorted descending by creation timestamp.
    """
    summaries: List[RecordSummary] = []
    if not records_dir.exists() or not records_dir.is_dir():
        return summaries

    for json_file in records_dir.glob("*.json"):
        if json_file.name == "batch_summary.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Verify basic schema
            if "schema_version" in data and "consultation_id" in data:
                record = ConsultationRecord.from_dict(data)
                summaries.append(RecordSummary.from_record(record, str(json_file.resolve())))
        except Exception:
            # Skip invalid or corrupt JSON files gracefully
            continue

    # Sort newest first
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def search_records(
    summaries: List[RecordSummary],
    query: Optional[str] = None,
    status_filter: Optional[str] = None,
    diagnosis_filter: Optional[str] = None,
    medication_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[RecordSummary]:
    """Filter and search consultation records by query, status, clinical findings, and date.
    
    Args:
        summaries: Full list of RecordSummary instances to search through.
        query: Case-insensitive search string matching patient_id, consultation_id, diagnoses, meds, symptoms.
        status_filter: Exact or partial match for status (e.g. 'approved', 'draft').
        diagnosis_filter: Case-insensitive substring matching diagnoses.
        medication_filter: Case-insensitive substring matching medications.
        date_from: Optional start date string (YYYY-MM-DD) inclusive.
        date_to: Optional end date string (YYYY-MM-DD) inclusive.
        
    Returns:
        List of matching RecordSummary instances.
    """
    results: List[RecordSummary] = []

    q = query.strip().lower() if query else None
    stat = status_filter.strip().lower() if status_filter and status_filter.lower() != "all" else None
    diag = diagnosis_filter.strip().lower() if diagnosis_filter else None
    med = medication_filter.strip().lower() if medication_filter else None

    # Parse date boundaries if provided
    dt_from = None
    if date_from:
        try:
            dt_from = datetime.fromisoformat(date_from[:10])
        except ValueError:
            pass

    dt_to = None
    if date_to:
        try:
            # Set to end of day for inclusive comparison
            dt_to = datetime.fromisoformat(date_to[:10]).replace(hour=23, minute=59, second=59)
        except ValueError:
            pass

    for summary in summaries:
        # Status filter
        if stat:
            if stat not in summary.review_status.lower():
                continue

        # Diagnosis filter
        if diag:
            if not any(diag in d.lower() for d in summary.diagnoses):
                continue

        # Medication filter
        if med:
            if not any(med in m.lower() for m in summary.medications):
                continue

        # Date range filter
        if dt_from or dt_to:
            try:
                rec_dt = datetime.fromisoformat(summary.created_at.replace("Z", "+00:00")[:19])
                if dt_from and rec_dt < dt_from:
                    continue
                if dt_to and rec_dt > dt_to:
                    continue
            except Exception:
                pass

        # Query filter (matches patient id, consultation id, snippet, diagnoses, meds, symptoms)
        if q:
            searchable_fields = [
                summary.patient_id.lower(),
                summary.consultation_id.lower(),
                summary.raw_snippet.lower(),
                " ".join(summary.diagnoses).lower(),
                " ".join(summary.medications).lower(),
                " ".join(summary.symptoms).lower(),
            ]
            if not any(q in field for field in searchable_fields):
                continue

        results.append(summary)

    return results
