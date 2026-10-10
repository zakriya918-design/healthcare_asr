"""Unit tests for the record search and indexing module."""

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest

from src.models.patient_record import (
    ConsultationRecord,
    PatientDemographics,
    ExtractedFinding,
    MedicationFinding,
    FindingStatus,
    ReviewStatus,
)
from src.processing.record_search import (
    RecordSummary,
    index_consultation_records,
    search_records,
)


def _create_sample_record(
    patient_id="PATIENT-101",
    age=34,
    gender="Female",
    symptoms=("migraine",),
    medications=("ibuprofen",),
    diagnoses=("tension headache",),
    review_status=ReviewStatus.DRAFT_PENDING_REVIEW,
    days_ago=0,
) -> ConsultationRecord:
    """Helper to construct a typed ConsultationRecord for testing."""
    timestamp = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    record = ConsultationRecord(
        created_at=timestamp,
        demographics=PatientDemographics(patient_id=patient_id, age=age, gender=gender),
        review_status=review_status,
        raw_transcript=f"Patient reports {', '.join(symptoms)} and takes {', '.join(medications)}.",
    )
    
    record.extracted_info.symptoms = [
        ExtractedFinding(category="symptom", name=s, status=FindingStatus.PRESENT) for s in symptoms
    ]
    record.extracted_info.medications = [
        MedicationFinding(name=m, dosage="200mg", frequency="daily") for m in medications
    ]
    record.extracted_info.diagnoses = [
        ExtractedFinding(category="diagnosis", name=d, status=FindingStatus.PRESENT) for d in diagnoses
    ]
    return record


def test_index_empty_or_nonexistent_directory(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist"
    summaries = index_consultation_records(nonexistent)
    assert summaries == []

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    summaries = index_consultation_records(empty_dir)
    assert summaries == []


def test_index_valid_records_and_ignore_batch_summary(tmp_path: Path):
    rec1 = _create_sample_record(patient_id="PT-JOHN", symptoms=["fever"], diagnoses=["influenza"])
    rec1.save_json(tmp_path)

    rec2 = _create_sample_record(
        patient_id="PT-JANE",
        symptoms=["cough"],
        medications=["amoxicillin"],
        review_status=ReviewStatus.REVIEWED_APPROVED,
        days_ago=2,
    )
    rec2.save_json(tmp_path)

    # Add a batch_summary.json file which must be ignored
    (tmp_path / "batch_summary.json").write_text(json.dumps({"total": 2}), encoding="utf-8")

    # Add a corrupt JSON file
    (tmp_path / "corrupt.json").write_text("{broken json", encoding="utf-8")

    summaries = index_consultation_records(tmp_path)
    assert len(summaries) == 2
    ids = [s.patient_id for s in summaries]
    assert "PT-JOHN" in ids
    assert "PT-JANE" in ids


def test_search_by_query_matching_name_diagnosis_symptom(tmp_path: Path):
    rec1 = _create_sample_record(patient_id="ROBERT-FOX", diagnoses=["hypertension"])
    rec2 = _create_sample_record(patient_id="MARIA-GARCIA", medications=["metformin"], symptoms=["fatigue"])
    rec3 = _create_sample_record(patient_id="DAVID-CLARK", diagnoses=["type 2 diabetes"])

    s1 = RecordSummary.from_record(rec1, "/path/1.json")
    s2 = RecordSummary.from_record(rec2, "/path/2.json")
    s3 = RecordSummary.from_record(rec3, "/path/3.json")
    all_summaries = [s1, s2, s3]

    # Search by patient ID
    res = search_records(all_summaries, query="robert")
    assert len(res) == 1
    assert res[0].patient_id == "ROBERT-FOX"

    # Search by medication name in query
    res = search_records(all_summaries, query="metformin")
    assert len(res) == 1
    assert res[0].patient_id == "MARIA-GARCIA"

    # Search by diagnosis in query
    res = search_records(all_summaries, query="diabetes")
    assert len(res) == 1
    assert res[0].patient_id == "DAVID-CLARK"


def test_search_with_status_filter():
    rec1 = _create_sample_record(patient_id="P1", review_status=ReviewStatus.DRAFT_PENDING_REVIEW)
    rec2 = _create_sample_record(patient_id="P2", review_status=ReviewStatus.REVIEWED_APPROVED)

    s1 = RecordSummary.from_record(rec1, "/1.json")
    s2 = RecordSummary.from_record(rec2, "/2.json")
    summaries = [s1, s2]

    approved = search_records(summaries, status_filter="approved")
    assert len(approved) == 1
    assert approved[0].patient_id == "P2"

    drafts = search_records(summaries, status_filter="draft")
    assert len(drafts) == 1
    assert drafts[0].patient_id == "P1"

    all_res = search_records(summaries, status_filter="all")
    assert len(all_res) == 2


def test_search_with_diagnosis_and_medication_filters():
    rec1 = _create_sample_record(diagnoses=["asthma"], medications=["albuterol"])
    rec2 = _create_sample_record(diagnoses=["pneumonia"], medications=["azithromycin"])
    rec3 = _create_sample_record(diagnoses=["asthma"], medications=["fluticasone"])

    s1 = RecordSummary.from_record(rec1, "/1.json")
    s2 = RecordSummary.from_record(rec2, "/2.json")
    s3 = RecordSummary.from_record(rec3, "/3.json")
    summaries = [s1, s2, s3]

    # Filter diagnosis
    asthma_records = search_records(summaries, diagnosis_filter="asthma")
    assert len(asthma_records) == 2

    # Filter diagnosis AND medication
    combo = search_records(summaries, diagnosis_filter="asthma", medication_filter="albuterol")
    assert len(combo) == 1
    assert combo[0].medications == ["albuterol"]


def test_search_with_date_range():
    today_rec = _create_sample_record(patient_id="Today Patient", days_ago=0)
    past_rec = _create_sample_record(patient_id="Past Patient", days_ago=10)

    s_today = RecordSummary.from_record(today_rec, "/today.json")
    s_past = RecordSummary.from_record(past_rec, "/past.json")
    summaries = [s_today, s_past]

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    recent = search_records(summaries, date_from=today_str)
    assert len(recent) == 1
    assert recent[0].patient_id == "Today Patient"
