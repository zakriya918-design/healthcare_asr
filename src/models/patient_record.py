"""Typed data models and serialization for Healthcare AI Clinical Documentation Assistant."""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union
import uuid


class SpeakerRole(str, Enum):
    """Estimated role of an utterance speaker."""
    DOCTOR = "Doctor"
    PATIENT = "Patient"
    CAREGIVER = "Caregiver"
    NURSE = "Nurse"
    UNKNOWN = "Unknown"


class FindingStatus(str, Enum):
    """Clinical status of an extracted finding."""
    PRESENT = "present"
    EXPLICITLY_ABSENT = "explicitly_absent"
    UNCERTAIN = "uncertain"
    NOT_MENTIONED = "not_mentioned"
    UNABLE_TO_EXTRACT = "unable_to_extract"


class ReviewStatus(str, Enum):
    """Review state of clinical notes."""
    DRAFT_PENDING_REVIEW = "draft_pending_review"
    REVIEWED_APPROVED = "reviewed_approved"
    REJECTED = "rejected"


@dataclass
class TranscriptSegment:
    """A segment or sentence from the recognized transcript."""
    id: int
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN
    speaker_id: Optional[str] = None  # e.g. "Speaker 1", "Speaker 2", "Speaker 3"
    role_confidence: float = 0.0  # 0.0 to 1.0 scale
    speaker_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker_role": self.speaker_role.value if isinstance(self.speaker_role, SpeakerRole) else str(self.speaker_role),
            "speaker_id": self.speaker_id or (self.speaker_role.value if self.speaker_role != SpeakerRole.UNKNOWN else "Speaker"),
            "role_confidence": self.role_confidence,
            "speaker_reason": self.speaker_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TranscriptSegment:
        role = data.get("speaker_role", SpeakerRole.UNKNOWN.value)
        try:
            speaker_role = SpeakerRole(role)
        except (ValueError, TypeError):
            speaker_role = SpeakerRole.UNKNOWN

        return cls(
            id=int(data.get("id", 0)),
            text=str(data.get("text", "")),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            speaker_role=speaker_role,
            speaker_id=data.get("speaker_id"),
            role_confidence=float(data.get("role_confidence", 0.0)),
            speaker_reason=str(data.get("speaker_reason", "")),
        )


@dataclass
class PatientDemographics:
    """Explicitly stated patient information."""
    patient_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None  # e.g. "Male", "Female", or None if not stated
    consultation_datetime: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "age": self.age,
            "gender": self.gender,
            "consultation_datetime": self.consultation_datetime,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PatientDemographics:
        return cls(
            patient_id=data.get("patient_id"),
            age=data.get("age"),
            gender=data.get("gender"),
            consultation_datetime=str(data.get("consultation_datetime", "")),
        )


@dataclass
class ExtractedFinding:
    """Structured clinical finding with explicit status and provenance."""
    category: str
    name: str
    status: FindingStatus = FindingStatus.PRESENT
    duration: Optional[str] = None
    details: Optional[str] = None
    source_phrase: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "status": self.status.value if isinstance(self.status, FindingStatus) else str(self.status),
            "duration": self.duration,
            "details": self.details,
            "source_phrase": self.source_phrase,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedFinding:
        status_val = data.get("status", FindingStatus.PRESENT.value)
        try:
            status = FindingStatus(status_val)
        except (ValueError, TypeError):
            status = FindingStatus.PRESENT

        return cls(
            category=str(data.get("category", "")),
            name=str(data.get("name", "")),
            status=status,
            duration=data.get("duration"),
            details=data.get("details"),
            source_phrase=data.get("source_phrase"),
        )


@dataclass
class MedicationFinding:
    """Extracted medication with dosage, frequency, and current vs past intent."""
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    status: FindingStatus = FindingStatus.PRESENT
    is_current_plan: bool = True
    source_phrase: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "status": self.status.value if isinstance(self.status, FindingStatus) else str(self.status),
            "is_current_plan": self.is_current_plan,
            "source_phrase": self.source_phrase,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MedicationFinding:
        status_val = data.get("status", FindingStatus.PRESENT.value)
        try:
            status = FindingStatus(status_val)
        except (ValueError, TypeError):
            status = FindingStatus.PRESENT

        return cls(
            name=str(data.get("name", "")),
            dosage=data.get("dosage"),
            frequency=data.get("frequency"),
            status=status,
            is_current_plan=bool(data.get("is_current_plan", True)),
            source_phrase=data.get("source_phrase"),
        )


@dataclass
class VitalSigns:
    """Explicitly measured vital signs. Missing values MUST remain None."""
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    blood_pressure_str: Optional[str] = None
    heart_rate_bpm: Optional[int] = None
    temperature_f: Optional[float] = None
    temperature_c: Optional[float] = None
    temperature_str: Optional[str] = None
    spo2_percent: Optional[int] = None
    blood_glucose_mg_dl: Optional[int] = None
    blood_glucose_str: Optional[str] = None
    source_phrase: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blood_pressure_systolic": self.blood_pressure_systolic,
            "blood_pressure_diastolic": self.blood_pressure_diastolic,
            "blood_pressure_str": self.blood_pressure_str,
            "heart_rate_bpm": self.heart_rate_bpm,
            "temperature_f": self.temperature_f,
            "temperature_c": self.temperature_c,
            "temperature_str": self.temperature_str,
            "spo2_percent": self.spo2_percent,
            "blood_glucose_mg_dl": self.blood_glucose_mg_dl,
            "blood_glucose_str": self.blood_glucose_str,
            "source_phrase": self.source_phrase,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VitalSigns:
        return cls(
            blood_pressure_systolic=data.get("blood_pressure_systolic"),
            blood_pressure_diastolic=data.get("blood_pressure_diastolic"),
            blood_pressure_str=data.get("blood_pressure_str"),
            heart_rate_bpm=data.get("heart_rate_bpm"),
            temperature_f=data.get("temperature_f"),
            temperature_c=data.get("temperature_c"),
            temperature_str=data.get("temperature_str"),
            spo2_percent=data.get("spo2_percent"),
            blood_glucose_mg_dl=data.get("blood_glucose_mg_dl"),
            blood_glucose_str=data.get("blood_glucose_str"),
            source_phrase=data.get("source_phrase"),
        )


@dataclass
class ExtractedClinicalInformation:
    """Categories of structured information extracted from the consultation."""
    symptoms: List[ExtractedFinding] = field(default_factory=list)
    relevant_history: List[ExtractedFinding] = field(default_factory=list)
    existing_conditions: List[ExtractedFinding] = field(default_factory=list)
    allergies: List[ExtractedFinding] = field(default_factory=list)
    medications: List[MedicationFinding] = field(default_factory=list)
    vital_signs: VitalSigns = field(default_factory=VitalSigns)
    diagnoses: List[ExtractedFinding] = field(default_factory=list)
    investigations: List[ExtractedFinding] = field(default_factory=list)
    treatment_plans: List[ExtractedFinding] = field(default_factory=list)
    advice_followup: List[ExtractedFinding] = field(default_factory=list)
    unclassified_statements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symptoms": [item.to_dict() for item in self.symptoms],
            "relevant_history": [item.to_dict() for item in self.relevant_history],
            "existing_conditions": [item.to_dict() for item in self.existing_conditions],
            "allergies": [item.to_dict() for item in self.allergies],
            "medications": [item.to_dict() for item in self.medications],
            "vital_signs": self.vital_signs.to_dict(),
            "diagnoses": [item.to_dict() for item in self.diagnoses],
            "investigations": [item.to_dict() for item in self.investigations],
            "treatment_plans": [item.to_dict() for item in self.treatment_plans],
            "advice_followup": [item.to_dict() for item in self.advice_followup],
            "unclassified_statements": list(self.unclassified_statements),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedClinicalInformation:
        return cls(
            symptoms=[ExtractedFinding.from_dict(x) for x in data.get("symptoms", [])],
            relevant_history=[ExtractedFinding.from_dict(x) for x in data.get("relevant_history", [])],
            existing_conditions=[ExtractedFinding.from_dict(x) for x in data.get("existing_conditions", [])],
            allergies=[ExtractedFinding.from_dict(x) for x in data.get("allergies", [])],
            medications=[MedicationFinding.from_dict(x) for x in data.get("medications", [])],
            vital_signs=VitalSigns.from_dict(data.get("vital_signs", {})),
            diagnoses=[ExtractedFinding.from_dict(x) for x in data.get("diagnoses", [])],
            investigations=[ExtractedFinding.from_dict(x) for x in data.get("investigations", [])],
            treatment_plans=[ExtractedFinding.from_dict(x) for x in data.get("treatment_plans", [])],
            advice_followup=[ExtractedFinding.from_dict(x) for x in data.get("advice_followup", [])],
            unclassified_statements=list(data.get("unclassified_statements", [])),
        )


@dataclass
class SOAPNote:
    """SOAP clinical note generated from consultation findings."""
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    missing_or_uncertain_info: List[str] = field(default_factory=list)
    review_warning: str = (
        "CLINICAL SAFETY NOTICE: This documentation is AI-generated draft text for "
        "licensed clinician review and verification only. It does not constitute "
        "independent medical advice, diagnosis, or prescription."
    )
    review_status: ReviewStatus = ReviewStatus.DRAFT_PENDING_REVIEW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subjective": self.subjective,
            "objective": self.objective,
            "assessment": self.assessment,
            "plan": self.plan,
            "missing_or_uncertain_info": list(self.missing_or_uncertain_info),
            "review_warning": self.review_warning,
            "review_status": self.review_status.value if isinstance(self.review_status, ReviewStatus) else str(self.review_status),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SOAPNote:
        rev_val = data.get("review_status", ReviewStatus.DRAFT_PENDING_REVIEW.value)
        try:
            review_status = ReviewStatus(rev_val)
        except (ValueError, TypeError):
            review_status = ReviewStatus.DRAFT_PENDING_REVIEW

        return cls(
            subjective=str(data.get("subjective", "")),
            objective=str(data.get("objective", "")),
            assessment=str(data.get("assessment", "")),
            plan=str(data.get("plan", "")),
            missing_or_uncertain_info=list(data.get("missing_or_uncertain_info", [])),
            review_warning=str(data.get("review_warning", "")),
            review_status=review_status,
        )


@dataclass
class ConsultationRecord:
    """Master record containing transcript, extracted clinical data, and SOAP note."""
    consultation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    demographics: PatientDemographics = field(default_factory=PatientDemographics)
    raw_transcript: str = ""
    segments: List[TranscriptSegment] = field(default_factory=list)
    extracted_info: ExtractedClinicalInformation = field(default_factory=ExtractedClinicalInformation)
    soap_note: SOAPNote = field(default_factory=SOAPNote)
    review_status: ReviewStatus = ReviewStatus.DRAFT_PENDING_REVIEW
    signed_by: Optional[str] = None
    signed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "consultation_id": self.consultation_id,
            "created_at": self.created_at,
            "demographics": self.demographics.to_dict(),
            "raw_transcript": self.raw_transcript,
            "segments": [seg.to_dict() for seg in self.segments],
            "extracted_info": self.extracted_info.to_dict(),
            "soap_note": self.soap_note.to_dict(),
            "review_status": self.review_status.value if isinstance(self.review_status, ReviewStatus) else str(self.review_status),
            "signed_by": self.signed_by,
            "signed_at": self.signed_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConsultationRecord:
        rev_val = data.get("review_status", ReviewStatus.DRAFT_PENDING_REVIEW.value)
        try:
            review_status = ReviewStatus(rev_val)
        except (ValueError, TypeError):
            review_status = ReviewStatus.DRAFT_PENDING_REVIEW

        return cls(
            consultation_id=str(data.get("consultation_id", uuid.uuid4())),
            created_at=str(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            demographics=PatientDemographics.from_dict(data.get("demographics", {})),
            raw_transcript=str(data.get("raw_transcript", "")),
            segments=[TranscriptSegment.from_dict(s) for s in data.get("segments", [])],
            extracted_info=ExtractedClinicalInformation.from_dict(data.get("extracted_info", {})),
            soap_note=SOAPNote.from_dict(data.get("soap_note", {})),
            review_status=review_status,
            signed_by=data.get("signed_by"),
            signed_at=data.get("signed_at"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ConsultationRecord:
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_json(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Atomically saves consultation record to JSON without overwriting existing files."""
        if output_dir is None:
            from src.utils.paths import get_output_dir
            output_path = get_output_dir()
        else:
            output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        # Unique filename using timestamp and shortened uuid
        safe_time = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_id = self.consultation_id[:8]
        filename = f"consultation_{safe_time}_{short_id}.json"
        target_file = output_path / filename

        # Ensure no accidental overwrite
        counter = 1
        while target_file.exists():
            target_file = output_path / f"consultation_{safe_time}_{short_id}_{counter}.json"
            counter += 1

        json_text = self.to_json(indent=2)

        # Atomic write via temporary file
        fd, temp_file_path = tempfile.mkstemp(prefix="temp_consultation_", suffix=".json", dir=str(output_path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json_text)
            # Atomic rename / replace
            os.replace(temp_file_path, str(target_file))
        except Exception:
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
            raise

        return target_file

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> ConsultationRecord:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Consultation JSON file not found: {path}")
        content = path.read_text(encoding="utf-8")
        return cls.from_json(content)
