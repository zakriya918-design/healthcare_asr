"""Rule-based speaker role estimation for clinical consultation transcripts.

DISCLAIMER:
This module implements transparent, rule-based heuristic role estimation (Doctor/Patient/Unknown).
It is NOT acoustic speaker diarization. It relies solely on conversational lexical cues,
question structures, and domain phrasing. It does not measure voice characteristics
or audio turns, and must be treated as an assistive heuristic subject to clinician review.
"""

from __future__ import annotations
import re
from typing import List, Optional, Tuple
from src.models.patient_record import SpeakerRole, TranscriptSegment


CLINICIAN_PATTERNS = [
    r"\bhow\s+(?:are\s+you|long\s+have\s+you|can\s+i\s+help|have\s+you\s+been)\b",
    r"\bwhat\s+brings\s+you\s+(?:in|here|today)\b",
    r"\b(?:let\s+me|i'm\s+going\s+to|let's)\s+(?:examine|listen|check|take\s+your|look\s+at)\b",
    r"\b(?:i'll|let's|we'll|i\s+am\s+going\s+to)\s+(?:prescribe|order|recommend|start\s+you\s+on)\b",
    r"\b(?:take\s+(?:this|one|two)|take\s+\d+\s*mg|twice\s+daily|three\s+times\s+a\s+day)\b",
    r"\b(?:any\s+fever|any\s+allergies|any\s+other\s+symptoms|do\s+you\s+have\s+any\s+allergies)\b",
    r"\b(?:follow\s+up|return\s+to\s+clinic|come\s+back\s+if|see\s+me\s+in)\b",
    r"\b(?:your\s+blood\s+pressure\s+is|your\s+temperature\s+is|your\s+heart\s+rate\s+is|lungs\s+sound)\b",
    r"\b(?:it\s+sounds\s+like|my\s+impression\s+is|could\s+be|likely\s+a|rule\s+out)\b",
]

PATIENT_PATTERNS = [
    r"\b(?:i\s+have\s+been\s+having|i've\s+had|i\s+have|i\s+feel|i'm\s+feeling|i\s+got)\b",
    r"\b(?:my\s+throat|my\s+head|my\s+chest|my\s+stomach|my\s+back|my\s+knee|my\s+ear)\s+(?:hurts|aches|is\s+sore)\b",
    r"\b(?:it\s+hurts\s+when|it\s+started|since\s+yesterday|for\s+about\s+\d+|for\s+the\s+past\s+\d+)\b",
    r"\b(?:i\s+noticed|i\s+can't\s+sleep|i\s+threw\s+up|i\s+don't\s+have\s+any)\b",
    r"\b(?:i\s+took\s+some|i\s+usually\s+take|i'm\s+allergic\s+to|no\s+known\s+allergies)\b",
    r"\b(?:no\s+fever|no\s+chest\s+pain|i\s+haven't\s+had|no\s+cough)\b",
    r"\b(?:my\s+wife|my\s+husband|my\s+daughter|my\s+son|at\s+work)\b",
]

PREFIX_DOCTOR_RE = re.compile(r"^(?:doctor|dr\.?|clinician|physician|provider):?\s*", re.IGNORECASE)
PREFIX_PATIENT_RE = re.compile(r"^(?:patient|pt):?\s*", re.IGNORECASE)


def estimate_speaker_role(
    text: str,
    previous_role: Optional[SpeakerRole] = None
) -> Tuple[SpeakerRole, float, str]:
    """
    Estimates speaker role for a given text snippet.
    Returns:
        (SpeakerRole, confidence [0.0 - 1.0], rationale_string)
    """
    cleaned = text.strip()
    if not cleaned:
        return SpeakerRole.UNKNOWN, 0.0, "Empty utterance"

    if PREFIX_DOCTOR_RE.match(cleaned):
        return SpeakerRole.DOCTOR, 0.95, "Explicit doctor label prefix"
    if PREFIX_PATIENT_RE.match(cleaned):
        return SpeakerRole.PATIENT, 0.95, "Explicit patient label prefix"

    doc_matches = 0
    for pat in CLINICIAN_PATTERNS:
        if re.search(pat, cleaned, re.IGNORECASE):
            doc_matches += 1

    pt_matches = 0
    for pat in PATIENT_PATTERNS:
        if re.search(pat, cleaned, re.IGNORECASE):
            pt_matches += 1

    if doc_matches > 0 and pt_matches == 0:
        conf = min(0.6 + (doc_matches * 0.15), 0.90)
        return SpeakerRole.DOCTOR, conf, f"Clinician phrasing cues matched ({doc_matches})"
    elif pt_matches > 0 and doc_matches == 0:
        conf = min(0.6 + (pt_matches * 0.15), 0.90)
        return SpeakerRole.PATIENT, conf, f"Patient symptom/history cues matched ({pt_matches})"
    elif doc_matches > 0 and pt_matches > 0:
        if doc_matches > pt_matches:
            return SpeakerRole.DOCTOR, 0.55, "Mixed phrasing cues, clinician predominant"
        elif pt_matches > doc_matches:
            return SpeakerRole.PATIENT, 0.55, "Mixed phrasing cues, patient predominant"
        else:
            return SpeakerRole.UNKNOWN, 0.30, "Conflicting clinician and patient cues"

    if re.match(r"^(?:yes|yeah|no|okay|ok|sure|i\s+see|thank\s+you|good\s+morning|hello|bye)[.?!]?$", cleaned, re.IGNORECASE):
        if previous_role == SpeakerRole.DOCTOR:
            return SpeakerRole.PATIENT, 0.40, "Conversational response to clinician"
        elif previous_role == SpeakerRole.PATIENT:
            return SpeakerRole.DOCTOR, 0.40, "Conversational response to patient"
        return SpeakerRole.UNKNOWN, 0.10, "Brief generic conversational token"

    return SpeakerRole.UNKNOWN, 0.0, "No definitive role patterns detected"


def process_transcript_segments(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """
    Annotates each segment with estimated speaker role and confidence.
    """
    last_role: Optional[SpeakerRole] = None
    processed: List[TranscriptSegment] = []

    for seg in segments:
        role, conf, reason = estimate_speaker_role(seg.text, previous_role=last_role)
        seg.speaker_role = role
        seg.role_confidence = conf
        seg.speaker_reason = reason
        if role != SpeakerRole.UNKNOWN:
            last_role = role
        processed.append(seg)

    return processed
