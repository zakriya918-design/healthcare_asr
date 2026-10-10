"""Rule-based speaker role estimation for clinical consultation transcripts.

DISCLAIMER:
This module implements transparent, rule-based heuristic role estimation (Doctor/Patient/Unknown).
It is NOT acoustic speaker diarization. It relies on conversational lexical cues,
question structures, clinician address patterns, and two-pass turn-taking context.
All outputs are draft suggestions subject to clinician review.
"""

from __future__ import annotations
import re
from typing import List, Optional, Tuple
from src.models.patient_record import SpeakerRole, TranscriptSegment


CLINICIAN_PATTERNS = [
    r"\bhow\s+(?:are\s+you|long\s+have\s+you|can\s+i\s+help|have\s+you\s+been)\b",
    r"\bwhat\s+brings\s+you\s+(?:in|here|today|back)\b",
    r"\b(?:let\s+me|i'm\s+going\s+to|let's)\s+(?:examine|listen|check|take\s+your|look\s+at|talk\s+about)\b",
    r"\b(?:i'll|let's|we'll|i\s+am\s+going\s+to|i'd\s+like\s+you\s+to)\s+(?:prescribe|order|recommend|start\s+you\s+on|try\s+using|stick\s+with)\b",
    r"\b(?:take\s+(?:this|one|two)|take\s+\d+\s*mg|twice\s+daily|three\s+times\s+a\s+day)\b",
    r"\b(?:any\s+fever|any\s+allergies|any\s+other\s+symptoms|any\s+other\s+costs|do\s+you\s+have|did\s+you\s+have)\b",
    r"\b(?:follow\s+up|return\s+to\s+clinic|come\s+back\s+if|see\s+me\s+in|set\s+a\s+time|early\s+next\s+week)\b",
    r"\b(?:your\s+blood\s+pressure\s+is|your\s+temperature\s+is|your\s+heart\s+rate\s+is|lungs\s+sound)\b",
    r"\b(?:it\s+sounds\s+like|my\s+impression\s+is|could\s+be|likely\s+a|rule\s+out|our\s+goal\s+is)\b",
    r"\b(?:have\s+you\s+been\s+keeping\s+up|we\s+had\s+you\s+(?:on|doing)|pills\s+were\s+only\s+meant|physical\s+therapy\s+is)\b",
    r"\b(?:why\s+was\s+that|do\s+you\s+think|can\s+you\s+tell\s+me\s+more|would\s+that\s+be\s+something)\b",
    r"\b(?:build\s+up\s+the\s+core|less\s+invasive\s+than|alleviate\s+the\s+need|cheapest\s+cost\s+of)\b",
    r"\b(?:alternative\s+therapies|yoga\s+or\s+tai\s+chi|generic\s+medication|non-prescription)\b",
    r"\b(?:i\s+would\s+encourage\s+you|during\s+open\s+season|insurance\s+options|cost\s+effective)\b",
    r"\b(?:what\s+i'll\s+do\s+is|my\s+office\s+team|talk\s+over\s+the\s+phone|see\s+if\s+those\s+would\s+work)\b",
    r"\b(?:hi|hello|welcome)\s*,\s*(?:mr\.|mrs\.|ms\.|miss)\s+\w+\b",
    r"\b(?:seen\s+you\s+a\s+number\s+of\s+times|thanks\s+to\s+see\s+you\s+again|glad\s+you\s+were\s+able\s+to\s+come)\b",
    r"\b(?:got\s+it\s*,\s*and\s+i\s+understand|i\s+understand|okay\s*,\s*well\s+there\s+are)\b",
]

PATIENT_PATTERNS = [
    r"\b(?:dr\.|doctor)\s+[a-z]+\b",
    r"\b(?:i\s+have\s+been\s+having|i've\s+had|i\s+have|i\s+feel|i'm\s+feeling|i\s+got)\b",
    r"\b(?:my\s+throat|my\s+head|my\s+chest|my\s+stomach|my\s+back|my\s+knee|my\s+ear)\s+(?:hurts|aches|is\s+sore|'s\s+been\s+hurting|is\s+hurting)\b",
    r"\b(?:it\s+hurts\s+when|it\s+started|since\s+yesterday|for\s+about\s+\d+|for\s+the\s+past\s+\d+)\b",
    r"\b(?:keeps\s+coming\s+back|got\s+hurt\s+on\s+the\s+job|bend\s+down|move\s+in\s+a\s+weird\s+way|just\s+go\s+out\s+again)\b",
    r"\b(?:i\s+noticed|i\s+can't\s+sleep|i\s+threw\s+up|i\s+don't\s+have\s+any)\b",
    r"\b(?:i\s+took\s+some|i\s+usually\s+take|i'm\s+allergic\s+to|no\s+known\s+allergies)\b",
    r"\b(?:no\s+fever|no\s+chest\s+pain|i\s+haven't\s+had|no\s+cough|swallowing\s+hurts)\b",
    r"\b(?:not\s+to\s+go\s+to\s+the\s+therapist|kind\s+of\s+a\s+hassle|with\s+my\s+work\s+schedule|cost\s+was\s+an\s+issue)\b",
    r"\b(?:get\s+back\s+to\s+work|could\s+use\s+the\s+money|slow\s+going\s+at\s+first)\b",
    r"\b(?:the\s+insurance\s+i\s+had|only\s+covered|moved\s+my\s+therapy\s+office|really\s+far\s+away|deal\s+with\s+parking|had\s+to\s+do\s+with\s+parking)\b",
    r"\b(?:started\s+to\s+get\s+really\s+expensive|don't\s+need\s+to\s+lose\s+any\s+more\s+time|lie\s+around\s+the\s+house)\b",
    r"\b(?:medicine\s+was\s+working\s+really\s+well|insurance\s+isn't\s+going\s+down|make\s+it\s+work\s+here)\b",
    r"\b(?:i'm\s+good|sure\s*,\s*that'd\s+be\s+great|that\s+sounds\s+(?:good|great))\b",
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

    # Addressing clinician by name or title indicates patient voice
    if re.search(r"\b(?:dr\.|doctor)\s+[a-z]+\b", cleaned, re.IGNORECASE) and doc_matches == 0:
        pt_matches += 2

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
    Annotates each segment with estimated speaker role and confidence using two-pass
    contextual turn-taking and conversational continuation smoothing.
    """
    if not segments:
        return []

    processed: List[TranscriptSegment] = []

    # Pass 1: Standalone rule-based estimation for anchor detection
    for seg in segments:
        role, conf, reason = estimate_speaker_role(seg.text)
        seg.speaker_role = role
        seg.role_confidence = conf
        seg.speaker_reason = reason
        processed.append(seg)

    # Pass 2: Contextual Turn Continuation & Alternation Smoothing
    for i in range(len(processed)):
        seg = processed[i]
        if seg.speaker_role != SpeakerRole.UNKNOWN:
            continue

        cleaned = seg.text.strip()
        prev_role = processed[i - 1].speaker_role if i > 0 else SpeakerRole.UNKNOWN
        prev_text = processed[i - 1].text.strip() if i > 0 else ""

        # Clinician short acknowledgment
        if re.match(r"^(?:i\s+see|got\s+it|good|okay|great|right)[.?!]?$", cleaned, re.IGNORECASE):
            if prev_role == SpeakerRole.PATIENT:
                seg.speaker_role = SpeakerRole.DOCTOR
                seg.role_confidence = 0.65
                seg.speaker_reason = "Clinician acknowledgment of patient utterance"
                continue
            elif prev_role == SpeakerRole.DOCTOR:
                seg.speaker_role = SpeakerRole.DOCTOR
                seg.role_confidence = 0.55
                seg.speaker_reason = "Clinician conversational wrap-up"
                continue

        # Patient short agreement following clinician statement
        if re.match(r"^(?:yeah|yes|sure|okay|ok|that\s+sounds\s+good|that\s+sounds\s+great|sounds\s+good)[.?!]?$", cleaned, re.IGNORECASE):
            if prev_role == SpeakerRole.DOCTOR:
                seg.speaker_role = SpeakerRole.PATIENT
                seg.role_confidence = 0.65
                seg.speaker_reason = "Patient agreement to clinician statement"
                continue

        # Continuation conjunctions (And, So, But, Because, It'll, Then, Or)
        if re.match(r"^(?:and|so|but|because|it\'ll|then|or|which|to\s+be|for\s+your|fully)\b", cleaned, re.IGNORECASE):
            if prev_role != SpeakerRole.UNKNOWN:
                seg.speaker_role = prev_role
                seg.role_confidence = 0.60
                seg.speaker_reason = f"Conversational continuation of {prev_role.value.lower()}"
                continue

        # Question response: if previous line was a question by Doctor, current line is Patient
        if prev_text.endswith("?") and prev_role == SpeakerRole.DOCTOR:
            seg.speaker_role = SpeakerRole.PATIENT
            seg.role_confidence = 0.60
            seg.speaker_reason = "Response to clinician inquiry"
            continue

        # In multi-sentence flow, default to previous active speaker if uninterrupted
        if prev_role != SpeakerRole.UNKNOWN:
            seg.speaker_role = prev_role
            seg.role_confidence = 0.50
            seg.speaker_reason = f"Dialogue flow continuation ({prev_role.value.lower()})"

    return processed
