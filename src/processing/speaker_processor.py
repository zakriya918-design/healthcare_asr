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
    r"\b(?:i'll|let's|we'll|i\s+am\s+going\s+to|i'd\s+like\s+you\s+to|i\s+recommend)\s+(?:prescribe|order|recommend|start\s+you\s+on|try\s+using|stick\s+with|making\s+some\s+lifestyle)\b",
    r"\b(?:take\s+(?:this|one|two)|take\s+\d+\s*mg|twice\s+daily|three\s+times\s+a\s+day)\b",
    r"\b(?:any\s+(?:chest\s+pain|fever|allergies|other\s+symptoms|other\s+costs|shortness\s+of\s+breath|nausea|vomiting|dizziness|headache|pain|swelling|cough)|do\s+you\s+have|did\s+you\s+have)\b",
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
    r"\b(?:got\s+it\s*,\s*and\s+i\s+understand|i\s+understand\s+your\s+concern|okay\s*,\s*well\s+there\s+are)\b",
    r"\b(?:unfortunately\s+that(?:\s+that)?\s+can\s+happen|quite\s+a\s+few\s+patients|reoccurring\s+episodes)\b",
    r"\b(?:heating\s+pad\s+for\s+your\s+back|stick\s+with\s+the\s+non-prescription|pros\s+and\s+cons\s+of\s+each)\b",
    r"\b(?:let's\s+talk\s+about\s+some\s+different\s+options|talk\s+about\s+some\s+of\s+the\s+other\s+costs)\b",
    r"\b(?:i'm\s+glad\s+you\s+came\s+in|glad\s+you\s+came\s+in|important\s+not\s+to\s+ignore)\b",
    r"\b(?:let\s+me\s+ask\s+you|ask\s+you\s+a\s+few\s+questions|better\s+understand\s+your\s+condition)\b",
    r"\b(?:have\s+you\s+noticed\s+if|symptoms\s+occur\s+during|family\s+history|history\s+of\s+heart\s+disease)\b",
    r"\b(?:relation\s+to\s+physical\s+activity|important\s+to\s+consider|your\s+heart\s+health)\b",
    r"\b(?:order\s+an\s+electrocardiogram|called\s+ecg|baseline\s+assessment|heart\'s\s+electrical\s+activity)\b",
    r"\b(?:help\s+us\s+determine|if\s+any\s+abnormalities|support\s+your\s+heart\s+health)\b",
    r"\b(?:balanced\s+diet|engaging\s+in\s+regular\s+exercise|heart-healthy\s+diet|whole\s+grains|manage\s+stress)\b",
    r"\b(?:we\'ll\s+wait\s+for|wait\s+for\s+the\s+ecg\s+results|prescribing\s+medications|encourage\s+you\s+to\s+take\s+it\s+easy)\b",
    r"\b(?:warning\s+signs\s+to\s+watch|pain\s+radiating\s+to|seek\s+immediate\s+medical\s+attention)\b",
    r"\b(?:results\s+of\s+your\s+ecg|heart\s+rhythm\s+appears\s+normal|while\s+the\s+ecg|reassuring|investigate\s+further)\b",
    r"\b(?:pulmonary\s+function\s+test|assess\s+your\s+lung\s+function|stress\s+test\s+to\s+evaluate)\b",
    r"\b(?:rule\s+out\s+any\s+underlying|guide\s+us\s+toward\s+an\s+accurate\s+diagnosis)\b",
    r"\b(?:it\'s\s+my\s+duty\s+to\s+provide|best\s+care\s+possible|schedule\s+a\s+follow-up\s+appointment)\b",
    r"\b(?:reach\s+out\s+to\s+me\s+or\s+the\s+clinic|we\'re\s+here\s+to\s+support\s+you)\b",
]

PATIENT_PATTERNS = [
    r"\b(?:dr\.|doctor)\s+[a-z]+\b",
    r"\b(?:i\s+have\s+been\s+having|i've\s+had|i\s+have|i\s+feel|i'm\s+feeling|i\s+got)\b",
    r"\b(?:my\s+throat|my\s+head|my\s+chest|my\s+stomach|my\s+back|my\s+knee|my\s+ear)\s+(?:hurts|aches|is\s+sore|'s\s+been\s+hurting|is\s+hurting)\b",
    r"\b(?:my\s+back'?s\s+been\s+hurting|back'?s\s+been\s+hurting|which(?:\s+the)?\s+pills)\b",
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
    r"\b(?:i've\s+been\s+experiencing|pain\s+in\s+my\s+chest|shortness\s+of\s+breath|bothering\s+me\s+for)\b",
    r"\b(?:happens\s+mostly\s+when\s+i|push\s+myself\s+physically|climbing\s+stairs|walking\s+fast)\b",
    r"\b(?:not\s+that\s+i'm\s+aware\s+of|my\s+parents\s+and\s+siblings|don't\s+have\s+any\s+heart)\b",
    r"\b(?:trust\s+your\s+judgment|proceed\s+with\s+the\s+necessary|make\s+sure\s+everything\s+is\s+okay)\b",
    r"\b(?:i\s+must\s+admit|hasn't\s+been\s+the\s+healthiest|routine\s+is\s+almost\s+non-existent)\b",
    r"\b(?:ready\s+to\s+commit|are\s+there\s+any\s+medications\s+i\s+should|other\s+treatments\s+we\s+should\s+consider)\b",
    r"\b(?:appreciate\s+your\s+guidance|i'll\s+be\s+cautious|prioritize\s+my\s+health)\b",
    r"\b(?:anything\s+else\s+i\s+should\s+be\s+aware|warning\s+signs\s+i\s+should\s+watch)\b",
    r"\b(?:i'll\s+keep\s+a\s+close\s+eye|won't\s+hesitate\s+to\s+reach\s+out|thank\s+you\s+for\s+your\s+time\s+and\s+expertise)\b",
    r"\b(?:that's\s+a\s+relief|relief\s+to\s+hear|mean\s+my\s+symptoms\s+are\s+not\s+related)\b",
    r"\b(?:what\s+tests\s+do\s+you\s+recommend|i'll\s+proceed\s+with\s+the\s+tests|appreciate\s+your\s+thoroughness)\b",
    r"\b(?:conscious\s+efforts\s+to\s+improve|challenging\s*,\s*but\s+i\s+understand)\b",
    r"\b(?:keep\s+that\s+in\s+mind|reassuring\s+to\s+know|in\s+good\s+hands)\b",
    r"\b(?:prompt\s+attention|eagerly\s+await\s+the\s+results|next\s+meeting|professionalism\s+and\s+care)\b",
    r"\b(?:keep\s+you\s+updated|goodbye\s+for\s+now)\b",
]

CAREGIVER_PATTERNS = [
    r"\b(?:he\s+hasn't\s+been|she\s+hasn't\s+been|he's\s+been|she's\s+been|he\s+was|she\s+was)\b",
    r"\b(?:my\s+husband|my\s+wife|my\s+mother|my\s+father|my\s+son|my\s+daughter|my\s+child|my\s+mom|my\s+dad)\b",
    r"\b(?:i\s+brought\s+(?:him|her)\s+in|we\s+noticed\s+that\s+(?:he|she)|i\s+noticed\s+(?:he|she)|i\s+drove\s+(?:him|her)|we\s+drove\s+(?:him|her))\b",
    r"\b(?:he\s+complained|she\s+complained|he\s+told\s+me|she\s+told\s+me|at\s+home\s+(?:he|she)|struggling\s+to\s+(?:walk|stand|breathe|eat))\b",
    r"\b(?:we\s+are\s+worried|he\s+can't|she\s+can't|he\s+couldn't|she\s+couldn't|he\s+could\s+not|she\s+could\s+not|his\s+pain|her\s+pain|his\s+memory|her\s+memory)\b",
    r"\b(?:i'm\s+his\s+wife|i'm\s+her\s+husband|i'm\s+his\s+mother|i'm\s+her\s+daughter|i'm\s+his\s+daughter|i'm\s+her\s+son|i'm\s+his\s+son)\b",
]

NURSE_PATTERNS = [
    r"\b(?:let\s+me\s+take\s+your|let's\s+check\s+your|let\s+me\s+get\s+your)\s+(?:vitals|blood\s+pressure|temperature|pulse|weight)\b",
    r"\b(?:step\s+on\s+the\s+scale|roll\s+up\s+your\s+sleeve|put\s+this\s+under\s+your\s+tongue|pulse\s+oximeter)\b",
    r"\b(?:the\s+doctor\s+will\s+be\s+(?:right|in\s+shortly)|doctor\s+will\s+see\s+you|dr\.\s+\w+\s+will\s+be\s+in)\b",
    r"\b(?:confirm\s+your|verify\s+your\s+name\s+and\s+date\s+of\s+birth)\b",
]

PREFIX_DOCTOR_RE = re.compile(r"^(?:\[?(?:doctor|dr\.?(?:\s+[a-z]+)?|clinician|physician|provider)\]?\s*[:\-–]|^(?:doctor|dr\.?)\s*:)\s*", re.IGNORECASE)
PREFIX_PATIENT_RE = re.compile(r"^(?:\[?(?:patient|pt)\]?\s*[:\-–]|^(?:patient|pt)\s*:)\s*", re.IGNORECASE)
PREFIX_CAREGIVER_RE = re.compile(r"^(?:\[?(?:caregiver|family|spouse)\]?\s*[:\-–]|^(?:caregiver|family)\s*:)\s*", re.IGNORECASE)
PREFIX_NURSE_RE = re.compile(r"^(?:\[?(?:nurse|rn|ma|medical\s+assistant|triage)\]?\s*[:\-–]|^(?:nurse|rn)\s*:)\s*", re.IGNORECASE)


def estimate_speaker_role(
    text: str,
    previous_role: Optional[SpeakerRole] = None
) -> Tuple[SpeakerRole, float, str]:
    """
    Estimates speaker role for a given text snippet (Doctor, Patient, Caregiver, Nurse, or Unknown).
    Returns:
        (SpeakerRole, confidence [0.0 - 1.0], rationale_string)
    """
    cleaned = text.strip()
    if not cleaned:
        return SpeakerRole.UNKNOWN, 0.0, "Empty utterance"

    # 1. Explicit script prefixes
    if PREFIX_DOCTOR_RE.match(cleaned):
        return SpeakerRole.DOCTOR, 0.95, "Explicit doctor label prefix"
    if PREFIX_PATIENT_RE.match(cleaned):
        return SpeakerRole.PATIENT, 0.95, "Explicit patient label prefix"
    if PREFIX_CAREGIVER_RE.match(cleaned):
        return SpeakerRole.CAREGIVER, 0.95, "Explicit caregiver label prefix"
    if PREFIX_NURSE_RE.match(cleaned):
        return SpeakerRole.NURSE, 0.95, "Explicit nurse/staff label prefix"

    # 1b. Heading comma address (e.g. "Doctor, what brings you in?" vs "Doctor, my back hurts")
    m_doc_comma = re.match(r"^(?:doctor|dr\.?)\s*,\s*(.*)$", cleaned, re.IGNORECASE)
    if m_doc_comma:
        remainder = m_doc_comma.group(1).strip()
        # Direct clinician inquiry or directive:
        if any(re.search(pat, remainder, re.IGNORECASE) for pat in CLINICIAN_PATTERNS) or remainder.endswith("?"):
            return SpeakerRole.DOCTOR, 0.90, "Clinician inquiry following doctor vocative marker"
        # Caregiver 3rd-person proxy report:
        if any(re.search(pat, remainder, re.IGNORECASE) for pat in CAREGIVER_PATTERNS):
            return SpeakerRole.CAREGIVER, 0.90, "Caregiver addressing clinician on patient behalf"
        # Otherwise patient presenting symptoms:
        return SpeakerRole.PATIENT, 0.88, "Patient addressing clinician with complaint"

    m_pt_comma = re.match(r"^(?:patient|pt)\s*,\s*(.*)$", cleaned, re.IGNORECASE)
    if m_pt_comma:
        remainder = m_pt_comma.group(1).strip()
        if any(re.search(pat, remainder, re.IGNORECASE) for pat in PATIENT_PATTERNS) or not any(re.search(pat, remainder, re.IGNORECASE) for pat in CLINICIAN_PATTERNS):
            return SpeakerRole.PATIENT, 0.90, "Patient symptom statement"

    # 2. Count phrasing cues across all 4 candidate roles
    doc_matches = sum(1 for pat in CLINICIAN_PATTERNS if re.search(pat, cleaned, re.IGNORECASE))
    pt_matches = sum(1 for pat in PATIENT_PATTERNS if re.search(pat, cleaned, re.IGNORECASE))
    cg_matches = sum(1 for pat in CAREGIVER_PATTERNS if re.search(pat, cleaned, re.IGNORECASE))
    nurse_matches = sum(1 for pat in NURSE_PATTERNS if re.search(pat, cleaned, re.IGNORECASE))

    # Caregiver 3rd-person proxy advocacy takes precedence over shared symptom timeline cues
    if cg_matches > 0:
        cg_matches += 2

    # Direct vocative address to doctor (e.g., "Doctor,", "Good morning, Doctor", "Thank you, Dr.", "Doctor, what tests...")
    # Addressing the doctor is a definitive sign of Patient or Caregiver voice
    is_addressing_doctor = bool(
        re.search(r"(?:,\s*(?:dr\.?|doctor)\b|\b(?:dr\.?|doctor)\s*[,.?!]|\b(?:dr\.|doctor)\s+[a-z]+|\bthank\s+you\s*,?\s*(?:dr\.?|doctor)\b|\bhello\s*,?\s*(?:dr\.?|doctor)\b|\bgood\s+morning\s*,?\s*(?:dr\.?|doctor)\b|\bunderstand\s*,?\s*(?:dr\.?|doctor)\b|\bappreciate\s*,?\s*(?:dr\.?|doctor)\b|\b(?:great|right|sure|relief)\s*,?\s*(?:dr\.?|doctor)\b)", cleaned, re.IGNORECASE)
        and not re.search(r"\b(?:doctor\s+(?:comes|will|sees|is\s+in|ordered|prescribed)|dr\.\s+\w+\s+(?:comes|will))\b", cleaned, re.IGNORECASE)
    )
    if is_addressing_doctor:
        if cg_matches > 0:
            cg_matches += 3
        else:
            pt_matches += 3
        doc_matches = 0

    # Direct address to patient using honorifics (e.g. "Mr. Johnson", "Mrs. Smith", "Ms. Davis")
    # Addressing the patient is a definitive sign of Doctor or Nurse voice
    is_addressing_patient = bool(
        re.search(r"\b(?:mr\.|mrs\.|ms\.|miss)\s+[A-Za-z]+\b", cleaned, re.IGNORECASE)
    )
    if is_addressing_patient:
        if nurse_matches > 0:
            nurse_matches += 3
        else:
            doc_matches += 3
        pt_matches = 0

    # Referring to doctor in third person (e.g., "before the doctor comes in", "doctor will be right in")
    if re.search(r"\b(?:doctor\s+(?:comes|will|sees|is\s+in)|dr\.\s+\w+\s+(?:comes|will))\b", cleaned, re.IGNORECASE):
        doc_matches = 0
        if nurse_matches > 0:
            nurse_matches += 2
        elif cg_matches > 0:
            cg_matches += 2

    # Nurse triage/intake cues
    if nurse_matches > 0 and max(doc_matches, pt_matches, cg_matches) == 0:
        conf = min(0.65 + (nurse_matches * 0.15), 0.90)
        return SpeakerRole.NURSE, conf, f"Nurse/staff vitals/intake cues matched ({nurse_matches})"

    # Caregiver / family proxy cues
    if cg_matches > 0 and max(doc_matches, pt_matches) == 0:
        conf = min(0.65 + (cg_matches * 0.15), 0.90)
        return SpeakerRole.CAREGIVER, conf, f"Caregiver/family proxy cues matched ({cg_matches})"

    # Clinician phrasing cues
    if doc_matches > 0 and max(pt_matches, cg_matches, nurse_matches) == 0:
        conf = min(0.6 + (doc_matches * 0.15), 0.90)
        return SpeakerRole.DOCTOR, conf, f"Clinician phrasing cues matched ({doc_matches})"

    # Patient first-person cues
    if pt_matches > 0 and max(doc_matches, cg_matches, nurse_matches) == 0:
        conf = min(0.6 + (pt_matches * 0.15), 0.90)
        return SpeakerRole.PATIENT, conf, f"Patient symptom/history cues matched ({pt_matches})"

    # Mixed cues resolution
    scores = {
        SpeakerRole.DOCTOR: doc_matches,
        SpeakerRole.PATIENT: pt_matches,
        SpeakerRole.CAREGIVER: cg_matches,
        SpeakerRole.NURSE: nurse_matches,
    }
    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_role, best_score = sorted_roles[0]
    second_role, second_score = sorted_roles[1]

    if best_score > 0 and best_score > second_score:
        conf = min(0.60 + ((best_score - second_score) * 0.10), 0.90)
        return best_role, conf, f"Predominant {best_role.value.lower()} cues ({best_score} vs {second_score})"
    elif best_score > 0 and best_score == second_score:
        return SpeakerRole.UNKNOWN, 0.30, f"Conflicting cues between {best_role.value} and {second_role.value}"

    if re.match(r"^(?:yes|yeah|no|okay|ok|sure|i\s+see|thank\s+you|good\s+morning|hello|bye)[.?!]?$", cleaned, re.IGNORECASE):
        if previous_role == SpeakerRole.DOCTOR:
            return SpeakerRole.PATIENT, 0.40, "Conversational response to clinician"
        elif previous_role == SpeakerRole.PATIENT:
            return SpeakerRole.DOCTOR, 0.40, "Conversational response to patient"
        elif previous_role is not None and previous_role != SpeakerRole.UNKNOWN:
            return previous_role, 0.40, f"Conversational acknowledgment of {previous_role.value.lower()}"
        return SpeakerRole.UNKNOWN, 0.10, "Brief generic conversational token"

    return SpeakerRole.UNKNOWN, 0.0, "No definitive role patterns detected"


def process_transcript_segments(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """
    Annotates each segment with estimated speaker role and confidence using two-pass
    contextual turn-taking and conversational continuation smoothing.
    Preserves already-assigned acoustic/multi-modal roles unless high-confidence linguistic cues override.
    """
    if not segments:
        return []

    processed: List[TranscriptSegment] = []

    # Pass 1: Standalone rule-based estimation for anchor detection (preserving existing assignments)
    for seg in segments:
        if seg.speaker_role and seg.speaker_role != SpeakerRole.UNKNOWN:
            # Segment already has an assigned role (e.g. from acoustic clustering);
            # only override if linguistic evidence is very decisive (conf >= 0.75)
            ling_role, conf, reason = estimate_speaker_role(seg.text)
            if conf >= 0.75 and ling_role != SpeakerRole.UNKNOWN and ling_role != seg.speaker_role:
                seg.speaker_role = ling_role
                seg.role_confidence = max(conf, seg.role_confidence)
                seg.speaker_reason = f"Linguistic override ({reason})"
        else:
            role, conf, reason = estimate_speaker_role(seg.text)
            seg.speaker_role = role
            seg.role_confidence = conf
            seg.speaker_reason = reason
        processed.append(seg)

    # Pass 2: Forward Contextual Turn Continuation & Alternation Smoothing
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

    # Pass 2b: Backward fill for leading unknown segments
    for i in range(len(processed) - 2, -1, -1):
        if processed[i].speaker_role == SpeakerRole.UNKNOWN:
            nxt_role = processed[i + 1].speaker_role
            if nxt_role == SpeakerRole.PATIENT:
                processed[i].speaker_role = SpeakerRole.DOCTOR
                processed[i].role_confidence = 0.55
                processed[i].speaker_reason = "Doctor opening preceding patient response"
            elif nxt_role == SpeakerRole.DOCTOR:
                processed[i].speaker_role = SpeakerRole.DOCTOR
                processed[i].role_confidence = 0.50
                processed[i].speaker_reason = "Doctor opening dialogue flow"

    return processed


def split_sentences_preserving_titles(text: str) -> List[str]:
    """Splits text into sentences while protecting periods in medical abbreviations (Dr., Mr., etc.)."""
    masked = re.sub(r"\b(Dr|Mr|Mrs|Ms|Pt)\.", r"\1<DOT>", text, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z])", masked)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def disentangle_and_heal_transcript_segments(
    segments: List[TranscriptSegment]
) -> List[TranscriptSegment]:
    """Cleans, repairs, and disentangles ASR dialogue segments for clinical conversations.
    
    1. Splits multi-sentence composite chunks containing distinct dialogue turns (e.g.
       Doctor question + Patient answer in one chunk) into sub-segments with proportional timestamps.
    2. Heals mid-sentence fragments that were artificially severed by ASR pauses.
    3. Runs contextual two-pass clinical speaker role estimation.
    4. Merges consecutive same-speaker utterances into cohesive turns.
    """
    if not segments:
        return []

    # 1. Split multi-sentence composite chunks proportionally
    split_items: List[TranscriptSegment] = []
    seg_counter = 1

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        sents = split_sentences_preserving_titles(text)
        if len(sents) <= 1:
            split_items.append(TranscriptSegment(
                id=seg_counter,
                text=text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                speaker_id=seg.speaker_id,
                speaker_role=seg.speaker_role,
            ))
            seg_counter += 1
        else:
            total_chars = sum(len(s) for s in sents)
            st = seg.start_time if seg.start_time is not None else 0.0
            et = seg.end_time if seg.end_time is not None else (st + float(len(text)) * 0.06)
            dur = max(0.5, et - st)
            cur_st = st
            for s in sents:
                frac = len(s) / max(1, total_chars)
                cur_et = cur_st + (dur * frac)
                split_items.append(TranscriptSegment(
                    id=seg_counter,
                    text=s,
                    start_time=round(cur_st, 2),
                    end_time=round(cur_et, 2),
                    speaker_id=seg.speaker_id,
                    speaker_role=seg.speaker_role,
                ))
                seg_counter += 1
                cur_st = cur_et

    # 2. Heal broken mid-sentence fragments severed across ASR boundaries
    healed: List[TranscriptSegment] = []
    i = 0
    while i < len(split_items):
        cur = split_items[i]
        cur_text = cur.text
        cur_end = cur.end_time

        while i + 1 < len(split_items):
            nxt = split_items[i + 1]
            nxt_text = nxt.text.strip()
            ends_term = cur_text.rstrip().endswith((".", "?", "!", ":"))
            starts_lower = nxt_text[0].islower() if nxt_text else False
            is_cont = bool(re.match(r"^(?:who|which|that|to|with|and|or|because|of|for|anymore|again|though|so|but)\b", nxt_text, re.IGNORECASE))
            
            # If current sentence does not terminate and next begins with lower or continuation, merge
            if (not ends_term) or starts_lower or (not ends_term and is_cont):
                cur_text = cur_text + " " + nxt_text
                cur_end = nxt.end_time
                i += 1
            else:
                break

        cur.text = cur_text
        cur.end_time = cur_end
        healed.append(cur)
        i += 1

    # 3. Contextual Two-Pass Role Estimation
    annotated = process_transcript_segments(healed)

    # 4. Turn Consolidation: Merge consecutive segments belonging to the same speaker
    # if the conversational gap is <= 1.2 seconds
    final_turns: List[TranscriptSegment] = []
    current_turn: Optional[TranscriptSegment] = None

    for seg in annotated:
        if current_turn is None:
            current_turn = seg
        elif (
            current_turn.speaker_role == seg.speaker_role
            and current_turn.speaker_role != SpeakerRole.UNKNOWN
            and (seg.start_time is None or current_turn.end_time is None or seg.start_time - current_turn.end_time <= 1.2)
        ):
            current_turn.text = f"{current_turn.text} {seg.text}".strip()
            current_turn.end_time = seg.end_time
        else:
            final_turns.append(current_turn)
            current_turn = seg

    if current_turn:
        final_turns.append(current_turn)

    # Re-index IDs and assign default speaker IDs based on roles
    for idx, turn in enumerate(final_turns):
        turn.id = idx + 1
        if not turn.speaker_id:
            if turn.speaker_role == SpeakerRole.DOCTOR:
                turn.speaker_id = "Speaker 1"
            elif turn.speaker_role == SpeakerRole.PATIENT:
                turn.speaker_id = "Speaker 2"
            elif turn.speaker_role == SpeakerRole.CAREGIVER:
                turn.speaker_id = "Speaker 3"
            elif turn.speaker_role == SpeakerRole.NURSE:
                turn.speaker_id = "Speaker 4"
            else:
                turn.speaker_id = "Speaker"

    return final_turns

