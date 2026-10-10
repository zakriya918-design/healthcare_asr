"""ICD-10 Clinical Coding Assistant for Healthcare AI Documentation Assistant.

Maps extracted clinical diagnoses, chronic conditions, and symptoms to standard
ICD-10-CM codes using deterministic clinical synonym tables and fuzzy lexical heuristics.
All suggestions carry explicit clinician review notices and confidence scoring.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from src.models.patient_record import ExtractedClinicalInformation, ExtractedFinding, FindingStatus


@dataclass
class ICD10CodeMatch:
    """A suggested ICD-10-CM code mapped from an extracted clinical entity."""
    code: str
    description: str
    matched_term: str
    category: str
    match_type: str  # "exact", "synonym", "fuzzy"
    confidence: float  # 0.0 to 1.0
    status: str = "suggested_pending_verification"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "description": self.description,
            "matched_term": self.matched_term,
            "category": self.category,
            "match_type": self.match_type,
            "confidence": round(self.confidence, 2),
            "status": self.status,
        }


# Standard Outpatient & Primary Care ICD-10-CM Clinical Lexicon
# Format: (Code, Official Description, List of trigger aliases/synonyms)
ICD10_DATABASE: List[Tuple[str, str, List[str]]] = [
    # Cardiovascular / Circulatory
    ("I10", "Essential (primary) hypertension", ["hypertension", "high blood pressure", "htn", "essential hypertension"]),
    ("I11.9", "Hypertensive heart disease without heart failure", ["hypertensive heart disease"]),
    ("I25.10", "Atherosclerotic heart disease of native coronary artery", ["coronary artery disease", "cad", "atherosclerosis", "coronary heart disease"]),
    ("I48.91", "Unspecified atrial fibrillation", ["atrial fibrillation", "afib", "a-fib"]),
    ("I50.9", "Heart failure, unspecified", ["heart failure", "congestive heart failure", "chf"]),
    ("E78.5", "Hyperlipidemia, unspecified", ["hyperlipidemia", "dyslipidemia", "high cholesterol", "hypercholesterolemia"]),

    # Endocrine & Metabolic
    ("E11.9", "Type 2 diabetes mellitus without complications", ["type 2 diabetes", "t2dm", "diabetes mellitus", "type ii diabetes", "adult onset diabetes"]),
    ("E10.9", "Type 1 diabetes mellitus without complications", ["type 1 diabetes", "t1dm", "type i diabetes", "juvenile diabetes"]),
    ("E03.9", "Hypothyroidism, unspecified", ["hypothyroidism", "underactive thyroid"]),
    ("E05.90", "Thyrotoxicosis without mentions of crisis, unspecified", ["hyperthyroidism", "overactive thyroid", "thyrotoxicosis"]),
    ("E66.9", "Obesity, unspecified", ["obesity", "overweight", "morbid obesity"]),
    ("E55.9", "Vitamin D deficiency, unspecified", ["vitamin d deficiency"]),

    # Respiratory
    ("J06.9", "Acute upper respiratory infection, unspecified", ["upper respiratory infection", "uri", "common cold", "head cold"]),
    ("J20.9", "Acute bronchitis, unspecified", ["acute bronchitis", "bronchitis"]),
    ("J45.909", "Unspecified asthma, uncomplicated", ["asthma", "bronchial asthma", "reactive airway disease"]),
    ("J44.9", "Chronic obstructive pulmonary disease, unspecified", ["copd", "chronic obstructive pulmonary disease", "emphysema"]),
    ("J18.9", "Pneumonia, unspecified organism", ["pneumonia", "chest infection", "bronchopneumonia"]),
    ("J30.9", "Allergic rhinitis, unspecified", ["allergic rhinitis", "hay fever", "seasonal allergies"]),
    ("J01.90", "Acute sinusitis, unspecified", ["sinusitis", "acute sinusitis", "sinus infection"]),
    ("J02.9", "Acute pharyngitis, unspecified", ["pharyngitis", "sore throat", "acute pharyngitis"]),

    # Musculoskeletal & Connective
    ("M54.50", "Low back pain, unspecified", ["low back pain", "lower back pain", "lumbago", "lumbar back pain"]),
    ("M54.2", "Cervicalgia (Neck pain)", ["neck pain", "cervicalgia", "cervical spine pain"]),
    ("M19.90", "Unspecified osteoarthritis, unspecified site", ["osteoarthritis", "oa", "degenerative joint disease"]),
    ("M77.9", "Enthesopathy, unspecified (Tendinitis)", ["tendinitis", "tendonitis", "tendinopathy"]),
    ("M79.7", "Fibromyalgia", ["fibromyalgia", "fibrositis"]),
    ("M25.50", "Pain in unspecified joint (Arthralgia)", ["joint pain", "arthralgia"]),

    # Neurological & Mental
    ("G43.909", "Migraine, unspecified, not intractable", ["migraine", "migraine headache", "hemicrania"]),
    ("G44.209", "Tension-type headache, unspecified", ["tension headache", "tension-type headache"]),
    ("R51.9", "Headache, unspecified", ["headache", "cephalalgia"]),
    ("F32.9", "Major depressive disorder, single episode, unspecified", ["depression", "major depressive disorder", "mdd", "depressive episode"]),
    ("F41.1", "Generalized anxiety disorder", ["generalized anxiety disorder", "gad", "anxiety", "chronic anxiety"]),
    ("G47.00", "Insomnia, unspecified", ["insomnia", "sleeplessness", "sleep disturbance"]),

    # Gastrointestinal
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis", ["gerd", "acid reflux", "gastroesophageal reflux", "heartburn"]),
    ("K29.70", "Gastritis, unspecified, without bleeding", ["gastritis", "stomach inflammation"]),
    ("K58.9", "Irritable bowel syndrome without diarrhea", ["irritable bowel syndrome", "ibs"]),
    ("K59.00", "Constipation, unspecified", ["constipation"]),

    # Common Signs & Symptoms (R-codes)
    ("R50.9", "Fever, unspecified", ["fever", "pyrexia", "febrile"]),
    ("R05.9", "Cough, unspecified", ["cough", "chronic cough", "dry cough", "productive cough"]),
    ("R07.9", "Chest pain, unspecified", ["chest pain", "thoracic pain"]),
    ("R06.02", "Shortness of breath (Dyspnea)", ["shortness of breath", "dyspnea", "sob", "difficulty breathing"]),
    ("R53.83", "Other fatigue (Fatigue/Lethargy)", ["fatigue", "tiredness", "exhaustion", "lethargy"]),
    ("R42", "Dizziness and giddiness", ["dizziness", "lightheadedness", "vertigo", "giddiness"]),
    ("R11.2", "Nausea with vomiting, unspecified", ["nausea and vomiting", "nausea with vomiting"]),
    ("R11.0", "Nausea", ["nausea", "queasiness"]),
    ("R19.7", "Diarrhea, unspecified", ["diarrhea", "loose stools"]),
    ("R55", "Syncope and collapse", ["syncope", "fainting", "blackout"]),
    ("J11.1", "Influenza due to unidentified influenza virus with other manifestations", ["influenza", "flu"]),
]


def _normalize_text(text: str) -> str:
    """Lowercase and strip non-alphanumeric punctuation for consistent matching."""
    cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
    return " ".join(cleaned.split())


def map_term_to_icd10(term: str, category: str = "diagnosis") -> List[ICD10CodeMatch]:
    """Search ICD-10 database for closest clinical code matches for a single term.
    
    Args:
        term: The extracted medical term (e.g., "hypertension", "type 2 diabetes").
        category: Clinical classification category ("diagnosis", "symptom", "condition").
        
    Returns:
        List of ICD10CodeMatch objects sorted descending by confidence score.
    """
    if not term or not term.strip():
        return []

    norm_term = _normalize_text(term)
    matches: List[ICD10CodeMatch] = []
    seen_codes = set()

    for code, desc, aliases in ICD10_DATABASE:
        best_conf = 0.0
        best_type = ""

        # Check exact and alias matches
        for alias in aliases:
            norm_alias = _normalize_text(alias)
            if norm_term == norm_alias:
                conf = 1.0
                mtype = "exact"
            elif norm_alias in norm_term:
                # Word boundary check for substring
                if re.search(rf"\b{re.escape(norm_alias)}\b", norm_term):
                    conf = 0.90
                    mtype = "synonym"
                else:
                    conf = 0.75
                    mtype = "partial"
            elif norm_term in norm_alias:
                if re.search(rf"\b{re.escape(norm_term)}\b", norm_alias):
                    conf = 0.85
                    mtype = "synonym"
                else:
                    conf = 0.70
                    mtype = "partial"
            else:
                continue

            if conf > best_conf:
                best_conf = conf
                best_type = mtype

        # Also check match against official description
        norm_desc = _normalize_text(desc)
        if norm_term in norm_desc and best_conf < 0.80:
            best_conf = 0.80
            best_type = "partial"

        if best_conf >= 0.70 and code not in seen_codes:
            seen_codes.add(code)
            matches.append(ICD10CodeMatch(
                code=code,
                description=desc,
                matched_term=term,
                category=category,
                match_type=best_type,
                confidence=best_conf,
            ))

    # Sort matches by confidence descending
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches


def suggest_icd10_codes(extracted_info: ExtractedClinicalInformation) -> List[ICD10CodeMatch]:
    """Analyze all confirmed diagnoses, conditions, and symptoms to suggest ICD-10 codes.
    
    Args:
        extracted_info: Structured ExtractedClinicalInformation from consultation.
        
    Returns:
        List of suggested ICD10CodeMatch objects sorted by category and confidence.
    """
    all_suggestions: List[ICD10CodeMatch] = []
    seen_codes = set()

    # 1. Process Diagnoses (Primary Priority)
    for diag in extracted_info.diagnoses:
        if diag.name and diag.status == FindingStatus.PRESENT:
            candidates = map_term_to_icd10(diag.name, category="diagnosis")
            for cand in candidates:
                if cand.code not in seen_codes:
                    seen_codes.add(cand.code)
                    all_suggestions.append(cand)

    # 2. Process Existing Conditions
    for cond in extracted_info.existing_conditions:
        if cond.name and cond.status == FindingStatus.PRESENT:
            candidates = map_term_to_icd10(cond.name, category="existing_condition")
            for cand in candidates:
                if cand.code not in seen_codes:
                    seen_codes.add(cand.code)
                    all_suggestions.append(cand)

    # 3. Process Confirmed Symptoms (Signs & Symptoms R-codes)
    for sym in extracted_info.symptoms:
        if sym.name and sym.status == FindingStatus.PRESENT:
            candidates = map_term_to_icd10(sym.name, category="symptom")
            for cand in candidates:
                if cand.code not in seen_codes:
                    seen_codes.add(cand.code)
                    all_suggestions.append(cand)

    return all_suggestions


def format_icd10_coding_block(suggestions: List[ICD10CodeMatch]) -> str:
    """Format ICD-10 suggestions into readable clinical documentation block."""
    if not suggestions:
        return "No specific ICD-10 diagnosis codes matched."

    lines = [
        "SUGGESTED ICD-10-CM CODES (CLINICIAN REVIEW REQUIRED):",
        "-------------------------------------------------------"
    ]
    for s in suggestions:
        lines.append(f"• {s.code} — {s.description}")
        lines.append(f"  [Matched: '{s.matched_term}' ({s.category}) | Confidence: {int(s.confidence * 100)}% | Type: {s.match_type}]")

    lines.append("\nNOTICE: Codes are auto-suggested draft text for clinical and billing verification.")
    return "\n".join(lines)
