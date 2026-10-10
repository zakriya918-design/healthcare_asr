"""Deterministic rule-based clinical information extraction for consultation transcripts.

DISCLAIMER:
This module extracts clinical mentions using deterministic standard-library regex patterns
and clinical linguistic heuristics. It does NOT use a large language model and does NOT
connect to third-party APIs. All outputs are draft suggestions requiring licensed clinician
review and validation before any clinical application.
"""

from __future__ import annotations
import re
from typing import List, Optional, Tuple
from src.models.patient_record import (
    ExtractedClinicalInformation,
    ExtractedFinding,
    FindingStatus,
    MedicationFinding,
    PatientDemographics,
    VitalSigns
)


# Negation triggers
NEGATION_PATTERNS = [
    r"\b(?:no|not|denies|denied|denying|without|free\s+of|ruled\s+out|negative\s+for|hasn't\s+had|haven't\s+had)\b"
]

# Uncertainty triggers
UNCERTAINTY_PATTERNS = [
    r"\b(?:possible|possibility\s+of|rule\s+out|r/o|suspected|suspect|likely|unlikely|could\s+be|may\s+be|concern\s+for|consider|evaluating\s+for)\b"
]

# Known symptoms and complaints
SYMPTOM_LEXICON = {
    "cough": ["cough", "coughing", "dry cough", "productive cough"],
    "sore throat": ["sore throat", "throat pain", "scratchy throat", "pharyngitis"],
    "fever": ["fever", "febrile", "high temperature", "chills"],
    "chills": ["chills", "shivering"],
    "shortness of breath": ["shortness of breath", "dyspnea", "trouble breathing", "hard to breathe"],
    "chest pain": ["chest pain", "chest tightness", "chest pressure"],
    "headache": ["headache", "migraine", "head throbbing"],
    "fatigue": ["fatigue", "tiredness", "exhaustion", "lethargy"],
    "nausea": ["nausea", "feeling sick", "queasy"],
    "vomiting": ["vomiting", "threw up", "throwing up"],
    "diarrhea": ["diarrhea", "loose stools"],
    "dizziness": ["dizziness", "lightheadedness", "dizzy"],
    "rash": ["rash", "hives", "skin eruption", "itching"],
    "congestion": ["congestion", "runny nose", "stuffy nose", "nasal congestion"],
    "abdominal pain": ["abdominal pain", "stomach ache", "belly pain", "cramps"],
    "back pain": ["back pain", "lower back pain", "back's been hurting", "back hurting", "hurting in my back", "back ache", "pain in back"],
    "joint pain": ["joint pain", "knee pain", "arthralgia"],
    "wheezing": ["wheezing", "stridor"],
    "ear pain": ["ear pain", "earache", "ear ache"],
    "insomnia": ["insomnia", "trouble sleeping", "cannot sleep", "difficulty sleeping"],
    "constipation": ["constipation", "hard stools"],
    "dysuria": ["painful urination", "burning urination", "dysuria", "hurts when i pee"],
    "palpitations": ["palpitations", "racing heart", "heart pounding", "fluttering in chest"],
    "swelling": ["swelling", "edema", "puffy ankles", "swollen ankles", "swollen feet", "swollen legs", "swollen"],
    "weakness": ["weakness", "muscle weakness", "feeling weak"],
}

# Known chronic conditions & past history
CONDITION_LEXICON = {
    "hypertension": ["hypertension", "high blood pressure", "htn"],
    "type 2 diabetes": ["type 2 diabetes", "t2dm", "diabetes mellitus", "diabetic", "diabetes"],
    "asthma": ["asthma", "bronchial asthma"],
    "copd": ["copd", "chronic bronchitis", "emphysema"],
    "hyperlipidemia": ["hyperlipidemia", "high cholesterol"],
    "hypothyroidism": ["hypothyroidism", "thyroid disease"],
    "coronary artery disease": ["coronary artery disease", "cad", "heart disease"],
    "depression": ["depression", "major depressive disorder"],
    "anxiety": ["anxiety", "generalized anxiety"],
    "gastroesophageal reflux": ["gerd", "acid reflux", "heartburn"],
    "chronic kidney disease": ["chronic kidney disease", "ckd", "renal disease"],
    "osteoarthritis": ["osteoarthritis", "degenerative joint disease"],
    "atrial fibrillation": ["atrial fibrillation", "afib", "a-fib"],
    "gout": ["gout", "hyperuricemia"],
    "chronic back pain": ["chronic back pain", "reoccurring episodes of back pain", "recurrent back pain", "lumbago", "back pain from coming back"],
    "work-related injury": ["hurt on the job", "injured at work", "work injury", "injury on the job"],
}

# Known common medications
MEDICATION_LEXICON = [
    "amoxicillin", "augmentin", "azithromycin", "ciprofloxacin", "cephalexin", "doxycycline",
    "ibuprofen", "acetaminophen", "paracetamol", "aspirin", "naproxen", "tramadol", "tram at all", "tram-at-all",
    "lisinopril", "amlodipine", "losartan", "atorvastatin", "simvastatin", "rosuvastatin",
    "metformin", "glipizide", "insulin", "albuterol", "fluticasone", "montelukast",
    "omeprazole", "pantoprazole", "levothyroxine", "prednisone", "furosemide",
    "hydrochlorothiazide", "metoprolol", "carvedilol", "gabapentin",
    "sertraline", "escitalopram", "duloxetine", "clopidogrel"
]

# Investigations triggers
INVESTIGATION_LEXICON = [
    "chest x-ray", "x-ray", "cxr", "blood test", "complete blood count", "cbc",
    "comprehensive metabolic panel", "cmp", "throat swab", "rapid strep test",
    "strep test", "covid test", "pcr test", "urinalysis", "urine test",
    "electrocardiogram", "ecg", "ekg", "mri", "ct scan", "ultrasound"
]


def _check_negation(text_segment: str) -> bool:
    """Detects if negation words appear in the text segment."""
    for pat in NEGATION_PATTERNS:
        if re.search(pat, text_segment, re.IGNORECASE):
            return True
    return False


def _check_uncertainty(text_segment: str) -> bool:
    """Detects if uncertainty words appear in the text segment."""
    for pat in UNCERTAINTY_PATTERNS:
        if re.search(pat, text_segment, re.IGNORECASE):
            return True
    return False


def extract_demographics(text: str) -> PatientDemographics:
    """Extracts explicitly stated patient age and gender."""
    demo = PatientDemographics()

    # Age extraction: "52-year-old female", "45 yo", "age 45", "patient is 34"
    age_match = re.search(r"\b(?:age\s*(?:is|of)?\s*)?(\d{1,3})\s*(?:years?\s*old|yo|y/o|year-old|-year-old)\b", text, re.IGNORECASE)
    if not age_match:
        age_match = re.search(r"\bpatient\s+is\s+(\d{1,3})\b", text, re.IGNORECASE)
    if not age_match:
        age_match = re.search(r"\b(\d{1,3})\s*yo\b", text, re.IGNORECASE)
    if age_match:
        age_val = int(age_match.group(1))
        if 0 <= age_val <= 125:
            demo.age = age_val

    # Gender extraction
    if re.search(r"\b(?:female|woman|lady|girl)\b", text, re.IGNORECASE):
        demo.gender = "Female"
    elif re.search(r"\b(?:male|man|gentleman|boy)\b", text, re.IGNORECASE):
        demo.gender = "Male"

    return demo


def extract_vital_signs(text: str) -> VitalSigns:
    """Extracts explicitly stated clinical vital signs. Unstated values remain None."""
    vitals = VitalSigns()
    vitals_sources = []

    # Blood Pressure: "128/82", "BP 130/85 mmHg", "118/76"
    bp_match = re.search(r"\b(?:bp|blood\s+pressure)?\s*(?:is|was|of)?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?\b", text, re.IGNORECASE)
    if bp_match:
        sys_val = int(bp_match.group(1))
        dia_val = int(bp_match.group(2))
        if 60 <= sys_val <= 260 and 30 <= dia_val <= 160:
            vitals.blood_pressure_systolic = sys_val
            vitals.blood_pressure_diastolic = dia_val
            vitals.blood_pressure_str = f"{sys_val}/{dia_val} mmHg"
            vitals_sources.append(bp_match.group(0))

    # Heart Rate / Pulse: "heart rate is 74", "pulse 80 bpm"
    hr_match = re.search(r"\b(?:heart\s+rate|hr|pulse)\s*(?:is|was|of)?\s*(\d{2,3})\s*(?:bpm|beats\s+per\s+minute)?\b", text, re.IGNORECASE)
    if hr_match:
        hr_val = int(hr_match.group(1))
        if 30 <= hr_val <= 230:
            vitals.heart_rate_bpm = hr_val
            vitals_sources.append(hr_match.group(0))

    # Temperature: "temp 99.1 F", "temperature 38.2 C", "101.2 degrees"
    temp_match = re.search(r"\b(?:temp|temperature)\s*(?:is|was|of)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:°?\s*(f|c|fahrenheit|celsius)|degrees)?\b", text, re.IGNORECASE)
    if temp_match:
        val = float(temp_match.group(1))
        unit = temp_match.group(2)
        if unit and unit.lower().startswith("c"):
            vitals.temperature_c = val
            vitals.temperature_str = f"{val} °C"
        elif unit and unit.lower().startswith("f"):
            vitals.temperature_f = val
            vitals.temperature_str = f"{val} °F"
        else:
            if val >= 85.0:
                vitals.temperature_f = val
                vitals.temperature_str = f"{val} °F"
            elif val <= 45.0:
                vitals.temperature_c = val
                vitals.temperature_str = f"{val} °C"
        vitals_sources.append(temp_match.group(0))

    # Oxygen saturation (SpO2): "SpO2 99%", "O2 98%"
    spo2_match = re.search(r"\b(?:spo2|oxygen\s+saturation|o2\s+sat|o2|sat)\s*(?:is|was|of|:)?\s*(\d{2,3})\s*%", text, re.IGNORECASE)
    if spo2_match:
        spo2_val = int(spo2_match.group(1))
        if 50 <= spo2_val <= 100:
            vitals.spo2_percent = spo2_val
            vitals_sources.append(spo2_match.group(0))

    # Blood glucose: "blood sugar 105", "glucose 110 mg/dL"
    glucose_match = re.search(r"\b(?:blood\s+sugar|blood\s+glucose|glucose|bsl)\s*(?:is|was|of)?\s*(\d{2,3})\s*(?:mg/dl|mgdl)?\b", text, re.IGNORECASE)
    if glucose_match:
        glu_val = int(glucose_match.group(1))
        if 30 <= glu_val <= 600:
            vitals.blood_glucose_mg_dl = glu_val
            vitals.blood_glucose_str = f"{glu_val} mg/dL"
            vitals_sources.append(glucose_match.group(0))

    if vitals_sources:
        vitals.source_phrase = "; ".join(vitals_sources)

    return vitals


def extract_symptoms(sentences: List[str]) -> List[ExtractedFinding]:
    """Extracts symptoms with negation, uncertainty, and duration."""
    findings: List[ExtractedFinding] = []
    seen = set()

    duration_re = re.compile(r"\b(?:for|since|past|last)\s+(\d+\s*(?:days?|weeks?|months?|hours?)|yesterday)\b", re.IGNORECASE)

    for sentence in sentences:
        s_clean = sentence.strip()
        if not s_clean:
            continue

        # Skip interrogative inquiries from asserting presence of symptoms
        is_inquiry = bool(re.search(r"\b(?:do|did|have|any)\s+you\s+(?:have|had|experience|feel|notice)\b", s_clean, re.IGNORECASE)) or s_clean.endswith("?")

        for canonical_name, variants in SYMPTOM_LEXICON.items():
            for var in variants:
                pattern = rf"\b{re.escape(var)}\b"
                match = re.search(pattern, s_clean, re.IGNORECASE)
                if match:
                    preceding_window = s_clean[max(0, match.start() - 40):match.start()]
                    following_window = s_clean[match.end():min(len(s_clean), match.end() + 30)]

                    is_negated = _check_negation(preceding_window) or bool(re.search(r"\b(?:ruled\s+out|is\s+not\s+present|not\s+noted|is\s+absent)\b", following_window, re.IGNORECASE))
                    is_uncertain = _check_uncertainty(preceding_window) or _check_uncertainty(following_window)

                    if is_negated:
                        status = FindingStatus.EXPLICITLY_ABSENT
                    elif is_uncertain:
                        status = FindingStatus.UNCERTAIN
                    elif is_inquiry:
                        continue
                    else:
                        status = FindingStatus.PRESENT

                    dur_match = duration_re.search(s_clean)
                    dur_val = dur_match.group(0) if dur_match else None

                    key = (canonical_name, status)
                    if key not in seen:
                        seen.add(key)
                        findings.append(ExtractedFinding(
                            category="symptom",
                            name=canonical_name,
                            status=status,
                            duration=dur_val,
                            source_phrase=s_clean
                        ))
                    break

    return findings


def extract_allergies(sentences: List[str]) -> List[ExtractedFinding]:
    """Extracts allergies or explicitly denies allergies."""
    allergies: List[ExtractedFinding] = []

    for s in sentences:
        if re.search(r"\b(?:no\s+known\s+(?:drug\s+)?allergies|no\s+allergies|denies\s+allergies|nkda)\b", s, re.IGNORECASE):
            allergies.append(ExtractedFinding(
                category="allergy",
                name="Allergies",
                status=FindingStatus.EXPLICITLY_ABSENT,
                details="No known drug allergies",
                source_phrase=s.strip()
            ))
            return allergies

    for s in sentences:
        match = re.search(r"\b(?:allergic\s+to|allergy\s+to|allergic\s+reaction\s+to)\s+([a-zA-Z0-9_\-\s]+?)(?:[.,;]|\band\b|$)", s, re.IGNORECASE)
        if match:
            allergen = match.group(1).strip()
            if allergen and not _check_negation(s):
                allergies.append(ExtractedFinding(
                    category="allergy",
                    name=allergen.title(),
                    status=FindingStatus.PRESENT,
                    details=f"Allergy to {allergen}",
                    source_phrase=s.strip()
                ))

    return allergies


def extract_conditions_and_history(sentences: List[str]) -> Tuple[List[ExtractedFinding], List[ExtractedFinding]]:
    """Extracts existing chronic conditions and relevant past medical history."""
    conditions: List[ExtractedFinding] = []
    history: List[ExtractedFinding] = []
    seen = set()

    for s in sentences:
        s_clean = s.strip()
        for canonical_name, variants in CONDITION_LEXICON.items():
            for var in variants:
                pattern = rf"\b{re.escape(var)}\b"
                if re.search(pattern, s_clean, re.IGNORECASE):
                    is_negated = _check_negation(s_clean)
                    status = FindingStatus.EXPLICITLY_ABSENT if is_negated else FindingStatus.PRESENT

                    is_past = bool(re.search(r"\b(?:history\s+of|past\s+medical\s+history|pmh|former|years\s+ago|as\s+a\s+child)\b", s_clean, re.IGNORECASE))

                    key = (canonical_name, status, is_past)
                    if key not in seen:
                        seen.add(key)
                        finding = ExtractedFinding(
                            category="condition" if not is_past else "history",
                            name=canonical_name,
                            status=status,
                            source_phrase=s_clean
                        )
                        if is_past:
                            history.append(finding)
                        else:
                            conditions.append(finding)
                    break

    return conditions, history


def extract_medications(sentences: List[str]) -> List[MedicationFinding]:
    """Extracts medications with dosage, frequency, and current plan status."""
    medications: List[MedicationFinding] = []
    seen = set()

    dosage_re = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units?|tablets?|capsules?|puffs?))\b", re.IGNORECASE)
    freq_re = re.compile(r"\b(once\s+daily|twice\s+daily|three\s+times\s+(?:a\s+day|daily)|four\s+times\s+(?:a\s+day|daily)|every\s+\d+\s+hours?|daily|bid|tid|qid|prn|as\s+needed|at\s+bedtime)\b", re.IGNORECASE)
    past_med_re = re.compile(r"\b(?:used\s+to\s+take|stopped\s+taking|previously\s+took|discontinued|past\s+prescription|in\s+the\s+past|had\s+you\s+on)\b", re.IGNORECASE)

    for s in sentences:
        s_clean = s.strip()
        for med_name in MEDICATION_LEXICON:
            pattern = rf"\b{re.escape(med_name)}\b"
            if re.search(pattern, s_clean, re.IGNORECASE):
                dosage_match = dosage_re.search(s_clean)
                dosage = dosage_match.group(1) if dosage_match else None

                freq_match = freq_re.search(s_clean)
                freq = freq_match.group(1) if freq_match else None

                is_negated = _check_negation(s_clean)
                status = FindingStatus.EXPLICITLY_ABSENT if is_negated else FindingStatus.PRESENT

                is_past = bool(past_med_re.search(s_clean))
                is_current = not is_past

                med_clean = med_name.lower()
                if med_clean in ("tram at all", "tram-at-all"):
                    canonical_name = "Tramadol"
                else:
                    canonical_name = med_name.title()

                key = (canonical_name, dosage, freq, status, is_current)
                if key not in seen:
                    seen.add(key)
                    medications.append(MedicationFinding(
                        name=canonical_name,
                        dosage=dosage,
                        frequency=freq,
                        status=status,
                        is_current_plan=is_current,
                        source_phrase=s_clean
                    ))

    return medications


def extract_diagnoses(sentences: List[str]) -> List[ExtractedFinding]:
    """Extracts clinician-stated assessment and diagnoses, preserving uncertainty."""
    diagnoses: List[ExtractedFinding] = []
    seen = set()

    diag_triggers = [
        r"\b(?:assessment\s+is|impression\s+is|diagnosed\s+with|diagnosis\s+is)\s+([^.,;\n]+)",
        r"\b(?:rule\s+out|r/o)\s+([^.,;\n]+)",
        r"\b(?:possible|suspected|likely|concern\s+for)\s+([^.,;\n]+)"
    ]

    for s in sentences:
        s_clean = s.strip()
        for trig in diag_triggers:
            match = re.search(trig, s_clean, re.IGNORECASE)
            if match:
                diag_term = match.group(1).strip()
                if len(diag_term) > 3 and not re.match(r"^(?:the|that|a|an|it|this)$", diag_term, re.IGNORECASE):
                    # Remove conversational fillers like 'as well', 'too', 'also'
                    diag_term = re.sub(r"\b(?:as\s+well|too|also)\b", "", diag_term, flags=re.IGNORECASE).strip()
                    is_rule_out = bool(re.search(r"\b(?:rule\s+out|r/o)\b", s_clean, re.IGNORECASE))
                    is_uncertain = _check_uncertainty(s_clean) or is_rule_out or ("possible" in s_clean.lower())
                    is_negated = _check_negation(s_clean) and not is_rule_out

                    if is_rule_out or is_uncertain:
                        status = FindingStatus.UNCERTAIN
                    elif is_negated:
                        status = FindingStatus.EXPLICITLY_ABSENT
                    else:
                        status = FindingStatus.PRESENT

                    key = (diag_term.title(), status)
                    if key not in seen:
                        seen.add(key)
                        diagnoses.append(ExtractedFinding(
                            category="diagnosis",
                            name=diag_term.title(),
                            status=status,
                            details="Clinician-stated diagnostic impression",
                            source_phrase=s_clean
                        ))

    return diagnoses


def extract_investigations(sentences: List[str]) -> List[ExtractedFinding]:
    """Extracts diagnostic tests and investigations ordered or discussed."""
    investigations: List[ExtractedFinding] = []
    seen = set()

    for s in sentences:
        s_clean = s.strip()
        for inv in INVESTIGATION_LEXICON:
            if re.search(rf"\b{re.escape(inv)}\b", s_clean, re.IGNORECASE):
                is_negated = _check_negation(s_clean)
                status = FindingStatus.EXPLICITLY_ABSENT if is_negated else FindingStatus.PRESENT
                key = (inv.title(), status)
                if key not in seen:
                    seen.add(key)
                    investigations.append(ExtractedFinding(
                        category="investigation",
                        name=inv.title(),
                        status=status,
                        source_phrase=s_clean
                    ))

    return investigations


def extract_plans_and_followup(sentences: List[str]) -> Tuple[List[ExtractedFinding], List[ExtractedFinding]]:
    """Extracts non-pharmacological treatment instructions and follow-up advice."""
    plans: List[ExtractedFinding] = []
    followup: List[ExtractedFinding] = []
    seen_plans = set()

    plan_cues = [
        "rest and fluids", "increase fluid intake", "fluids and rest",
        "warm salt water gargle", "ice pack", "elevate leg",
        "saline nasal spray", "steam inhalation", "throat lozenges",
        "physical therapy", "heating pad", "heat therapy", "yoga",
        "tai chi", "meditation", "core strengthening", "exercise therapy", "stretching"
    ]

    for s in sentences:
        s_clean = s.strip()
        for cue in plan_cues:
            if re.search(rf"\b{re.escape(cue)}\b", s_clean, re.IGNORECASE):
                cue_name = cue.title()
                if cue_name not in seen_plans:
                    seen_plans.add(cue_name)
                    plans.append(ExtractedFinding(
                        category="treatment_plan",
                        name=cue_name,
                        status=FindingStatus.PRESENT,
                        source_phrase=s_clean
                    ))

        if re.search(r"\b(?:follow\s+up|see\s+me\s+again|come\s+back|return\s+if|talk\s+over\s+the\s+phone|next\s+week|call\s+back)\b", s_clean, re.IGNORECASE):
            followup.append(ExtractedFinding(
                category="advice_followup",
                name="Follow-up instructions",
                status=FindingStatus.PRESENT,
                details=s_clean,
                source_phrase=s_clean
            ))

    return plans, followup


def process_clinical_information(transcript_text: str) -> ExtractedClinicalInformation:
    """
    Main extraction pipeline. Splits transcript into sentences, applies deterministic rules,
    and returns a structured ExtractedClinicalInformation object.
    """
    raw_sentences = [
        line.strip() for line in re.split(r"(?:(?<=[.?!])\s+)|\r|\n", transcript_text) if line.strip()
    ]

    info = ExtractedClinicalInformation()

    info.vital_signs = extract_vital_signs(transcript_text)
    info.symptoms = extract_symptoms(raw_sentences)
    info.allergies = extract_allergies(raw_sentences)
    conditions, history = extract_conditions_and_history(raw_sentences)
    info.existing_conditions = conditions
    info.relevant_history = history
    info.medications = extract_medications(raw_sentences)
    info.diagnoses = extract_diagnoses(raw_sentences)
    info.investigations = extract_investigations(raw_sentences)
    plans, followup = extract_plans_and_followup(raw_sentences)
    info.treatment_plans = plans
    info.advice_followup = followup

    for s in raw_sentences:
        matched = False
        for finding_list in [info.symptoms, info.allergies, info.existing_conditions,
                             info.relevant_history, info.diagnoses, info.investigations,
                             info.treatment_plans, info.advice_followup]:
            if any(f.source_phrase and s in f.source_phrase for f in finding_list):
                matched = True
                break
        if not matched and any(m.source_phrase and s in m.source_phrase for m in info.medications):
            matched = True
        if not matched and info.vital_signs.source_phrase and any(p in s for p in info.vital_signs.source_phrase.split("; ")):
            matched = True

        if not matched and len(s.split()) >= 3:
            info.unclassified_statements.append(s)

    return info
