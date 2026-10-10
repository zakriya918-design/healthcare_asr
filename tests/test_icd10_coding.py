"""Unit tests for the ICD-10 Clinical Coding Assistant."""

import pytest
from src.models.patient_record import (
    ExtractedClinicalInformation,
    ExtractedFinding,
    FindingStatus,
)
from src.processing.icd10_coding import (
    ICD10CodeMatch,
    map_term_to_icd10,
    suggest_icd10_codes,
    format_icd10_coding_block,
    _normalize_text,
)


def test_normalize_text():
    assert _normalize_text("  Type 2 Diabetes,   Mellitus!  ") == "type 2 diabetes mellitus"
    assert _normalize_text("A-fib (cardiac)") == "a-fib cardiac"


def test_map_term_exact_and_synonyms():
    # Exact match
    res_htn = map_term_to_icd10("hypertension")
    assert len(res_htn) > 0
    assert res_htn[0].code == "I10"
    assert res_htn[0].confidence == 1.0
    assert res_htn[0].match_type == "exact"

    # Synonym matches
    res_hbp = map_term_to_icd10("high blood pressure")
    assert len(res_hbp) > 0
    assert res_hbp[0].code == "I10"

    res_t2dm = map_term_to_icd10("type 2 diabetes")
    assert len(res_t2dm) > 0
    assert res_t2dm[0].code == "E11.9"

    res_gerd = map_term_to_icd10("acid reflux")
    assert len(res_gerd) > 0
    assert res_gerd[0].code == "K21.9"


def test_map_symptoms_to_r_codes():
    res_cough = map_term_to_icd10("cough", category="symptom")
    assert any(m.code == "R05.9" for m in res_cough)

    res_fever = map_term_to_icd10("fever", category="symptom")
    assert any(m.code == "R50.9" for m in res_fever)

    res_chest = map_term_to_icd10("chest pain", category="symptom")
    assert any(m.code == "R07.9" for m in res_chest)


def test_map_unmatched_term():
    res = map_term_to_icd10("completely unknown alien condition xyz")
    assert res == []

    res_empty = map_term_to_icd10("")
    assert res_empty == []


def test_suggest_icd10_codes_full_record():
    info = ExtractedClinicalInformation()
    # Confirmed diagnosis
    info.diagnoses.append(ExtractedFinding(category="diagnosis", name="asthma", status=FindingStatus.PRESENT))
    # Confirmed condition
    info.existing_conditions.append(ExtractedFinding(category="condition", name="hypertension", status=FindingStatus.PRESENT))
    # Confirmed symptom
    info.symptoms.append(ExtractedFinding(category="symptom", name="shortness of breath", status=FindingStatus.PRESENT))
    # Negated symptom (must NOT be coded as present diagnosis)
    info.symptoms.append(ExtractedFinding(category="symptom", name="chest pain", status=FindingStatus.EXPLICITLY_ABSENT))

    codes = suggest_icd10_codes(info)
    code_list = [c.code for c in codes]

    assert "J45.909" in code_list  # Asthma
    assert "I10" in code_list      # Hypertension
    assert "R06.02" in code_list  # Dyspnea / Shortness of breath
    assert "R07.9" not in code_list  # Negated chest pain excluded


def test_format_icd10_coding_block():
    empty_block = format_icd10_coding_block([])
    assert "No specific ICD-10" in empty_block

    matches = [
        ICD10CodeMatch(
            code="I10",
            description="Essential (primary) hypertension",
            matched_term="hypertension",
            category="diagnosis",
            match_type="exact",
            confidence=1.0,
        )
    ]
    block = format_icd10_coding_block(matches)
    assert "I10" in block
    assert "Essential (primary) hypertension" in block
    assert "CLINICIAN REVIEW REQUIRED" in block


def test_icd10_code_match_serialization():
    m = ICD10CodeMatch(
        code="E11.9",
        description="Type 2 diabetes",
        matched_term="t2dm",
        category="diagnosis",
        match_type="synonym",
        confidence=0.9,
    )
    d = m.to_dict()
    assert d["code"] == "E11.9"
    assert d["confidence"] == 0.9
    assert d["status"] == "suggested_pending_verification"
