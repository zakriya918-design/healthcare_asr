"""Unit tests for medical information extraction rules and safeguards."""

import pytest
from src.processing.medical_processor import (
    extract_demographics,
    extract_vital_signs,
    extract_symptoms,
    extract_allergies,
    extract_conditions_and_history,
    extract_medications,
    extract_diagnoses,
    extract_investigations,
    extract_plans_and_followup,
    process_clinical_information
)
from src.models.patient_record import FindingStatus


def test_demographics_extraction():
    text = "The patient is a 52-year-old female presenting with cough."
    demo = extract_demographics(text)
    assert demo.age == 52
    assert demo.gender == "Female"

    text_male = "He is a 34 yo man who fell yesterday."
    demo_male = extract_demographics(text_male)
    assert demo_male.age == 34
    assert demo_male.gender == "Male"

    text_no_demo = "Good morning doctor. I have had ear pain."
    demo_empty = extract_demographics(text_no_demo)
    assert demo_empty.age is None
    assert demo_empty.gender is None


def test_vital_signs_explicit_and_missing():
    # Only BP and HR mentioned
    text = "Her BP is 128/82 mmHg and heart rate is 74 bpm."
    vitals = extract_vital_signs(text)
    assert vitals.blood_pressure_systolic == 128
    assert vitals.blood_pressure_diastolic == 82
    assert vitals.blood_pressure_str == "128/82 mmHg"
    assert vitals.heart_rate_bpm == 74
    # Missing vitals must remain None!
    assert vitals.temperature_f is None
    assert vitals.temperature_c is None
    assert vitals.spo2_percent is None
    assert vitals.blood_glucose_mg_dl is None

    # Full set of vitals
    text_full = "Vitals: BP 118/76, pulse 80, temperature 99.1 F, SpO2 99%, blood sugar 105 mg/dL."
    v_full = extract_vital_signs(text_full)
    assert v_full.blood_pressure_systolic == 118
    assert v_full.blood_pressure_diastolic == 76
    assert v_full.heart_rate_bpm == 80
    assert v_full.temperature_f == 99.1
    assert v_full.spo2_percent == 99
    assert v_full.blood_glucose_mg_dl == 105


def test_symptoms_positive_and_negation():
    sentences = [
        "I have had a severe sore throat and cough for 4 days.",
        "The patient denies chest pain and has had no fever.",
        "There is also mild fatigue."
    ]
    symptoms = extract_symptoms(sentences)

    symptom_map = {s.name: s for s in symptoms}
    assert "sore throat" in symptom_map
    assert symptom_map["sore throat"].status == FindingStatus.PRESENT
    assert symptom_map["sore throat"].duration == "for 4 days"

    assert "cough" in symptom_map
    assert symptom_map["cough"].status == FindingStatus.PRESENT

    assert "chest pain" in symptom_map
    assert symptom_map["chest pain"].status == FindingStatus.EXPLICITLY_ABSENT

    assert "fever" in symptom_map
    assert symptom_map["fever"].status == FindingStatus.EXPLICITLY_ABSENT

    assert "fatigue" in symptom_map
    assert symptom_map["fatigue"].status == FindingStatus.PRESENT


def test_allergies_positive_and_absent():
    sentences_absent = ["The patient has no known drug allergies."]
    allergies_absent = extract_allergies(sentences_absent)
    assert len(allergies_absent) == 1
    assert allergies_absent[0].status == FindingStatus.EXPLICITLY_ABSENT

    sentences_present = ["The patient is allergic to penicillin."]
    allergies_present = extract_allergies(sentences_present)
    assert len(allergies_present) == 1
    assert allergies_present[0].status == FindingStatus.PRESENT
    assert "Penicillin" in allergies_present[0].name


def test_conditions_and_history():
    sentences = [
        "She has chronic hypertension and type 2 diabetes.",
        "There is a history of asthma as a child."
    ]
    conditions, history = extract_conditions_and_history(sentences)
    cond_names = [c.name for c in conditions]
    assert "hypertension" in cond_names
    assert "type 2 diabetes" in cond_names

    hist_names = [h.name for h in history]
    assert "asthma" in hist_names


def test_medications_dosage_and_frequency():
    sentences = [
        "Take amoxicillin 500 mg twice daily for ten days.",
        "You may also take ibuprofen 400 mg as needed for throat pain."
    ]
    meds = extract_medications(sentences)
    med_map = {m.name: m for m in meds}

    assert "Amoxicillin" in med_map
    assert med_map["Amoxicillin"].dosage == "500 mg"
    assert med_map["Amoxicillin"].frequency == "twice daily"
    assert med_map["Amoxicillin"].is_current_plan is True

    assert "Ibuprofen" in med_map
    assert med_map["Ibuprofen"].dosage == "400 mg"
    assert med_map["Ibuprofen"].frequency == "as needed"


def test_diagnoses_uncertainty_preservation():
    sentences = [
        "My assessment is acute pharyngitis.",
        "We need to rule out pneumonia.",
        "Possible bronchitis as well."
    ]
    diagnoses = extract_diagnoses(sentences)
    diag_map = {d.name.lower(): d for d in diagnoses}

    assert any("pharyngitis" in k and d.status == FindingStatus.PRESENT for k, d in diag_map.items())
    assert any("pneumonia" in k and d.status == FindingStatus.UNCERTAIN for k, d in diag_map.items())
    assert any("bronchitis" in k and d.status == FindingStatus.UNCERTAIN for k, d in diag_map.items())


def test_investigations_and_followup():
    sentences = [
        "Let's order a chest x-ray and a rapid strep test.",
        "Increase fluid intake and rest and fluids.",
        "Please follow up in 2 weeks or if symptoms worsen."
    ]
    investigations = extract_investigations(sentences)
    inv_names = [i.name.lower() for i in investigations]
    assert any("x-ray" in n for n in inv_names)
    assert any("strep test" in n for n in inv_names)

    plans, followup = extract_plans_and_followup(sentences)
    assert len(plans) >= 1
    assert len(followup) >= 1
    assert "follow up in 2 weeks" in followup[0].details.lower()


def test_full_pipeline_preserves_unclassified_text():
    raw_transcript = (
        "Doctor: Hello there. What brings you in today?\n"
        "Patient: I work at a grocery store. I have had a cough for 3 days.\n"
        "Doctor: Any fever? Patient: No fever."
    )
    info = process_clinical_information(raw_transcript)
    assert any(s.name == "cough" and s.status == FindingStatus.PRESENT for s in info.symptoms)
    assert any(s.name == "fever" and s.status == FindingStatus.EXPLICITLY_ABSENT for s in info.symptoms)
    # Unclassified text should contain conversational background like grocery store
    assert any("grocery store" in u for u in info.unclassified_statements)


def test_expanded_lexicons():
    text = (
        "Patient complains of insomnia and painful urination for three days. "
        "Also reports heart palpitations and swollen ankles. "
        "Past medical history includes osteoarthritis and chronic kidney disease. "
        "Prescribed tramadol 50 mg twice daily and furosemide 20 mg once daily."
    )
    info = process_clinical_information(text)
    symptom_names = [s.name for s in info.symptoms]
    assert "insomnia" in symptom_names
    assert "dysuria" in symptom_names
    assert "palpitations" in symptom_names
    assert "swelling" in symptom_names

    history_names = [h.name for h in info.relevant_history]
    assert "osteoarthritis" in history_names
    assert "chronic kidney disease" in history_names

    med_names = [m.name.lower() for m in info.medications]
    assert "tramadol" in med_names
    assert "furosemide" in med_names


def test_back_pain_consultation_extraction_and_phonetic_normalization():
    transcript = (
        "Well, my back's been hurting again. "
        "Ever since I got hurt on the job three years ago. "
        "I get reoccurring episodes of back pain. "
        "In the past we had you on some tram at all because physical therapy alone was slow. "
        "Try using a heating pad for your back pain. "
        "There are some alternative therapies like yoga or Tai Chi classes or meditation therapies. "
        "Let's set a time early next week where we can talk over the phone."
    )
    info = process_clinical_information(transcript)
    symptom_names = [s.name for s in info.symptoms]
    assert "back pain" in symptom_names

    cond_names = [c.name for c in info.existing_conditions]
    assert "chronic back pain" in cond_names

    hist_names = [h.name for h in info.relevant_history]
    assert "work-related injury" in hist_names

    med_names = [m.name for m in info.medications]
    assert "Tramadol" in med_names
    tramadol_med = next(m for m in info.medications if m.name == "Tramadol")
    assert tramadol_med.is_current_plan is False

    plan_names = [p.name for p in info.treatment_plans]
    assert "Physical Therapy" in plan_names
    assert "Heating Pad" in plan_names
    assert "Yoga" in plan_names

    assert len(info.advice_followup) >= 1
