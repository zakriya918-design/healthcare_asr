"""Deterministic SOAP clinical note generator from structured clinical findings.

DISCLAIMER:
This module compiles structured findings into a draft SOAP note.
It does NOT independently diagnose patients, predict medical outcomes, or suggest unmentioned
treatments. All generated notes are marked as draft_pending_review and must be reviewed,
verified, and signed by a licensed clinician before any medical use.
"""

from __future__ import annotations
from typing import List, Optional
from src.models.patient_record import (
    ConsultationRecord,
    ExtractedClinicalInformation,
    ExtractedFinding,
    FindingStatus,
    MedicationFinding,
    ReviewStatus,
    SOAPNote,
    VitalSigns
)


def format_subjective_section(info: ExtractedClinicalInformation, demographics_summary: str = "") -> str:
    """Formats the Subjective (S) section including patient complaints, negatives, and history."""
    lines = []
    if demographics_summary:
        lines.append(f"Demographics: {demographics_summary}")

    # Positive symptoms
    positive_symptoms = [f for f in info.symptoms if f.status == FindingStatus.PRESENT]
    if positive_symptoms:
        symptom_strs = []
        for s in positive_symptoms:
            if s.duration:
                symptom_strs.append(f"{s.name} ({s.duration})")
            else:
                symptom_strs.append(s.name)
        lines.append(f"Reported Symptoms: {', '.join(symptom_strs)}.")
    else:
        lines.append("Reported Symptoms: None explicitly reported.")

    # Pertinent negatives
    negative_symptoms = [f for f in info.symptoms if f.status == FindingStatus.EXPLICITLY_ABSENT]
    if negative_symptoms:
        neg_names = [f.name for f in negative_symptoms]
        lines.append(f"Pertinent Negatives: Denies {', '.join(neg_names)}.")

    # Chronic conditions & medical history
    active_conditions = [f.name for f in info.existing_conditions if f.status == FindingStatus.PRESENT]
    if active_conditions:
        lines.append(f"Existing Conditions: {', '.join(active_conditions)}.")

    past_hist = [f.name for f in info.relevant_history if f.status == FindingStatus.PRESENT]
    if past_hist:
        lines.append(f"Past Medical History: {', '.join(past_hist)}.")

    # Allergies
    if info.allergies:
        allergy_strs = []
        for a in info.allergies:
            if a.status == FindingStatus.EXPLICITLY_ABSENT:
                allergy_strs.append("No known drug allergies (NKDA)")
            else:
                allergy_strs.append(a.name)
        lines.append(f"Allergies: {', '.join(allergy_strs)}.")
    else:
        lines.append("Allergies: Not explicitly documented.")

    # Patient reported medications (past or home)
    home_meds = [m for m in info.medications if not m.is_current_plan]
    if home_meds:
        med_strs = [f"{m.name} ({m.dosage or 'dose unstated'}, {m.frequency or 'frequency unstated'})" for m in home_meds]
        lines.append(f"Prior / Home Medications: {', '.join(med_strs)}.")

    return "\n".join(lines)


def format_objective_section(vitals: VitalSigns) -> str:
    """Formats the Objective (O) section. Missing vital signs are explicitly marked as not recorded."""
    lines = ["Vital Signs:"]

    # Blood pressure
    if vitals.blood_pressure_str:
        lines.append(f"- Blood Pressure: {vitals.blood_pressure_str}")
    elif vitals.blood_pressure_systolic is not None and vitals.blood_pressure_diastolic is not None:
        lines.append(f"- Blood Pressure: {vitals.blood_pressure_systolic}/{vitals.blood_pressure_diastolic} mmHg")
    else:
        lines.append("- Blood Pressure: Not recorded")

    # Heart rate
    if vitals.heart_rate_bpm is not None:
        lines.append(f"- Heart Rate: {vitals.heart_rate_bpm} bpm")
    else:
        lines.append("- Heart Rate: Not recorded")

    # Temperature
    if vitals.temperature_str:
        lines.append(f"- Body Temperature: {vitals.temperature_str}")
    elif vitals.temperature_f is not None:
        lines.append(f"- Body Temperature: {vitals.temperature_f} °F")
    elif vitals.temperature_c is not None:
        lines.append(f"- Body Temperature: {vitals.temperature_c} °C")
    else:
        lines.append("- Body Temperature: Not recorded")

    # Oxygen Saturation (SpO2)
    if vitals.spo2_percent is not None:
        lines.append(f"- Oxygen Saturation (SpO2): {vitals.spo2_percent}%")
    else:
        lines.append("- Oxygen Saturation (SpO2): Not recorded")

    # Blood Glucose
    if vitals.blood_glucose_str:
        lines.append(f"- Blood Glucose: {vitals.blood_glucose_str}")
    elif vitals.blood_glucose_mg_dl is not None:
        lines.append(f"- Blood Glucose: {vitals.blood_glucose_mg_dl} mg/dL")
    else:
        lines.append("- Blood Glucose: Not recorded")

    return "\n".join(lines)


def format_assessment_section(info: ExtractedClinicalInformation) -> str:
    """Formats the Assessment (A) section. Preserves clinician diagnostic statements and uncertainty."""
    lines = []
    confirmed_diags = [d for d in info.diagnoses if d.status == FindingStatus.PRESENT]
    uncertain_diags = [d for d in info.diagnoses if d.status == FindingStatus.UNCERTAIN]

    if confirmed_diags:
        lines.append("Clinician Assessment:")
        for d in confirmed_diags:
            lines.append(f"- {d.name}")

    if uncertain_diags:
        lines.append("Differential / Tentative Impressions (Uncertain):")
        for d in uncertain_diags:
            lines.append(f"- {d.name} (Tentative / To rule out)")

    if not confirmed_diags and not uncertain_diags:
        lines.append("Assessment not documented in consultation transcript. Requires clinician clinical evaluation.")

    return "\n".join(lines)


def format_plan_section(info: ExtractedClinicalInformation) -> str:
    """Formats the Plan (P) section covering clinician-stated medications, tests, and instructions."""
    lines = []

    # Prescribed / Planned Medications
    plan_meds = [m for m in info.medications if m.is_current_plan and m.status == FindingStatus.PRESENT]
    if plan_meds:
        lines.append("Pharmacological Plan:")
        for m in plan_meds:
            details = []
            if m.dosage:
                details.append(m.dosage)
            if m.frequency:
                details.append(m.frequency)
            detail_str = f" ({', '.join(details)})" if details else ""
            lines.append(f"- {m.name}{detail_str}")
    else:
        lines.append("Pharmacological Plan: No new medications explicitly prescribed.")

    # Investigations Ordered
    if info.investigations:
        lines.append("\nDiagnostic Investigations Ordered:")
        for inv in info.investigations:
            lines.append(f"- {inv.name}")

    # Treatment / Supportive Care
    if info.treatment_plans:
        lines.append("\nPatient Instructions & Supportive Care:")
        for tp in info.treatment_plans:
            lines.append(f"- {tp.name}")

    # Follow-up
    if info.advice_followup:
        lines.append("\nFollow-up & Warnings:")
        for af in info.advice_followup:
            lines.append(f"- {af.details or af.name}")
    else:
        lines.append("\nFollow-up: As advised by clinician or if symptoms worsen.")

    return "\n".join(lines)


def identify_missing_or_uncertain_information(info: ExtractedClinicalInformation) -> List[str]:
    """Identifies missing clinical baseline parameters and tentative findings."""
    missing = []

    # Vital sign gaps
    v = info.vital_signs
    if v.blood_pressure_systolic is None and not v.blood_pressure_str:
        missing.append("Blood pressure not recorded.")
    if v.heart_rate_bpm is None:
        missing.append("Heart rate / pulse not recorded.")
    if v.temperature_f is None and v.temperature_c is None and not v.temperature_str:
        missing.append("Body temperature not recorded.")
    if v.spo2_percent is None:
        missing.append("Oxygen saturation (SpO2) not recorded.")

    # Allergies gap
    if not info.allergies:
        missing.append("Allergy status not explicitly confirmed.")

    # Medication gaps
    for m in info.medications:
        if not m.dosage:
            missing.append(f"Medication '{m.name}' dosage unspecified.")
        if not m.frequency:
            missing.append(f"Medication '{m.name}' frequency unspecified.")

    # Uncertain diagnoses
    for d in info.diagnoses:
        if d.status == FindingStatus.UNCERTAIN:
            missing.append(f"Diagnostic impression '{d.name}' is tentative and requires clinical confirmation.")

    return missing


def generate_soap_note(
    info: ExtractedClinicalInformation,
    demographics_summary: str = ""
) -> SOAPNote:
    """Generates a complete, deterministic draft SOAP note from extracted clinical data."""
    subjective_text = format_subjective_section(info, demographics_summary)
    objective_text = format_objective_section(info.vital_signs)
    assessment_text = format_assessment_section(info)
    plan_text = format_plan_section(info)
    missing_items = identify_missing_or_uncertain_information(info)

    return SOAPNote(
        subjective=subjective_text,
        objective=objective_text,
        assessment=assessment_text,
        plan=plan_text,
        missing_or_uncertain_info=missing_items,
        review_status=ReviewStatus.DRAFT_PENDING_REVIEW
    )
