"""HL7 FHIR R4 Exporter for Healthcare AI Clinical Documentation Assistant.

Generates standard HL7 FHIR (Fast Healthcare Interoperability Resources) R4 JSON documents
from ConsultationRecord instances for hospital Electronic Medical Record (EMR/EHR) interoperability.

Conforms to FHIR R4 Specification:
- Bundle (document type) with Composition, Patient, Condition, Observation,
  MedicationStatement, and AllergyIntolerance resources.
- Standard LOINC codes for SOAP sections and clinical vital signs.
- Standard ICD-10-CM coding for clinical conditions and symptoms.
- Explicit verificationStatus: "confirmed" for present findings vs "refuted" for negated findings.
- Clinician review status reflection: preliminary (draft) vs final (approved).
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid

from src.models.patient_record import (
    ConsultationRecord,
    FindingStatus,
    ReviewStatus,
    ExtractedFinding,
    MedicationFinding
)
from src.processing.icd10_coding import suggest_icd10_codes

logger = logging.getLogger(__name__)


def _sanitize_id(val: Optional[str], default_prefix: str = "item") -> str:
    r"""Ensures an ID is non-empty and contains only valid FHIR ID characters [A-Za-z0-9\-\.]{1,64}."""
    if not val:
        return f"{default_prefix}-{uuid.uuid4().hex[:8]}"
    cleaned = "".join(c if c.isalnum() or c in "-." else "-" for c in val).strip("-.")
    return cleaned[:64] if cleaned else f"{default_prefix}-{uuid.uuid4().hex[:8]}"


def build_fhir_bundle(record: ConsultationRecord) -> Dict[str, Any]:
    """
    Transforms a ConsultationRecord into a fully-structured HL7 FHIR R4 Document Bundle.

    Returns:
        Python dictionary conforming to FHIR R4 Bundle schema.
    """
    consult_id = _sanitize_id(record.consultation_id, "consult")
    raw_pid = record.demographics.patient_id or "UNKNOWN"
    patient_id = _sanitize_id(raw_pid, "patient")
    patient_ref = f"Patient/{patient_id}"

    entries: List[Dict[str, Any]] = []

    # Map review status to FHIR Composition status
    is_approved = record.review_status == ReviewStatus.REVIEWED_APPROVED
    composition_status = "final" if is_approved else "preliminary"

    # 1. Composition Resource (The Document Header & SOAP Note Sections)
    comp_sections: List[Dict[str, Any]] = [
        {
            "title": "Subjective (S)",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "61150-9",
                        "display": "Subjective narrative"
                    }
                ]
            },
            "text": {
                "status": "generated",
                "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\"><p>{record.soap_note.subjective or 'No subjective complaints recorded.'}</p></div>"
            }
        },
        {
            "title": "Objective (O)",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "61149-1",
                        "display": "Objective narrative"
                    }
                ]
            },
            "text": {
                "status": "generated",
                "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\"><p>{record.soap_note.objective or 'No objective vitals recorded.'}</p></div>"
            }
        },
        {
            "title": "Assessment (A)",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "51848-0",
                        "display": "Assessment narrative"
                    }
                ]
            },
            "text": {
                "status": "generated",
                "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\"><p>{record.soap_note.assessment or 'Assessment not documented.'}</p></div>"
            }
        },
        {
            "title": "Plan (P)",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "18776-5",
                        "display": "Plan of treatment"
                    }
                ]
            },
            "text": {
                "status": "generated",
                "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\"><p>{record.soap_note.plan or 'No explicit plan documented.'}</p></div>"
            }
        }
    ]

    # Add missing info section if present
    if record.soap_note.missing_or_uncertain_info:
        missing_bullets = "".join(f"<li>{item}</li>" for item in record.soap_note.missing_or_uncertain_info)
        comp_sections.append({
            "title": "Data Gaps & Uncertainties",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "86645-9",
                        "display": "Discrepancies and uncertainties narrative"
                    }
                ]
            },
            "text": {
                "status": "generated",
                "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\"><ul>{missing_bullets}</ul></div>"
            }
        })

    composition_resource = {
        "resourceType": "Composition",
        "id": f"comp-{consult_id}",
        "status": composition_status,
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11506-3",
                    "display": "Progress note"
                }
            ],
            "text": "Outpatient Consultation Note"
        },
        "category": [
            {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "LP173121-7",
                        "display": "Clinical documentation"
                    }
                ]
            }
        ],
        "subject": {
            "reference": patient_ref,
            "display": f"Patient ID: {raw_pid}"
        },
        "date": record.created_at,
        "title": "Consultation Encounter SOAP Record",
        "section": comp_sections
    }

    entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": composition_resource
    })

    # 2. Patient Resource
    gender_map = {"male": "male", "m": "male", "female": "female", "f": "female"}
    fhir_gender = gender_map.get((record.demographics.gender or "").strip().lower(), "unknown")

    patient_resource: Dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [
            {
                "system": "urn:oid:healthcare-asr:patient-id",
                "value": raw_pid
            }
        ],
        "gender": fhir_gender
    }

    if record.demographics.age is not None:
        patient_resource["extension"] = [
            {
                "url": "http://hl7.org/fhir/StructureDefinition/patient-age",
                "valueQuantity": {
                    "value": record.demographics.age,
                    "unit": "years",
                    "system": "http://unitsofmeasure.org",
                    "code": "a"
                }
            }
        ]

    entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": patient_resource
    })

    # Pre-index ICD-10 suggestions for coding Conditions
    icd10_matches = suggest_icd10_codes(record.extracted_info)
    icd_map: Dict[str, Any] = {}
    for m in icd10_matches:
        icd_map[m.matched_term.lower()] = m

    # 3. Condition Resources (Symptoms, Conditions, Diagnoses)
    def add_condition(finding: ExtractedFinding, category_code: str):
        is_negated = finding.status == FindingStatus.EXPLICITLY_ABSENT
        clinical_status = "resolved" if is_negated else "active"
        verif_status = "refuted" if is_negated else "confirmed"

        code_codings = []
        matched_icd = icd_map.get(finding.name.lower())
        if matched_icd and not is_negated:
            code_codings.append({
                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                "code": matched_icd.code,
                "display": matched_icd.description
            })

        cond_resource = {
            "resourceType": "Condition",
            "id": f"cond-{uuid.uuid4().hex[:8]}",
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": clinical_status
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": verif_status
                    }
                ]
            },
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                            "code": category_code
                        }
                    ]
                }
            ],
            "code": {
                "coding": code_codings,
                "text": finding.name
            },
            "subject": {"reference": patient_ref},
            "recordedDate": record.created_at
        }
        if finding.duration:
            cond_resource["note"] = [{"text": f"Duration: {finding.duration}"}]

        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": cond_resource
        })

    for sym in record.extracted_info.symptoms:
        add_condition(sym, "symptom")

    for cond in record.extracted_info.existing_conditions:
        add_condition(cond, "problem-list-item")

    for diag in record.extracted_info.diagnoses:
        add_condition(diag, "encounter-diagnosis")

    # 4. Observation Resources for Vital Signs
    vitals = record.extracted_info.vital_signs

    def add_vital_obs(code: str, display: str, value_num: Optional[float], unit: str, ucum_code: str, raw_str: Optional[str] = None):
        if value_num is None and not raw_str:
            return

        obs_resource: Dict[str, Any] = {
            "resourceType": "Observation",
            "id": f"obs-{uuid.uuid4().hex[:8]}",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": code,
                        "display": display
                    }
                ],
                "text": display
            },
            "subject": {"reference": patient_ref},
            "effectiveDateTime": record.created_at
        }

        if value_num is not None:
            obs_resource["valueQuantity"] = {
                "value": value_num,
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": ucum_code
            }
        elif raw_str:
            obs_resource["valueString"] = raw_str

        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": obs_resource
        })

    # Blood Pressure (Systolic + Diastolic component observation)
    if vitals.blood_pressure_systolic is not None or vitals.blood_pressure_diastolic is not None or vitals.blood_pressure_str:
        bp_obs: Dict[str, Any] = {
            "resourceType": "Observation",
            "id": f"obs-bp-{uuid.uuid4().hex[:8]}",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "85354-9",
                        "display": "Blood pressure panel with all children optional"
                    }
                ],
                "text": "Blood Pressure"
            },
            "subject": {"reference": patient_ref},
            "effectiveDateTime": record.created_at
        }

        components = []
        if vitals.blood_pressure_systolic is not None:
            components.append({
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic blood pressure"}]
                },
                "valueQuantity": {"value": vitals.blood_pressure_systolic, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
            })
        if vitals.blood_pressure_diastolic is not None:
            components.append({
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic blood pressure"}]
                },
                "valueQuantity": {"value": vitals.blood_pressure_diastolic, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
            })

        if components:
            bp_obs["component"] = components
        elif vitals.blood_pressure_str:
            bp_obs["valueString"] = vitals.blood_pressure_str

        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": bp_obs
        })

    # Heart Rate
    add_vital_obs("8867-4", "Heart rate", vitals.heart_rate_bpm, "/min", "/min")

    # Body Temperature
    temp_val = vitals.temperature_f if vitals.temperature_f is not None else vitals.temperature_c
    temp_unit = "[degF]" if vitals.temperature_f is not None else "Cel"
    temp_disp = "deg F" if vitals.temperature_f is not None else "deg C"
    add_vital_obs("8310-5", "Body temperature", temp_val, temp_disp, temp_unit, vitals.temperature_str)

    # Oxygen Saturation SpO2
    add_vital_obs("2708-6", "Oxygen saturation in Arterial blood by Pulse oximetry", vitals.spo2_percent, "%", "%")

    # Blood Glucose
    add_vital_obs("2339-0", "Glucose [Mass/volume] in Blood", vitals.blood_glucose_mg_dl, "mg/dL", "mg/dL", vitals.blood_glucose_str)

    # 5. MedicationStatement Resources
    for med in record.extracted_info.medications:
        med_resource = {
            "resourceType": "MedicationStatement",
            "id": f"med-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "medicationCodeableConcept": {
                "text": med.name
            },
            "subject": {"reference": patient_ref},
            "effectiveDateTime": record.created_at
        }
        dosage_parts = []
        if med.dosage:
            dosage_parts.append(f"Dose: {med.dosage}")
        if med.frequency:
            dosage_parts.append(f"Frequency: {med.frequency}")
        route_val = getattr(med, "route", None)
        if route_val:
            dosage_parts.append(f"Route: {route_val}")

        if dosage_parts:
            med_resource["dosage"] = [{"text": ", ".join(dosage_parts)}]

        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": med_resource
        })

    # 6. AllergyIntolerance Resources
    for alg in record.extracted_info.allergies:
        alg_resource = {
            "resourceType": "AllergyIntolerance",
            "id": f"alg-{uuid.uuid4().hex[:8]}",
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                        "code": "active"
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
                        "code": "confirmed"
                    }
                ]
            },
            "code": {
                "text": alg.name
            },
            "patient": {"reference": patient_ref},
            "recordedDate": record.created_at
        }
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": alg_resource
        })

    # Assemble FHIR Bundle
    bundle: Dict[str, Any] = {
        "resourceType": "Bundle",
        "id": f"bundle-{consult_id}",
        "type": "document",
        "timestamp": record.created_at,
        "entry": entries
    }

    return bundle


def export_fhir_json(
    record: ConsultationRecord,
    output_path: Union[str, Path],
    indent: int = 2
) -> Path:
    """
    Exports a ConsultationRecord as an HL7 FHIR R4 JSON document.

    Args:
        record: The ConsultationRecord instance.
        output_path: Destination path for the .json file.
        indent: Indentation spaces for formatted JSON.

    Returns:
        Path to the written JSON file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    bundle = build_fhir_bundle(record)
    content = json.dumps(bundle, indent=indent, ensure_ascii=False)
    dest.write_text(content, encoding="utf-8")

    logger.info(f"Successfully exported FHIR R4 document bundle to: {dest}")
    return dest


def validate_fhir_bundle(bundle_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates structural conformance of a FHIR document bundle.

    Returns:
        (is_valid, error_list)
    """
    errors: List[str] = []

    if bundle_dict.get("resourceType") != "Bundle":
        errors.append('Root resourceType must be "Bundle"')

    if bundle_dict.get("type") != "document":
        errors.append('Bundle type must be "document"')

    entries = bundle_dict.get("entry", [])
    if not isinstance(entries, list) or len(entries) == 0:
        errors.append("Bundle must contain non-empty entries list")
        return False, errors

    # First entry in a FHIR document bundle must be a Composition resource
    first_res = entries[0].get("resource", {})
    if first_res.get("resourceType") != "Composition":
        errors.append('First entry in a FHIR document bundle must be a "Composition" resource')

    # Verify every resource has resourceType and id
    for idx, e in enumerate(entries):
        r = e.get("resource", {})
        if not r.get("resourceType"):
            errors.append(f"Entry {idx} is missing resourceType")
        if not r.get("id"):
            errors.append(f"Entry {idx} ({r.get('resourceType')}) is missing id")

    return len(errors) == 0, errors
