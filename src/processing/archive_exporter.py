"""Consultation Bundle Archive Exporter for Healthcare AI Clinical Documentation Assistant.

Packages complete consultation encounters into standardized ZIP archives containing:
- Consultation Record JSON (schema v1.0)
- HTML Clinical Report
- PDF Clinical Report
- Audio WAV Recording (when available)
- SHA-256 Cryptographic Integrity Manifest for medico-legal audit and tamper verification.
"""

from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Optional, Dict, Any, Tuple
import zipfile

from src.models.patient_record import ConsultationRecord
from src.processing.exporter import export_html_report, export_pdf_report, ClinicBranding

logger = logging.getLogger(__name__)


def compute_sha256_bytes(data: bytes) -> str:
    """Computes SHA-256 hexadecimal hash string of bytes data."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def compute_sha256_file(file_path: Path) -> str:
    """Computes SHA-256 hexadecimal hash string of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_consultation_zip(
    record: ConsultationRecord,
    output_dir: Optional[Path] = None,
    audio_path: Optional[Path] = None,
    branding: Optional[ClinicBranding] = None,
    zip_filename: Optional[str] = None
) -> Path:
    """
    Exports a complete consultation package as a ZIP archive with SHA-256 checksum manifest.

    Args:
        record: The ConsultationRecord instance to package.
        output_dir: Destination folder. Defaults to C:\\Users\\Aziza\\HEALTHCARE_ASR\\output.
        audio_path: Optional path to raw WAV recording file.
        branding: Optional clinic branding customization for HTML/PDF reports.
        zip_filename: Optional custom ZIP filename.

    Returns:
        Path to the generated ZIP archive.
    """
    if output_dir is None:
        target_dir = Path(r"C:\Users\Aziza\HEALTHCARE_ASR\output")
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    patient_clean = (record.demographics.patient_id or "patient").replace(" ", "_")
    cid_clean = record.consultation_id[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if zip_filename:
        zip_path = target_dir / zip_filename
    else:
        zip_path = target_dir / f"consultation_bundle_{patient_clean}_{cid_clean}_{timestamp}.zip"

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # 1. Save record.json
        json_path = tmp_dir / "record.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(record.to_json(indent=2))

        # 2. Generate report.html
        html_path = tmp_dir / "report.html"
        export_html_report(record, output_path=html_path, branding=branding)

        # 3. Generate report.pdf
        pdf_path = tmp_dir / "report.pdf"
        export_pdf_report(record, output_path=pdf_path, branding=branding)

        # 4. Copy audio WAV if provided and exists
        audio_dest_in_zip: Optional[str] = None
        if audio_path and Path(audio_path).is_file():
            src_audio = Path(audio_path)
            copied_audio = tmp_dir / f"audio_{src_audio.name}"
            with open(src_audio, "rb") as rf, open(copied_audio, "wb") as wf:
                wf.write(rf.read())
            audio_dest_in_zip = copied_audio.name

        # 5. Build SHA-256 Manifest
        files_manifest: Dict[str, Dict[str, Any]] = {}

        bundle_files = [
            ("record.json", json_path),
            ("report.html", html_path),
            ("report.pdf", pdf_path)
        ]
        if audio_dest_in_zip:
            bundle_files.append((audio_dest_in_zip, tmp_dir / audio_dest_in_zip))

        for name_in_zip, local_fpath in bundle_files:
            file_bytes = local_fpath.read_bytes()
            files_manifest[name_in_zip] = {
                "sha256": compute_sha256_bytes(file_bytes),
                "size_bytes": len(file_bytes)
            }

        manifest_data = {
            "consultation_id": record.consultation_id,
            "patient_id": record.demographics.patient_id,
            "patient_age": record.demographics.age,
            "patient_gender": record.demographics.gender,
            "review_status": record.review_status.value,
            "export_timestamp": record.created_at,
            "generator": "Healthcare AI Clinical Documentation Assistant",
            "schema_version": "1.0",
            "verification_algorithm": "SHA-256",
            "integrity_notice": "Cryptographically verifiable consultation archive. Do not modify contents.",
            "files": files_manifest
        }

        manifest_path = tmp_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 6. Write all into ZIP
        with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname="record.json")
            zf.write(html_path, arcname="report.html")
            zf.write(pdf_path, arcname="report.pdf")
            if audio_dest_in_zip:
                zf.write(tmp_dir / audio_dest_in_zip, arcname=audio_dest_in_zip)
            zf.write(manifest_path, arcname="manifest.json")

    logger.info(f"Exported consultation ZIP bundle: {zip_path}")
    return zip_path


def verify_consultation_zip(zip_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verifies the integrity of a consultation ZIP archive against its SHA-256 manifest.

    Returns:
        (is_valid, message, manifest_dict)
    """
    if not Path(zip_path).is_file():
        return False, f"ZIP file not found: {zip_path}", {}

    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            namelist = zf.namelist()
            if "manifest.json" not in namelist:
                return False, "Archive is missing manifest.json", {}

            manifest_raw = zf.read("manifest.json").decode("utf-8")
            manifest = json.loads(manifest_raw)

            expected_files = manifest.get("files", {})
            if not expected_files:
                return False, "Manifest does not contain any file specifications", manifest

            for filename, file_info in expected_files.items():
                if filename not in namelist:
                    return False, f"Missing expected file in archive: {filename}", manifest

                file_data = zf.read(filename)
                computed_hash = compute_sha256_bytes(file_data)
                expected_hash = file_info.get("sha256", "")

                if computed_hash.lower() != expected_hash.lower():
                    return False, f"Checksum mismatch for {filename} (tampered or corrupted)", manifest

            return True, "All files verified against SHA-256 integrity manifest", manifest
    except zipfile.BadZipFile:
        return False, "File is not a valid or readable ZIP archive", {}
    except Exception as e:
        return False, f"Verification failed with exception: {e}", {}
