"""Report export module for HTML and PDF clinical documentation.

DISCLAIMER:
All generated reports are marked with mandatory clinician-review warnings.
No output constitutes final medical records until reviewed, verified, and signed by a licensed clinician.
"""

from __future__ import annotations
import html
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from src.models.patient_record import ConsultationRecord, FindingStatus
from src.processing.icd10_coding import suggest_icd10_codes

logger = logging.getLogger(__name__)


@dataclass
class ClinicBranding:
    """Custom clinic branding metadata for report headers."""
    clinic_name: str = "Apex Healthcare Clinic"
    department: str = "Ambulatory Care & Family Medicine"
    physician_name: str = "Dr. Attending Physician, MD"
    facility_address: str = "100 Medical Center Blvd, Suite 400"
    contact_info: str = "Tel: (555) 019-2834 | records@apexclinic.health"


def generate_html_report(record: ConsultationRecord, branding: Optional[ClinicBranding] = None) -> str:
    """Generates a styled, print-friendly HTML clinical document from a ConsultationRecord."""
    brand = branding or ClinicBranding()
    demo = record.demographics
    soap = record.soap_note
    info = record.extracted_info
    vitals = info.vital_signs

    age_str = f"{demo.age} years" if demo.age else "Unspecified"
    gender_str = demo.gender if demo.gender else "Unspecified"
    consult_dt = demo.consultation_datetime if demo.consultation_datetime else record.created_at

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Documentation - {html.escape(record.consultation_id)}</title>
    <style>
        :root {{
            --primary-color: #0f4c81;
            --secondary-color: #2b6cb0;
            --bg-color: #f7fafc;
            --card-bg: #ffffff;
            --text-color: #2d3748;
            --border-color: #e2e8f0;
            --warning-bg: #fff5f5;
            --warning-border: #feb2b2;
            --warning-text: #c53030;
            --success-color: #276749;
            --absent-color: #718096;
            --uncertain-color: #dd6b20;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            margin: 0;
            padding: 24px;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: var(--card-bg);
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            padding: 32px;
            border: 1px solid var(--border-color);
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}

        .header h1 {{
            margin: 0;
            color: var(--primary-color);
            font-size: 24px;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            background: #edf2f7;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 24px;
            font-size: 14px;
        }}

        .warning-banner {{
            background-color: var(--warning-bg);
            border-left: 4px solid var(--warning-text);
            padding: 16px;
            margin-bottom: 24px;
            border-radius: 4px;
            color: var(--warning-text);
            font-weight: 600;
            font-size: 14px;
        }}

        .section-title {{
            color: var(--primary-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 6px;
            margin-top: 24px;
            margin-bottom: 12px;
            font-size: 18px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .soap-block {{
            white-space: pre-wrap;
            background: #faf5ff;
            border: 1px solid #e9d8fd;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 16px;
            font-family: inherit;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            font-size: 14px;
        }}

        th, td {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: #edf2f7;
            color: var(--primary-color);
            font-weight: 600;
        }}

        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .badge-present {{ background: #c6f6d5; color: #22543d; }}
        .badge-absent {{ background: #e2e8f0; color: #4a5568; }}
        .badge-uncertain {{ background: #feebc8; color: #742a2a; }}

        .transcript-box {{
            background: #1a202c;
            color: #e2e8f0;
            padding: 16px;
            border-radius: 6px;
            font-family: "Courier New", Courier, monospace;
            font-size: 13px;
            max-height: 300px;
            overflow-y: auto;
        }}

        .segment-item {{
            margin-bottom: 8px;
        }}

        .role-doctor {{ color: #63b3ed; font-weight: bold; }}
        .role-patient {{ color: #68d391; font-weight: bold; }}
        .role-unknown {{ color: #cbd5e0; font-weight: bold; }}

        @media print {{
            body {{ background: #fff; padding: 0; }}
            .container {{ box-shadow: none; border: none; padding: 0; }}
            .transcript-box {{ max-height: none; background: #fff; color: #000; border: 1px solid #ccc; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Clinic Letterhead Header -->
        <div style="border-bottom: 2px solid var(--primary-color); padding-bottom: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <div style="font-size: 22px; font-weight: bold; color: var(--primary-color); letter-spacing: 0.5px;">{html.escape(brand.clinic_name)}</div>
                <div style="font-size: 13px; color: #4a5568; margin-top: 3px;">{html.escape(brand.department)} &bull; {html.escape(brand.physician_name)}</div>
                <div style="font-size: 12px; color: #718096; margin-top: 2px;">{html.escape(brand.facility_address)} &bull; {html.escape(brand.contact_info)}</div>
            </div>
            <div>
                <span class="badge badge-uncertain">{html.escape(record.review_status.value.upper())}</span>
            </div>
        </div>

        <div class="header">
            <div>
                <h1>Clinical Consultation Record</h1>
                <div style="font-size: 13px; color: #718096;">Consultation ID: {html.escape(record.consultation_id)}</div>
            </div>
        </div>

        <div class="warning-banner">
            ⚠️ {html.escape(soap.review_warning)}
        </div>

        <div class="meta-grid">
            <div><strong>Patient ID:</strong> {html.escape(str(demo.patient_id or 'N/A'))}</div>
            <div><strong>Age / Gender:</strong> {html.escape(age_str)} / {html.escape(gender_str)}</div>
            <div><strong>Consultation Date:</strong> {html.escape(consult_dt)}</div>
            <div><strong>Schema Version:</strong> 1.0</div>
        </div>

        <div class="section-title">SOAP Note — Subjective (S)</div>
        <div class="soap-block">{html.escape(soap.subjective or 'No subjective data recorded.')}</div>

        <div class="section-title">SOAP Note — Objective (O)</div>
        <div class="soap-block">{html.escape(soap.objective or 'No objective vitals recorded.')}</div>

        <div class="section-title">SOAP Note — Assessment (A)</div>
        <div class="soap-block">{html.escape(soap.assessment or 'Assessment not documented.')}</div>

        <div class="section-title">SOAP Note — Plan (P)</div>
        <div class="soap-block">{html.escape(soap.plan or 'No explicit plan documented.')}</div>

        {f'''
        <div class="section-title">Missing or Uncertain Clinical Information</div>
        <ul style="background: #fffaf0; border: 1px solid #fbd38d; padding: 16px 32px; border-radius: 6px; color: #742a2a;">
            {"".join(f"<li>{html.escape(item)}</li>" for item in soap.missing_or_uncertain_info)}
        </ul>
        ''' if soap.missing_or_uncertain_info else ''}

        <div class="section-title">Extracted Clinical Findings</div>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Finding Name</th>
                    <th>Status</th>
                    <th>Duration / Details</th>
                </tr>
            </thead>
            <tbody>
    """

    all_findings = (
        info.symptoms + info.relevant_history + info.existing_conditions +
        info.allergies + info.diagnoses + info.investigations +
        info.treatment_plans + info.advice_followup
    )

    if all_findings:
        for f in all_findings:
            status_val = f.status.value if isinstance(f.status, FindingStatus) else str(f.status)
            badge_cls = "badge-present"
            if status_val == "explicitly_absent":
                badge_cls = "badge-absent"
            elif status_val == "uncertain":
                badge_cls = "badge-uncertain"

            html_code += f"""
                <tr>
                    <td><strong>{html.escape(f.category.title())}</strong></td>
                    <td>{html.escape(f.name)}</td>
                    <td><span class="badge {badge_cls}">{html.escape(status_val.replace('_', ' '))}</span></td>
                    <td>{html.escape(f.duration or f.details or '-')}</td>
                </tr>
            """
    else:
        html_code += """
            <tr><td colspan="4" style="text-align: center; color: #718096;">No structured clinical findings extracted.</td></tr>
        """

    html_code += """
            </tbody>
        </table>
    """

    if info.medications:
        html_code += """
        <div class="section-title">Extracted Medications</div>
        <table>
            <thead>
                <tr>
                    <th>Medication Name</th>
                    <th>Dosage</th>
                    <th>Frequency</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        for m in info.medications:
            html_code += f"""
                <tr>
                    <td><strong>{html.escape(m.name)}</strong></td>
                    <td>{html.escape(m.dosage or 'Unspecified')}</td>
                    <td>{html.escape(m.frequency or 'Unspecified')}</td>
                    <td><span class="badge badge-present">{html.escape(m.status.value.replace('_', ' '))}</span></td>
                </tr>
            """
        html_code += "</tbody></table>"

    # ICD-10 Diagnostic Coding Table
    icd_codes = suggest_icd10_codes(info)
    if icd_codes:
        html_code += """
        <div class="section-title">Suggested ICD-10-CM Diagnostic Codes (Draft)</div>
        <table>
            <thead>
                <tr>
                    <th style="width: 15%;">ICD-10 Code</th>
                    <th style="width: 45%;">Official Description</th>
                    <th style="width: 25%;">Matched Clinical Finding</th>
                    <th style="width: 15%;">Confidence</th>
                </tr>
            </thead>
            <tbody>
        """
        for cm in icd_codes:
            html_code += f"""
                <tr>
                    <td><strong style="color: #6b21a8;">{html.escape(cm.code)}</strong></td>
                    <td>{html.escape(cm.description)}</td>
                    <td>{html.escape(cm.matched_term)} <span style="font-size: 11px; color: #718096;">({html.escape(cm.category)})</span></td>
                    <td><span class="badge badge-present">{int(cm.confidence * 100)}%</span></td>
                </tr>
            """
        html_code += "</tbody></table>"

    if record.segments:
        html_code += """
        <div class="section-title">Recognized Dialogue & Speaker Roles</div>
        <div class="transcript-box">
        """
        for seg in record.segments:
            role_val = seg.speaker_role.value if hasattr(seg.speaker_role, 'value') else str(seg.speaker_role)
            role_cls = f"role-{role_val.lower()}"
            html_code += f"""
            <div class="segment-item">
                <span class="{role_cls}">[{html.escape(role_val.upper())}]</span>
                <span style="color: #a0aec0;">[{seg.start_time:.1f}s - {seg.end_time:.1f}s]:</span>
                {html.escape(seg.text)}
            </div>
            """
        html_code += "</div>"

    html_code += """
        <div style="margin-top: 32px; text-align: center; font-size: 12px; color: #a0aec0; border-top: 1px solid #e2e8f0; padding-top: 16px;">
            Healthcare AI Clinical Documentation Assistant &bull; Windows Local CPU Execution &bull; Strict Privacy Enforced
        </div>
    </div>
</body>
</html>
    """

    return html_code


def export_html_report(
    record: ConsultationRecord,
    output_path: Union[str, Path],
    branding: Optional[ClinicBranding] = None
) -> Path:
    """Saves the clinical consultation record as a styled HTML file with clinic branding."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_content = generate_html_report(record, branding=branding)
    path.write_text(html_content, encoding="utf-8")
    logger.info(f"Successfully exported HTML report to: {path}")
    return path


def export_pdf_report(
    record: ConsultationRecord,
    output_path: Union[str, Path],
    branding: Optional[ClinicBranding] = None
) -> Path:
    """Generates a professional PDF document using ReportLab with customized clinic letterhead."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except ImportError as e:
        raise RuntimeError("ReportLab library is required for PDF export. Install via: pip install reportlab") from e

    brand = branding or ClinicBranding()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f4c81"),
        spaceAfter=6
    )

    sub_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#718096"),
        spaceAfter=12
    )

    warning_style = ParagraphStyle(
        "WarningBox",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#c53030"),
        backColor=colors.HexColor("#fff5f5"),
        borderColor=colors.HexColor("#feb2b2"),
        borderWidth=1,
        borderPadding=8,
        spaceAfter=14
    )

    heading2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f4c81"),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=8
    )

    story = []

    # Clinic Letterhead
    clinic_title_style = ParagraphStyle(
        "ClinicTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f4c81"),
        spaceAfter=2
    )

    clinic_sub_style = ParagraphStyle(
        "ClinicSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=1
    )

    clinic_contact_style = ParagraphStyle(
        "ClinicContact",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#718096"),
        spaceAfter=6
    )

    story.append(Paragraph(html.escape(brand.clinic_name), clinic_title_style))
    story.append(Paragraph(f"{html.escape(brand.department)} &bull; {html.escape(brand.physician_name)}", clinic_sub_style))
    story.append(Paragraph(f"{html.escape(brand.facility_address)} &bull; {html.escape(brand.contact_info)}", clinic_contact_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f4c81"), spaceBefore=2, spaceAfter=8))

    # Title & Metadata Header
    story.append(Paragraph("Healthcare AI Clinical Documentation Report", title_style))
    story.append(Paragraph(f"Consultation ID: {record.consultation_id} | Created: {record.created_at[:19]}", sub_style))

    # Warning Box
    story.append(Paragraph(f"⚠️ {record.soap_note.review_warning}", warning_style))

    # Patient Metadata Table
    demo = record.demographics
    meta_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {demo.patient_id or 'N/A'}", body_style),
            Paragraph(f"<b>Age:</b> {demo.age or 'Unspecified'}", body_style),
            Paragraph(f"<b>Gender:</b> {demo.gender or 'Unspecified'}", body_style),
            Paragraph(f"<b>Status:</b> {record.review_status.value.upper()}", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[130, 130, 130, 150])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#edf2f7')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # SOAP Sections
    soap = record.soap_note
    sections = [
        ("SOAP Note — Subjective (S)", soap.subjective),
        ("SOAP Note — Objective (O)", soap.objective),
        ("SOAP Note — Assessment (A)", soap.assessment),
        ("SOAP Note — Plan (P)", soap.plan),
    ]

    for title, content in sections:
        story.append(Paragraph(title, heading2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e0"), spaceAfter=4))
        formatted_content = html.escape(content or "Not documented.").replace("\n", "<br/>")
        story.append(Paragraph(formatted_content, body_style))

    # Missing & Uncertain Information
    if soap.missing_or_uncertain_info:
        story.append(Paragraph("Missing or Uncertain Clinical Information", heading2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#feb2b2"), spaceAfter=4))
        for item in soap.missing_or_uncertain_info:
            story.append(Paragraph(f"• {html.escape(item)}", body_style))

    # Extracted Findings Table
    info = record.extracted_info
    all_findings = (
        info.symptoms + info.relevant_history + info.existing_conditions +
        info.allergies + info.diagnoses + info.investigations +
        info.treatment_plans + info.advice_followup
    )

    if all_findings:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Extracted Clinical Findings", heading2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e0"), spaceAfter=6))

        table_data = [["Category", "Finding Name", "Status", "Details"]]
        for f in all_findings:
            table_data.append([
                Paragraph(f.category.title(), body_style),
                Paragraph(f.name, body_style),
                Paragraph(f.status.value.replace("_", " ").title(), body_style),
                Paragraph(f.duration or f.details or "-", body_style)
            ])

        findings_table = Table(table_data, colWidths=[110, 150, 120, 160])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f4c81')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ]))
        story.append(findings_table)

    # ICD-10 Coding Table for PDF
    icd_matches = suggest_icd10_codes(info)
    if icd_matches:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Suggested ICD-10-CM Diagnostic Codes (Draft)", heading2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#6b21a8"), spaceAfter=6))

        icd_table_data = [["Code", "Official Description", "Matched Clinical Finding", "Confidence"]]
        for cm in icd_matches:
            icd_table_data.append([
                Paragraph(f"<b>{cm.code}</b>", body_style),
                Paragraph(cm.description, body_style),
                Paragraph(f"{cm.matched_term} ({cm.category})", body_style),
                Paragraph(f"{int(cm.confidence * 100)}%", body_style)
            ])

        icd_table = Table(icd_table_data, colWidths=[80, 230, 160, 70])
        icd_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#6b21a8')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ]))
        story.append(icd_table)

    # Footer note
    story.append(Spacer(1, 14))
    story.append(Paragraph("Healthcare AI Clinical Documentation Assistant • Strict Privacy & Clinician Verification Enforced", sub_style))

    doc.build(story)
    logger.info(f"Successfully exported PDF report to: {path}")
    return path
