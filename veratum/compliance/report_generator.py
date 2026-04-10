"""Professional PDF compliance report generation for AI systems.

Generates regulatory-grade PDF reports suitable for CISOs, auditors,
and regulators. Combines Veratum receipt data, crosswalk analysis,
and DPIA findings into polished, data-driven compliance documentation.

Supported report types:
    - executive_summary: Key metrics, risk overview, compliance dashboard
    - audit_trail: Receipt-by-receipt detailed audit
    - framework_report: Framework-specific compliance mapping
    - dpia_pdf: Formatted DPIA with risk assessments

Examples:
    >>> from veratum.compliance import ComplianceReportGenerator
    >>>
    >>> generator = ComplianceReportGenerator(
    ...     org_name="Acme Corp",
    ...     report_title="Q1 2026 AI Compliance Report"
    ... )
    >>>
    >>> # Generate executive summary
    >>> pdf_bytes = generator.generate_executive_summary(
    ...     receipts=sdk.get_receipts(limit=500),
    ...     dpia={...}
    ... )
    >>> with open("compliance_report.pdf", "wb") as f:
    ...     f.write(pdf_bytes)
    >>>
    >>> # Convenience function
    >>> pdf_bytes = generate_report(
    ...     receipts=receipts,
    ...     org_name="Acme Corp",
    ...     report_type="framework_report",
    ...     framework="eu_ai_act"
    ... )
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether, PageTemplate, Frame
)
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

from .crosswalk import crosswalk, list_frameworks, FRAMEWORKS
from .dpia import DPIAGenerator, DPIAReport

logger = logging.getLogger("veratum.compliance.report_generator")


# ---------------------------------------------------------------------------
# Color scheme for reports
# ---------------------------------------------------------------------------

COLORS = {
    "veratum_blue": colors.HexColor("#0066CC"),
    "veratum_dark": colors.HexColor("#003366"),
    "compliant_green": colors.HexColor("#2E7D32"),
    "warning_amber": colors.HexColor("#F57C00"),
    "risk_red": colors.HexColor("#C62828"),
    "neutral_gray": colors.HexColor("#424242"),
    "light_gray": colors.HexColor("#F5F5F5"),
    "border_gray": colors.HexColor("#BDBDBD"),
}


# ---------------------------------------------------------------------------
# Page template with headers and footers
# ---------------------------------------------------------------------------

class VeratumPageTemplate(PageTemplate):
    """Custom page template with VERATUM branding and footers."""

    def __init__(self, page_height, page_width, org_name, report_title,
                 generated_at):
        self.org_name = org_name
        self.report_title = report_title
        self.generated_at = generated_at

        # Main content frame
        frame = Frame(
            0.5 * inch, 0.5 * inch,
            page_width - inch, page_height - 1 * inch,
            leftPadding=0, rightPadding=0,
            topPadding=0.6 * inch, bottomPadding=0.6 * inch
        )

        PageTemplate.__init__(self, [frame], onPage=self._on_page)

    def _on_page(self, canvas, doc):
        """Draw header and footer on each page."""
        page_width = doc.pagesize[0]
        page_height = doc.pagesize[1]

        # Header background
        canvas.setFillColor(COLORS["veratum_dark"])
        canvas.rect(0, page_height - 0.5 * inch, page_width, 0.5 * inch, fill=1)

        # VERATUM branding in header
        canvas.setFont("Helvetica-Bold", 14)
        canvas.setFillColor(colors.white)
        canvas.drawString(0.5 * inch, page_height - 0.32 * inch, "VERATUM")

        canvas.setFont("Helvetica", 9)
        canvas.drawString(1.3 * inch, page_height - 0.32 * inch, self.report_title)

        # Footer with page numbers and timestamp
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(COLORS["neutral_gray"])

        # Page number on right
        page_num = getattr(self, '_pagenum', 1)
        canvas.drawRightString(
            page_width - 0.5 * inch,
            0.3 * inch,
            f"Page {doc.page}"
        )

        # Generated timestamp on left
        canvas.drawString(0.5 * inch, 0.3 * inch, f"Generated: {self.generated_at}")

        # Horizontal rule
        canvas.setStrokeColor(COLORS["border_gray"])
        canvas.setLineWidth(0.5)
        canvas.line(0.5 * inch, 0.45 * inch, page_width - 0.5 * inch, 0.45 * inch)


# ---------------------------------------------------------------------------
# ComplianceReportGenerator class
# ---------------------------------------------------------------------------

class ComplianceReportGenerator:
    """
    Generates professional PDF compliance reports.

    Combines receipt data, crosswalk analysis, and DPIA findings into
    polished reports suitable for regulatory submission and board review.

    Args:
        org_name: Organization name for report header.
        report_title: Report title (default: "AI Compliance Report").
    """

    def __init__(
        self,
        org_name: str,
        report_title: str = "AI Compliance Report",
    ):
        self.org_name = org_name
        self.report_title = report_title
        self.generated_at = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    def generate_executive_summary(
        self,
        receipts: List[Dict[str, Any]],
        dpia: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Generate executive summary PDF report.

        Includes key metrics dashboard, risk overview, compliance coverage by
        framework, top findings, and evidence integrity summary.

        Args:
            receipts: List of Veratum receipt dictionaries.
            dpia: DPIA report dictionary (optional, from DPIAGenerator).

        Returns:
            PDF content as bytes.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
        )

        # Build story
        story = []
        styles = getSampleStyleSheet()

        # Cover page
        story.extend(self._cover_page())
        story.append(PageBreak())

        # Key metrics dashboard
        story.extend(self._metrics_dashboard(receipts))
        story.append(Spacer(1, 0.3 * inch))

        # Risk overview
        if dpia:
            story.extend(self._risk_overview(dpia))
            story.append(Spacer(1, 0.3 * inch))

        # Compliance coverage by framework
        story.extend(self._framework_coverage(receipts))
        story.append(PageBreak())

        # Detailed findings
        story.extend(self._compliance_findings(receipts))
        story.append(Spacer(1, 0.3 * inch))

        # Evidence integrity
        story.extend(self._evidence_integrity(receipts))

        # Build PDF
        doc.build(
            story,
            onFirstPage=lambda c, d: self._draw_header_footer(c, d),
            onLaterPages=lambda c, d: self._draw_header_footer(c, d),
        )

        return buffer.getvalue()

    def generate_audit_trail(self, receipts: List[Dict[str, Any]]) -> bytes:
        """
        Generate detailed receipt-by-receipt audit trail.

        Lists each receipt with key fields, compliance status, and any
        issues or gaps identified.

        Args:
            receipts: List of Veratum receipt dictionaries.

        Returns:
            PDF content as bytes.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
        )

        story = []
        styles = getSampleStyleSheet()

        # Title
        story.extend(self._cover_page())
        story.append(PageBreak())

        # Section header
        story.append(self._section_header("Audit Trail"))
        story.append(Spacer(1, 0.2 * inch))

        # Summary stats
        summary_text = f"Total Receipts: {len(receipts)}"
        story.append(Paragraph(summary_text, styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        # Table of receipts
        table_data = [
            [
                "Timestamp",
                "Model",
                "Decision Type",
                "Human Review",
                "Compliance Score",
            ]
        ]

        for receipt in receipts[:100]:  # Limit to first 100 for readability
            timestamp = receipt.get("timestamp", "N/A")[:19]
            model = receipt.get("model", "N/A")[:20]
            decision_type = receipt.get("decision_type", "N/A")[:15]
            human_review = receipt.get("human_review_state", "N/A")[:10]

            # Calculate basic compliance score
            score = self._calculate_receipt_score(receipt)

            table_data.append([timestamp, model, decision_type, human_review, f"{score:.0%}"])

        table = Table(table_data, colWidths=[1.2 * inch, 1.2 * inch, 1.2 * inch, 1 * inch, 1.1 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["veratum_dark"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border_gray"]),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["light_gray"]]),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
        ]))

        story.append(table)
        if len(receipts) > 100:
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(
                f"<i>Showing first 100 of {len(receipts)} receipts</i>",
                styles["Normal"]
            ))

        doc.build(
            story,
            onFirstPage=lambda c, d: self._draw_header_footer(c, d),
            onLaterPages=lambda c, d: self._draw_header_footer(c, d),
        )

        return buffer.getvalue()

    def generate_framework_report(
        self,
        framework: str,
        receipts: List[Dict[str, Any]],
    ) -> bytes:
        """
        Generate framework-specific compliance report.

        Maps receipts to a single framework (e.g., "eu_ai_act") and shows
        requirement-by-requirement compliance status.

        Args:
            framework: Framework ID (e.g., "eu_ai_act", "nist_ai_rmf").
            receipts: List of Veratum receipt dictionaries.

        Returns:
            PDF content as bytes.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
        )

        story = []
        styles = getSampleStyleSheet()

        # Cover with framework-specific info
        story.extend(self._cover_page())
        story.append(PageBreak())

        # Framework title
        framework_name = self._get_framework_name(framework)
        story.append(self._section_header(f"{framework_name} Compliance"))
        story.append(Spacer(1, 0.2 * inch))

        # Run crosswalk analysis
        try:
            crosswalk_result = crosswalk(
                receipts[0] if receipts else {},
                frameworks=[framework],
                include_recommended=True,
            )
        except Exception as e:
            logger.error(f"Crosswalk failed: {e}")
            crosswalk_result = {}

        # Framework details
        if framework in crosswalk_result.get("frameworks", {}):
            fw_data = crosswalk_result["frameworks"][framework]

            # Overall score
            score = fw_data.get("score", 0)
            status_text = self._get_compliance_status_text(score)
            color = self._get_risk_color(score)

            story.append(Paragraph(
                f"<b>Overall Compliance Score: {score:.1%}</b> ({status_text})",
                styles["Normal"]
            ))
            story.append(Spacer(1, 0.2 * inch))

            # Requirements table
            req_data = [["Requirement ID", "Description", "Status", "Score"]]
            for req_id, req_info in fw_data.get("requirements", {}).items():
                status = req_info.get("status", "unknown").upper()
                req_score = req_info.get("score", 0)
                description = req_info.get("description", "")[:40]

                req_data.append([
                    req_id,
                    description,
                    status,
                    f"{req_score:.0%}"
                ])

            if req_data:
                table = Table(req_data, colWidths=[1.5 * inch, 2.2 * inch, 0.9 * inch, 0.9 * inch])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COLORS["veratum_dark"]),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border_gray"]),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["light_gray"]]),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                ]))
                story.append(table)

            # Gaps
            gaps = fw_data.get("gaps", [])
            if gaps:
                story.append(PageBreak())
                story.append(self._section_header("Gaps and Missing Fields"))
                story.append(Spacer(1, 0.2 * inch))

                for gap in gaps:
                    missing = gap.get("missing_fields", [])
                    if missing:
                        story.append(Paragraph(
                            f"<b>{gap.get('requirement', 'Unknown')}</b>: "
                            f"Missing {', '.join(missing)}",
                            styles["Normal"]
                        ))
                        story.append(Spacer(1, 0.1 * inch))

        doc.build(
            story,
            onFirstPage=lambda c, d: self._draw_header_footer(c, d),
            onLaterPages=lambda c, d: self._draw_header_footer(c, d),
        )

        return buffer.getvalue()

    def generate_dpia_pdf(self, dpia_data: Dict[str, Any]) -> bytes:
        """
        Render DPIA as a formatted PDF report.

        Converts DPIAReport structure into a professional PDF with
        risk matrices, mitigation tables, and evidence references.

        Args:
            dpia_data: DPIA report dictionary from DPIAGenerator.generate().

        Returns:
            PDF content as bytes.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
        )

        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=COLORS["veratum_dark"],
            spaceAfter=6,
            alignment=TA_CENTER,
        )
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("DPIA Report", title_style))
        story.append(Spacer(1, 0.1 * inch))

        # Header info
        system_name = dpia_data.get("system_name", "Unknown System")
        story.append(Paragraph(f"<b>System:</b> {system_name}", styles["Normal"]))
        story.append(Paragraph(
            f"<b>Data Controller:</b> {dpia_data.get('data_controller', 'N/A')}",
            styles["Normal"]
        ))
        story.append(Paragraph(
            f"<b>Generated:</b> {dpia_data.get('generated_at', 'N/A')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.3 * inch))

        # Risk level summary
        risk_level = dpia_data.get("overall_risk_level", "unknown").upper()
        risk_color = self._get_risk_color_name(risk_level)
        story.append(Paragraph(
            f"<b>Overall Risk Level:</b> <font color='{risk_color}'>{risk_level}</font>",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.2 * inch))

        # Risks
        story.append(self._section_header("Identified Risks"))
        risks = dpia_data.get("risks", [])
        if risks:
            risk_table_data = [["Category", "Description", "Likelihood", "Severity", "Mitigation Status"]]
            for risk in risks:
                risk_table_data.append([
                    risk.get("category", ""),
                    risk.get("description", "")[:30],
                    risk.get("likelihood", ""),
                    risk.get("severity", ""),
                    risk.get("mitigation_status", ""),
                ])

            table = Table(risk_table_data, colWidths=[1.1 * inch, 1.8 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLORS["veratum_dark"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border_gray"]),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["light_gray"]]),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No risks identified.", styles["Normal"]))

        story.append(Spacer(1, 0.2 * inch))

        # Safeguards
        story.append(self._section_header("Safeguards and Mitigations"))
        safeguards = dpia_data.get("safeguards", [])
        if safeguards:
            for sg in safeguards:
                story.append(Paragraph(
                    f"<b>{sg.get('name', 'Safeguard')}:</b> {sg.get('description', '')}",
                    styles["Normal"]
                ))
                story.append(Spacer(1, 0.1 * inch))
        else:
            story.append(Paragraph("No safeguards documented.", styles["Normal"]))

        story.append(Spacer(1, 0.2 * inch))

        # Recommendations
        story.append(self._section_header("Recommendations"))
        recommendations = dpia_data.get("recommendations", [])
        if recommendations:
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", styles["Normal"]))
        else:
            story.append(Paragraph("No recommendations at this time.", styles["Normal"]))

        doc.build(
            story,
            onFirstPage=lambda c, d: self._draw_header_footer(c, d),
            onLaterPages=lambda c, d: self._draw_header_footer(c, d),
        )

        return buffer.getvalue()

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _cover_page(self) -> List:
        """Build cover page story elements."""
        styles = getSampleStyleSheet()
        story = []

        story.append(Spacer(1, 1 * inch))

        # Title
        title_style = ParagraphStyle(
            "CoverTitle",
            parent=styles["Heading1"],
            fontSize=28,
            textColor=COLORS["veratum_dark"],
            spaceAfter=12,
            alignment=TA_CENTER,
        )
        story.append(Paragraph(self.report_title, title_style))

        # Org name
        org_style = ParagraphStyle(
            "CoverOrg",
            parent=styles["Normal"],
            fontSize=16,
            textColor=COLORS["veratum_blue"],
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        story.append(Paragraph(self.org_name, org_style))

        story.append(Spacer(1, 0.5 * inch))

        # Generated date
        date_style = ParagraphStyle(
            "CoverDate",
            parent=styles["Normal"],
            fontSize=11,
            alignment=TA_CENTER,
        )
        story.append(Paragraph(f"<b>Generated:</b> {self.generated_at}", date_style))

        story.append(Spacer(1, 1.5 * inch))

        # VERATUM branding
        branding_style = ParagraphStyle(
            "Branding",
            parent=styles["Normal"],
            fontSize=10,
            textColor=COLORS["neutral_gray"],
            alignment=TA_CENTER,
        )
        story.append(Paragraph(
            "Compliance Report powered by VERATUM<br/>Evidence Engine for AI Systems",
            branding_style
        ))

        return story

    def _section_header(self, title: str) -> Paragraph:
        """Create a styled section header."""
        styles = getSampleStyleSheet()
        header_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=COLORS["veratum_dark"],
            spaceAfter=6,
            borderColor=COLORS["veratum_blue"],
            borderWidth=2,
            borderPadding=6,
        )
        return Paragraph(title, header_style)

    def _metrics_dashboard(self, receipts: List[Dict[str, Any]]) -> List:
        """Build metrics dashboard section."""
        styles = getSampleStyleSheet()
        story = []

        story.append(self._section_header("Key Metrics Dashboard"))
        story.append(Spacer(1, 0.2 * inch))

        # Calculate metrics
        total_receipts = len(receipts)
        verified_count = sum(1 for r in receipts if r.get("entry_hash") and r.get("signature"))
        verified_pct = (verified_count / total_receipts * 100) if total_receipts > 0 else 0

        human_review_count = sum(1 for r in receipts if r.get("human_review_state") == "approved")
        human_review_pct = (human_review_count / total_receipts * 100) if total_receipts > 0 else 0

        frameworks_covered = self._count_frameworks_covered(receipts)

        # Metrics table
        metrics_data = [
            ["Total Decisions", str(total_receipts)],
            ["Cryptographically Verified", f"{verified_pct:.1f}%"],
            ["Human Review Rate", f"{human_review_pct:.1f}%"],
            ["Frameworks Covered", str(frameworks_covered)],
        ]

        table = Table(metrics_data, colWidths=[2.5 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), COLORS["light_gray"]),
            ("BACKGROUND", (1, 0), (1, -1), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border_gray"]),
        ]))

        story.append(table)
        return story

    def _risk_overview(self, dpia: Dict[str, Any]) -> List:
        """Build risk overview section from DPIA."""
        styles = getSampleStyleSheet()
        story = []

        story.append(self._section_header("Risk Overview"))
        story.append(Spacer(1, 0.2 * inch))

        risk_level = dpia.get("overall_risk_level", "unknown").upper()
        story.append(Paragraph(f"<b>Overall Risk Level:</b> {risk_level}", styles["Normal"]))

        # Risk count by category
        risks = dpia.get("risks", [])
        if risks:
            risk_counts = {}
            for risk in risks:
                category = risk.get("category", "unknown")
                risk_counts[category] = risk_counts.get(category, 0) + 1

            story.append(Spacer(1, 0.1 * inch))
            for category, count in sorted(risk_counts.items()):
                story.append(Paragraph(f"• {category}: {count} risk(s)", styles["Normal"]))

        return story

    def _framework_coverage(self, receipts: List[Dict[str, Any]]) -> List:
        """Build framework coverage section."""
        styles = getSampleStyleSheet()
        story = []

        story.append(self._section_header("Compliance Coverage by Framework"))
        story.append(Spacer(1, 0.2 * inch))

        # Run crosswalk on a sample receipt
        if not receipts:
            story.append(Paragraph("No receipts to analyze.", styles["Normal"]))
            return story

        try:
            result = crosswalk(receipts[0], include_recommended=True)
        except Exception as e:
            logger.warning(f"Crosswalk failed: {e}")
            story.append(Paragraph("Framework analysis unavailable.", styles["Normal"]))
            return story

        frameworks = result.get("frameworks", {})
        if not frameworks:
            story.append(Paragraph("No frameworks analyzed.", styles["Normal"]))
            return story

        # Table of framework scores
        fw_data = [["Framework", "Score", "Status"]]
        for fw_id, fw_info in sorted(frameworks.items()):
            score = fw_info.get("score", 0)
            status = self._get_compliance_status_text(score)
            fw_name = fw_info.get("name", fw_id)[:30]

            fw_data.append([fw_name, f"{score:.0%}", status])

        table = Table(fw_data, colWidths=[2.2 * inch, 1 * inch, 1.3 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["veratum_dark"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border_gray"]),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["light_gray"]]),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
        ]))

        story.append(table)
        return story

    def _compliance_findings(self, receipts: List[Dict[str, Any]]) -> List:
        """Build detailed findings section."""
        styles = getSampleStyleSheet()
        story = []

        story.append(self._section_header("Findings and Observations"))
        story.append(Spacer(1, 0.2 * inch))

        # Calculate stats
        total = len(receipts)
        if total == 0:
            story.append(Paragraph("No receipts to analyze.", styles["Normal"]))
            return story

        # Decision types
        decision_types = {}
        for r in receipts:
            dt = r.get("decision_type", "unspecified")
            decision_types[dt] = decision_types.get(dt, 0) + 1

        story.append(Paragraph("<b>Decision Types:</b>", styles["Normal"]))
        for dt, count in sorted(decision_types.items(), key=lambda x: -x[1])[:5]:
            pct = count / total * 100
            story.append(Paragraph(f"• {dt}: {count} ({pct:.1f}%)", styles["Normal"]))

        story.append(Spacer(1, 0.2 * inch))

        # Models
        models = {}
        for r in receipts:
            model = r.get("model", "unknown")
            models[model] = models.get(model, 0) + 1

        story.append(Paragraph("<b>Models Used:</b>", styles["Normal"]))
        for model, count in sorted(models.items(), key=lambda x: -x[1])[:5]:
            story.append(Paragraph(f"• {model}: {count} calls", styles["Normal"]))

        return story

    def _evidence_integrity(self, receipts: List[Dict[str, Any]]) -> List:
        """Build evidence integrity section."""
        styles = getSampleStyleSheet()
        story = []

        story.append(self._section_header("Evidence Integrity Summary"))
        story.append(Spacer(1, 0.2 * inch))

        total = len(receipts)
        if total == 0:
            story.append(Paragraph("No receipts to verify.", styles["Normal"]))
            return story

        # Hash presence
        hashed = sum(1 for r in receipts if r.get("entry_hash"))
        hash_pct = hashed / total * 100

        # Signature presence
        signed = sum(1 for r in receipts if r.get("signature"))
        sig_pct = signed / total * 100

        # Merkle chain
        merkle = sum(1 for r in receipts if r.get("merkle_proof"))
        merkle_pct = merkle / total * 100

        story.append(Paragraph(
            f"<b>Hash Coverage:</b> {hash_pct:.1f}% ({hashed}/{total})",
            styles["Normal"]
        ))
        story.append(Paragraph(
            f"<b>Digital Signatures:</b> {sig_pct:.1f}% ({signed}/{total})",
            styles["Normal"]
        ))
        story.append(Paragraph(
            f"<b>Merkle Chain Proofs:</b> {merkle_pct:.1f}% ({merkle}/{total})",
            styles["Normal"]
        ))

        return story

    def _draw_header_footer(self, canvas_obj, doc):
        """Draw header and footer on pages."""
        canvas_obj.saveState()

        page_height = doc.pagesize[1]
        page_width = doc.pagesize[0]

        # Header
        canvas_obj.setFillColor(COLORS["veratum_dark"])
        canvas_obj.rect(0, page_height - 0.5 * inch, page_width, 0.5 * inch, fill=1)

        canvas_obj.setFont("Helvetica-Bold", 12)
        canvas_obj.setFillColor(colors.white)
        canvas_obj.drawString(0.5 * inch, page_height - 0.32 * inch, "VERATUM")

        canvas_obj.setFont("Helvetica", 9)
        canvas_obj.drawString(1.3 * inch, page_height - 0.32 * inch, self.report_title)

        # Footer
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(COLORS["neutral_gray"])

        canvas_obj.drawString(0.5 * inch, 0.3 * inch, f"Generated: {self.generated_at}")
        canvas_obj.drawRightString(
            page_width - 0.5 * inch,
            0.3 * inch,
            f"Page {doc.page}"
        )

        canvas_obj.setStrokeColor(COLORS["border_gray"])
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(0.5 * inch, 0.45 * inch, page_width - 0.5 * inch, 0.45 * inch)

        canvas_obj.restoreState()

    def _calculate_receipt_score(self, receipt: Dict[str, Any]) -> float:
        """Calculate a basic compliance score for a receipt."""
        score = 0.0
        max_score = 0.0

        # Check for key fields
        key_fields = [
            "timestamp", "model", "provider", "decision_type",
            "human_review_state", "entry_hash", "signature",
        ]

        for field in key_fields:
            max_score += 1
            if receipt.get(field):
                score += 1

        return score / max_score if max_score > 0 else 0.0

    def _count_frameworks_covered(self, receipts: List[Dict[str, Any]]) -> int:
        """Count how many frameworks are satisfied by receipt data."""
        if not receipts:
            return 0

        try:
            result = crosswalk(receipts[0], include_recommended=True)
            frameworks = result.get("frameworks", {})
            return sum(1 for fw_info in frameworks.values() if fw_info.get("score", 0) > 0.5)
        except Exception:
            return 0

    def _get_framework_name(self, framework_id: str) -> str:
        """Get human-readable framework name."""
        frameworks = list_frameworks()
        for fw in frameworks:
            if fw.get("id") == framework_id:
                return fw.get("name", framework_id)
        return framework_id

    def _get_compliance_status_text(self, score: float) -> str:
        """Get compliance status text from score."""
        if score >= 0.9:
            return "Compliant"
        elif score >= 0.7:
            return "Substantially Compliant"
        elif score >= 0.5:
            return "Partially Compliant"
        else:
            return "Non-Compliant"

    def _get_risk_color(self, score: float) -> str:
        """Get color name for risk level."""
        if score >= 0.9:
            return COLORS["compliant_green"]
        elif score >= 0.7:
            return COLORS["compliant_green"]
        elif score >= 0.5:
            return COLORS["warning_amber"]
        else:
            return COLORS["risk_red"]

    def _get_risk_color_name(self, risk_level: str) -> str:
        """Get HTML color name for risk level."""
        level = risk_level.lower()
        if level == "critical":
            return "#C62828"
        elif level == "high":
            return "#F57C00"
        elif level == "medium":
            return "#FBC02D"
        else:
            return "#2E7D32"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_report(
    receipts: List[Dict[str, Any]],
    org_name: str,
    report_type: str = "executive_summary",
    **kwargs,
) -> bytes:
    """
    Convenience function to generate a compliance report.

    Args:
        receipts: List of Veratum receipt dictionaries.
        org_name: Organization name.
        report_type: Report type ("executive_summary", "audit_trail",
                     "framework_report", "dpia_pdf").
        **kwargs: Additional arguments (report_title, framework, dpia, etc.).

    Returns:
        PDF content as bytes.

    Examples:
        >>> # Executive summary
        >>> pdf = generate_report(receipts, "Acme Corp")
        >>>
        >>> # Framework report
        >>> pdf = generate_report(
        ...     receipts, "Acme Corp",
        ...     report_type="framework_report",
        ...     framework="eu_ai_act"
        ... )
        >>>
        >>> # DPIA PDF
        >>> pdf = generate_report(
        ...     receipts, "Acme Corp",
        ...     report_type="dpia_pdf",
        ...     dpia=dpia_data
        ... )
    """
    report_title = kwargs.get("report_title", "AI Compliance Report")
    generator = ComplianceReportGenerator(org_name, report_title)

    if report_type == "executive_summary":
        dpia = kwargs.get("dpia")
        return generator.generate_executive_summary(receipts, dpia)

    elif report_type == "audit_trail":
        return generator.generate_audit_trail(receipts)

    elif report_type == "framework_report":
        framework = kwargs.get("framework", "eu_ai_act")
        return generator.generate_framework_report(framework, receipts)

    elif report_type == "dpia_pdf":
        dpia = kwargs.get("dpia", {})
        return generator.generate_dpia_pdf(dpia)

    else:
        raise ValueError(f"Unknown report type: {report_type}")
