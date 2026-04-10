"""Regulatory compliance — crosswalk, validation, policy engine, DPIA, bias analysis, reporting.

Modules:
    crosswalk        — Maps receipts to 17 regulatory frameworks simultaneously
    validation       — Jurisdiction-aware receipt validation (NYC LL144, GDPR, EU AI Act, etc.)
    prevention       — Pre-execution policy engine with BLOCKED receipt evidence
    dpia             — Automated GDPR Article 35 DPIA generation from receipt data
    bias             — EEOC 4/5ths rule, adverse impact analysis, NYC LL144 bias audits
    report_generator — Professional PDF compliance report generation (requires reportlab)
"""

from .crosswalk import crosswalk, list_frameworks, get_required_fields, get_gaps_for_frameworks
from .validation import validate_receipt, list_all_jurisdictions
from .prevention import (
    VeratumPolicyEngine,
    PolicyResult,
    PolicyViolation,
    PolicyViolationError,
    BUILT_IN_POLICIES,
)
from .dpia import DPIAGenerator, DPIAReport, generate_dpia
from .bias import (
    selection_rate,
    impact_ratio,
    four_fifths_rule,
    nyc_ll144_bias_audit,
    adverse_impact_analysis,
)


def __getattr__(name):
    """Lazy-load report_generator to avoid hard dependency on reportlab."""
    if name in ("ComplianceReportGenerator", "generate_report"):
        from .report_generator import ComplianceReportGenerator, generate_report
        return {"ComplianceReportGenerator": ComplianceReportGenerator, "generate_report": generate_report}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "crosswalk", "list_frameworks", "get_required_fields", "get_gaps_for_frameworks",
    "validate_receipt", "list_all_jurisdictions",
    "VeratumPolicyEngine", "PolicyResult", "PolicyViolation", "PolicyViolationError", "BUILT_IN_POLICIES",
    "DPIAGenerator", "DPIAReport", "generate_dpia",
    "selection_rate", "impact_ratio", "four_fifths_rule", "nyc_ll144_bias_audit", "adverse_impact_analysis",
    "ComplianceReportGenerator", "generate_report",
]
