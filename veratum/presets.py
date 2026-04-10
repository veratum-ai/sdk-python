"""Industry-specific compliance presets for Veratum SDK.

This module provides ready-to-deploy policy packs for major industries.
Customers no longer need to manually configure compliance frameworks —
select an industry and get CFPB+FINRA+SOC2+GDPR (finance), HIPAA+GDPR (healthcare),
or NYC LL144+EEOC+Illinois AIVA (hiring) automatically.

Design philosophy:
- One call, complete compliance setup
- Real regulatory knowledge (not boilerplate)
- Sensible defaults a CISO would expect
- Fully composable (merge presets, extend with custom rules)
- All settings auditable and override-able
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ThreatLevel(Enum):
    """Security threat level for PromptGuard/ThreatDetector settings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IndustryPreset:
    """
    A complete compliance configuration for a vertical industry.

    Attributes:
        name: Internal identifier (e.g., "financial_services", "healthcare")
        display_name: Human-readable name (e.g., "Financial Services")
        description: Regulatory context and use cases
        frameworks: List of framework IDs to enable (e.g., "eu_ai_act", "gdpr", "cfpb_ecoa")
        required_receipt_fields: Fields that MUST be present for compliance
        security_config: PromptGuard/ThreatDetector settings (threat levels, jailbreak detection)
        cost_config: Budget thresholds and alerts
        policies: Auto-applied policy rules (what decisions require human review, etc.)
        dpia_required: Data Protection Impact Assessment mandatory
        human_review_required: Human review required for all decisions
        pii_detection: Automatic PII detection and handling
        bias_audit_required: Mandatory bias/fairness audit on outputs
        data_retention_days: How long audit records are kept
        audit_level: "light" (hash only) or "full" (VC + timestamp + bias hooks)
        severity_override: Force high-risk classification (EU AI Act Annex III)
        intersectional_bias_analysis: Check bias across demographic combinations
    """

    name: str
    display_name: str
    description: str
    frameworks: List[str]
    required_receipt_fields: List[str]
    security_config: Dict[str, Any]
    cost_config: Dict[str, Any]
    policies: List[Dict[str, Any]]
    dpia_required: bool
    human_review_required: bool
    pii_detection: bool
    bias_audit_required: bool
    data_retention_days: int = 2555  # 7 years default
    audit_level: str = "full"
    severity_override: Optional[str] = None  # "high_risk" or None
    intersectional_bias_analysis: bool = False


# ============================================================================
# INDUSTRY PRESETS — Real regulatory knowledge
# ============================================================================

FINANCIAL_SERVICES_PRESET = IndustryPreset(
    name="financial_services",
    display_name="Financial Services",
    description=(
        "Banks, lending platforms, investment advisors. Enables EU AI Act, "
        "GDPR, CFPB/ECOA (fair lending), FINRA (investments), SOC 2 (security), "
        "NIST AI RMF. Mandatory human review for credit/lending decisions. "
        "PII detection + redaction. Bias audit on lending/credit products. "
        "Cost budget warning at $10K/month."
    ),
    frameworks=[
        "eu_ai_act",
        "gdpr",
        "cfpb_ecoa",
        "finra",
        "soc2",
        "nist_ai_rmf",
    ],
    required_receipt_fields=[
        "timestamp",
        "model",
        "provider",
        "prompt_hash",
        "response_hash",
        "tokens_in",
        "tokens_out",
        "decision_type",
        "human_review_state",
        "bias_audit",
        "applicable_jurisdictions",
        "explainability",
    ],
    security_config={
        "threat_detection": ThreatLevel.CRITICAL.value,
        "prompt_injection_detection": True,
        "jailbreak_detection": True,
        "data_exfiltration_prevention": True,
        "pii_masking": True,
        "sensitive_data_redaction": True,
        "tokenization_for_pii": True,
    },
    cost_config={
        "budget_monthly_usd": 10000.0,
        "warn_at_percent": 80,
        "alert_at_percent": 95,
        "enforce_limit": True,
        "per_transaction_limit_usd": 5.0,
    },
    policies=[
        {
            "name": "credit_decision_human_review",
            "description": "All credit/lending decisions require human review",
            "trigger": "decision_type == 'credit_decision' or decision_type == 'lending'",
            "action": "require_human_review",
            "reviewer_competence_level": "qualified_examiner",
            "max_response_time_minutes": 30,
        },
        {
            "name": "adverse_action_notice",
            "description": "Generate adverse action notice for denied credit",
            "trigger": "decision_outcome == 'deny' and decision_type == 'credit_decision'",
            "action": "generate_notice",
            "notice_type": "adverse_action",
        },
        {
            "name": "fair_lending_monitoring",
            "description": "Monitor for disparate impact (four-fifths rule, ECOA/FHA)",
            "trigger": True,
            "action": "quarterly_bias_audit",
            "protected_classes": ["race", "color", "religion", "sex", "national_origin", "age", "disability"],
            "rule": "four_fifths_rule",
        },
    ],
    dpia_required=True,
    human_review_required=True,
    pii_detection=True,
    bias_audit_required=True,
    data_retention_days=1825,  # 5 years per CFPB/FINRA
    audit_level="full",
    severity_override="high_risk",
    intersectional_bias_analysis=True,
)

HEALTHCARE_PRESET = IndustryPreset(
    name="healthcare",
    display_name="Healthcare",
    description=(
        "Hospitals, clinics, health tech. Enables EU AI Act high-risk, GDPR, "
        "HIPAA, ISO 42001 (AI governance), NIST AI RMF. DPIA mandatory. "
        "Human review required for diagnosis/treatment recommendations. "
        "PII detection + redaction mandatory (HIPAA strictness). "
        "10-year retention for medical records."
    ),
    frameworks=[
        "eu_ai_act",
        "gdpr",
        "hipaa",
        "iso_42001",
        "nist_ai_rmf",
    ],
    required_receipt_fields=[
        "timestamp",
        "model",
        "provider",
        "prompt_hash",
        "response_hash",
        "tokens_in",
        "tokens_out",
        "decision_type",
        "human_review_state",
        "explainability",
        "applicable_jurisdictions",
        "bias_audit",
    ],
    security_config={
        "threat_detection": ThreatLevel.CRITICAL.value,
        "prompt_injection_detection": True,
        "jailbreak_detection": True,
        "data_exfiltration_prevention": True,
        "pii_masking": True,
        "hipaa_phi_redaction": True,  # HIPAA Safe Harbor
        "encryption_at_rest": True,
        "encryption_in_transit": True,
        "access_logging": True,
    },
    cost_config={
        "budget_monthly_usd": 25000.0,  # Higher for healthcare institutions
        "warn_at_percent": 75,
        "alert_at_percent": 90,
        "enforce_limit": True,
    },
    policies=[
        {
            "name": "clinical_decision_human_review",
            "description": "Diagnosis/treatment recommendations require licensed clinician review",
            "trigger": "decision_type in ['diagnosis', 'treatment_recommendation', 'clinical_decision']",
            "action": "require_human_review",
            "reviewer_qualifications": "licensed_clinician",
            "max_response_time_minutes": 15,
            "escalation_required": True,
        },
        {
            "name": "patient_notification",
            "description": "Patient must be notified that AI was used in care decision",
            "trigger": True,
            "action": "require_patient_notification",
            "notification_form": "informed_consent",
        },
    ],
    dpia_required=True,
    human_review_required=True,
    pii_detection=True,
    bias_audit_required=True,
    data_retention_days=3650,  # 10 years for medical records
    audit_level="full",
    severity_override="high_risk",
    intersectional_bias_analysis=False,
)

HIRING_PRESET = IndustryPreset(
    name="hiring",
    display_name="Hiring & Recruitment",
    description=(
        "Recruiting platforms, talent acquisition. Enables EU AI Act, GDPR, "
        "NYC Local Law 144, EEOC Uniform Guidelines, Illinois AIVA, "
        "Colorado SB24-205, NIST AI RMF. Bias audit MANDATORY (four-fifths rule + "
        "intersectional analysis). Human review required for all employment decisions. "
        "NYC LL144 adverse action notice + audit trail required."
    ),
    frameworks=[
        "eu_ai_act",
        "gdpr",
        "nyc_ll144",
        "eeoc",
        "illinois_aiva",
        "colorado_sb24_205",
        "nist_ai_rmf",
    ],
    required_receipt_fields=[
        "timestamp",
        "model",
        "provider",
        "prompt_hash",
        "response_hash",
        "tokens_in",
        "tokens_out",
        "decision_type",
        "decision_outcome",
        "human_review_state",
        "bias_audit",
        "explainability",
        "applicable_jurisdictions",
        "reviewer_id",
        "review_outcome",
    ],
    security_config={
        "threat_detection": ThreatLevel.HIGH.value,
        "prompt_injection_detection": True,
        "jailbreak_detection": True,
        "pii_masking": True,
        "ssn_redaction": True,
        "date_of_birth_redaction": True,
    },
    cost_config={
        "budget_monthly_usd": 5000.0,
        "warn_at_percent": 80,
        "alert_at_percent": 95,
        "enforce_limit": True,
    },
    policies=[
        {
            "name": "employment_decision_human_review",
            "description": "All employment decisions (screening, ranking, offer, reject) require human review",
            "trigger": "decision_type in ['employment_screening', 'hiring_recommendation', 'candidate_ranking']",
            "action": "require_human_review",
            "reviewer_competence_level": "hiring_manager_or_recruiter",
            "max_response_time_minutes": 24 * 60,  # 24 hours
        },
        {
            "name": "nyc_ll144_adverse_action",
            "description": "NYC LL144: Provide reason for automated employment decision + opportunity to appeal",
            "trigger": "decision_outcome == 'reject' or decision_outcome == 'screen_out'",
            "action": "provide_adverse_action_notice",
            "notice_includes": [
                "specific_factors_used",
                "how_to_appeal",
                "manual_review_option",
            ],
            "language_requirement": "plain_language",
        },
        {
            "name": "disparate_impact_four_fifths",
            "description": "Monitor hiring outcomes for adverse impact (four-fifths rule per EEOC)",
            "trigger": True,
            "action": "continuous_monitoring",
            "protected_classes": ["race", "color", "religion", "sex", "national_origin", "age", "disability", "genetic_info"],
            "rule": "four_fifths_rule",
            "reporting_frequency": "quarterly",
        },
        {
            "name": "intersectional_bias_analysis",
            "description": "Check bias across demographic intersections (e.g., Black women, older men)",
            "trigger": True,
            "action": "continuous_monitoring",
            "intersections": ["race_gender", "age_gender", "disability_gender"],
        },
    ],
    dpia_required=True,
    human_review_required=True,
    pii_detection=True,
    bias_audit_required=True,
    data_retention_days=1095,  # 3 years per EEOC
    audit_level="full",
    severity_override="high_risk",
    intersectional_bias_analysis=True,
)

LEGAL_PRESET = IndustryPreset(
    name="legal",
    display_name="Legal Tech",
    description=(
        "Law firms, legal research platforms, contract automation. Enables EU AI Act, "
        "GDPR, SOC 2, NIST AI RMF, ISO 42001. Human review required for case "
        "recommendations + legal strategy. PII detection mandatory (attorney-client privilege). "
        "Comprehensive audit trail for litigation support."
    ),
    frameworks=[
        "eu_ai_act",
        "gdpr",
        "soc2",
        "nist_ai_rmf",
        "iso_42001",
    ],
    required_receipt_fields=[
        "timestamp",
        "model",
        "provider",
        "prompt_hash",
        "response_hash",
        "tokens_in",
        "tokens_out",
        "decision_type",
        "human_review_state",
        "explainability",
        "applicable_jurisdictions",
    ],
    security_config={
        "threat_detection": ThreatLevel.HIGH.value,
        "prompt_injection_detection": True,
        "jailbreak_detection": True,
        "pii_masking": True,
        "attorney_client_privilege_protection": True,
        "access_logging": True,
    },
    cost_config={
        "budget_monthly_usd": 15000.0,
        "warn_at_percent": 80,
        "alert_at_percent": 95,
        "enforce_limit": True,
    },
    policies=[
        {
            "name": "case_recommendation_review",
            "description": "Case strategy + recommendations reviewed by licensed attorney",
            "trigger": "decision_type in ['case_recommendation', 'legal_strategy', 'motion_review']",
            "action": "require_human_review",
            "reviewer_qualifications": "licensed_attorney",
            "max_response_time_minutes": 60,
        },
        {
            "name": "litigation_audit_trail",
            "description": "Maintain discoverable audit trail for litigation support",
            "trigger": True,
            "action": "enhanced_logging",
            "includes": ["full_prompt", "full_response", "reviewer_analysis"],
        },
    ],
    dpia_required=True,
    human_review_required=True,
    pii_detection=True,
    bias_audit_required=False,
    data_retention_days=2555,  # 7 years
    audit_level="full",
    severity_override=None,
    intersectional_bias_analysis=False,
)

INSURANCE_PRESET = IndustryPreset(
    name="insurance",
    display_name="Insurance",
    description=(
        "Insurance underwriting, claims, pricing. Enables EU AI Act, GDPR, NAIC Model "
        "Bulletin, SOC 2, NIST AI RMF. Bias audit mandatory on underwriting (unfair "
        "discrimination monitoring). Human review for claims decisions. Actuarial "
        "justification required for pricing models."
    ),
    frameworks=[
        "eu_ai_act",
        "gdpr",
        "naic",
        "soc2",
        "nist_ai_rmf",
    ],
    required_receipt_fields=[
        "timestamp",
        "model",
        "provider",
        "prompt_hash",
        "response_hash",
        "tokens_in",
        "tokens_out",
        "decision_type",
        "decision_outcome",
        "human_review_state",
        "bias_audit",
        "explainability",
        "applicable_jurisdictions",
    ],
    security_config={
        "threat_detection": ThreatLevel.HIGH.value,
        "prompt_injection_detection": True,
        "jailbreak_detection": True,
        "data_exfiltration_prevention": True,
        "pii_masking": True,
    },
    cost_config={
        "budget_monthly_usd": 20000.0,
        "warn_at_percent": 80,
        "alert_at_percent": 95,
        "enforce_limit": True,
    },
    policies=[
        {
            "name": "underwriting_decision_review",
            "description": "High-dollar or complex underwriting decisions reviewed by underwriter",
            "trigger": "decision_type == 'underwriting' and (decision_amount_usd > 100000 or complexity == 'high')",
            "action": "require_human_review",
            "reviewer_qualifications": "underwriter",
            "max_response_time_minutes": 120,
        },
        {
            "name": "claims_decision_review",
            "description": "Claims decisions reviewed before payment/denial",
            "trigger": "decision_type == 'claims_decision' and decision_outcome in ['deny', 'payout']",
            "action": "require_human_review",
            "reviewer_qualifications": "claims_adjuster",
        },
        {
            "name": "unfair_discrimination_monitoring",
            "description": "Monitor pricing for unfair discrimination per NAIC (protected classes)",
            "trigger": True,
            "action": "continuous_monitoring",
            "protected_classes": ["race", "color", "religion", "sex", "national_origin", "age", "disability"],
            "rule": "disparate_impact_analysis",
            "reporting_frequency": "quarterly",
        },
        {
            "name": "actuarial_justification",
            "description": "Rate changes require actuarial studies per insurance law",
            "trigger": "decision_type == 'rate_adjustment' and change_percent > 5",
            "action": "require_actuarial_study",
        },
    ],
    dpia_required=True,
    human_review_required=True,
    pii_detection=True,
    bias_audit_required=True,
    data_retention_days=2190,  # 6 years per NAIC
    audit_level="full",
    severity_override="high_risk",
    intersectional_bias_analysis=True,
)

GOVERNMENT_PRESET = IndustryPreset(
    name="government",
    display_name="Government & Public Sector",
    description=(
        "Federal/state agencies, public safety, benefits determination. Maximum "
        "security baseline: EU AI Act, NIST AI RMF, NIST 800-53, FedRAMP (if federal), "
        "ISO 42001. All decisions require human review. DPIA mandatory. "
        "Comprehensive audit trail for transparency/accountability."
    ),
    frameworks=[
        "eu_ai_act",
        "nist_ai_rmf",
        "nist_800_53",
        "fedramp",
        "iso_42001",
    ],
    required_receipt_fields=[
        "timestamp",
        "model",
        "provider",
        "prompt_hash",
        "response_hash",
        "tokens_in",
        "tokens_out",
        "decision_type",
        "decision_outcome",
        "human_review_state",
        "human_review_notes",
        "bias_audit",
        "explainability",
        "applicable_jurisdictions",
        "reviewer_id",
        "review_outcome",
    ],
    security_config={
        "threat_detection": ThreatLevel.CRITICAL.value,
        "prompt_injection_detection": True,
        "jailbreak_detection": True,
        "data_exfiltration_prevention": True,
        "pii_masking": True,
        "encryption_at_rest": True,
        "encryption_in_transit": True,
        "access_logging": True,
        "audit_logging": True,
        "anomaly_detection": True,
    },
    cost_config={
        "budget_monthly_usd": None,  # No limit for government
        "warn_at_percent": 90,
        "alert_at_percent": 100,
        "enforce_limit": False,
    },
    policies=[
        {
            "name": "all_decisions_human_review",
            "description": "Every consequential decision requires documented human review",
            "trigger": True,
            "action": "require_human_review",
            "reviewer_qualifications": "government_official",
            "max_response_time_minutes": 60,
            "documentation_required": True,
        },
        {
            "name": "transparency_explanation",
            "description": "Provide clear explanation of decision factors to citizen/applicant",
            "trigger": True,
            "action": "generate_explanation",
            "audience": "citizen",
            "language": "plain_language",
            "includes": ["key_factors", "how_to_appeal"],
        },
        {
            "name": "bias_fairness_audit",
            "description": "Continuous monitoring for bias in benefit determination, licensing, etc.",
            "trigger": True,
            "action": "continuous_monitoring",
            "protected_classes": ["race", "color", "religion", "sex", "national_origin", "age", "disability"],
            "reporting_frequency": "monthly",
        },
    ],
    dpia_required=True,
    human_review_required=True,
    pii_detection=True,
    bias_audit_required=True,
    data_retention_days=2555,  # 7 years
    audit_level="full",
    severity_override="high_risk",
    intersectional_bias_analysis=True,
)

EDUCATION_PRESET = IndustryPreset(
    name="education",
    display_name="Education & Academic Institutions",
    description=(
        "Universities, K-12, edtech platforms. Enables EU AI Act, GDPR, NIST AI RMF. "
        "Bias audit mandatory on grading/admissions (disproportionate impact on protected classes). "
        "PII of minors protected under FERPA-equivalent rules. Human review for "
        "admissions + scholarship decisions."
    ),
    frameworks=[
        "eu_ai_act",
        "gdpr",
        "nist_ai_rmf",
    ],
    required_receipt_fields=[
        "timestamp",
        "model",
        "provider",
        "prompt_hash",
        "response_hash",
        "tokens_in",
        "tokens_out",
        "decision_type",
        "decision_outcome",
        "human_review_state",
        "bias_audit",
        "explainability",
        "applicable_jurisdictions",
    ],
    security_config={
        "threat_detection": ThreatLevel.HIGH.value,
        "prompt_injection_detection": True,
        "jailbreak_detection": True,
        "pii_masking": True,
        "ferpa_compliance": True,  # Family Educational Rights and Privacy Act
        "minor_data_protection": True,
    },
    cost_config={
        "budget_monthly_usd": 8000.0,
        "warn_at_percent": 80,
        "alert_at_percent": 95,
        "enforce_limit": True,
    },
    policies=[
        {
            "name": "admissions_decision_review",
            "description": "Admissions + scholarship decisions reviewed by admissions officer",
            "trigger": "decision_type in ['admissions', 'scholarship_award']",
            "action": "require_human_review",
            "reviewer_qualifications": "admissions_officer_or_faculty",
            "max_response_time_minutes": 120,
        },
        {
            "name": "grading_bias_audit",
            "description": "Monitor automated grading for bias against protected groups",
            "trigger": "decision_type == 'grading' or decision_type == 'academic_assessment'",
            "action": "continuous_monitoring",
            "protected_classes": ["race", "color", "religion", "sex", "national_origin", "age", "disability"],
            "intersectional_analysis": True,
        },
        {
            "name": "minor_data_protection",
            "description": "Heightened protection for records of students under 18",
            "trigger": "student_age < 18",
            "action": "enforce_minor_protections",
            "includes": [
                "parental_notification_for_automated_decisions",
                "shorter_data_retention",
                "no_profiling",
            ],
        },
    ],
    dpia_required=True,
    human_review_required=True,
    pii_detection=True,
    bias_audit_required=True,
    data_retention_days=1095,  # 3 years
    audit_level="full",
    severity_override=None,
    intersectional_bias_analysis=True,
)

GENERAL_PRESET = IndustryPreset(
    name="general",
    display_name="General Purpose",
    description=(
        "Default/general-purpose AI applications. Enables EU AI Act, NIST AI RMF, "
        "OWASP Top 10 for LLM Applications. Basic security + sensible compliance defaults. "
        "Appropriate for most applications that aren't specifically high-risk "
        "(hiring, lending, healthcare, etc.)."
    ),
    frameworks=[
        "eu_ai_act",
        "nist_ai_rmf",
        "owasp_llm_top_10",
    ],
    required_receipt_fields=[
        "timestamp",
        "model",
        "provider",
        "prompt_hash",
        "response_hash",
        "tokens_in",
        "tokens_out",
        "decision_type",
        "explainability",
    ],
    security_config={
        "threat_detection": ThreatLevel.MEDIUM.value,
        "prompt_injection_detection": True,
        "jailbreak_detection": False,
        "data_exfiltration_prevention": False,
        "pii_masking": False,
    },
    cost_config={
        "budget_monthly_usd": 2000.0,
        "warn_at_percent": 80,
        "alert_at_percent": 95,
        "enforce_limit": True,
    },
    policies=[],
    dpia_required=False,
    human_review_required=False,
    pii_detection=False,
    bias_audit_required=False,
    data_retention_days=2555,  # 7 years
    audit_level="light",
    severity_override=None,
    intersectional_bias_analysis=False,
)


# ============================================================================
# Preset Registry
# ============================================================================

PRESETS_REGISTRY: Dict[str, IndustryPreset] = {
    "financial_services": FINANCIAL_SERVICES_PRESET,
    "healthcare": HEALTHCARE_PRESET,
    "hiring": HIRING_PRESET,
    "legal": LEGAL_PRESET,
    "insurance": INSURANCE_PRESET,
    "government": GOVERNMENT_PRESET,
    "education": EDUCATION_PRESET,
    "general": GENERAL_PRESET,
}


# ============================================================================
# Public API
# ============================================================================


def get_preset(name: str) -> Optional[IndustryPreset]:
    """
    Retrieve a preset by name.

    Args:
        name: Preset name (e.g., "financial_services", "healthcare")

    Returns:
        IndustryPreset object, or None if not found.

    Raises:
        ValueError: If preset name is not recognized (with list of valid presets)
    """
    if name not in PRESETS_REGISTRY:
        valid = list(PRESETS_REGISTRY.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Valid presets: {', '.join(valid)}"
        )
    return PRESETS_REGISTRY[name]


def list_presets() -> List[str]:
    """
    List all available preset names.

    Returns:
        List of preset names in alphabetical order.
    """
    return sorted(PRESETS_REGISTRY.keys())


def get_preset_for_vertical(vertical: str) -> Optional[IndustryPreset]:
    """
    Map a vertical string to the most appropriate preset.

    Handles common aliases and variations.

    Args:
        vertical: Vertical name (e.g., "finance", "bank", "hospital", "recruit")

    Returns:
        IndustryPreset object, or None if no suitable preset found.
    """
    # Normalize input
    vertical_lower = vertical.lower().strip()

    # Vertical → preset mapping (handle aliases)
    vertical_map = {
        "finance": "financial_services",
        "fintech": "financial_services",
        "bank": "financial_services",
        "banking": "financial_services",
        "lending": "financial_services",
        "loans": "financial_services",
        "credit": "financial_services",
        "investment": "financial_services",
        "stock": "financial_services",
        "hospital": "healthcare",
        "clinic": "healthcare",
        "medical": "healthcare",
        "doctor": "healthcare",
        "health": "healthcare",
        "pharma": "healthcare",
        "biotech": "healthcare",
        "hiring": "hiring",
        "recruitment": "hiring",
        "recruiter": "hiring",
        "hr": "hiring",
        "talent": "hiring",
        "job": "hiring",
        "resume": "hiring",
        "law": "legal",
        "lawyer": "legal",
        "attorney": "legal",
        "legal_tech": "legal",
        "contract": "legal",
        "insurance": "insurance",
        "underwriting": "insurance",
        "claims": "insurance",
        "actuarial": "insurance",
        "government": "government",
        "government_agency": "government",
        "public_sector": "government",
        "federal": "government",
        "state": "government",
        "benefits": "government",
        "welfare": "government",
        "education": "education",
        "school": "education",
        "university": "education",
        "college": "education",
        "edtech": "education",
        "student": "education",
        "academic": "education",
    }

    preset_name = vertical_map.get(vertical_lower)
    if preset_name:
        return PRESETS_REGISTRY.get(preset_name)

    # Fallback: try exact match
    if vertical_lower in PRESETS_REGISTRY:
        return PRESETS_REGISTRY[vertical_lower]

    return None


def create_custom_preset(
    base: str,
    name: str,
    display_name: str,
    **overrides: Any,
) -> IndustryPreset:
    """
    Create a custom preset by extending an existing one.

    Useful for customers who need slight variations of standard presets.

    Args:
        base: Name of preset to extend (e.g., "financial_services")
        name: Custom preset name
        display_name: Human-readable name for custom preset
        **overrides: Fields to override (frameworks, policies, security_config, etc.)

    Returns:
        New IndustryPreset with overrides applied.

    Example:
        custom = create_custom_preset(
            base="financial_services",
            name="fintech_startup",
            display_name="Fintech Startup (Light Compliance)",
            audit_level="light",
            dpia_required=False,
        )
    """
    base_preset = get_preset(base)

    # Start with base preset fields
    preset_dict = asdict(base_preset)

    # Apply overrides
    preset_dict["name"] = name
    preset_dict["display_name"] = display_name
    for key, value in overrides.items():
        if key in preset_dict:
            preset_dict[key] = value

    return IndustryPreset(**preset_dict)


def merge_presets(*names: str) -> IndustryPreset:
    """
    Merge multiple presets into a single composite preset.

    Useful for organizations operating across multiple regulated industries
    (e.g., a fintech + insurance joint venture).

    Merge strategy:
    - frameworks: Union of all frameworks
    - required_receipt_fields: Union of all fields
    - security_config: Highest threat level wins, other settings merged
    - cost_config: Budget is the sum of all presets
    - policies: All policies combined
    - boolean flags (dpia_required, human_review_required, etc.): True if any is True
    - data_retention_days: Maximum value

    Args:
        *names: Names of presets to merge (at least 2)

    Returns:
        New composite IndustryPreset.

    Raises:
        ValueError: If fewer than 2 presets provided
    """
    if len(names) < 2:
        raise ValueError("merge_presets() requires at least 2 presets")

    presets = [get_preset(name) for name in names]

    # Merge frameworks (union)
    frameworks: Set[str] = set()
    for p in presets:
        frameworks.update(p.frameworks)

    # Merge receipt fields (union)
    receipt_fields: Set[str] = set()
    for p in presets:
        receipt_fields.update(p.required_receipt_fields)

    # Merge security config (highest threat level wins)
    threat_levels = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }
    max_threat_level = ThreatLevel.LOW.value
    max_threat_score = 0
    for p in presets:
        threat_level = p.security_config.get("threat_detection", ThreatLevel.LOW.value)
        score = threat_levels.get(threat_level, 0)
        if score > max_threat_score:
            max_threat_score = score
            max_threat_level = threat_level

    merged_security_config: Dict[str, Any] = {}
    for p in presets:
        merged_security_config.update(p.security_config)
    merged_security_config["threat_detection"] = max_threat_level

    # Merge cost config (sum budgets, use strictest limits)
    merged_cost_config: Dict[str, Any] = {}
    total_budget = 0.0
    min_warn_percent = 100
    min_alert_percent = 100
    for p in presets:
        if p.cost_config.get("budget_monthly_usd"):
            total_budget += p.cost_config["budget_monthly_usd"]
        min_warn_percent = min(
            min_warn_percent, p.cost_config.get("warn_at_percent", 80)
        )
        min_alert_percent = min(
            min_alert_percent, p.cost_config.get("alert_at_percent", 95)
        )

    merged_cost_config = {
        "budget_monthly_usd": total_budget if total_budget > 0 else None,
        "warn_at_percent": min_warn_percent,
        "alert_at_percent": min_alert_percent,
        "enforce_limit": any(p.cost_config.get("enforce_limit") for p in presets),
    }

    # Merge policies (all combined)
    merged_policies: List[Dict[str, Any]] = []
    for p in presets:
        merged_policies.extend(p.policies)

    # Boolean flags (True if any is True)
    dpia_required = any(p.dpia_required for p in presets)
    human_review_required = any(p.human_review_required for p in presets)
    pii_detection = any(p.pii_detection for p in presets)
    bias_audit_required = any(p.bias_audit_required for p in presets)
    intersectional_bias_analysis = any(
        p.intersectional_bias_analysis for p in presets
    )

    # Data retention: maximum
    data_retention_days = max(p.data_retention_days for p in presets)

    # Audit level: FULL if any is FULL
    audit_level = "full" if any(p.audit_level == "full" for p in presets) else "light"

    # Severity override: high_risk if any is high_risk
    severity_override = None
    if any(p.severity_override == "high_risk" for p in presets):
        severity_override = "high_risk"

    # Create merged preset
    merged_name = "_".join(names)
    merged_display_name = " + ".join(
        [get_preset(name).display_name for name in names]
    )
    merged_description = f"Composite preset merging: {merged_display_name}"

    return IndustryPreset(
        name=merged_name,
        display_name=merged_display_name,
        description=merged_description,
        frameworks=sorted(list(frameworks)),
        required_receipt_fields=sorted(list(receipt_fields)),
        security_config=merged_security_config,
        cost_config=merged_cost_config,
        policies=merged_policies,
        dpia_required=dpia_required,
        human_review_required=human_review_required,
        pii_detection=pii_detection,
        bias_audit_required=bias_audit_required,
        data_retention_days=data_retention_days,
        audit_level=audit_level,
        severity_override=severity_override,
        intersectional_bias_analysis=intersectional_bias_analysis,
    )


def apply_preset_config(preset: IndustryPreset) -> Dict[str, Any]:
    """
    Convert an IndustryPreset into a configuration dict suitable for
    passing to VeratumSDK or VeratumInstance.

    This is the bridge between presets and actual SDK configuration.

    Args:
        preset: IndustryPreset to convert

    Returns:
        Configuration dict with all preset settings organized by subsystem:
        {
            "compliance": {
                "frameworks": [...],
                "required_receipt_fields": [...],
                "dpia_required": bool,
                ...
            },
            "security": {...},
            "cost": {...},
            "policies": {...},
            "audit": {...},
        }
    """
    return {
        "compliance": {
            "frameworks": preset.frameworks,
            "required_receipt_fields": preset.required_receipt_fields,
            "dpia_required": preset.dpia_required,
            "applicable_jurisdictions": preset.frameworks,  # Map frameworks to jurisdictions
        },
        "security": {
            "threat_detection": preset.security_config.get("threat_detection"),
            "prompt_injection_detection": preset.security_config.get(
                "prompt_injection_detection"
            ),
            "jailbreak_detection": preset.security_config.get("jailbreak_detection"),
            "data_exfiltration_prevention": preset.security_config.get(
                "data_exfiltration_prevention"
            ),
            "pii_masking": preset.security_config.get("pii_masking"),
            "sensitive_data_redaction": preset.security_config.get(
                "sensitive_data_redaction"
            ),
        },
        "cost": {
            "budget_monthly_usd": preset.cost_config.get("budget_monthly_usd"),
            "warn_at_percent": preset.cost_config.get("warn_at_percent"),
            "alert_at_percent": preset.cost_config.get("alert_at_percent"),
            "enforce_limit": preset.cost_config.get("enforce_limit"),
        },
        "policies": {
            "human_review_required": preset.human_review_required,
            "policies": preset.policies,
        },
        "audit": {
            "audit_level": preset.audit_level,
            "pii_detection": preset.pii_detection,
            "bias_audit_required": preset.bias_audit_required,
            "intersectional_bias_analysis": preset.intersectional_bias_analysis,
            "data_retention_days": preset.data_retention_days,
            "severity_override": preset.severity_override,
        },
        "metadata": {
            "preset_name": preset.name,
            "display_name": preset.display_name,
            "description": preset.description,
        },
    }


# ============================================================================
# Utility Functions
# ============================================================================


def describe_preset(name: str) -> str:
    """
    Get a human-readable description of a preset.

    Args:
        name: Preset name

    Returns:
        Multi-line description with frameworks, policies, and settings.
    """
    preset = get_preset(name)

    lines = [
        f"Preset: {preset.display_name}",
        f"Name: {preset.name}",
        "",
        f"Description:",
        f"  {preset.description}",
        "",
        f"Frameworks Enabled ({len(preset.frameworks)}):",
    ]

    for fw in preset.frameworks:
        lines.append(f"  - {fw}")

    lines.extend([
        "",
        "Key Settings:",
        f"  - Audit Level: {preset.audit_level.upper()}",
        f"  - Human Review Required: {preset.human_review_required}",
        f"  - PII Detection: {preset.pii_detection}",
        f"  - Bias Audit Required: {preset.bias_audit_required}",
        f"  - DPIA Required: {preset.dpia_required}",
        f"  - Data Retention: {preset.data_retention_days} days (~{preset.data_retention_days // 365} years)",
        "",
        f"Security Threat Level: {preset.security_config.get('threat_detection').upper()}",
        f"Cost Budget: ${preset.cost_config.get('budget_monthly_usd')} /month",
        "",
    ])

    if preset.policies:
        lines.append(f"Auto-Applied Policies ({len(preset.policies)}):")
        for policy in preset.policies:
            lines.append(f"  - {policy['name']}: {policy['description']}")
    else:
        lines.append("No auto-applied policies.")

    return "\n".join(lines)
