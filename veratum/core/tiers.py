"""Tiered audit modes for Veratum SDK.

Addresses the customer concern: "Why am I paying for full cryptographic
audit on every chatbot reply?"

Design decision — two tiers, not three or five:
- LIGHT: Hash chain only. No VC, no RFC 3161 timestamp, no bias audit.
  Perfect for: chatbots, internal tools, low-risk summarization.
  Cost: ~0.1ms overhead, no network call.
- FULL: Everything — VC, RFC 3161 timestamp, bias audit hooks,
  Merkle batch anchoring. Required for: hiring, lending, insurance,
  any consequential decision under EU AI Act Art.6 or Colorado SB24-205.

Why not a MEDIUM tier? Compliance is binary — either you need the
evidence trail or you don't. A middle tier creates ambiguity about
what's "enough" for regulators. Two tiers make the decision simple:
"Is this a consequential decision? → FULL. Otherwise → LIGHT."

Vertical presets map to tiers automatically:
- hiring → FULL (NYC LL144 requires it)
- lending → FULL (CFPB/ECOA requires it)
- insurance → FULL (NAIC requires it)
- chatbot → LIGHT
- internal → LIGHT
- custom → user chooses
"""

from enum import Enum
from typing import Any, Dict, List, Optional


class AuditLevel(Enum):
    """Audit level for receipts."""
    LIGHT = "light"  # Hash chain only — no VC, no timestamp
    FULL = "full"    # Full compliance — VC, timestamp, bias hooks


# Vertical → default audit level mapping
VERTICAL_DEFAULTS: Dict[str, AuditLevel] = {
    # Consequential decisions — FULL is mandatory
    "hiring": AuditLevel.FULL,
    "lending": AuditLevel.FULL,
    "insurance": AuditLevel.FULL,
    "healthcare": AuditLevel.FULL,
    "financial_advice": AuditLevel.FULL,
    "credit": AuditLevel.FULL,
    "housing": AuditLevel.FULL,
    "education": AuditLevel.FULL,
    "criminal_justice": AuditLevel.FULL,
    # Low-risk — LIGHT is appropriate
    "chatbot": AuditLevel.LIGHT,
    "internal": AuditLevel.LIGHT,
    "content_generation": AuditLevel.LIGHT,
    "summarization": AuditLevel.LIGHT,
    "translation": AuditLevel.LIGHT,
    "code_review": AuditLevel.LIGHT,
    "general": AuditLevel.FULL,
}


# Vertical presets: pre-configured compliance fields per industry
VERTICAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "hiring": {
        "audit_level": AuditLevel.FULL,
        "decision_type": "employment_screening",
        "vertical": "hiring",
        "jurisdictions": ["nyc_ll144", "eeoc", "illinois_aiva"],
        "compliance_fields": {
            "requires_bias_audit": True,
            "requires_adverse_action_notice": True,
            "human_review_required": True,
            "data_processing_basis": "legitimate_interest",
            "retention_period": "P3Y",
        },
        "description": (
            "NYC Local Law 144 compliant hiring preset. Includes bias audit "
            "hooks, adverse action notice fields, and mandatory human review "
            "flags. Supports intersectional analysis (race × gender)."
        ),
    },
    "lending": {
        "audit_level": AuditLevel.FULL,
        "decision_type": "credit_decision",
        "vertical": "lending",
        "jurisdictions": ["cfpb_ecoa", "eeoc", "colorado_sb24_205"],
        "compliance_fields": {
            "requires_adverse_action_notice": True,
            "requires_explainability": True,
            "human_review_required": True,
            "data_processing_basis": "legal_obligation",
            "retention_period": "P5Y",
        },
        "description": (
            "CFPB/ECOA compliant lending preset. Includes adverse action "
            "reason codes, explainability requirements, and fair lending "
            "monitoring hooks."
        ),
    },
    "insurance": {
        "audit_level": AuditLevel.FULL,
        "decision_type": "insurance_underwriting",
        "vertical": "insurance",
        "jurisdictions": ["naic", "colorado_sb24_205"],
        "compliance_fields": {
            "requires_bias_audit": True,
            "requires_explainability": True,
            "actuarial_justification_required": True,
            "data_processing_basis": "contractual_necessity",
            "retention_period": "P6Y",
        },
        "description": (
            "NAIC Model Bulletin compliant insurance preset. Includes "
            "actuarial justification hooks, unfair discrimination monitoring, "
            "and rate-filing documentation support."
        ),
    },
    "financial_advice": {
        "audit_level": AuditLevel.FULL,
        "decision_type": "financial_recommendation",
        "vertical": "financial_advice",
        "jurisdictions": ["finra", "colorado_sb24_205"],
        "compliance_fields": {
            "requires_suitability_check": True,
            "requires_explainability": True,
            "human_review_required": True,
            "data_processing_basis": "legal_obligation",
            "retention_period": "P6Y",
        },
        "description": (
            "FINRA Rules 3110/17a-3/17a-4 compliant preset. Includes "
            "suitability check hooks, 6-year retention, and supervision "
            "documentation."
        ),
    },
    "healthcare": {
        "audit_level": AuditLevel.FULL,
        "decision_type": "clinical_decision_support",
        "vertical": "healthcare",
        "jurisdictions": ["eu_ai_act", "colorado_sb24_205"],
        "compliance_fields": {
            "requires_explainability": True,
            "human_review_required": True,
            "data_processing_basis": "vital_interest",
            "retention_period": "P10Y",
            "hipaa_applicable": True,
        },
        "description": (
            "Healthcare AI decision support preset. EU AI Act high-risk "
            "classification (Annex III). Includes clinical explainability "
            "and 10-year retention for medical records."
        ),
    },
    "general": {
        "audit_level": AuditLevel.FULL,
        "decision_type": "general_ai_usage",
        "vertical": "general",
        "jurisdictions": [
            "eu_ai_act",
            "colorado_sb24_205",
            "nyc_ll144",
            "eeoc",
            "cfpb_ecoa",
        ],
        "compliance_fields": {
            "requires_explainability": True,
            "human_review_required": False,
            "data_processing_basis": "legitimate_interest",
            "retention_period": "P7Y",
        },
        "description": (
            "General-purpose AI usage preset. Comprehensive compliance across "
            "major jurisdictions (EU AI Act, Colorado SB24-205, NYC LL144, EEOC, CFPB/ECOA) "
            "with 7-year retention and full audit level. Suitable for most applications "
            "that aren't specifically categorized as high-risk decision systems."
        ),
    },
}


def get_audit_level(vertical: str) -> AuditLevel:
    """
    Get the default audit level for a vertical.

    Args:
        vertical: Industry vertical name.

    Returns:
        AuditLevel.FULL for consequential decisions, AuditLevel.LIGHT otherwise.
    """
    return VERTICAL_DEFAULTS.get(vertical, AuditLevel.LIGHT)


def get_preset(vertical: str) -> Optional[Dict[str, Any]]:
    """
    Get the full compliance preset for a vertical.

    Args:
        vertical: Industry vertical name.

    Returns:
        Preset configuration dict, or None if no preset exists.
    """
    return VERTICAL_PRESETS.get(vertical)


def list_presets() -> List[Dict[str, str]]:
    """
    List all available vertical presets.

    Returns:
        List of dicts with name, decision_type, and description.
    """
    return [
        {
            "name": name,
            "decision_type": preset["decision_type"],
            "audit_level": preset["audit_level"].value,
            "description": preset["description"],
        }
        for name, preset in VERTICAL_PRESETS.items()
    ]


def apply_preset(
    receipt_kwargs: Dict[str, Any],
    vertical: str,
    *,
    override_level: Optional[AuditLevel] = None,
) -> Dict[str, Any]:
    """
    Apply a vertical preset to receipt generation kwargs.

    Merges preset compliance fields into the receipt kwargs without
    overwriting user-provided values (user always wins).

    Args:
        receipt_kwargs: Existing receipt generation kwargs.
        vertical: Vertical name to apply preset for.
        override_level: Force a specific audit level (overrides preset).

    Returns:
        Updated receipt kwargs with preset fields merged.
    """
    preset = VERTICAL_PRESETS.get(vertical)
    if not preset:
        return receipt_kwargs

    result = dict(receipt_kwargs)

    # Set audit level
    level = override_level or preset["audit_level"]
    result.setdefault("_audit_level", level)

    # Merge simple fields (user-provided values take precedence)
    result.setdefault("decision_type", preset["decision_type"])
    result.setdefault("vertical", preset["vertical"])

    # Merge jurisdictions (additive)
    existing_j = result.get("jurisdictions", [])
    preset_j = preset.get("jurisdictions", [])
    merged_j = list(dict.fromkeys(existing_j + preset_j))  # deduplicate, preserve order
    result["jurisdictions"] = merged_j

    # Merge compliance fields (user-provided values take precedence)
    preset_compliance = preset.get("compliance_fields", {})
    for key, value in preset_compliance.items():
        result.setdefault(key, value)

    return result
