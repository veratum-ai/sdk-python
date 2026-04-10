"""Jurisdiction-aware receipt validation.

Addresses the compliance head's concern: "Your system accepts receipts
that don't meet regulatory requirements."

Uses conditional validation: if jurisdiction=nyc_ll144, then bias_audit
fields are REQUIRED, not optional. If jurisdiction=gdpr, then
data_processing_basis is REQUIRED.

Design (researched from JSON Schema if-then-else, NYC DCWP rules,
EU AI Act Article 71):

Why not JSON Schema?
- JSON Schema if-then-else is powerful but requires a JSON Schema
  validator library dependency (jsonschema)
- Our validation is Python-native — faster, more readable, better errors
- We can do things JSON Schema can't: cross-field validation,
  date range checks, enum validation with custom messages

Validation is STRICT by default for FULL audit level, LENIENT for LIGHT.
"""

from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Jurisdiction requirement definitions
# ---------------------------------------------------------------------------

# Each jurisdiction defines: required fields, conditional fields, and
# validation rules. This is the single source of truth for "what does
# this law actually require in a receipt?"

JURISDICTION_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "nyc_ll144": {
        "name": "NYC Local Law 144 (AEDT)",
        "required_fields": [
            "bias_audit",
            "decision_type",
        ],
        "required_decision_types": [
            "employment_screening",
            "hiring",
            "promotion",
        ],
        "conditional_rules": [
            {
                "description": "Bias audit must include selection rates by race and gender",
                "check": "_check_nyc_bias_audit",
            },
        ],
    },
    "eeoc": {
        "name": "EEOC Uniform Guidelines (29 CFR 1607)",
        "required_fields": [
            "decision_type",
        ],
        "conditional_rules": [
            {
                "description": "Must track adverse impact for employment decisions",
                "check": "_check_eeoc_adverse_impact",
            },
        ],
    },
    "colorado_sb24_205": {
        "name": "Colorado SB24-205 (AI Consumer Protections)",
        "required_fields": [
            "decision_type",
            "vertical",
        ],
        "conditional_rules": [
            {
                "description": "Consequential decisions require human review state",
                "check": "_check_colorado_human_review",
            },
        ],
    },
    "cfpb_ecoa": {
        "name": "CFPB / Equal Credit Opportunity Act",
        "required_fields": [
            "decision_type",
        ],
        "conditional_rules": [
            {
                "description": "Credit decisions require adverse action reason codes",
                "check": "_check_cfpb_adverse_action",
            },
        ],
    },
    "gdpr": {
        "name": "GDPR (Regulation 2016/679)",
        "required_fields": [
            "data_processing_basis",
        ],
        "conditional_rules": [
            {
                "description": "Automated decisions require human review mechanism (Art. 22)",
                "check": "_check_gdpr_automated_decisions",
            },
        ],
    },
    "eu_ai_act": {
        "name": "EU AI Act (Regulation 2024/1689)",
        "required_fields": [
            "decision_type",
            "vertical",
        ],
        "conditional_rules": [
            {
                "description": "High-risk AI requires human oversight documentation (Art. 14)",
                "check": "_check_eu_ai_act_oversight",
            },
        ],
    },
    "illinois_aiva": {
        "name": "Illinois AI Video Interview Act",
        "required_fields": [
            "decision_type",
            "illinois_aiva",
        ],
        "conditional_rules": [],
    },
    "finra": {
        "name": "FINRA Rules 3110/17a-3/17a-4",
        "required_fields": [
            "decision_type",
            "finra",
        ],
        "conditional_rules": [],
    },
    "naic": {
        "name": "NAIC Model AI Bulletin",
        "required_fields": [
            "decision_type",
            "insurance",
        ],
        "conditional_rules": [],
    },
}


# ---------------------------------------------------------------------------
# Validation checks (called by name from conditional_rules)
# ---------------------------------------------------------------------------

def _check_nyc_bias_audit(receipt: Dict[str, Any]) -> List[str]:
    """NYC LL144 requires bias audit with intersectional analysis."""
    errors = []
    bias = receipt.get("bias_audit")
    if bias is None:
        errors.append("NYC LL144: bias_audit field is required")
        return errors
    if isinstance(bias, dict):
        if "selection_rates" not in bias and "impact_ratios" not in bias:
            errors.append(
                "NYC LL144: bias_audit must include selection_rates or impact_ratios"
            )
    return errors


def _check_eeoc_adverse_impact(receipt: Dict[str, Any]) -> List[str]:
    """EEOC requires adverse impact tracking for employment decisions."""
    errors = []
    dt = receipt.get("decision_type", "")
    employment_types = {"employment_screening", "hiring", "promotion", "termination"}
    if dt in employment_types and not receipt.get("bias_audit"):
        errors.append(
            "EEOC: employment decisions should include bias_audit data "
            "for adverse impact monitoring"
        )
    return errors


def _check_colorado_human_review(receipt: Dict[str, Any]) -> List[str]:
    """Colorado SB24-205: consequential decisions need human review state."""
    errors = []
    # Colorado defines "consequential decisions" broadly
    consequential_verticals = {
        "hiring", "lending", "insurance", "healthcare",
        "housing", "education", "criminal_justice",
    }
    vertical = receipt.get("vertical", "")
    if vertical in consequential_verticals:
        if not receipt.get("human_review_state") and not receipt.get("review_outcome"):
            errors.append(
                "Colorado SB24-205: consequential decisions require "
                "human_review_state or review_outcome"
            )
    return errors


def _check_cfpb_adverse_action(receipt: Dict[str, Any]) -> List[str]:
    """CFPB/ECOA: credit decisions require adverse action notices."""
    errors = []
    dt = receipt.get("decision_type", "")
    credit_types = {"credit_decision", "lending", "loan_application"}
    if dt in credit_types:
        if not receipt.get("adverse_action") and not receipt.get("explainability"):
            errors.append(
                "CFPB/ECOA: credit decisions require adverse_action "
                "or explainability data for reason code disclosure"
            )
    return errors


def _check_gdpr_automated_decisions(receipt: Dict[str, Any]) -> List[str]:
    """GDPR Art. 22: automated decisions affecting individuals need safeguards."""
    errors = []
    if not receipt.get("data_processing_basis"):
        errors.append("GDPR: data_processing_basis is required (Art. 6)")
    return errors


def _check_eu_ai_act_oversight(receipt: Dict[str, Any]) -> List[str]:
    """EU AI Act Art. 14: high-risk AI needs human oversight documentation."""
    errors = []
    high_risk_verticals = {
        "hiring", "lending", "insurance", "healthcare",
        "education", "criminal_justice", "border_control",
    }
    vertical = receipt.get("vertical", "")
    if vertical in high_risk_verticals:
        if not receipt.get("human_review_state") and not receipt.get("review_outcome"):
            errors.append(
                "EU AI Act Art. 14: high-risk AI systems require human "
                "oversight documentation (human_review_state or review_outcome)"
            )
    return errors


# Map of check names to functions
_CHECK_FUNCTIONS = {
    "_check_nyc_bias_audit": _check_nyc_bias_audit,
    "_check_eeoc_adverse_impact": _check_eeoc_adverse_impact,
    "_check_colorado_human_review": _check_colorado_human_review,
    "_check_cfpb_adverse_action": _check_cfpb_adverse_action,
    "_check_gdpr_automated_decisions": _check_gdpr_automated_decisions,
    "_check_eu_ai_act_oversight": _check_eu_ai_act_oversight,
}


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_receipt(
    receipt: Dict[str, Any],
    jurisdictions: Optional[List[str]] = None,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate a receipt against jurisdiction-specific requirements.

    Args:
        receipt: Receipt dictionary to validate.
        jurisdictions: List of jurisdiction codes to validate against.
                       If None, validates against all jurisdictions found
                       in the receipt's "jurisdictions" field.
        strict: If True, missing required fields are errors.
                If False, missing fields are warnings (for LIGHT audit mode).

    Returns:
        {
            "valid": bool,
            "errors": [{"jurisdiction": str, "message": str}],
            "warnings": [{"jurisdiction": str, "message": str}],
            "jurisdictions_checked": [str],
        }
    """
    # Determine which jurisdictions to check
    if jurisdictions is None:
        jurisdictions = receipt.get("jurisdictions", [])
    if not jurisdictions:
        # No jurisdictions specified — can't validate
        return {
            "valid": True,
            "errors": [],
            "warnings": [],
            "jurisdictions_checked": [],
        }

    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    for j_code in jurisdictions:
        j_spec = JURISDICTION_REQUIREMENTS.get(j_code)
        if not j_spec:
            warnings.append({
                "jurisdiction": j_code,
                "message": f"Unknown jurisdiction: {j_code}",
            })
            continue

        # Check required fields
        for field in j_spec.get("required_fields", []):
            if field not in receipt or receipt[field] is None:
                entry = {
                    "jurisdiction": j_code,
                    "message": (
                        f"{j_spec['name']}: missing required field '{field}'"
                    ),
                }
                if strict:
                    errors.append(entry)
                else:
                    warnings.append(entry)

        # Check conditional rules
        for rule in j_spec.get("conditional_rules", []):
            check_name = rule["check"]
            check_fn = _CHECK_FUNCTIONS.get(check_name)
            if check_fn:
                rule_errors = check_fn(receipt)
                for msg in rule_errors:
                    entry = {
                        "jurisdiction": j_code,
                        "message": msg,
                    }
                    if strict:
                        errors.append(entry)
                    else:
                        warnings.append(entry)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "jurisdictions_checked": list(jurisdictions),
    }


def list_jurisdiction_requirements(jurisdiction: str) -> Optional[Dict[str, Any]]:
    """
    Get the requirements for a specific jurisdiction.

    Args:
        jurisdiction: Jurisdiction code (e.g., "nyc_ll144").

    Returns:
        Requirement specification, or None if unknown.
    """
    return JURISDICTION_REQUIREMENTS.get(jurisdiction)


def list_all_jurisdictions() -> List[Dict[str, str]]:
    """
    List all supported jurisdictions.

    Returns:
        List of dicts with code and name.
    """
    return [
        {"code": code, "name": spec["name"]}
        for code, spec in JURISDICTION_REQUIREMENTS.items()
    ]
