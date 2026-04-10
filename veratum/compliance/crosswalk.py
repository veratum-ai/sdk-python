"""Multi-framework compliance crosswalk engine.

Maps a single Veratum receipt to requirements across multiple regulatory
and standards frameworks simultaneously. One audit record satisfies
many frameworks — customers don't need separate systems per regulation.

Supported frameworks (17 total):
- EU AI Act (Regulation 2024/1689) — Articles 9, 12, 13, 14, 26
- EU AI Act — GPAI Obligations (Articles 51-56)
- NIST AI Risk Management Framework (AI RMF 1.0)
- ISO/IEC 42001:2023 (AI Management System)
- ISO/IEC 27001:2022 (Information Security)
- GDPR (Regulation 2016/679)
- Colorado SB24-205
- NYC Local Law 144
- EEOC Uniform Guidelines
- CFPB/ECOA
- Illinois AIVA
- Texas RAIGA
- FINRA Rules
- NAIC Model Bulletin
- SOC 2 Type II (Trust Services Criteria)
- W3C VC Data Model 2.0 (credential structure compliance)
- OWASP Top 10 for LLM Applications 2025
"""

from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Framework definitions — requirement → receipt field mapping
# ---------------------------------------------------------------------------

# Each framework maps requirement IDs to the receipt fields that satisfy them.
# "required" means the field MUST be present for compliance.
# "recommended" means the field SHOULD be present for best practice.

FRAMEWORKS: Dict[str, Dict[str, Any]] = {
    "eu_ai_act": {
        "name": "EU AI Act (Regulation 2024/1689)",
        "effective_date": "2026-08-01",
        "requirements": {
            "art12_logging": {
                "description": "Automatic logging of AI system operations",
                "fields_required": [
                    "timestamp", "model", "provider", "prompt_hash",
                    "response_hash", "tokens_in", "tokens_out",
                ],
                "fields_recommended": ["decision_type", "vertical", "metadata"],
                "severity": "mandatory",
            },
            "art13_transparency": {
                "description": "Transparency and provision of information to deployers",
                "fields_required": ["explainability"],
                "fields_recommended": ["ai_score", "ai_threshold", "decision_outcome"],
                "severity": "mandatory",
            },
            "art14_human_oversight": {
                "description": "Human oversight measures",
                "fields_required": ["human_review_state"],
                "fields_recommended": [
                    "reviewer_id", "reviewer_name", "reviewer_role",
                    "reviewer_authority_scope", "reviewer_competence_level",
                    "reviewer_training_date", "review_duration_seconds",
                    "review_method", "review_outcome",
                ],
                "severity": "mandatory",
            },
            "art9_risk_management": {
                "description": "Risk management system requirements",
                "fields_required": ["bias_audit"],
                "fields_recommended": ["applicable_jurisdictions", "compliance_metadata"],
                "severity": "mandatory",
            },
            "art26_deployer_obligations": {
                "description": "Obligations of deployers of high-risk AI",
                "fields_required": ["human_review_state", "applicable_jurisdictions"],
                "fields_recommended": ["dpia_reference"],
                "severity": "mandatory",
            },
        },
    },
    "nist_ai_rmf": {
        "name": "NIST AI Risk Management Framework 1.0",
        "effective_date": "2023-01-26",
        "requirements": {
            "govern_1": {
                "description": "Policies, processes, procedures, and practices are in place",
                "fields_required": ["compliance_metadata"],
                "fields_recommended": ["applicable_jurisdictions"],
                "severity": "recommended",
            },
            "map_1": {
                "description": "Context is established and understood",
                "fields_required": ["decision_type", "vertical"],
                "fields_recommended": ["decision_category", "decision_outcome"],
                "severity": "recommended",
            },
            "measure_2": {
                "description": "AI systems are evaluated for trustworthy characteristics",
                "fields_required": ["bias_audit"],
                "fields_recommended": ["ai_score", "ai_threshold", "explainability"],
                "severity": "recommended",
            },
            "manage_1": {
                "description": "AI risks based on assessments are prioritized and acted upon",
                "fields_required": ["human_review_state"],
                "fields_recommended": [
                    "reviewer_id", "review_outcome", "override_reason",
                ],
                "severity": "recommended",
            },
            "manage_4": {
                "description": "Negative residual risks are documented",
                "fields_required": ["metadata"],
                "fields_recommended": ["explainability", "bias_audit"],
                "severity": "recommended",
            },
        },
    },
    "iso_42001": {
        "name": "ISO/IEC 42001:2023 AI Management System",
        "effective_date": "2023-12-18",
        "requirements": {
            "clause_6_risk": {
                "description": "Actions to address AI risks and opportunities",
                "fields_required": ["bias_audit", "human_review_state"],
                "fields_recommended": ["explainability", "compliance_metadata"],
                "severity": "mandatory",
            },
            "clause_8_operation": {
                "description": "Operational planning and control",
                "fields_required": [
                    "timestamp", "model", "provider",
                    "prompt_hash", "response_hash",
                ],
                "fields_recommended": ["tokens_in", "tokens_out", "metadata"],
                "severity": "mandatory",
            },
            "clause_9_evaluation": {
                "description": "Performance evaluation and monitoring",
                "fields_required": ["ai_score"],
                "fields_recommended": ["ai_threshold", "review_outcome"],
                "severity": "mandatory",
            },
            "annex_b_impact": {
                "description": "AI system impact assessment",
                "fields_required": ["decision_category", "decision_outcome"],
                "fields_recommended": [
                    "affected_individual_notified", "appeal_available",
                    "bias_audit",
                ],
                "severity": "recommended",
            },
        },
    },
    "iso_27001": {
        "name": "ISO/IEC 27001:2022 Information Security",
        "effective_date": "2022-10-25",
        "requirements": {
            "a8_16_monitoring": {
                "description": "Monitoring activities — networks, systems, applications",
                "fields_required": ["timestamp", "entry_hash", "prev_hash", "sequence_no"],
                "fields_recommended": ["signature", "merkle_proof"],
                "severity": "mandatory",
            },
            "a8_10_data_deletion": {
                "description": "Information deletion (supports GDPR right to erasure)",
                "fields_required": ["data_subject_id_hash"],
                "fields_recommended": [
                    "retention_legal_basis", "data_processing_purpose",
                ],
                "severity": "mandatory",
            },
            "a5_34_privacy": {
                "description": "Privacy and protection of PII",
                "fields_required": ["data_processing_basis", "data_processing_purpose"],
                "fields_recommended": [
                    "special_categories_present", "data_subject_consent",
                    "dpia_reference",
                ],
                "severity": "mandatory",
            },
        },
    },
    "gdpr": {
        "name": "GDPR (Regulation 2016/679)",
        "effective_date": "2018-05-25",
        "requirements": {
            "art5_principles": {
                "description": "Principles relating to processing of personal data",
                "fields_required": [
                    "data_processing_basis", "data_processing_purpose",
                    "timestamp",
                ],
                "fields_recommended": ["retention_legal_basis"],
                "severity": "mandatory",
            },
            "art22_automated_decisions": {
                "description": "Automated individual decision-making, including profiling",
                "fields_required": [
                    "human_review_state", "explainability",
                    "data_processing_basis",
                ],
                "fields_recommended": [
                    "appeal_available", "appeal_mechanism",
                    "affected_individual_notified",
                ],
                "severity": "mandatory",
            },
            "art30_records": {
                "description": "Records of processing activities",
                "fields_required": [
                    "data_processing_basis", "data_processing_purpose",
                    "data_subject_id_hash",
                ],
                "fields_recommended": [
                    "special_categories_present", "retention_legal_basis",
                ],
                "severity": "mandatory",
            },
            "art35_dpia": {
                "description": "Data protection impact assessment",
                "fields_required": ["dpia_reference"],
                "fields_recommended": ["bias_audit", "explainability"],
                "severity": "conditional",
            },
            "art17_erasure": {
                "description": "Right to erasure (right to be forgotten)",
                "fields_required": ["data_subject_id_hash"],
                "fields_recommended": ["retention_legal_basis"],
                "severity": "mandatory",
            },
        },
    },
    "colorado_sb24_205": {
        "name": "Colorado SB24-205 (AI Consumer Protections)",
        "effective_date": "2026-02-01",
        "requirements": {
            "s5_impact_assessment": {
                "description": "Impact assessment for high-risk AI systems",
                "fields_required": [
                    "decision_category", "bias_audit", "human_review_state",
                ],
                "fields_recommended": [
                    "decision_outcome", "explainability",
                    "affected_individual_notified",
                ],
                "severity": "mandatory",
            },
            "s6_disclosure": {
                "description": "Consumer disclosure and right to appeal",
                "fields_required": [
                    "affected_individual_notified", "appeal_available",
                ],
                "fields_recommended": [
                    "notification_timestamp", "appeal_mechanism",
                    "correction_opportunity",
                ],
                "severity": "mandatory",
            },
        },
    },
    "nyc_ll144": {
        "name": "NYC Local Law 144 of 2021",
        "effective_date": "2023-07-05",
        "requirements": {
            "bias_audit_annual": {
                "description": "Annual bias audit of AEDT",
                "fields_required": ["bias_audit"],
                "fields_recommended": ["ai_score", "ai_threshold"],
                "severity": "mandatory",
            },
            "candidate_notice": {
                "description": "Notice to candidates about AEDT use",
                "fields_required": [
                    "affected_individual_notified",
                    "ai_disclosure_provided",
                ],
                "fields_recommended": ["notification_timestamp"],
                "severity": "mandatory",
            },
        },
    },
    "eeoc": {
        "name": "EEOC Uniform Guidelines (29 CFR 1607)",
        "effective_date": "1978-09-25",
        "requirements": {
            "adverse_impact_analysis": {
                "description": "Four-fifths rule adverse impact analysis",
                "fields_required": ["bias_audit"],
                "fields_recommended": [
                    "decision_category", "decision_outcome",
                ],
                "severity": "mandatory",
            },
        },
    },
    "cfpb_ecoa": {
        "name": "CFPB / Equal Credit Opportunity Act",
        "effective_date": "1974-10-28",
        "requirements": {
            "adverse_action_notice": {
                "description": "Adverse action notice with specific reasons",
                "fields_required": ["explainability"],
                "fields_recommended": [
                    "adverse_action_notice_sent",
                    "adverse_action_notice_date",
                    "decision_outcome",
                ],
                "severity": "mandatory",
            },
        },
    },
    "illinois_aiva": {
        "name": "Illinois AI Video Interview Act",
        "effective_date": "2020-01-01",
        "requirements": {
            "consent_and_disclosure": {
                "description": "Informed consent and AI disclosure",
                "fields_required": [
                    "consent_obtained", "ai_disclosure_provided",
                ],
                "fields_recommended": ["consent_timestamp"],
                "severity": "mandatory",
            },
        },
    },
    "texas_raiga": {
        "name": "Texas Responsible AI Governance Act",
        "effective_date": "2025-09-01",
        "requirements": {
            "safe_harbor": {
                "description": "Safe harbor through NIST AI RMF compliance",
                "fields_required": ["compliance_metadata"],
                "fields_recommended": ["bias_audit", "human_review_state"],
                "severity": "mandatory",
            },
        },
    },
    "finra": {
        "name": "FINRA Rules 3110, 17a-3, 17a-4",
        "effective_date": "various",
        "requirements": {
            "supervision_records": {
                "description": "Supervisory system and recordkeeping",
                "fields_required": [
                    "timestamp", "entry_hash", "human_review_state",
                    "finra_rule_ref",
                ],
                "fields_recommended": [
                    "reviewer_id", "review_outcome", "signature",
                ],
                "severity": "mandatory",
            },
        },
    },
    "naic": {
        "name": "NAIC Model AI Bulletin",
        "effective_date": "2024-01-01",
        "requirements": {
            "unfair_discrimination": {
                "description": "Testing for unfair discrimination in insurance",
                "fields_required": [
                    "bias_audit", "insurance_line",
                ],
                "fields_recommended": [
                    "actuarial_justification", "decision_outcome",
                ],
                "severity": "mandatory",
            },
        },
    },
    "soc2": {
        "name": "SOC 2 Type II (Trust Services Criteria)",
        "effective_date": "ongoing",
        "requirements": {
            "cc6_logical_access": {
                "description": "Logical and physical access controls",
                "fields_required": ["entry_hash", "signature"],
                "fields_recommended": ["merkle_proof", "verifiable_credential"],
                "severity": "mandatory",
            },
            "cc7_system_operations": {
                "description": "System operations monitoring",
                "fields_required": [
                    "timestamp", "model", "provider",
                    "tokens_in", "tokens_out",
                ],
                "fields_recommended": ["metadata"],
                "severity": "mandatory",
            },
            "cc8_change_management": {
                "description": "Change management controls",
                "fields_required": ["schema_version", "sdk_version"],
                "fields_recommended": ["prev_hash", "sequence_no"],
                "severity": "recommended",
            },
        },
    },
    "w3c_vc_20": {
        "name": "W3C Verifiable Credentials Data Model 2.0",
        "effective_date": "2025-05-15",
        "requirements": {
            "credential_integrity": {
                "description": "Cryptographic integrity and tamper evidence",
                "fields_required": ["entry_hash", "prev_hash", "sequence_no"],
                "fields_recommended": ["signature", "verifiable_credential"],
                "severity": "mandatory",
            },
            "issuer_identification": {
                "description": "Issuer identity via DID or URL",
                "fields_required": ["provider", "schema_version", "sdk_version"],
                "fields_recommended": ["compliance_metadata"],
                "severity": "mandatory",
            },
            "subject_claims": {
                "description": "credentialSubject with verifiable claims",
                "fields_required": [
                    "timestamp", "model", "prompt_hash", "response_hash",
                    "decision_type",
                ],
                "fields_recommended": [
                    "tokens_in", "tokens_out", "vertical",
                ],
                "severity": "mandatory",
            },
            "validity_period": {
                "description": "validFrom/validUntil temporal bounds",
                "fields_required": ["timestamp"],
                "fields_recommended": ["applicable_jurisdictions"],
                "severity": "mandatory",
            },
        },
    },
    "owasp_llm_top10": {
        "name": "OWASP Top 10 for LLM Applications 2025",
        "effective_date": "2025-01-01",
        "requirements": {
            "llm01_prompt_injection": {
                "description": "LLM01: Prompt Injection — detect and log prompt manipulation attempts",
                "fields_required": ["prompt_hash", "prompt_guard"],
                "fields_recommended": ["threat_detection", "policy_evaluation"],
                "severity": "mandatory",
            },
            "llm02_sensitive_info_disclosure": {
                "description": "LLM02: Sensitive Information Disclosure — prevent PII/secrets in outputs",
                "fields_required": ["response_hash", "pii_detected"],
                "fields_recommended": ["data_processing_basis", "data_subject_id_hash"],
                "severity": "mandatory",
            },
            "llm03_supply_chain": {
                "description": "LLM03: Supply Chain Vulnerabilities — track model provenance",
                "fields_required": ["model", "provider", "sdk_version"],
                "fields_recommended": ["schema_version", "compliance_metadata"],
                "severity": "recommended",
            },
            "llm04_data_model_poisoning": {
                "description": "LLM04: Data and Model Poisoning — evidence of input validation",
                "fields_required": ["prompt_hash", "entry_hash"],
                "fields_recommended": ["bias_audit", "threat_detection"],
                "severity": "recommended",
            },
            "llm05_improper_output_handling": {
                "description": "LLM05: Improper Output Handling — log and validate outputs",
                "fields_required": ["response_hash"],
                "fields_recommended": ["policy_evaluation", "content_safety_score"],
                "severity": "mandatory",
            },
            "llm06_excessive_agency": {
                "description": "LLM06: Excessive Agency — human oversight for autonomous actions",
                "fields_required": ["human_review_state"],
                "fields_recommended": [
                    "reviewer_id", "reviewer_authority_scope",
                    "tool_calls_logged",
                ],
                "severity": "mandatory",
            },
            "llm07_system_prompt_leakage": {
                "description": "LLM07: System Prompt Leakage — protect system instructions",
                "fields_required": ["prompt_hash"],
                "fields_recommended": ["prompt_guard", "threat_detection"],
                "severity": "recommended",
            },
            "llm08_vector_embedding_weaknesses": {
                "description": "LLM08: Vector and Embedding Weaknesses — RAG integrity",
                "fields_required": ["prompt_hash", "response_hash"],
                "fields_recommended": ["retrieval_context_hash", "metadata"],
                "severity": "recommended",
            },
            "llm09_misinformation": {
                "description": "LLM09: Misinformation — track factual grounding and hallucination",
                "fields_required": ["response_hash", "explainability"],
                "fields_recommended": ["grounding_score", "retrieval_context_hash"],
                "severity": "mandatory",
            },
            "llm10_unbounded_consumption": {
                "description": "LLM10: Unbounded Consumption — cost and resource controls",
                "fields_required": ["tokens_in", "tokens_out"],
                "fields_recommended": [
                    "cost_usd", "latency_ms", "budget_remaining",
                ],
                "severity": "mandatory",
            },
        },
    },
    "eu_ai_act_gpai": {
        "name": "EU AI Act — General-Purpose AI Model Obligations (Art. 51-56)",
        "effective_date": "2025-08-02",
        "requirements": {
            "art53_technical_documentation": {
                "description": "Technical documentation for GPAI models",
                "fields_required": [
                    "model", "provider", "timestamp",
                    "tokens_in", "tokens_out",
                ],
                "fields_recommended": [
                    "schema_version", "sdk_version", "metadata",
                ],
                "severity": "mandatory",
            },
            "art53_copyright_policy": {
                "description": "Copyright compliance policy for training data",
                "fields_required": ["compliance_metadata"],
                "fields_recommended": ["data_processing_basis"],
                "severity": "mandatory",
            },
            "art55_systemic_risk": {
                "description": "Systemic risk assessment for high-capability GPAI",
                "fields_required": ["bias_audit", "explainability"],
                "fields_recommended": [
                    "threat_detection", "prompt_guard", "policy_evaluation",
                ],
                "severity": "mandatory",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Crosswalk engine
# ---------------------------------------------------------------------------

def crosswalk(
    receipt: Dict[str, Any],
    *,
    frameworks: Optional[List[str]] = None,
    include_recommended: bool = True,
) -> Dict[str, Any]:
    """
    Map a single receipt against multiple compliance frameworks.

    For each framework, evaluates which requirements are satisfied,
    partially satisfied, or missing based on receipt field presence.

    Args:
        receipt: Veratum receipt dictionary.
        frameworks: List of framework IDs to evaluate (default: all).
        include_recommended: Include recommended fields in scoring (default True).

    Returns:
        Crosswalk report:
        {
            "frameworks": {
                "eu_ai_act": {
                    "name": str,
                    "score": float (0-1),
                    "requirements": {
                        "art12_logging": {
                            "status": "met" | "partial" | "not_met",
                            "required_present": [...],
                            "required_missing": [...],
                            "recommended_present": [...],
                            "recommended_missing": [...],
                        },
                    },
                    "gaps": [...],
                },
            },
            "overall_score": float,
            "total_requirements": int,
            "requirements_met": int,
            "requirements_partial": int,
            "requirements_not_met": int,
        }
    """
    target_frameworks = frameworks or list(FRAMEWORKS.keys())
    receipt_fields = _get_receipt_fields(receipt)

    report: Dict[str, Any] = {
        "frameworks": {},
        "overall_score": 0.0,
        "total_requirements": 0,
        "requirements_met": 0,
        "requirements_partial": 0,
        "requirements_not_met": 0,
    }

    total_score = 0.0
    total_reqs = 0

    for fw_id in target_frameworks:
        fw_def = FRAMEWORKS.get(fw_id)
        if not fw_def:
            continue

        fw_report = _evaluate_framework(receipt_fields, fw_def, include_recommended)
        report["frameworks"][fw_id] = fw_report

        # Aggregate
        for req_id, req_data in fw_report["requirements"].items():
            total_reqs += 1
            if req_data["status"] == "met":
                report["requirements_met"] += 1
                total_score += 1.0
            elif req_data["status"] == "partial":
                report["requirements_partial"] += 1
                total_score += 0.5
            else:
                report["requirements_not_met"] += 1

    report["total_requirements"] = total_reqs
    report["overall_score"] = round(total_score / total_reqs, 4) if total_reqs > 0 else 0.0

    return report


def _get_receipt_fields(receipt: Dict[str, Any]) -> Set[str]:
    """
    Get set of non-None, non-empty fields present in the receipt.

    Handles nested dicts and lists — considers the parent field "present"
    if it contains any non-None content.
    """
    present: Set[str] = set()
    for key, value in receipt.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, dict) and not value:
            continue
        if isinstance(value, list) and not value:
            continue
        present.add(key)
    return present


def _evaluate_framework(
    receipt_fields: Set[str],
    fw_def: Dict[str, Any],
    include_recommended: bool,
) -> Dict[str, Any]:
    """Evaluate a single framework against receipt fields."""
    requirements = fw_def.get("requirements", {})
    fw_report: Dict[str, Any] = {
        "name": fw_def["name"],
        "effective_date": fw_def.get("effective_date", ""),
        "score": 0.0,
        "requirements": {},
        "gaps": [],
    }

    total_score = 0.0
    total_reqs = 0

    for req_id, req_def in requirements.items():
        req_fields = req_def.get("fields_required", [])
        rec_fields = req_def.get("fields_recommended", []) if include_recommended else []

        req_present = [f for f in req_fields if f in receipt_fields]
        req_missing = [f for f in req_fields if f not in receipt_fields]
        rec_present = [f for f in rec_fields if f in receipt_fields]
        rec_missing = [f for f in rec_fields if f not in receipt_fields]

        # Status determination
        if not req_missing:
            status = "met"
            score = 1.0
        elif req_present:
            status = "partial"
            score = len(req_present) / len(req_fields) if req_fields else 0.0
        else:
            status = "not_met"
            score = 0.0

        total_score += score
        total_reqs += 1

        fw_report["requirements"][req_id] = {
            "description": req_def.get("description", ""),
            "severity": req_def.get("severity", "mandatory"),
            "status": status,
            "score": round(score, 4),
            "required_present": req_present,
            "required_missing": req_missing,
            "recommended_present": rec_present,
            "recommended_missing": rec_missing,
        }

        if req_missing:
            fw_report["gaps"].append({
                "requirement": req_id,
                "description": req_def.get("description", ""),
                "severity": req_def.get("severity", "mandatory"),
                "missing_fields": req_missing,
            })

    fw_report["score"] = round(total_score / total_reqs, 4) if total_reqs > 0 else 0.0
    return fw_report


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def list_frameworks() -> List[Dict[str, str]]:
    """
    List all supported frameworks.

    Returns:
        List of {"id": str, "name": str, "effective_date": str}.
    """
    return [
        {
            "id": fw_id,
            "name": fw_def["name"],
            "effective_date": fw_def.get("effective_date", ""),
        }
        for fw_id, fw_def in FRAMEWORKS.items()
    ]


def get_required_fields(framework_id: str) -> List[str]:
    """
    Get all required fields for a specific framework.

    Args:
        framework_id: Framework identifier (e.g., "eu_ai_act").

    Returns:
        Deduplicated list of required field names.
    """
    fw = FRAMEWORKS.get(framework_id, {})
    fields: Set[str] = set()
    for req in fw.get("requirements", {}).values():
        fields.update(req.get("fields_required", []))
    return sorted(fields)


def get_gaps_for_frameworks(
    receipt: Dict[str, Any],
    framework_ids: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Quick gap analysis for specific frameworks.

    Args:
        receipt: Receipt dictionary.
        framework_ids: List of framework IDs.

    Returns:
        Dict mapping framework_id → list of gaps.
    """
    report = crosswalk(receipt, frameworks=framework_ids)
    return {
        fw_id: fw_data.get("gaps", [])
        for fw_id, fw_data in report.get("frameworks", {}).items()
    }
