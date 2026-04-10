"""
Automated Data Protection Impact Assessment (DPIA) generation.

Generates GDPR Article 35 compliant DPIAs from Veratum receipt data.
No competitor generates DPIAs automatically from evidence — Credo AI
and Certainity require manual questionnaire-based workflows.

Veratum's approach: your AI system already generates receipts with
decision types, data processing bases, bias audits, and human review
states. This module assembles that evidence into a structured DPIA
that satisfies GDPR Article 35(7) requirements:

  (a) systematic description of processing operations and purposes
  (b) assessment of necessity and proportionality
  (c) assessment of risks to rights and freedoms
  (d) measures to address risks (safeguards, security, mechanisms)

Also supports:
- EU AI Act Article 26(9) — deployer DPIA for high-risk AI
- Colorado SB24-205 §5 — impact assessment for high-risk AI
- Canada AIDA — algorithmic impact assessment

Example:
    >>> from veratum.dpia import DPIAGenerator
    >>>
    >>> generator = DPIAGenerator()
    >>> receipts = sdk.get_receipts(limit=1000)
    >>> dpia = generator.generate(
    ...     receipts=receipts,
    ...     system_name="Hiring Screening AI",
    ...     system_description="Automated resume screening for engineering roles",
    ...     data_controller="Acme Corp",
    ...     dpo_contact="dpo@acme.com",
    ... )
    >>> print(dpia["risk_level"])  # "high"
    >>> print(dpia["article35_compliant"])  # True
    >>> dpia_text = generator.to_markdown(dpia)
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("veratum.dpia")


# ---------------------------------------------------------------------------
# DPIA data structures
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    """A single risk identified in the DPIA."""
    risk_id: str
    category: str  # "rights_and_freedoms", "discrimination", "transparency", "security", "accuracy"
    description: str
    likelihood: str  # "low", "medium", "high"
    severity: str  # "low", "medium", "high"
    risk_level: str  # "low", "medium", "high", "critical"
    mitigation: str
    mitigation_status: str  # "implemented", "partial", "planned", "none"
    evidence_references: List[str] = field(default_factory=list)
    regulation_references: List[str] = field(default_factory=list)


@dataclass
class DPIAReport:
    """Complete DPIA report structure per GDPR Article 35(7)."""

    # Header
    dpia_id: str = ""
    generated_at: str = ""
    system_name: str = ""
    system_description: str = ""
    data_controller: str = ""
    dpo_contact: str = ""
    assessment_period_start: str = ""
    assessment_period_end: str = ""

    # Article 35(7)(a) — Systematic description
    processing_description: Dict[str, Any] = field(default_factory=dict)

    # Article 35(7)(b) — Necessity and proportionality
    necessity_assessment: Dict[str, Any] = field(default_factory=dict)

    # Article 35(7)(c) — Risk assessment
    risks: List[RiskAssessment] = field(default_factory=list)
    overall_risk_level: str = "unknown"

    # Article 35(7)(d) — Mitigation measures
    safeguards: List[Dict[str, Any]] = field(default_factory=list)

    # Evidence summary
    evidence_summary: Dict[str, Any] = field(default_factory=dict)

    # Compliance status
    article35_compliant: bool = False
    frameworks_assessed: List[str] = field(default_factory=list)
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["risks"] = [asdict(r) for r in self.risks]
        return d


# ---------------------------------------------------------------------------
# Risk level matrix
# ---------------------------------------------------------------------------

_RISK_MATRIX: Dict[Tuple[str, str], str] = {
    ("low", "low"): "low",
    ("low", "medium"): "low",
    ("low", "high"): "medium",
    ("medium", "low"): "low",
    ("medium", "medium"): "medium",
    ("medium", "high"): "high",
    ("high", "low"): "medium",
    ("high", "medium"): "high",
    ("high", "high"): "critical",
}


# ---------------------------------------------------------------------------
# High-risk verticals per EU AI Act Annex III
# ---------------------------------------------------------------------------

HIGH_RISK_VERTICALS: Set[str] = {
    "employment", "hiring", "hr", "recruitment",
    "lending", "credit", "finance",
    "insurance", "underwriting",
    "healthcare", "medical", "triage",
    "education", "academic",
    "criminal_justice", "law_enforcement",
    "border_control", "immigration",
    "housing",
    "social_scoring",
    "biometric",
}

CONSEQUENTIAL_DECISION_TYPES: Set[str] = {
    "employment_screening", "hiring", "promotion", "termination",
    "credit_decision", "loan_application", "loan_approval",
    "insurance_underwriting", "claims_decision",
    "medical_diagnosis", "triage",
    "benefits_eligibility", "housing_application",
    "criminal_risk_assessment", "sentencing",
    "academic_admission", "grading",
}


# ---------------------------------------------------------------------------
# DPIA Generator
# ---------------------------------------------------------------------------

class DPIAGenerator:
    """
    Generates GDPR Article 35 compliant DPIAs from Veratum receipt data.

    Analyzes receipt evidence to automatically:
    - Identify data processing operations and their bases
    - Assess risks to data subjects' rights and freedoms
    - Evaluate existing safeguards (human review, bias audits, etc.)
    - Generate mitigation recommendations
    - Produce a structured DPIA document

    This replaces manual questionnaire-based DPIA workflows.
    """

    def __init__(
        self,
        *,
        risk_threshold: float = 0.5,
        include_evidence_hashes: bool = True,
    ):
        self.risk_threshold = risk_threshold
        self.include_evidence_hashes = include_evidence_hashes

    def generate(
        self,
        receipts: List[Dict[str, Any]],
        system_name: str,
        system_description: str = "",
        data_controller: str = "",
        dpo_contact: str = "",
        *,
        frameworks: Optional[List[str]] = None,
    ) -> DPIAReport:
        """
        Generate a complete DPIA from receipt evidence.

        Args:
            receipts: List of Veratum receipt dictionaries.
            system_name: Name of the AI system being assessed.
            system_description: Description of the system's purpose.
            data_controller: Name of the data controller organization.
            dpo_contact: Contact info for the Data Protection Officer.
            frameworks: Additional frameworks to assess (default: GDPR + EU AI Act).

        Returns:
            DPIAReport with complete assessment.
        """
        if not receipts:
            raise ValueError("At least one receipt is required to generate a DPIA")

        report = DPIAReport(
            dpia_id=f"dpia_{hashlib.sha256(json.dumps(system_name).encode()).hexdigest()[:12]}",
            generated_at=datetime.now(timezone.utc).isoformat(),
            system_name=system_name,
            system_description=system_description,
            data_controller=data_controller,
            dpo_contact=dpo_contact,
        )

        # Determine assessment period from receipt timestamps
        timestamps = [r.get("timestamp", "") for r in receipts if r.get("timestamp")]
        if timestamps:
            report.assessment_period_start = min(timestamps)
            report.assessment_period_end = max(timestamps)

        # (a) Systematic description of processing
        report.processing_description = self._describe_processing(receipts)

        # (b) Necessity and proportionality
        report.necessity_assessment = self._assess_necessity(receipts)

        # (c) Risk assessment
        report.risks = self._assess_risks(receipts)
        report.overall_risk_level = self._compute_overall_risk(report.risks)

        # (d) Safeguards and measures
        report.safeguards = self._identify_safeguards(receipts)

        # Evidence summary
        report.evidence_summary = self._summarize_evidence(receipts)

        # Compliance check
        report.frameworks_assessed = frameworks or ["gdpr", "eu_ai_act"]
        report.gaps = self._identify_gaps(receipts, report)
        report.recommendations = self._generate_recommendations(report)
        report.article35_compliant = self._check_article35_compliance(report)

        return report

    # --- (a) Systematic description ----------------------------------------

    def _describe_processing(self, receipts: List[Dict]) -> Dict[str, Any]:
        """Describe processing operations from receipt evidence."""
        models = set()
        providers = set()
        decision_types = set()
        verticals = set()
        processing_bases = set()
        processing_purposes = set()
        total_decisions = len(receipts)
        total_tokens_in = 0
        total_tokens_out = 0

        for r in receipts:
            if r.get("model"):
                models.add(r["model"])
            if r.get("provider"):
                providers.add(r["provider"])
            if r.get("decision_type"):
                decision_types.add(r["decision_type"])
            if r.get("vertical"):
                verticals.add(r["vertical"])
            if r.get("data_processing_basis"):
                processing_bases.add(r["data_processing_basis"])
            if r.get("data_processing_purpose"):
                processing_purposes.add(r["data_processing_purpose"])
            total_tokens_in += r.get("tokens_in", 0) or 0
            total_tokens_out += r.get("tokens_out", 0) or 0

        # Check for personal data processing indicators
        has_personal_data = any(
            r.get("data_subject_id_hash") or r.get("pii_detected")
            for r in receipts
        )
        has_special_categories = any(
            r.get("special_categories_present") for r in receipts
        )

        return {
            "ai_models_used": sorted(models),
            "providers": sorted(providers),
            "decision_types": sorted(decision_types),
            "verticals": sorted(verticals),
            "processing_bases": sorted(processing_bases),
            "processing_purposes": sorted(processing_purposes),
            "total_decisions_assessed": total_decisions,
            "total_tokens_processed": total_tokens_in + total_tokens_out,
            "personal_data_processed": has_personal_data,
            "special_categories_present": has_special_categories,
            "automated_decision_making": any(
                r.get("decision_outcome") for r in receipts
            ),
            "profiling_detected": any(
                r.get("decision_type") in CONSEQUENTIAL_DECISION_TYPES
                for r in receipts
            ),
        }

    # --- (b) Necessity and proportionality ----------------------------------

    def _assess_necessity(self, receipts: List[Dict]) -> Dict[str, Any]:
        """Assess necessity and proportionality of processing."""
        has_purpose = any(r.get("data_processing_purpose") for r in receipts)
        has_basis = any(r.get("data_processing_basis") for r in receipts)
        has_explanation = any(r.get("explainability") for r in receipts)

        # Check data minimization — are tokens reasonable?
        tokens_in = [r.get("tokens_in", 0) or 0 for r in receipts]
        avg_tokens = statistics.mean(tokens_in) if tokens_in else 0

        return {
            "purpose_documented": has_purpose,
            "legal_basis_documented": has_basis,
            "explainability_provided": has_explanation,
            "average_input_tokens": round(avg_tokens),
            "data_minimization_assessment": (
                "adequate" if avg_tokens < 10000
                else "review_recommended" if avg_tokens < 50000
                else "excessive_data_collection_risk"
            ),
            "proportionality_score": sum([
                0.25 if has_purpose else 0,
                0.25 if has_basis else 0,
                0.25 if has_explanation else 0,
                0.25 if avg_tokens < 10000 else 0.1,
            ]),
        }

    # --- (c) Risk assessment ------------------------------------------------

    def _assess_risks(self, receipts: List[Dict]) -> List[RiskAssessment]:
        """Identify and assess risks from receipt evidence."""
        risks: List[RiskAssessment] = []

        # Risk 1: Discrimination / bias
        risks.append(self._assess_discrimination_risk(receipts))

        # Risk 2: Lack of transparency
        risks.append(self._assess_transparency_risk(receipts))

        # Risk 3: Insufficient human oversight
        risks.append(self._assess_oversight_risk(receipts))

        # Risk 4: Data security
        risks.append(self._assess_security_risk(receipts))

        # Risk 5: Accuracy / hallucination
        risks.append(self._assess_accuracy_risk(receipts))

        # Risk 6: Rights restriction (automated decisions)
        risks.append(self._assess_rights_risk(receipts))

        # Risk 7: Prompt injection / adversarial attacks
        risks.append(self._assess_adversarial_risk(receipts))

        # Risk 8: Excessive data collection
        risks.append(self._assess_data_minimization_risk(receipts))

        return risks

    def _assess_discrimination_risk(self, receipts: List[Dict]) -> RiskAssessment:
        """Assess risk of discriminatory outcomes."""
        has_bias_audit = sum(1 for r in receipts if r.get("bias_audit"))
        audit_rate = has_bias_audit / len(receipts) if receipts else 0
        is_high_risk = any(
            r.get("vertical") in HIGH_RISK_VERTICALS for r in receipts
        )

        if audit_rate > 0.8:
            likelihood, mitigation_status = "low", "implemented"
        elif audit_rate > 0.3:
            likelihood, mitigation_status = "medium", "partial"
        else:
            likelihood, mitigation_status = "high", "none"

        severity = "high" if is_high_risk else "medium"
        risk_level = _RISK_MATRIX.get((likelihood, severity), "medium")

        return RiskAssessment(
            risk_id="RISK-001",
            category="discrimination",
            description=(
                "Risk of discriminatory outcomes from AI-assisted decisions. "
                f"{'HIGH-RISK vertical detected. ' if is_high_risk else ''}"
                f"Bias audit coverage: {audit_rate:.0%} of decisions."
            ),
            likelihood=likelihood,
            severity=severity,
            risk_level=risk_level,
            mitigation=(
                f"Bias audit conducted on {audit_rate:.0%} of decisions. "
                "EEOC 4/5ths rule and adverse impact analysis available."
            ),
            mitigation_status=mitigation_status,
            evidence_references=[f"{has_bias_audit} receipts with bias_audit"],
            regulation_references=[
                "GDPR Article 22", "EU AI Act Article 9",
                "EEOC 29 CFR 1607", "NYC LL144",
            ],
        )

    def _assess_transparency_risk(self, receipts: List[Dict]) -> RiskAssessment:
        """Assess risk of insufficient transparency."""
        has_explanation = sum(1 for r in receipts if r.get("explainability"))
        explain_rate = has_explanation / len(receipts) if receipts else 0

        if explain_rate > 0.8:
            likelihood, mitigation_status = "low", "implemented"
        elif explain_rate > 0.3:
            likelihood, mitigation_status = "medium", "partial"
        else:
            likelihood, mitigation_status = "high", "none"

        return RiskAssessment(
            risk_id="RISK-002",
            category="transparency",
            description=(
                f"Risk of opaque AI decision-making. "
                f"Explainability coverage: {explain_rate:.0%}."
            ),
            likelihood=likelihood,
            severity="medium",
            risk_level=_RISK_MATRIX.get((likelihood, "medium"), "medium"),
            mitigation=(
                f"Explainability data present in {explain_rate:.0%} of receipts."
            ),
            mitigation_status=mitigation_status,
            evidence_references=[f"{has_explanation} receipts with explainability"],
            regulation_references=[
                "GDPR Article 13/14", "EU AI Act Article 13",
            ],
        )

    def _assess_oversight_risk(self, receipts: List[Dict]) -> RiskAssessment:
        """Assess risk of insufficient human oversight."""
        has_review = sum(
            1 for r in receipts
            if r.get("human_review_state") and r["human_review_state"] != "none"
        )
        review_rate = has_review / len(receipts) if receipts else 0

        # Higher risk if consequential decisions lack review
        consequential_no_review = sum(
            1 for r in receipts
            if r.get("decision_type") in CONSEQUENTIAL_DECISION_TYPES
            and (not r.get("human_review_state") or r["human_review_state"] == "none")
        )

        if consequential_no_review > 0:
            likelihood = "high"
            severity = "high"
        elif review_rate > 0.7:
            likelihood = "low"
            severity = "medium"
        else:
            likelihood = "medium"
            severity = "medium"

        return RiskAssessment(
            risk_id="RISK-003",
            category="rights_and_freedoms",
            description=(
                f"Risk of insufficient human oversight. "
                f"Human review rate: {review_rate:.0%}. "
                f"Consequential decisions without review: {consequential_no_review}."
            ),
            likelihood=likelihood,
            severity=severity,
            risk_level=_RISK_MATRIX.get((likelihood, severity), "medium"),
            mitigation=(
                f"Human review implemented for {review_rate:.0%} of decisions."
            ),
            mitigation_status="implemented" if review_rate > 0.7 else "partial",
            evidence_references=[
                f"{has_review} receipts with human_review_state",
                f"{consequential_no_review} consequential decisions without review",
            ],
            regulation_references=[
                "GDPR Article 22(3)", "EU AI Act Article 14",
                "Colorado SB24-205 §5",
            ],
        )

    def _assess_security_risk(self, receipts: List[Dict]) -> RiskAssessment:
        """Assess data security risk."""
        has_hash_chain = sum(
            1 for r in receipts if r.get("entry_hash") and r.get("prev_hash")
        )
        chain_rate = has_hash_chain / len(receipts) if receipts else 0
        has_signature = sum(1 for r in receipts if r.get("signature"))

        if chain_rate > 0.9 and has_signature > 0:
            likelihood = "low"
            mitigation_status = "implemented"
        elif chain_rate > 0.5:
            likelihood = "medium"
            mitigation_status = "partial"
        else:
            likelihood = "high"
            mitigation_status = "none"

        return RiskAssessment(
            risk_id="RISK-004",
            category="security",
            description=(
                f"Risk of evidence tampering or data breach. "
                f"Hash chain coverage: {chain_rate:.0%}."
            ),
            likelihood=likelihood,
            severity="high",
            risk_level=_RISK_MATRIX.get((likelihood, "high"), "high"),
            mitigation=(
                f"Cryptographic hash chain on {chain_rate:.0%} of receipts. "
                f"{has_signature} receipts with HMAC signatures."
            ),
            mitigation_status=mitigation_status,
            evidence_references=[
                f"{has_hash_chain} receipts in hash chain",
                f"{has_signature} signed receipts",
            ],
            regulation_references=[
                "GDPR Article 32", "ISO/IEC 27001 A.8.16",
            ],
        )

    def _assess_accuracy_risk(self, receipts: List[Dict]) -> RiskAssessment:
        """Assess accuracy and hallucination risk."""
        has_grounding = sum(
            1 for r in receipts
            if r.get("grounding_score") or r.get("retrieval_context_hash")
        )
        grounding_rate = has_grounding / len(receipts) if receipts else 0

        return RiskAssessment(
            risk_id="RISK-005",
            category="accuracy",
            description=(
                f"Risk of inaccurate or hallucinated AI outputs. "
                f"Grounding/RAG evidence: {grounding_rate:.0%}."
            ),
            likelihood="medium" if grounding_rate > 0.5 else "high",
            severity="medium",
            risk_level="medium",
            mitigation=(
                f"Factual grounding tracked in {grounding_rate:.0%} of receipts."
            ),
            mitigation_status="implemented" if grounding_rate > 0.5 else "planned",
            evidence_references=[f"{has_grounding} receipts with grounding data"],
            regulation_references=["OWASP LLM09 Misinformation"],
        )

    def _assess_rights_risk(self, receipts: List[Dict]) -> RiskAssessment:
        """Assess risk of restricting data subjects' rights."""
        has_appeal = sum(1 for r in receipts if r.get("appeal_available"))
        has_notification = sum(
            1 for r in receipts if r.get("affected_individual_notified")
        )
        appeal_rate = has_appeal / len(receipts) if receipts else 0
        notification_rate = has_notification / len(receipts) if receipts else 0

        if appeal_rate > 0.8 and notification_rate > 0.8:
            likelihood = "low"
        elif appeal_rate > 0.3 or notification_rate > 0.3:
            likelihood = "medium"
        else:
            likelihood = "high"

        return RiskAssessment(
            risk_id="RISK-006",
            category="rights_and_freedoms",
            description=(
                f"Risk of restricting data subjects' rights. "
                f"Appeal mechanism: {appeal_rate:.0%}. "
                f"Individual notification: {notification_rate:.0%}."
            ),
            likelihood=likelihood,
            severity="high",
            risk_level=_RISK_MATRIX.get((likelihood, "high"), "high"),
            mitigation=(
                f"Appeal available in {appeal_rate:.0%} of decisions. "
                f"Individuals notified in {notification_rate:.0%}."
            ),
            mitigation_status="implemented" if appeal_rate > 0.8 else "partial",
            evidence_references=[
                f"{has_appeal} receipts with appeal_available",
                f"{has_notification} receipts with notification",
            ],
            regulation_references=[
                "GDPR Articles 13, 14, 22", "Colorado SB24-205 §6",
            ],
        )

    def _assess_adversarial_risk(self, receipts: List[Dict]) -> RiskAssessment:
        """Assess prompt injection / adversarial attack risk."""
        has_prompt_guard = sum(1 for r in receipts if r.get("prompt_guard"))
        has_threat_detection = sum(
            1 for r in receipts if r.get("threat_detection")
        )
        guard_rate = has_prompt_guard / len(receipts) if receipts else 0

        return RiskAssessment(
            risk_id="RISK-007",
            category="security",
            description=(
                f"Risk of adversarial attacks (prompt injection, jailbreaking). "
                f"Prompt guard coverage: {guard_rate:.0%}."
            ),
            likelihood="medium" if guard_rate > 0.5 else "high",
            severity="high",
            risk_level="high" if guard_rate < 0.5 else "medium",
            mitigation=(
                f"Prompt guard active on {guard_rate:.0%} of requests. "
                f"{has_threat_detection} requests with threat detection."
            ),
            mitigation_status="implemented" if guard_rate > 0.8 else "partial" if guard_rate > 0 else "planned",
            evidence_references=[
                f"{has_prompt_guard} receipts with prompt_guard",
                f"{has_threat_detection} receipts with threat_detection",
            ],
            regulation_references=[
                "OWASP LLM01 Prompt Injection",
                "EU AI Act Article 15 (Accuracy, Robustness, Cybersecurity)",
            ],
        )

    def _assess_data_minimization_risk(self, receipts: List[Dict]) -> RiskAssessment:
        """Assess data minimization risk."""
        tokens_in = [r.get("tokens_in", 0) or 0 for r in receipts]
        avg_tokens = statistics.mean(tokens_in) if tokens_in else 0
        has_pii = sum(1 for r in receipts if r.get("pii_detected"))

        if avg_tokens < 5000 and has_pii == 0:
            likelihood = "low"
        elif avg_tokens < 20000:
            likelihood = "medium"
        else:
            likelihood = "high"

        return RiskAssessment(
            risk_id="RISK-008",
            category="rights_and_freedoms",
            description=(
                f"Risk of excessive data collection. "
                f"Average input: {avg_tokens:.0f} tokens. "
                f"PII detected in {has_pii} requests."
            ),
            likelihood=likelihood,
            severity="medium",
            risk_level=_RISK_MATRIX.get((likelihood, "medium"), "medium"),
            mitigation=(
                f"Token usage tracked. PII detection in {has_pii} receipts."
            ),
            mitigation_status="implemented" if has_pii > 0 else "planned",
            evidence_references=[
                f"Average tokens: {avg_tokens:.0f}",
                f"{has_pii} receipts with PII detection",
            ],
            regulation_references=[
                "GDPR Article 5(1)(c) — data minimization",
            ],
        )

    # --- Risk aggregation ---------------------------------------------------

    def _compute_overall_risk(self, risks: List[RiskAssessment]) -> str:
        """Compute overall risk level from individual assessments."""
        level_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        if not risks:
            return "unknown"

        scores = [level_scores.get(r.risk_level, 2) for r in risks]
        avg = statistics.mean(scores)
        max_score = max(scores)

        # Overall is driven by worst risk but tempered by average
        if max_score >= 4 or avg >= 3:
            return "critical"
        elif max_score >= 3 or avg >= 2.5:
            return "high"
        elif avg >= 1.5:
            return "medium"
        return "low"

    # --- (d) Safeguards -----------------------------------------------------

    def _identify_safeguards(self, receipts: List[Dict]) -> List[Dict[str, Any]]:
        """Identify existing safeguards from receipt evidence."""
        safeguards = []

        # Hash chain integrity
        chained = sum(1 for r in receipts if r.get("entry_hash"))
        if chained > 0:
            safeguards.append({
                "name": "Cryptographic Hash Chain",
                "description": "SHA-256 hash chain ensures tamper-evident evidence trail",
                "type": "technical",
                "status": "active",
                "coverage": f"{chained}/{len(receipts)} receipts",
                "regulation": "ISO/IEC 27001 A.8.16, SOC 2 CC6",
            })

        # Human review
        reviewed = sum(
            1 for r in receipts
            if r.get("human_review_state") and r["human_review_state"] != "none"
        )
        if reviewed > 0:
            safeguards.append({
                "name": "Human Oversight Process",
                "description": "Human review of AI decisions before or after execution",
                "type": "organizational",
                "status": "active",
                "coverage": f"{reviewed}/{len(receipts)} decisions reviewed",
                "regulation": "EU AI Act Article 14, GDPR Article 22",
            })

        # Bias audit
        audited = sum(1 for r in receipts if r.get("bias_audit"))
        if audited > 0:
            safeguards.append({
                "name": "Bias Audit Program",
                "description": "Statistical bias analysis using EEOC 4/5ths rule",
                "type": "technical",
                "status": "active",
                "coverage": f"{audited}/{len(receipts)} decisions audited",
                "regulation": "EEOC 29 CFR 1607, NYC LL144",
            })

        # Policy engine
        has_policy = sum(1 for r in receipts if r.get("policy_evaluation"))
        if has_policy > 0:
            safeguards.append({
                "name": "Pre-Execution Policy Engine",
                "description": "Real-time policy enforcement blocks non-compliant decisions",
                "type": "technical",
                "status": "active",
                "coverage": f"{has_policy}/{len(receipts)} decisions evaluated",
                "regulation": "EU AI Act Article 9",
            })

        # PII redaction
        has_pii_redaction = sum(
            1 for r in receipts if r.get("pii_detected") is not None
        )
        if has_pii_redaction > 0:
            safeguards.append({
                "name": "PII Detection and Redaction",
                "description": "Automated detection and handling of personal data in AI inputs/outputs",
                "type": "technical",
                "status": "active",
                "coverage": f"{has_pii_redaction}/{len(receipts)} requests scanned",
                "regulation": "GDPR Article 25 (data protection by design)",
            })

        # Prompt guard
        has_guard = sum(1 for r in receipts if r.get("prompt_guard"))
        if has_guard > 0:
            safeguards.append({
                "name": "Prompt Injection Detection",
                "description": "Automated detection of prompt manipulation attempts",
                "type": "technical",
                "status": "active",
                "coverage": f"{has_guard}/{len(receipts)} requests screened",
                "regulation": "OWASP LLM01, EU AI Act Article 15",
            })

        return safeguards

    # --- Evidence summary ---------------------------------------------------

    def _summarize_evidence(self, receipts: List[Dict]) -> Dict[str, Any]:
        """Summarize the evidence base for this DPIA."""
        return {
            "total_receipts_analyzed": len(receipts),
            "receipt_fields_coverage": self._field_coverage(receipts),
            "unique_models": len(set(r.get("model", "") for r in receipts if r.get("model"))),
            "unique_providers": len(set(r.get("provider", "") for r in receipts if r.get("provider"))),
            "hash_chain_integrity": sum(1 for r in receipts if r.get("entry_hash")) / len(receipts) if receipts else 0,
        }

    def _field_coverage(self, receipts: List[Dict]) -> Dict[str, float]:
        """Calculate coverage rate for key DPIA fields."""
        key_fields = [
            "timestamp", "model", "provider", "prompt_hash", "response_hash",
            "decision_type", "vertical", "human_review_state", "bias_audit",
            "explainability", "data_processing_basis", "entry_hash",
            "policy_evaluation", "prompt_guard", "pii_detected",
        ]
        coverage = {}
        for f in key_fields:
            present = sum(1 for r in receipts if r.get(f) is not None)
            coverage[f] = round(present / len(receipts), 4) if receipts else 0
        return coverage

    # --- Gap analysis -------------------------------------------------------

    def _identify_gaps(
        self, receipts: List[Dict], report: DPIAReport
    ) -> List[Dict[str, Any]]:
        """Identify compliance gaps."""
        gaps = []

        if not any(r.get("data_processing_basis") for r in receipts):
            gaps.append({
                "field": "data_processing_basis",
                "severity": "critical",
                "regulation": "GDPR Article 6",
                "description": "No legal basis for data processing documented",
            })

        if not any(r.get("data_processing_purpose") for r in receipts):
            gaps.append({
                "field": "data_processing_purpose",
                "severity": "high",
                "regulation": "GDPR Article 5(1)(b)",
                "description": "Processing purpose not documented",
            })

        # Check high-risk verticals without bias audit
        high_risk_no_audit = sum(
            1 for r in receipts
            if r.get("vertical") in HIGH_RISK_VERTICALS and not r.get("bias_audit")
        )
        if high_risk_no_audit > 0:
            gaps.append({
                "field": "bias_audit",
                "severity": "critical",
                "regulation": "EU AI Act Article 9, EEOC",
                "description": f"{high_risk_no_audit} high-risk decisions without bias audit",
            })

        # Consequential decisions without human review
        consequential_no_review = sum(
            1 for r in receipts
            if r.get("decision_type") in CONSEQUENTIAL_DECISION_TYPES
            and (not r.get("human_review_state") or r["human_review_state"] == "none")
        )
        if consequential_no_review > 0:
            gaps.append({
                "field": "human_review_state",
                "severity": "critical",
                "regulation": "EU AI Act Article 14, GDPR Article 22",
                "description": f"{consequential_no_review} consequential decisions without human review",
            })

        if not any(r.get("prompt_guard") for r in receipts):
            gaps.append({
                "field": "prompt_guard",
                "severity": "medium",
                "regulation": "OWASP LLM01, EU AI Act Article 15",
                "description": "No prompt injection detection implemented",
            })

        return gaps

    # --- Recommendations ----------------------------------------------------

    def _generate_recommendations(self, report: DPIAReport) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        for gap in report.gaps:
            if gap["severity"] == "critical":
                recs.append(
                    f"CRITICAL: {gap['description']}. "
                    f"Required by {gap['regulation']}. "
                    f"Add '{gap['field']}' to all receipts."
                )
            elif gap["severity"] == "high":
                recs.append(
                    f"HIGH: {gap['description']}. "
                    f"Recommended by {gap['regulation']}."
                )

        high_risks = [r for r in report.risks if r.risk_level in ("high", "critical")]
        for risk in high_risks:
            if risk.mitigation_status != "implemented":
                recs.append(
                    f"RISK: {risk.description} "
                    f"Mitigation status: {risk.mitigation_status}."
                )

        if not recs:
            recs.append(
                "No critical gaps identified. Continue monitoring receipt "
                "completeness and maintain bias audit schedule."
            )

        return recs

    # --- Article 35 compliance check ----------------------------------------

    def _check_article35_compliance(self, report: DPIAReport) -> bool:
        """Check if the DPIA satisfies GDPR Article 35(7) requirements."""
        # (a) systematic description
        has_description = bool(report.processing_description.get("decision_types"))
        # (b) necessity
        has_necessity = report.necessity_assessment.get("proportionality_score", 0) > 0
        # (c) risks
        has_risks = len(report.risks) > 0
        # (d) safeguards
        has_safeguards = len(report.safeguards) > 0

        return all([has_description, has_necessity, has_risks, has_safeguards])

    # --- Export -------------------------------------------------------------

    def to_markdown(self, report: DPIAReport) -> str:
        """Export DPIA as Markdown document."""
        lines = [
            f"# Data Protection Impact Assessment",
            f"## {report.system_name}",
            "",
            f"**DPIA ID:** {report.dpia_id}",
            f"**Generated:** {report.generated_at}",
            f"**Data Controller:** {report.data_controller}",
            f"**DPO Contact:** {report.dpo_contact}",
            f"**Assessment Period:** {report.assessment_period_start} to {report.assessment_period_end}",
            f"**Overall Risk Level:** {report.overall_risk_level.upper()}",
            f"**Article 35 Compliant:** {'Yes' if report.article35_compliant else 'No'}",
            "",
            "---",
            "",
            "## 1. Systematic Description of Processing (Art. 35(7)(a))",
            "",
        ]

        pd = report.processing_description
        lines.append(f"- **AI Models:** {', '.join(pd.get('ai_models_used', []))}")
        lines.append(f"- **Providers:** {', '.join(pd.get('providers', []))}")
        lines.append(f"- **Decision Types:** {', '.join(pd.get('decision_types', []))}")
        lines.append(f"- **Verticals:** {', '.join(pd.get('verticals', []))}")
        lines.append(f"- **Total Decisions Assessed:** {pd.get('total_decisions_assessed', 0)}")
        lines.append(f"- **Personal Data Processed:** {'Yes' if pd.get('personal_data_processed') else 'No'}")
        lines.append(f"- **Automated Decision-Making:** {'Yes' if pd.get('automated_decision_making') else 'No'}")
        lines.append("")

        lines.append("## 2. Necessity and Proportionality (Art. 35(7)(b))")
        lines.append("")
        na = report.necessity_assessment
        lines.append(f"- **Legal Basis Documented:** {'Yes' if na.get('legal_basis_documented') else 'No'}")
        lines.append(f"- **Purpose Documented:** {'Yes' if na.get('purpose_documented') else 'No'}")
        lines.append(f"- **Proportionality Score:** {na.get('proportionality_score', 0):.2f}/1.00")
        lines.append("")

        lines.append("## 3. Risk Assessment (Art. 35(7)(c))")
        lines.append("")
        for risk in report.risks:
            lines.append(f"### {risk.risk_id}: {risk.category.replace('_', ' ').title()}")
            lines.append(f"- **Description:** {risk.description}")
            lines.append(f"- **Likelihood:** {risk.likelihood} | **Severity:** {risk.severity} | **Risk Level:** {risk.risk_level.upper()}")
            lines.append(f"- **Mitigation:** {risk.mitigation}")
            lines.append(f"- **Status:** {risk.mitigation_status}")
            lines.append(f"- **Regulations:** {', '.join(risk.regulation_references)}")
            lines.append("")

        lines.append("## 4. Safeguards and Measures (Art. 35(7)(d))")
        lines.append("")
        for s in report.safeguards:
            lines.append(f"### {s['name']}")
            lines.append(f"- {s['description']}")
            lines.append(f"- **Status:** {s['status']} | **Coverage:** {s['coverage']}")
            lines.append(f"- **Regulation:** {s['regulation']}")
            lines.append("")

        if report.gaps:
            lines.append("## 5. Identified Gaps")
            lines.append("")
            for gap in report.gaps:
                lines.append(f"- **[{gap['severity'].upper()}]** {gap['description']} ({gap['regulation']})")
            lines.append("")

        if report.recommendations:
            lines.append("## 6. Recommendations")
            lines.append("")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Generated by Veratum DPIA Engine from {report.evidence_summary.get('total_receipts_analyzed', 0)} evidence receipts.*")

        return "\n".join(lines)

    def to_json(self, report: DPIAReport) -> str:
        """Export DPIA as JSON."""
        return json.dumps(report.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_dpia(
    receipts: List[Dict[str, Any]],
    system_name: str,
    **kwargs,
) -> DPIAReport:
    """
    One-liner DPIA generation.

    Usage:
        from veratum.dpia import generate_dpia

        dpia = generate_dpia(receipts, "My AI System", data_controller="Acme Corp")
        print(dpia.overall_risk_level)
    """
    generator = DPIAGenerator()
    return generator.generate(receipts, system_name, **kwargs)


__all__ = [
    "DPIAGenerator",
    "DPIAReport",
    "RiskAssessment",
    "generate_dpia",
    "HIGH_RISK_VERTICALS",
    "CONSEQUENTIAL_DECISION_TYPES",
]
