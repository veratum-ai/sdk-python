"""
Veratum Policy Prevention Engine.

Evaluates AI decisions against compliance policies BEFORE they take effect.
Blocks decisions that violate configured rules and creates "BLOCKED" receipts
as evidence that harm was actively prevented.

A 'BLOCKED' receipt is the strongest possible compliance evidence.
It proves: (1) the AI system detected a potential violation,
(2) the system actively prevented harm, (3) the exact rule that
was triggered, (4) the exact timestamp of prevention.
Under EU AI Act Article 9 (risk management), demonstrating active
risk mitigation is not just good practice — it is a legal requirement.

Example:
    >>> from veratum.prevention import VeratumPolicyEngine
    >>>
    >>> engine = VeratumPolicyEngine(
    ...     policies=["eu_ai_act_hiring", "eeoc_hiring"],
    ...     custom_rules={"max_rejection_rate": 0.3}
    ... )
    >>>
    >>> result = engine.evaluate(
    ...     decision={"score": 0.35, "outcome": "reject", "decision_type": "screening"},
    ...     context={"vertical": "employment", "jurisdiction": "US_NYC"}
    ... )
    >>>
    >>> if not result.allowed:
    ...     print(f"BLOCKED: {result.blocked_reason}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .bias import four_fifths_rule, impact_ratio, selection_rate

logger = logging.getLogger("veratum.prevention")


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class PolicyViolation:
    """A single policy violation."""

    policy: str
    """Which policy was violated (e.g. 'eeoc_hiring')."""

    rule: str
    """Human-readable description of the violated rule."""

    severity: str
    """'critical' (blocks decision), 'high' (blocks), 'medium' (flags), 'low' (warns)."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional context (metric values, thresholds, etc.)."""


@dataclass
class PolicyResult:
    """Result of evaluating a decision against policies."""

    allowed: bool
    """Whether the decision is allowed to proceed."""

    result: str
    """'allowed', 'blocked', or 'flagged'."""

    violations: List[PolicyViolation] = field(default_factory=list)
    """List of policy violations found."""

    risk_score: float = 0.0
    """Overall risk score (0.0 = safe, 1.0 = maximum risk)."""

    required_actions: List[str] = field(default_factory=list)
    """Actions required before this decision can proceed."""

    blocked_reason: Optional[str] = None
    """If blocked, the primary reason why."""

    policies_checked: List[str] = field(default_factory=list)
    """Which policies were evaluated."""

    evaluation_time_ms: int = 0
    """How long the evaluation took."""

    def to_receipt_fields(self) -> Dict[str, Any]:
        """Convert to receipt policy_evaluation fields."""
        return {
            "policy_evaluation": {
                "policies_checked": self.policies_checked,
                "result": self.result,
                "violations": [
                    {
                        "policy": v.policy,
                        "rule": v.rule,
                        "severity": v.severity,
                        "details": v.details,
                    }
                    for v in self.violations
                ],
                "risk_score": round(self.risk_score, 4),
                "required_actions": self.required_actions,
                "blocked_reason": self.blocked_reason,
                "evaluation_time_ms": self.evaluation_time_ms,
            }
        }


class PolicyViolationError(Exception):
    """Raised when a decision is blocked by the policy engine."""

    def __init__(self, result: PolicyResult):
        self.result = result
        super().__init__(result.blocked_reason or "Decision blocked by policy engine")


# ─── Built-in Policy Definitions ─────────────────────────────────────────────


BUILT_IN_POLICIES: Dict[str, Dict[str, Any]] = {
    "eu_ai_act_hiring": {
        "description": "EU AI Act Article 22 — high-risk AI in employment",
        "verticals": ["employment", "hiring", "hr"],
        "rules": [
            {
                "id": "eu_art14_human_review",
                "description": "Human review required for reject decisions on high-risk AI",
                "check": "_check_eu_human_review",
                "severity": "critical",
            },
            {
                "id": "eu_art13_transparency",
                "description": "AI decision must include explainability information",
                "check": "_check_eu_transparency",
                "severity": "high",
            },
            {
                "id": "eu_art9_risk_threshold",
                "description": "Decisions below confidence threshold require escalation",
                "check": "_check_confidence_threshold",
                "severity": "high",
            },
        ],
    },
    "eeoc_hiring": {
        "description": "US EEOC adverse impact compliance — 4/5ths rule",
        "verticals": ["employment", "hiring", "hr"],
        "rules": [
            {
                "id": "eeoc_four_fifths",
                "description": "4/5ths rule: selection rate for any protected group >= 80% of highest group",
                "check": "_check_four_fifths_rule",
                "severity": "critical",
            },
            {
                "id": "eeoc_adverse_action",
                "description": "Adverse action notice required for all rejections",
                "check": "_check_adverse_action_notice",
                "severity": "high",
            },
        ],
    },
    "nyc_ll144": {
        "description": "NYC Local Law 144 — automated employment decision tools",
        "verticals": ["employment", "hiring", "hr"],
        "rules": [
            {
                "id": "ll144_bias_audit",
                "description": "Annual bias audit required with published results",
                "check": "_check_bias_audit_required",
                "severity": "critical",
            },
            {
                "id": "ll144_candidate_notice",
                "description": "Candidates must be notified of AI use 10 days prior",
                "check": "_check_candidate_notice",
                "severity": "high",
            },
        ],
    },
    "fcra_credit": {
        "description": "Fair Credit Reporting Act — credit decision compliance",
        "verticals": ["finance", "lending", "credit", "insurance"],
        "rules": [
            {
                "id": "fcra_adverse_action",
                "description": "Adverse action notice with specific reason codes required for denials",
                "check": "_check_fcra_adverse_action",
                "severity": "critical",
            },
            {
                "id": "fcra_human_review",
                "description": "Human review required above configurable threshold",
                "check": "_check_human_review_threshold",
                "severity": "high",
            },
        ],
    },
    "hipaa_triage": {
        "description": "HIPAA — AI-assisted medical triage compliance",
        "verticals": ["healthcare", "medical", "triage"],
        "rules": [
            {
                "id": "hipaa_human_oversight",
                "description": "Clinical decisions require licensed professional review",
                "check": "_check_clinical_review",
                "severity": "critical",
            },
            {
                "id": "hipaa_high_risk_escalation",
                "description": "High-severity triage scores must be escalated immediately",
                "check": "_check_triage_escalation",
                "severity": "critical",
            },
        ],
    },
    "naic_insurance": {
        "description": "NAIC Model Law — AI in insurance underwriting",
        "verticals": ["insurance", "underwriting"],
        "rules": [
            {
                "id": "naic_actuarial_justification",
                "description": "Actuarial justification required for risk-based pricing decisions",
                "check": "_check_actuarial_justification",
                "severity": "high",
            },
            {
                "id": "naic_unfair_discrimination",
                "description": "Decision must not unfairly discriminate based on protected class",
                "check": "_check_unfair_discrimination",
                "severity": "critical",
            },
        ],
    },
    "content_safety": {
        "description": "Content moderation safety policies",
        "verticals": ["content_moderation", "content", "moderation"],
        "rules": [
            {
                "id": "csam_block",
                "description": "Block all content flagged for CSAM",
                "check": "_check_csam_flag",
                "severity": "critical",
            },
            {
                "id": "hate_speech_threshold",
                "description": "Block content above hate speech confidence threshold",
                "check": "_check_hate_speech",
                "severity": "critical",
            },
        ],
    },
}


# ─── Policy Engine ────────────────────────────────────────────────────────────


class VeratumPolicyEngine:
    """
    Real-time policy enforcement engine for AI decisions.

    Evaluates decisions against configured compliance policies before they
    take effect. Blocks decisions that violate rules and creates "BLOCKED"
    receipts as evidence of active harm prevention.

    The engine supports built-in policies for common regulations (EU AI Act,
    EEOC, FCRA, HIPAA, etc.) and custom rules for customer-specific needs.
    """

    def __init__(
        self,
        policies: Optional[List[str]] = None,
        custom_rules: Optional[Dict[str, Any]] = None,
        *,
        confidence_threshold: float = 0.5,
        human_review_threshold: float = 0.7,
        block_on_missing_fields: bool = False,
        strict_mode: bool = False,
    ):
        """
        Initialize policy engine.

        Args:
            policies: List of built-in policy names to enforce.
                     e.g. ["eu_ai_act_hiring", "eeoc_hiring", "fcra_credit"]
            custom_rules: Customer-defined additional rules as key-value pairs.
                         e.g. {"max_rejection_rate": 0.3, "require_explanation": True}
            confidence_threshold: Below this AI confidence, require human review.
            human_review_threshold: Above this score, human review is mandatory.
            block_on_missing_fields: If True, block decisions missing required fields.
            strict_mode: If True, any violation blocks. If False, only critical/high block.
        """
        self.active_policies: Dict[str, Dict[str, Any]] = {}
        self.custom_rules = custom_rules or {}
        self.confidence_threshold = confidence_threshold
        self.human_review_threshold = human_review_threshold
        self.block_on_missing_fields = block_on_missing_fields
        self.strict_mode = strict_mode

        # Historical decisions for batch fairness analysis
        self._decision_history: List[Dict[str, Any]] = []
        self._max_history = 10000

        # Load requested built-in policies
        for name in (policies or []):
            if name in BUILT_IN_POLICIES:
                self.active_policies[name] = BUILT_IN_POLICIES[name]
            else:
                available = ", ".join(BUILT_IN_POLICIES.keys())
                raise ValueError(
                    f"Unknown policy '{name}'. Available: {available}"
                )

    @staticmethod
    def list_policies() -> Dict[str, str]:
        """List all available built-in policies with descriptions."""
        return {k: v["description"] for k, v in BUILT_IN_POLICIES.items()}

    def evaluate(
        self,
        decision: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyResult:
        """
        Evaluate a decision against all active policies.

        This is the core method called before every AI decision takes effect.
        It runs all configured policy checks and returns whether the decision
        should be allowed, blocked, or flagged for review.

        Args:
            decision: The AI decision to evaluate. Expected fields:
                - score (float): AI confidence or decision score
                - outcome (str): "approve", "reject", "flag", etc.
                - decision_type (str): Type of decision being made
                - protected_attributes (dict, optional): For fairness checks
                - explanation (str, optional): AI's reasoning
                - adverse_action_codes (list, optional): Reason codes
            context: Additional context for evaluation. Fields:
                - vertical (str): Industry vertical
                - jurisdiction (str): Legal jurisdiction
                - human_review_state (str): Current review status
                - bias_audit (dict): Existing bias audit data
                - candidate_notified (bool): Whether subject was notified

        Returns:
            PolicyResult with allowed/blocked/flagged status and details.
        """
        start = time.time()
        context = context or {}
        violations: List[PolicyViolation] = []
        required_actions: List[str] = []
        policies_checked: List[str] = []

        # Run each active policy's rules
        for policy_name, policy_def in self.active_policies.items():
            policies_checked.append(policy_name)

            for rule in policy_def.get("rules", []):
                check_fn_name = rule["check"]
                check_fn = getattr(self, check_fn_name, None)
                if check_fn is None:
                    logger.warning("Check function %s not found", check_fn_name)
                    continue

                violation = check_fn(decision, context, policy_name, rule)
                if violation:
                    violations.append(violation)

        # Run custom rules
        custom_violations = self._evaluate_custom_rules(decision, context)
        violations.extend(custom_violations)

        # Determine outcome
        critical_violations = [v for v in violations if v.severity in ("critical", "high")]
        medium_violations = [v for v in violations if v.severity == "medium"]

        if critical_violations or (self.strict_mode and violations):
            result = "blocked"
            allowed = False
            blocked_reason = critical_violations[0].rule if critical_violations else violations[0].rule
        elif medium_violations:
            result = "flagged"
            allowed = True
            required_actions.append("human_review_recommended")
        else:
            result = "allowed"
            allowed = True

        # Compute risk score
        risk_score = self._compute_risk_score(violations)

        # Collect required actions from violations
        for v in violations:
            if v.severity in ("critical", "high"):
                if "human_review_required" not in required_actions:
                    required_actions.append("human_review_required")
            action = v.details.get("required_action")
            if action and action not in required_actions:
                required_actions.append(action)

        # Track for historical analysis
        self._track_decision(decision, context, result)

        elapsed_ms = int((time.time() - start) * 1000)

        return PolicyResult(
            allowed=allowed,
            result=result,
            violations=violations,
            risk_score=risk_score,
            required_actions=required_actions,
            blocked_reason=blocked_reason if not allowed else None,
            policies_checked=policies_checked,
            evaluation_time_ms=elapsed_ms,
        )

    # ─── Built-in Rule Checks ─────────────────────────────────────────────────

    def _check_eu_human_review(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """EU AI Act Art.14 — human review required for reject decisions."""
        outcome = decision.get("outcome", "").lower()
        review_state = context.get("human_review_state", "none")

        if outcome in ("reject", "deny", "decline", "terminate") and review_state == "none":
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity=rule["severity"],
                details={
                    "outcome": outcome,
                    "human_review_state": review_state,
                    "required_action": "human_review_required",
                    "regulation": "EU AI Act Article 14",
                },
            )
        return None

    def _check_eu_transparency(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """EU AI Act Art.13 — transparency/explainability required."""
        explanation = decision.get("explanation") or context.get("explanation")
        if not explanation:
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity=rule["severity"],
                details={
                    "required_action": "provide_explanation",
                    "regulation": "EU AI Act Article 13",
                },
            )
        return None

    def _check_confidence_threshold(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """Block low-confidence decisions that need escalation."""
        score = decision.get("score")
        if score is not None and score < self.confidence_threshold:
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity=rule["severity"],
                details={
                    "score": score,
                    "threshold": self.confidence_threshold,
                    "required_action": "escalate_to_human",
                    "regulation": "EU AI Act Article 9",
                },
            )
        return None

    def _check_four_fifths_rule(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """EEOC 4/5ths rule — check against historical batch data."""
        protected = decision.get("protected_attributes", {})
        if not protected:
            return None  # Can't check without demographic data

        # Check if we have enough historical data
        if len(self._decision_history) < 30:
            return None  # Not enough data for statistical significance

        # Run 4/5ths rule from existing bias module
        try:
            batch = self._get_batch_by_group(protected)
            if not batch:
                return None

            for group_name, group_data in batch.items():
                if group_data["total"] == 0:
                    continue
                rate = group_data["selected"] / group_data["total"]

                for other_name, other_data in batch.items():
                    if other_name == group_name or other_data["total"] == 0:
                        continue
                    other_rate = other_data["selected"] / other_data["total"]

                    if other_rate > 0:
                        ratio = rate / other_rate
                        if ratio < 0.8:
                            return PolicyViolation(
                                policy=policy,
                                rule=rule["description"],
                                severity=rule["severity"],
                                details={
                                    "group": group_name,
                                    "reference_group": other_name,
                                    "selection_rate": round(rate, 4),
                                    "reference_rate": round(other_rate, 4),
                                    "impact_ratio": round(ratio, 4),
                                    "threshold": 0.8,
                                    "required_action": "disparate_impact_review",
                                    "regulation": "EEOC 29 CFR 1607.4D",
                                },
                            )
        except Exception as e:
            logger.warning("4/5ths rule check failed: %s", e)

        return None

    def _check_adverse_action_notice(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """EEOC — adverse action notice required for rejections."""
        outcome = decision.get("outcome", "").lower()
        if outcome in ("reject", "deny", "decline"):
            notice_sent = context.get("adverse_action_notice_sent", False)
            if not notice_sent:
                return PolicyViolation(
                    policy=policy,
                    rule=rule["description"],
                    severity=rule["severity"],
                    details={
                        "outcome": outcome,
                        "required_action": "send_adverse_action_notice",
                        "regulation": "EEOC / FCRA",
                    },
                )
        return None

    def _check_bias_audit_required(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """NYC LL144 — bias audit must exist."""
        bias_audit = context.get("bias_audit")
        if not bias_audit:
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity=rule["severity"],
                details={
                    "required_action": "conduct_bias_audit",
                    "regulation": "NYC Local Law 144",
                },
            )
        return None

    def _check_candidate_notice(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """NYC LL144 — candidate must be notified of AI use."""
        notified = context.get("candidate_notified", False)
        if not notified:
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity=rule["severity"],
                details={
                    "required_action": "notify_candidate",
                    "regulation": "NYC Local Law 144 §20-871(b)",
                },
            )
        return None

    def _check_fcra_adverse_action(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """FCRA — adverse action notice with reason codes for denials."""
        outcome = decision.get("outcome", "").lower()
        if outcome in ("reject", "deny", "decline"):
            reason_codes = decision.get("adverse_action_codes") or context.get("adverse_action_codes")
            if not reason_codes:
                return PolicyViolation(
                    policy=policy,
                    rule=rule["description"],
                    severity=rule["severity"],
                    details={
                        "outcome": outcome,
                        "required_action": "provide_adverse_action_notice_with_codes",
                        "regulation": "FCRA §615(a)",
                    },
                )
        return None

    def _check_human_review_threshold(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """Human review required above configurable threshold."""
        score = decision.get("score")
        if score is not None and score > self.human_review_threshold:
            review_state = context.get("human_review_state", "none")
            if review_state == "none":
                return PolicyViolation(
                    policy=policy,
                    rule=rule["description"],
                    severity=rule["severity"],
                    details={
                        "score": score,
                        "threshold": self.human_review_threshold,
                        "required_action": "human_review_required",
                    },
                )
        return None

    def _check_clinical_review(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """HIPAA — clinical decisions require licensed professional review."""
        review_state = context.get("human_review_state", "none")
        reviewer_role = context.get("reviewer_role", "")

        if review_state == "none" or reviewer_role not in (
            "physician", "nurse_practitioner", "physician_assistant", "licensed_clinician"
        ):
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity=rule["severity"],
                details={
                    "human_review_state": review_state,
                    "reviewer_role": reviewer_role,
                    "required_action": "clinical_review_required",
                    "regulation": "HIPAA / FDA AI-ML Guidance",
                },
            )
        return None

    def _check_triage_escalation(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """HIPAA — high-severity triage must escalate."""
        score = decision.get("score", 0)
        severity = decision.get("triage_severity", "").lower()

        if severity in ("critical", "emergency") or score > 0.9:
            escalated = context.get("escalated", False)
            if not escalated:
                return PolicyViolation(
                    policy=policy,
                    rule=rule["description"],
                    severity=rule["severity"],
                    details={
                        "triage_severity": severity,
                        "score": score,
                        "required_action": "immediate_escalation",
                    },
                )
        return None

    def _check_actuarial_justification(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """NAIC — actuarial justification required for pricing."""
        justification = context.get("actuarial_justification")
        if not justification:
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity=rule["severity"],
                details={
                    "required_action": "provide_actuarial_justification",
                    "regulation": "NAIC AI Model Law",
                },
            )
        return None

    def _check_unfair_discrimination(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """NAIC — no unfair discrimination in insurance."""
        protected = decision.get("protected_attributes", {})
        if protected and decision.get("uses_protected_class_data", False):
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity=rule["severity"],
                details={
                    "protected_attributes_used": list(protected.keys()),
                    "required_action": "remove_protected_class_inputs",
                    "regulation": "NAIC AI Model Law §5",
                },
            )
        return None

    def _check_csam_flag(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """Block all CSAM-flagged content."""
        if decision.get("csam_detected", False) or decision.get("csam_score", 0) > 0.01:
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity="critical",
                details={
                    "required_action": "block_and_report",
                    "regulation": "18 USC §2258A / EU CSAM Regulation",
                },
            )
        return None

    def _check_hate_speech(
        self, decision: Dict, context: Dict, policy: str, rule: Dict
    ) -> Optional[PolicyViolation]:
        """Block high-confidence hate speech."""
        hate_score = decision.get("hate_speech_score", 0)
        threshold = self.custom_rules.get("hate_speech_threshold", 0.8)
        if hate_score > threshold:
            return PolicyViolation(
                policy=policy,
                rule=rule["description"],
                severity="critical",
                details={
                    "hate_speech_score": hate_score,
                    "threshold": threshold,
                    "required_action": "block_content",
                    "regulation": "DSA Art. 16",
                },
            )
        return None

    # ─── Custom Rules ─────────────────────────────────────────────────────────

    def _evaluate_custom_rules(
        self, decision: Dict, context: Dict
    ) -> List[PolicyViolation]:
        """Evaluate customer-defined custom rules."""
        violations: List[PolicyViolation] = []

        # Max rejection rate
        max_rej = self.custom_rules.get("max_rejection_rate")
        if max_rej is not None and len(self._decision_history) >= 20:
            recent = self._decision_history[-100:]
            reject_count = sum(
                1 for d in recent if d.get("outcome", "").lower() in ("reject", "deny", "decline")
            )
            reject_rate = reject_count / len(recent)
            if reject_rate > max_rej:
                violations.append(PolicyViolation(
                    policy="custom",
                    rule=f"Rejection rate ({reject_rate:.1%}) exceeds maximum ({max_rej:.1%})",
                    severity="high",
                    details={"reject_rate": round(reject_rate, 4), "max_allowed": max_rej},
                ))

        # Require explanation
        if self.custom_rules.get("require_explanation") and not decision.get("explanation"):
            violations.append(PolicyViolation(
                policy="custom",
                rule="Explanation required for all decisions",
                severity="medium",
                details={"required_action": "provide_explanation"},
            ))

        # Score bounds
        min_score = self.custom_rules.get("min_allowed_score")
        if min_score is not None:
            score = decision.get("score", 1.0)
            if score < min_score:
                violations.append(PolicyViolation(
                    policy="custom",
                    rule=f"Score ({score}) below minimum allowed ({min_score})",
                    severity="high",
                    details={"score": score, "min_allowed": min_score},
                ))

        max_score = self.custom_rules.get("max_allowed_score")
        if max_score is not None:
            score = decision.get("score", 0.0)
            if score > max_score:
                violations.append(PolicyViolation(
                    policy="custom",
                    rule=f"Score ({score}) above maximum allowed ({max_score})",
                    severity="high",
                    details={"score": score, "max_allowed": max_score},
                ))

        return violations

    # ─── Batch Fairness Analysis ──────────────────────────────────────────────

    def check_disparate_impact(
        self,
        decisions_batch: List[Dict[str, Any]],
        protected_attribute: str,
        favorable_outcome: str = "approve",
    ) -> Dict[str, Any]:
        """
        Compute disparate impact metrics from a batch of decisions.

        Uses the same statistical methods as IBM AIF360's disparate_impact metric.
        Returns the ratio and whether it passes the 4/5ths rule.

        Args:
            decisions_batch: List of decision dicts with 'outcome' and
                           protected_attribute fields.
            protected_attribute: Name of the protected attribute to analyze.
            favorable_outcome: What counts as a favorable outcome.

        Returns:
            Dict with:
                - disparate_impact (float): min group rate / max group rate
                - statistical_parity_difference (float): min rate - max rate
                - group_rates (dict): selection rate per group
                - four_fifths_compliant (bool): Whether ratio >= 0.8
                - sample_size (int): Total decisions analyzed
        """
        # Group decisions by protected attribute
        groups: Dict[str, Dict[str, int]] = {}
        for d in decisions_batch:
            group = str(d.get(protected_attribute, "unknown"))
            if group not in groups:
                groups[group] = {"total": 0, "selected": 0}
            groups[group]["total"] += 1
            if d.get("outcome", "").lower() == favorable_outcome.lower():
                groups[group]["selected"] += 1

        # Compute rates
        group_rates: Dict[str, float] = {}
        for name, data in groups.items():
            group_rates[name] = data["selected"] / data["total"] if data["total"] > 0 else 0.0

        rates = [r for r in group_rates.values() if r > 0]
        if not rates:
            return {
                "disparate_impact": 1.0,
                "statistical_parity_difference": 0.0,
                "group_rates": group_rates,
                "four_fifths_compliant": True,
                "sample_size": len(decisions_batch),
            }

        min_rate = min(group_rates.values())
        max_rate = max(rates)
        di = min_rate / max_rate if max_rate > 0 else 1.0
        spd = min_rate - max_rate

        return {
            "disparate_impact": round(di, 4),
            "statistical_parity_difference": round(spd, 4),
            "group_rates": {k: round(v, 4) for k, v in group_rates.items()},
            "four_fifths_compliant": di >= 0.8,
            "sample_size": len(decisions_batch),
        }

    # ─── Internal Helpers ─────────────────────────────────────────────────────

    def _track_decision(
        self, decision: Dict, context: Dict, result: str
    ) -> None:
        """Track decision in history for batch analysis."""
        entry = {
            **decision,
            "_policy_result": result,
            "_timestamp": time.time(),
        }
        self._decision_history.append(entry)
        if len(self._decision_history) > self._max_history:
            self._decision_history = self._decision_history[-self._max_history:]

    def _get_batch_by_group(
        self, protected_attributes: Dict[str, Any]
    ) -> Dict[str, Dict[str, int]]:
        """Get selection rates grouped by protected attribute from history."""
        groups: Dict[str, Dict[str, int]] = {}

        # Use the first protected attribute for grouping
        attr_name = next(iter(protected_attributes), None)
        if not attr_name:
            return {}

        for d in self._decision_history:
            pa = d.get("protected_attributes", {})
            group = str(pa.get(attr_name, "unknown"))
            if group not in groups:
                groups[group] = {"total": 0, "selected": 0}
            groups[group]["total"] += 1
            outcome = d.get("outcome", "").lower()
            if outcome in ("approve", "accept", "pass", "advance"):
                groups[group]["selected"] += 1

        return groups

    def _compute_risk_score(self, violations: List[PolicyViolation]) -> float:
        """Compute aggregate risk score from violations."""
        if not violations:
            return 0.0

        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
        }

        total = sum(severity_weights.get(v.severity, 0.1) for v in violations)
        return min(1.0, total / 2.0)  # Normalize: 2+ critical violations = 1.0
