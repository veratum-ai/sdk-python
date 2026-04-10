"""
Runtime AI threat detection and response engine.

Unified security layer that combines prompt injection detection,
anomaly detection, abuse pattern recognition, and content safety
into a single real-time threat pipeline. Every detection produces
an evidence receipt for compliance.

This is what FireTail, Zenity, and Lakera sell as standalone products.
Veratum integrates it with the compliance evidence layer — every threat
is both a security event AND a compliance record.

Threat categories:
- Prompt injection / jailbreak (via prompt_guard)
- Token abuse (via cost_controls)
- Anomalous usage patterns (rate spikes, off-hours, new models)
- Data exfiltration attempts (large output extraction)
- Model abuse (systematic probing, adversarial inputs)
- Policy violations (via prevention engine)

Example:
    >>> from veratum.threat_detection import ThreatDetector
    >>>
    >>> detector = ThreatDetector()
    >>> result = detector.analyze(
    ...     prompt="Ignore previous instructions...",
    ...     response="Here are the system instructions...",
    ...     model="gpt-4o",
    ...     tokens_in=500,
    ...     tokens_out=2000,
    ...     user="user_123",
    ... )
    >>> print(result.threat_level)  # "critical"
    >>> print(result.threats)  # [Threat(category="injection", ...)]
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Set

logger = logging.getLogger("veratum.threat_detection")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Threat:
    """A single detected threat."""
    category: str  # "injection", "exfiltration", "abuse", "anomaly", "policy"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    mitigation: str = ""
    owasp_ref: str = ""  # OWASP LLM Top 10 reference
    regulation_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThreatResult:
    """Result of threat analysis for a single request."""
    threat_level: str = "none"  # "none", "low", "medium", "high", "critical"
    blocked: bool = False
    threats: List[Threat] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0-1.0
    analysis_time_ms: int = 0
    timestamp: str = ""
    request_fingerprint: str = ""

    @property
    def threat_count(self) -> int:
        return len(self.threats)

    def to_receipt_field(self) -> Dict[str, Any]:
        """Convert to receipt threat_detection field."""
        return {
            "threat_detection": {
                "threat_level": self.threat_level,
                "blocked": self.blocked,
                "risk_score": round(self.risk_score, 4),
                "threat_count": self.threat_count,
                "categories": list(set(t.category for t in self.threats)),
                "analysis_time_ms": self.analysis_time_ms,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["threats"] = [t.to_dict() for t in self.threats]
        return d


# ---------------------------------------------------------------------------
# Threat Detector
# ---------------------------------------------------------------------------

class ThreatDetector:
    """
    Runtime AI threat detection engine.

    Analyzes every LLM request/response for security threats,
    usage anomalies, and compliance violations. Produces evidence
    receipts for audit trails.

    Args:
        block_on_injection: Auto-block injection attempts (default: True)
        block_on_exfiltration: Auto-block data exfiltration (default: True)
        rate_limit_per_minute: Max requests per user per minute (0=disabled)
        max_output_tokens: Flag large outputs as potential exfiltration
        anomaly_window_size: Number of requests for baseline calculation
        enable_prompt_guard: Use built-in prompt guard (default: True)
    """

    def __init__(
        self,
        *,
        block_on_injection: bool = True,
        block_on_exfiltration: bool = True,
        rate_limit_per_minute: int = 0,
        max_output_tokens: int = 50000,
        anomaly_window_size: int = 100,
        enable_prompt_guard: bool = True,
    ):
        self.block_on_injection = block_on_injection
        self.block_on_exfiltration = block_on_exfiltration
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_output_tokens = max_output_tokens
        self.anomaly_window_size = anomaly_window_size
        self.enable_prompt_guard = enable_prompt_guard

        # User activity tracking
        self._user_requests: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._user_tokens: Dict[str, Deque[int]] = defaultdict(
            lambda: deque(maxlen=self.anomaly_window_size)
        )

        # Global baselines
        self._token_history: Deque[int] = deque(maxlen=anomaly_window_size)
        self._latency_history: Deque[float] = deque(maxlen=anomaly_window_size)
        self._model_usage: Dict[str, int] = defaultdict(int)

        # Threat log
        self._threat_log: List[Dict[str, Any]] = []
        self._total_analyzed = 0
        self._total_blocked = 0

        # Prompt guard instance
        self._prompt_guard = None
        if enable_prompt_guard:
            try:
                from .prompt_guard import PromptGuard
                self._prompt_guard = PromptGuard(
                    block_on_injection=block_on_injection,
                    block_on_pii=False,  # PII handled separately
                    block_on_toxicity=True,
                )
            except ImportError:
                logger.warning("prompt_guard module not available")

    def analyze(
        self,
        *,
        prompt: str = "",
        response: str = "",
        model: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        user: Optional[str] = None,
        latency_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThreatResult:
        """
        Analyze a request/response pair for threats.

        Call this for every LLM interaction to detect threats
        in real-time.

        Args:
            prompt: User prompt (input to model).
            response: Model response (output).
            model: Model name.
            tokens_in: Input token count.
            tokens_out: Output token count.
            user: User identifier for per-user tracking.
            latency_ms: Request latency in milliseconds.
            metadata: Additional context.

        Returns:
            ThreatResult with detected threats.
        """
        start = time.time()
        threats: List[Threat] = []
        now = time.time()

        # 1. Prompt injection detection
        if prompt and self._prompt_guard:
            guard_result = self._prompt_guard.scan(prompt)
            for t in guard_result.threats:
                threats.append(Threat(
                    category="injection",
                    severity=t.severity,
                    description=t.description,
                    details={"matched_pattern": t.matched_pattern, "confidence": t.confidence},
                    owasp_ref="LLM01: Prompt Injection",
                    regulation_ref="EU AI Act Article 15",
                ))

            # Check output for PII leakage
            if response:
                output_result = self._prompt_guard.scan_output(response)
                for t in output_result.threats:
                    if t.threat_type == "pii":
                        threats.append(Threat(
                            category="exfiltration",
                            severity="high",
                            description=f"PII leaked in output: {t.description}",
                            details={"pii_type": t.matched_pattern},
                            owasp_ref="LLM02: Sensitive Information Disclosure",
                            regulation_ref="GDPR Article 5(1)(f)",
                        ))

        # 2. Rate limiting
        if user and self.rate_limit_per_minute > 0:
            rate_threat = self._check_rate_limit(user, now)
            if rate_threat:
                threats.append(rate_threat)

        # 3. Token anomaly detection
        if tokens_out > 0:
            anomaly_threats = self._check_token_anomaly(
                tokens_in, tokens_out, user
            )
            threats.extend(anomaly_threats)

        # 4. Data exfiltration detection
        if tokens_out > self.max_output_tokens:
            threats.append(Threat(
                category="exfiltration",
                severity="high",
                description=(
                    f"Large output detected: {tokens_out} tokens "
                    f"(threshold: {self.max_output_tokens})"
                ),
                details={
                    "tokens_out": tokens_out,
                    "threshold": self.max_output_tokens,
                },
                mitigation="Review output for sensitive data extraction",
                owasp_ref="LLM02: Sensitive Information Disclosure",
            ))

        # 5. Output-to-input ratio anomaly (possible extraction)
        if tokens_in > 0 and tokens_out > 0:
            ratio = tokens_out / tokens_in
            if ratio > 20:  # >20x amplification
                threats.append(Threat(
                    category="exfiltration",
                    severity="medium",
                    description=(
                        f"High output amplification: {ratio:.1f}x "
                        f"({tokens_in} in → {tokens_out} out)"
                    ),
                    details={"ratio": round(ratio, 2)},
                    owasp_ref="LLM10: Unbounded Consumption",
                ))

        # 6. Model probing detection (rapid different-model usage)
        if model and user:
            probing_threat = self._check_model_probing(model, user)
            if probing_threat:
                threats.append(probing_threat)

        # Update baselines
        if tokens_out:
            self._token_history.append(tokens_out)
        if user:
            self._user_requests[user].append(now)
            self._user_tokens[user].append(tokens_out)
        if model:
            self._model_usage[model] += 1
        if latency_ms:
            self._latency_history.append(latency_ms)

        # Compute result
        blocked = False
        if self.block_on_injection and any(t.category == "injection" for t in threats):
            blocked = True
        if self.block_on_exfiltration and any(
            t.category == "exfiltration" and t.severity in ("high", "critical")
            for t in threats
        ):
            blocked = True

        risk_score = self._compute_risk_score(threats)
        threat_level = self._compute_threat_level(risk_score, threats)
        elapsed_ms = int((time.time() - start) * 1000)

        self._total_analyzed += 1
        if blocked:
            self._total_blocked += 1

        result = ThreatResult(
            threat_level=threat_level,
            blocked=blocked,
            threats=threats,
            risk_score=risk_score,
            analysis_time_ms=elapsed_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Log significant threats
        if threats:
            self._threat_log.append({
                "timestamp": result.timestamp,
                "threat_level": threat_level,
                "blocked": blocked,
                "categories": [t.category for t in threats],
                "user": user,
                "model": model,
            })

        return result

    # --- Detection methods --------------------------------------------------

    def _check_rate_limit(self, user: str, now: float) -> Optional[Threat]:
        """Check if user exceeds rate limit."""
        window_start = now - 60  # 1-minute window
        recent = [t for t in self._user_requests[user] if t > window_start]

        if len(recent) >= self.rate_limit_per_minute:
            return Threat(
                category="abuse",
                severity="high",
                description=(
                    f"Rate limit exceeded: {len(recent)} requests/min "
                    f"(limit: {self.rate_limit_per_minute})"
                ),
                details={
                    "requests_per_minute": len(recent),
                    "limit": self.rate_limit_per_minute,
                    "user": user,
                },
                mitigation="Throttle or block user requests",
                owasp_ref="LLM10: Unbounded Consumption",
            )
        return None

    def _check_token_anomaly(
        self, tokens_in: int, tokens_out: int, user: Optional[str]
    ) -> List[Threat]:
        """Detect anomalous token usage patterns."""
        threats = []

        # Global anomaly — compare to baseline
        if len(self._token_history) >= 20:
            mean_tokens = statistics.mean(self._token_history)
            stdev_tokens = statistics.stdev(self._token_history) if len(self._token_history) > 1 else mean_tokens

            if stdev_tokens > 0 and tokens_out > mean_tokens + 3 * stdev_tokens:
                threats.append(Threat(
                    category="anomaly",
                    severity="medium",
                    description=(
                        f"Token usage anomaly: {tokens_out} tokens "
                        f"(baseline: {mean_tokens:.0f} ± {stdev_tokens:.0f})"
                    ),
                    details={
                        "tokens_out": tokens_out,
                        "baseline_mean": round(mean_tokens),
                        "baseline_stdev": round(stdev_tokens),
                        "z_score": round((tokens_out - mean_tokens) / stdev_tokens, 2),
                    },
                ))

        # Per-user anomaly
        if user and len(self._user_tokens[user]) >= 10:
            user_mean = statistics.mean(self._user_tokens[user])
            user_stdev = statistics.stdev(self._user_tokens[user]) if len(self._user_tokens[user]) > 1 else user_mean

            if user_stdev > 0 and tokens_out > user_mean + 3 * user_stdev:
                threats.append(Threat(
                    category="anomaly",
                    severity="medium",
                    description=(
                        f"User token anomaly for {user}: {tokens_out} tokens "
                        f"(user baseline: {user_mean:.0f} ± {user_stdev:.0f})"
                    ),
                    details={
                        "user": user,
                        "tokens_out": tokens_out,
                        "user_baseline_mean": round(user_mean),
                    },
                ))

        return threats

    def _check_model_probing(self, model: str, user: str) -> Optional[Threat]:
        """Detect rapid model switching (possible probing)."""
        # This is a simplified check — in production you'd track model
        # sequences per user more precisely
        return None  # TODO: implement with per-user model history

    # --- Scoring ------------------------------------------------------------

    @staticmethod
    def _compute_risk_score(threats: List[Threat]) -> float:
        """Compute aggregate risk score."""
        if not threats:
            return 0.0

        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
        }

        max_weight = max(severity_weights.get(t.severity, 0.1) for t in threats)
        bonus = min(0.2, len(threats) * 0.03)
        return min(1.0, max_weight + bonus)

    @staticmethod
    def _compute_threat_level(risk_score: float, threats: List[Threat]) -> str:
        """Compute overall threat level."""
        if not threats:
            return "none"

        has_critical = any(t.severity == "critical" for t in threats)
        has_high = any(t.severity == "high" for t in threats)

        if has_critical or risk_score >= 0.9:
            return "critical"
        elif has_high or risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"

    # --- Stats and reporting ------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get threat detection statistics."""
        category_counts: Dict[str, int] = defaultdict(int)
        for entry in self._threat_log:
            for cat in entry.get("categories", []):
                category_counts[cat] += 1

        return {
            "total_analyzed": self._total_analyzed,
            "total_blocked": self._total_blocked,
            "block_rate": (
                round(self._total_blocked / self._total_analyzed, 4)
                if self._total_analyzed > 0 else 0
            ),
            "threats_by_category": dict(category_counts),
            "threat_log_size": len(self._threat_log),
            "models_tracked": dict(self._model_usage),
            "users_tracked": len(self._user_requests),
        }

    def get_recent_threats(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent threat events."""
        return self._threat_log[-limit:]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def analyze_request(
    prompt: str = "",
    response: str = "",
    **kwargs,
) -> ThreatResult:
    """
    One-liner threat analysis.

    Usage:
        from veratum.threat_detection import analyze_request
        result = analyze_request(prompt="...", response="...", model="gpt-4o")
    """
    detector = ThreatDetector()
    return detector.analyze(prompt=prompt, response=response, **kwargs)


__all__ = [
    "ThreatDetector",
    "ThreatResult",
    "Threat",
    "analyze_request",
]
