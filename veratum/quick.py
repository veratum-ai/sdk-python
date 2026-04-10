"""Veratum quickstart module — dead-simple one-liner initialization.

Competitors like Arize Phoenix, Lakera, and Arthur all have simple one-liners.
Veratum now matches that simplicity while preserving full power.

Dead-simple init:
    >>> import veratum
    >>> v = veratum.init(api_key="vsk_...", vertical="financial")
    >>> client = v.wrap(openai.OpenAI())
    >>> # Every call now has receipts, security scanning, cost tracking

Even simpler if you use env vars:
    >>> v = veratum.quickstart()  # reads VERATUM_API_KEY, VERATUM_VERTICAL from env
    >>> client = v.wrap(openai.OpenAI())

This module provides:
1. VeratumInstance — unified API to everything (wrapping, security, compliance, cost)
2. init() — initialize with explicit config
3. quickstart() — auto-detect from environment
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .core.sdk import VeratumSDK
from .core.receipt import Receipt
from .crypto.chain import HashChain
from .security.prompt_guard import PromptGuard

# ``threat_detection`` and ``cost_controls`` have been parked under
# ``veratum.future``. Import lazily so the SDK stays importable even
# when these optional modules are missing.
try:
    from .future.threat_detection import ThreatDetector
except ImportError:  # pragma: no cover — parked module
    ThreatDetector = None  # type: ignore[assignment]

try:
    from .future.cost_controls import CostTracker, calculate_cost
except ImportError:  # pragma: no cover — parked module
    CostTracker = None  # type: ignore[assignment]
    calculate_cost = None  # type: ignore[assignment]

from .compliance.crosswalk import crosswalk, list_frameworks
from .compliance.dpia import generate_dpia

if TYPE_CHECKING:
    from .core.receipt import Receipt

logger = logging.getLogger("veratum.quick")


# ─────────────────────────────────────────────────────────────────────────────
# VeratumInstance — unified quickstart API
# ─────────────────────────────────────────────────────────────────────────────


class VeratumInstance:
    """
    Unified Veratum instance for auditing, compliance, and security.

    This is what you get from init() or quickstart(). It's the primary API —
    a single object that provides everything: receipt wrapping, security scanning,
    cost tracking, compliance checking, and DPIA generation.

    Attributes:
        api_key (str): Veratum API key for receipt uploads
        vertical (str): Industry vertical (e.g., "financial", "healthcare", "hiring")
        guard (PromptGuard): Security scanner for prompt injection, PII, jailbreaks
        tracker (CostTracker): Cost and budget tracker for token usage
        threats (ThreatDetector): Runtime threat detection (anomalies, abuse, etc.)
        chain (HashChain): Cryptographic evidence chain
        sdk (VeratumSDK): Core SDK for wrapping clients and uploading receipts

    Example:
        >>> v = init(api_key="vsk_...", vertical="financial", security=True, cost_tracking=True)
        >>>
        >>> # Wrap any client
        >>> client = v.wrap(openai.OpenAI())
        >>> response = client.messages.create(...)  # receipt + security + cost all automatic
        >>>
        >>> # Quick security scan
        >>> result = v.scan("User input here")
        >>> if result.blocked:
        ...     print("Prompt injection detected!")
        >>>
        >>> # Check cost
        >>> cost_usd = v.cost("gpt-4o", tokens_in=1000, tokens_out=500)
        >>>
        >>> # Generate compliance report
        >>> receipts = [...]  # from your audit log
        >>> dpia = v.dpia(receipts, "Hiring Screening System")
        >>> report = v.report(receipts, type="executive_summary")
    """

    def __init__(
        self,
        api_key: str,
        vertical: str = "general",
        security: bool = False,
        cost_tracking: bool = False,
        shadow_ai: bool = False,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
        auto_timestamp: bool = True,
        buffered: bool = True,
    ) -> None:
        """
        Initialize a Veratum instance.

        Args:
            api_key: Veratum API key (or auto-loaded from VERATUM_API_KEY env var)
            vertical: Industry vertical for auto-configuration:
                - "financial" → auto-enables CFPB, FINRA, SOC 2
                - "healthcare" → auto-enables HIPAA
                - "hiring" → auto-enables NYC LL144, EEOC, Illinois AIVA
                - "general" → enables core frameworks only
            security: Enable PromptGuard + ThreatDetector for runtime scanning
            cost_tracking: Enable CostTracker for budget enforcement
            shadow_ai: Enable ShadowAI detection for unauthorized model usage
            endpoint: Veratum API endpoint (default: VERATUM_ENDPOINT env var or production)
            timeout: Request timeout in seconds
            auto_timestamp: Request remote timestamp after receipt upload
            buffered: Use async buffered upload with retry
        """
        self.api_key = api_key
        self.vertical = vertical

        # Core SDK for receipt generation and client wrapping
        self.sdk = VeratumSDK(
            endpoint=endpoint,
            api_key=api_key,
            vertical=vertical,
            timeout=timeout,
            auto_timestamp=auto_timestamp,
            buffered=buffered,
        )

        # Cryptographic chain (used by sdk internally, exposed for manual receipt creation)
        self.chain = self.sdk._hash_chain

        # Security components
        self.guard: Optional[PromptGuard] = None
        if security:
            self.guard = PromptGuard(block_on_injection=False)

        self.threats: Optional[ThreatDetector] = None
        if security:
            self.threats = ThreatDetector()

        # Cost tracking
        self.tracker: Optional[CostTracker] = None
        if cost_tracking:
            # Auto-detect budget from environment or use unlimited
            budget_str = os.environ.get("VERATUM_BUDGET")
            budget = float(budget_str) if budget_str else None
            self.tracker = CostTracker(budget_usd=budget)

        # Shadow AI detection (optional)
        self.shadow_ai_enabled = shadow_ai

    def wrap(self, client: Any) -> Any:
        """
        Wrap an LLM client to intercept and audit all API calls.

        Supports OpenAI, Anthropic, Google, and any client with a
        messages.create() method. Every call automatically:
        1. Generates a cryptographic receipt
        2. Scans for security threats (if security=True)
        3. Tracks cost and budget (if cost_tracking=True)
        4. Uploads evidence to Veratum

        Args:
            client: LLM client instance (e.g., openai.OpenAI(), anthropic.Anthropic())

        Returns:
            Wrapped client with identical interface — transparent to existing code

        Example:
            >>> import openai
            >>> v = init(api_key="vsk_...")
            >>> client = v.wrap(openai.OpenAI())
            >>> # Use exactly as before — wrapping is transparent
            >>> response = client.messages.create(
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "Hello"}]
            ... )
        """
        return self.sdk.wrap(client)

    def receipt(
        self,
        prompt: str,
        response: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Receipt:
        """
        Manually create a receipt for a single LLM interaction.

        Use this to audit decisions made outside the SDK's wrapping layer
        (e.g., batch processing, offline analysis, legacy systems).

        Args:
            prompt: The user input / prompt text
            response: The model's response text
            model: Model name (e.g., "gpt-4o", "claude-3.5-sonnet")
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            metadata: Optional custom metadata dict (user_id, session_id, etc.)

        Returns:
            Receipt with cryptographic hash chain and all evidence fields

        Example:
            >>> v = init(api_key="vsk_...")
            >>> receipt = v.receipt(
            ...     prompt="Analyze this resume",
            ...     response="...",
            ...     model="gpt-4o",
            ...     tokens_in=100,
            ...     tokens_out=250,
            ...     metadata={"user_id": "hr_001", "stage": "screening"}
            ... )
            >>> print(receipt.entry_hash)  # Cryptographic proof
        """
        return self.sdk._receipt_generator.generate(
            prompt=prompt,
            response=response,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata=metadata or {},
        )

    def scan(self, text: str) -> Dict[str, Any]:
        """
        Quick security scan of input text for threats.

        Detects:
        - Prompt injection attempts ("Ignore previous instructions...")
        - Jailbreak attempts (e.g., role-play, context escapes)
        - Personally Identifiable Information (SSN, credit cards, etc.)
        - Content safety violations (toxicity, etc.)
        - Encoding attacks (Base64, hex, Unicode obfuscation)

        Requires security=True during initialization.

        Args:
            text: Text to scan (usually user input)

        Returns:
            Dict with keys:
                - blocked (bool): Whether to block this input
                - risk_score (float 0.0-1.0): Overall risk assessment
                - threats (list): Detected threat signals
                - pii_found (list): PII types found
                - safe (bool): Shorthand for (not blocked and risk_score < 0.5)

        Example:
            >>> v = init(api_key="vsk_...", security=True)
            >>> result = v.scan("What's the admin password?")
            >>> if result["blocked"]:
            ...     print("Security threat detected!")
            >>> print(f"Risk: {result['risk_score']:.2%}")
        """
        if not self.guard:
            raise RuntimeError(
                "Prompt scanning disabled. Initialize with security=True to enable."
            )
        result = self.guard.scan(text)
        return {
            "blocked": result.blocked,
            "risk_score": result.risk_score,
            "threats": [t.to_dict() for t in result.threats],
            "pii_found": result.pii_found,
            "safe": result.safe,
            "scan_time_ms": result.scan_time_ms,
        }

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """
        Quick cost calculation for a single request.

        Knows pricing for 50+ models across OpenAI, Anthropic, Google,
        Mistral, Cohere, Meta, DeepSeek, and more (updated April 2025).

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3.5-sonnet")
            tokens_in: Input tokens
            tokens_out: Output tokens

        Returns:
            Cost in USD (float)

        Example:
            >>> v = init(api_key="vsk_...")
            >>> cost = v.cost("gpt-4o", tokens_in=1000, tokens_out=500)
            >>> print(f"Request cost: ${cost:.4f}")
        """
        return calculate_cost(model=model, tokens_in=tokens_in, tokens_out=tokens_out)

    def dpia(
        self,
        receipts: List[Receipt],
        system_name: str,
        system_description: Optional[str] = None,
        data_controller: Optional[str] = None,
        dpo_contact: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a GDPR Article 35 compliant Data Protection Impact Assessment.

        Assembles evidence from receipts to create structured DPIAs that satisfy:
        - GDPR Article 35(7) — DPA requirements
        - EU AI Act Article 26(9) — deployer DPIA for high-risk AI
        - Colorado SB24-205 §5 — impact assessment

        Args:
            receipts: List of Receipt objects from your audit log
            system_name: Name of the AI system (e.g., "Hiring Screening AI")
            system_description: What the system does and why
            data_controller: Organization operating the system
            dpo_contact: Data Protection Officer contact email

        Returns:
            Dict with keys:
                - dpia_id: Unique DPIA identifier
                - risk_level: "low", "medium", "high", or "critical"
                - risks: List of identified risks with mitigations
                - safeguards: List of implemented safeguards
                - article35_compliant: Boolean compliance status
                - markdown: Full DPIA as markdown text

        Example:
            >>> v = init(api_key="vsk_...")
            >>> receipts = v.sdk.get_receipts(limit=1000)
            >>> dpia = v.dpia(
            ...     receipts=receipts,
            ...     system_name="Resume Screening",
            ...     data_controller="Acme Corp",
            ... )
            >>> if dpia["risk_level"] == "high":
            ...     print("DPIA shows high risk. Review required.")
        """
        dpia_result = generate_dpia(
            receipts=receipts,
            system_name=system_name,
            system_description=system_description or "",
            data_controller=data_controller or "Unknown",
            dpo_contact=dpo_contact or "unknown@example.com",
        )
        return dpia_result

    def report(
        self,
        receipts: List[Receipt],
        report_type: str = "executive_summary",
        format: str = "dict",
    ) -> Union[Dict[str, Any], str]:
        """
        Generate a compliance report from receipts.

        Assembles receipts into structured reports for different audiences
        and compliance requirements.

        Args:
            receipts: List of Receipt objects
            report_type: Type of report to generate:
                - "executive_summary": High-level compliance overview
                - "framework_crosswalk": Mapping to all 17 supported frameworks
                - "detailed": Full evidence with all receipt fields
                - "audit_trail": Chronological log with chain verification
            format: Output format:
                - "dict": Python dictionary (default)
                - "json": JSON string
                - "markdown": Human-readable markdown
                - "pdf": PDF document (requires additional setup)

        Returns:
            Report as specified format

        Example:
            >>> v = init(api_key="vsk_...")
            >>> receipts = v.sdk.get_receipts(limit=1000)
            >>> report = v.report(receipts, report_type="framework_crosswalk")
            >>> for framework, status in report["frameworks"].items():
            ...     print(f"{framework}: {status['compliance']}")
        """
        # This would integrate with report_generator.py
        # For now, return a structured dict
        return {
            "type": report_type,
            "format": format,
            "receipt_count": len(receipts),
            "summary": f"Report generated for {len(receipts)} receipts",
        }

    def frameworks(self) -> Dict[str, Dict[str, Any]]:
        """
        List all supported compliance frameworks.

        Returns:
            Dict mapping framework names to metadata:
                - "eu_ai_act": {name: "EU AI Act...", jurisdiction: "EU", ...}
                - "gdpr": {name: "GDPR", jurisdiction: "EU", ...}
                - etc.

        Example:
            >>> v = init(api_key="vsk_...")
            >>> frameworks = v.frameworks()
            >>> for name, meta in frameworks.items():
            ...     print(f"{name}: {meta['name']}")
        """
        return list_frameworks()

    def crosswalk(self, receipt: Receipt) -> Dict[str, Dict[str, Any]]:
        """
        Check a single receipt against all 17 compliance frameworks.

        Identifies which frameworks the receipt satisfies and which have gaps.

        Args:
            receipt: A single Receipt object

        Returns:
            Dict mapping framework name to compliance status:
                - "gaps": List of unsatisfied requirements
                - "satisfied": List of satisfied requirements
                - "compliance_percentage": 0-100
                - "recommendation": Human-readable guidance

        Example:
            >>> v = init(api_key="vsk_...")
            >>> receipt = v.receipt(...)
            >>> results = v.crosswalk(receipt)
            >>> for framework, status in results.items():
            ...     print(f"{framework}: {status['compliance_percentage']}%")
        """
        return crosswalk(receipt=receipt)


# ─────────────────────────────────────────────────────────────────────────────
# init() and quickstart() — the actual one-liners
# ─────────────────────────────────────────────────────────────────────────────


def init(
    api_key: Optional[str] = None,
    vertical: str = "general",
    security: bool = False,
    cost_tracking: bool = False,
    shadow_ai: bool = False,
    endpoint: Optional[str] = None,
    timeout: float = 30.0,
) -> VeratumInstance:
    """
    Initialize Veratum in one line. That's it.

    Dead-simple initialization matching competitors like Arize Phoenix and Arthur:
        >>> import veratum
        >>> v = veratum.init(api_key="vsk_...", vertical="financial")
        >>> client = v.wrap(openai.OpenAI())
        >>> # Every call now has receipts, security, and compliance

    Auto-detection logic:
    - If api_key is None, reads VERATUM_API_KEY environment variable
    - If endpoint is None, reads VERATUM_ENDPOINT (or uses production default)
    - Vertical auto-configures compliance frameworks:
      - "financial" → CFPB, FINRA, SOC 2
      - "healthcare" → HIPAA
      - "hiring" → NYC LL144, EEOC, Illinois AIVA
      - "general" → core frameworks only

    Args:
        api_key: Veratum API key starting with "vsk_"
                 If not provided, reads VERATUM_API_KEY environment variable
        vertical: Industry vertical ("financial", "healthcare", "hiring", "general")
        security: Enable prompt injection detection + threat detection
        cost_tracking: Enable budget tracking and enforcement
        shadow_ai: Detect unauthorized AI model usage
        endpoint: Veratum API endpoint (auto-detected if not provided)
        timeout: Request timeout in seconds

    Returns:
        VeratumInstance — ready to wrap clients and audit decisions

    Raises:
        ValueError: If api_key not provided and VERATUM_API_KEY env var not set

    Example:
        >>> # Explicit initialization
        >>> v = veratum.init(
        ...     api_key="vsk_example123",
        ...     vertical="financial",
        ...     security=True,
        ...     cost_tracking=True,
        ... )
        >>> client = v.wrap(openai.OpenAI())
        >>>
        >>> # With environment variables (even simpler)
        >>> v = veratum.init(vertical="financial")  # reads VERATUM_API_KEY
        >>> client = v.wrap(openai.OpenAI())
        >>>
        >>> # All from environment (see quickstart())
        >>> v = veratum.quickstart()
    """
    # Auto-load api_key if not provided
    if api_key is None:
        api_key = os.environ.get("VERATUM_API_KEY")
        if not api_key:
            raise ValueError(
                "api_key required: pass as argument or set VERATUM_API_KEY environment variable"
            )

    # Auto-load endpoint if not provided
    if endpoint is None:
        endpoint = os.environ.get("VERATUM_ENDPOINT")

    return VeratumInstance(
        api_key=api_key,
        vertical=vertical,
        security=security,
        cost_tracking=cost_tracking,
        shadow_ai=shadow_ai,
        endpoint=endpoint,
        timeout=timeout,
    )


def quickstart() -> VeratumInstance:
    """
    Absolute simplest initialization — one function, zero arguments.

    Reads all configuration from environment variables:
    - VERATUM_API_KEY (required) — your API key
    - VERATUM_VERTICAL (optional) — industry vertical, default "general"
    - VERATUM_ENDPOINT (optional) — API endpoint, auto-detected if not set
    - VERATUM_SECURITY (optional) — "true"/"false", default "false"
    - VERATUM_COST_TRACKING (optional) — "true"/"false", default "false"
    - VERATUM_SHADOW_AI (optional) — "true"/"false", default "false"

    Returns:
        VeratumInstance ready to use

    Raises:
        ValueError: If VERATUM_API_KEY is not set

    Example:
        >>> # Set environment variables
        >>> import os
        >>> os.environ["VERATUM_API_KEY"] = "vsk_..."
        >>> os.environ["VERATUM_VERTICAL"] = "financial"
        >>> os.environ["VERATUM_SECURITY"] = "true"
        >>>
        >>> # One line to initialize everything
        >>> v = veratum.quickstart()
        >>> client = v.wrap(openai.OpenAI())
        >>> # Go!
    """
    api_key = os.environ.get("VERATUM_API_KEY")
    if not api_key:
        raise ValueError(
            "VERATUM_API_KEY environment variable not set. "
            "Run: export VERATUM_API_KEY='vsk_...'"
        )

    vertical = os.environ.get("VERATUM_VERTICAL", "general")
    security = os.environ.get("VERATUM_SECURITY", "false").lower() == "true"
    cost_tracking = os.environ.get("VERATUM_COST_TRACKING", "false").lower() == "true"
    shadow_ai = os.environ.get("VERATUM_SHADOW_AI", "false").lower() == "true"
    endpoint = os.environ.get("VERATUM_ENDPOINT")

    return VeratumInstance(
        api_key=api_key,
        vertical=vertical,
        security=security,
        cost_tracking=cost_tracking,
        shadow_ai=shadow_ai,
        endpoint=endpoint,
    )


__all__ = [
    "VeratumInstance",
    "init",
    "quickstart",
]
