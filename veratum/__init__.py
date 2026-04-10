"""Veratum — AI decision evidence infrastructure.

The complete SDK for AI auditability, compliance, and accountability.
Every AI decision gets a cryptographic receipt. Every receipt maps to
17 regulatory frameworks. Every framework gap gets flagged automatically.

Subpackages:
    veratum.core        — SDK, receipts, evidence engine, instrumentation
    veratum.crypto      — Hash chains, Merkle trees, signing, verification
    veratum.compliance  — Crosswalk (17 frameworks), policy engine, DPIA, bias analysis
    veratum.security    — Prompt guard, PII redaction
    veratum.future      — Parked modules (shadow_ai, threat_detection,
                          cost_controls, zk); do not import from production.

Supported frameworks (17):
    EU AI Act, EU AI Act GPAI, NIST AI RMF, ISO 42001, ISO 27001,
    GDPR, Colorado SB24-205, NYC LL144, EEOC, CFPB/ECOA,
    Illinois AIVA, Texas RAIGA, FINRA, NAIC, SOC 2, W3C VC 2.0,
    OWASP Top 10 for LLM Applications 2025.
"""

# ── Quickstart ──────────────────────────────────────────────────────────────
from .quick import (
    VeratumInstance,
    init,
    quickstart,
)

# ── Core ─────────────────────────────────────────────────────────────────────
from .core import (
    VeratumSDK,
    wrap,
    Receipt,
    EvidenceEngine,
    get_evidence_engine,
    Instrument,
    AuditLevel,
    get_audit_level,
    get_preset,
    list_presets,
    apply_preset,
    ReceiptBuffer,
)

# ── Multi-Provider Support ──────────────────────────────────────────────────────
from .providers import (
    detect_provider,
    auto_wrap,
    WrapConfig,
    UnsupportedProviderError,
    wrap_openai,
    wrap_anthropic,
    wrap_google,
    wrap_mistral,
    wrap_cohere,
    wrap_bedrock,
)

# ── Crypto ───────────────────────────────────────────────────────────────────
from .crypto import (
    HashChain,
    jcs_canonicalize,
    MerkleTree,
    BatchAnchor,
    verify_proof,
    hmac_sign_receipt,
    verify_hmac_signature,
    SequenceCheckpoint,
    verify_checkpoint,
    TransparencyLog,
    WitnessRegistry,
    verify_inclusion,
    verify_consistency,
    hash_receipt,
    verify_receipt,
    verify_chain,
    export_verification_report,
)

# ── Compliance ───────────────────────────────────────────────────────────────
from .compliance import (
    crosswalk,
    list_frameworks,
    get_required_fields,
    get_gaps_for_frameworks,
    VeratumPolicyEngine,
    PolicyResult,
    PolicyViolation,
    PolicyViolationError,
    BUILT_IN_POLICIES,
    DPIAGenerator,
    DPIAReport,
    generate_dpia,
    selection_rate,
    impact_ratio,
    four_fifths_rule,
    nyc_ll144_bias_audit,
    adverse_impact_analysis,
)
from .compliance.validation import validate_receipt as validate_receipt_compliance
from .compliance.validation import list_all_jurisdictions

# ── Security ─────────────────────────────────────────────────────────────────
from .security import (
    PromptGuard,
    PromptGuardResult,
    PromptBlockedError,
    scan_prompt,
    scan_output,
    PIIRedactor,
    PrivacyLayer,
    create_commitment,
    verify_commitment,
)


__version__ = "2.3.1"

__all__ = [
    # ── Quickstart ──
    "init", "quickstart", "VeratumInstance",
    # ── Core ──
    "VeratumSDK", "wrap",
    "Receipt",
    "EvidenceEngine", "get_evidence_engine",
    "Instrument",
    "AuditLevel", "get_audit_level", "get_preset", "list_presets", "apply_preset",
    "ReceiptBuffer",
    # ── Multi-Provider Support ──
    "detect_provider", "auto_wrap", "WrapConfig", "UnsupportedProviderError",
    "wrap_openai", "wrap_anthropic", "wrap_google", "wrap_mistral", "wrap_cohere", "wrap_bedrock",
    # ── Crypto ──
    "HashChain", "jcs_canonicalize",
    "MerkleTree", "BatchAnchor", "verify_proof",
    "hmac_sign_receipt", "verify_hmac_signature", "SequenceCheckpoint", "verify_checkpoint",
    "TransparencyLog", "WitnessRegistry", "verify_inclusion", "verify_consistency", "hash_receipt",
    "verify_receipt", "verify_chain", "export_verification_report",
    # ── Compliance ──
    "crosswalk", "list_frameworks", "get_required_fields", "get_gaps_for_frameworks",
    "validate_receipt_compliance", "list_all_jurisdictions",
    "VeratumPolicyEngine", "PolicyResult", "PolicyViolation", "PolicyViolationError", "BUILT_IN_POLICIES",
    "DPIAGenerator", "DPIAReport", "generate_dpia",
    "selection_rate", "impact_ratio", "four_fifths_rule", "nyc_ll144_bias_audit", "adverse_impact_analysis",
    # ── Security ──
    "PromptGuard", "PromptGuardResult", "PromptBlockedError", "scan_prompt", "scan_output",
    "PIIRedactor", "PrivacyLayer", "create_commitment", "verify_commitment",
]
