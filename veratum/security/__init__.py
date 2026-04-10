"""Security — prompt injection defense and PII redaction.

Modules:
    prompt_guard      — Prompt injection, jailbreak, and PII detection
    privacy           — PII redaction and cryptographic commitments

Note: ``shadow_ai`` and ``threat_detection`` have been moved to
``veratum.future`` until product-market fit is established.
"""

from .prompt_guard import PromptGuard, PromptGuardResult, PromptBlockedError, scan_prompt, scan_output
from .privacy import PIIRedactor, PrivacyLayer, create_commitment, verify_commitment

__all__ = [
    "PromptGuard", "PromptGuardResult", "PromptBlockedError", "scan_prompt", "scan_output",
    "PIIRedactor", "PrivacyLayer", "create_commitment", "verify_commitment",
]
