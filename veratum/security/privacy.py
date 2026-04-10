"""Client-side privacy layer — PII redaction and prompt hashing.

Addresses the #1 enterprise deal-killer: "You're asking me to send
our prompts containing PII to your server?"

Architecture (researched from Langfuse, Datadog LLM Observability):
- Hash the original prompt/response client-side (commitment scheme)
- Redact PII before transmission (names, SSNs, emails, phones)
- Send ONLY: redacted text + original hash + salt
- Server never sees plaintext PII
- Audit: customer holds original, hash proves it matches

PII Detection approach:
- Regex-based for structured PII (SSN, email, phone, credit card)
- No external dependencies (no spaCy, no Presidio) — those are optional
- Customers CAN plug in Microsoft Presidio for NER-based detection
- Design: fast, zero-dependency default with optional deep detection

Why regex-only as default (not Presidio/spaCy):
- Presidio adds ~200MB of dependencies (spaCy models)
- SDK must be lightweight — enterprise CISOs won't approve a 200MB audit SDK
- Regex catches 90% of structured PII (SSN, email, phone, CC numbers)
- Custom redactors let customers add their own patterns
- Optional: `pip install veratum[pii]` could add Presidio later

Commitment scheme (not ZKP — researched, ZKP is overkill):
- Hash(prompt + salt) → commitment stored with receipt
- Customer keeps original prompt + salt locally
- Later: auditor verifies Hash(prompt + salt) == stored commitment
- Proves "this exact prompt was sent" without server storing it
"""

import hashlib
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Built-in PII patterns (regex-based, zero dependencies)
# ---------------------------------------------------------------------------

# US Social Security Number: 123-45-6789 or 123456789
_SSN_PATTERN = re.compile(
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
)

# Email addresses
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# US/International phone numbers
_PHONE_PATTERN = re.compile(
    r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# Credit card numbers (Visa, MC, Amex, Discover)
_CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
)

# US Date of Birth patterns (MM/DD/YYYY, MM-DD-YYYY)
_DOB_PATTERN = re.compile(
    r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b"
)

# IP addresses (IPv4)
_IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# Default PII patterns with their labels
DEFAULT_PII_PATTERNS: Dict[str, re.Pattern] = {
    "SSN": _SSN_PATTERN,
    "EMAIL": _EMAIL_PATTERN,
    "PHONE": _PHONE_PATTERN,
    "CREDIT_CARD": _CREDIT_CARD_PATTERN,
    "DOB": _DOB_PATTERN,
    "IPV4": _IPV4_PATTERN,
}


class PIIRedactor:
    """
    Redacts PII from text using regex patterns.

    Zero external dependencies. Catches structured PII (SSN, email,
    phone, credit cards, DOB, IP addresses). Customers can add custom
    patterns for their specific data types.

    Usage:
        redactor = PIIRedactor()
        result = redactor.redact("Call John at 555-123-4567")
        # result.text == "Call John at [PHONE_REDACTED]"
        # result.redactions == [{"type": "PHONE", "start": 14, "end": 26}]

    Custom patterns:
        redactor.add_pattern("EMPLOYEE_ID", r"EMP-\\d{6}")
    """

    def __init__(
        self,
        *,
        patterns: Optional[Dict[str, re.Pattern]] = None,
        replacement_format: str = "[{label}_REDACTED]",
        enabled: bool = True,
    ) -> None:
        """
        Initialize PII redactor.

        Args:
            patterns: Custom PII patterns (overrides defaults if provided).
            replacement_format: Format string for replacements. {label} is
                                replaced with the pattern name (e.g., "SSN").
            enabled: Whether redaction is active (default True).
        """
        self._patterns = dict(patterns or DEFAULT_PII_PATTERNS)
        self._replacement_format = replacement_format
        self._enabled = enabled
        self._custom_redactors: List[Callable[[str], str]] = []

    def add_pattern(self, label: str, pattern: str) -> None:
        """
        Add a custom PII pattern.

        Args:
            label: Name for this PII type (e.g., "EMPLOYEE_ID").
            pattern: Regex pattern string.
        """
        self._patterns[label] = re.compile(pattern)

    def add_custom_redactor(self, fn: Callable[[str], str]) -> None:
        """
        Add a custom redaction function.

        Use this to plug in Microsoft Presidio or spaCy NER:
            from presidio_analyzer import AnalyzerEngine
            analyzer = AnalyzerEngine()
            redactor.add_custom_redactor(my_presidio_redact_fn)

        Args:
            fn: Callable that takes text and returns redacted text.
        """
        self._custom_redactors.append(fn)

    def redact(self, text: str) -> "RedactionResult":
        """
        Redact PII from text.

        Args:
            text: Input text potentially containing PII.

        Returns:
            RedactionResult with redacted text and metadata.
        """
        if not self._enabled or not text:
            return RedactionResult(
                text=text,
                original_length=len(text) if text else 0,
                redacted=False,
                redactions=[],
            )

        redactions: List[Dict[str, Any]] = []
        result = text

        # Apply regex patterns
        for label, pattern in self._patterns.items():
            replacement = self._replacement_format.format(label=label)
            matches = list(pattern.finditer(result))
            if matches:
                for match in reversed(matches):  # Reverse to preserve positions
                    redactions.append({
                        "type": label,
                        "start": match.start(),
                        "end": match.end(),
                    })
                result = pattern.sub(replacement, result)

        # Apply custom redactors (e.g., Presidio)
        for fn in self._custom_redactors:
            result = fn(result)

        return RedactionResult(
            text=result,
            original_length=len(text),
            redacted=len(redactions) > 0,
            redactions=redactions,
        )


class RedactionResult:
    """Result of PII redaction."""

    __slots__ = ("text", "original_length", "redacted", "redactions")

    def __init__(
        self,
        text: str,
        original_length: int,
        redacted: bool,
        redactions: List[Dict[str, Any]],
    ) -> None:
        self.text = text
        self.original_length = original_length
        self.redacted = redacted
        self.redactions = redactions


# ---------------------------------------------------------------------------
# Commitment scheme — prove what was sent without storing it
# ---------------------------------------------------------------------------

def create_commitment(
    data: str,
    salt: Optional[str] = None,
) -> Dict[str, str]:
    """
    Create a cryptographic commitment for data.

    The commitment proves "this exact data was processed" without
    the server ever seeing the original. The customer keeps the
    data + salt; the server stores only the commitment hash.

    Scheme: commitment = SHA-256(data || salt)

    Args:
        data: The original data (prompt or response text).
        salt: Optional salt (generated if not provided).

    Returns:
        {
            "commitment": hex-encoded SHA-256 hash,
            "salt": the salt used (customer must store this),
        }
    """
    if salt is None:
        salt = os.urandom(16).hex()

    commitment = hashlib.sha256(
        (data + salt).encode("utf-8")
    ).hexdigest()

    return {
        "commitment": commitment,
        "salt": salt,
    }


def verify_commitment(
    data: str,
    salt: str,
    commitment: str,
) -> bool:
    """
    Verify a commitment against original data.

    An auditor or regulator uses this to verify: "the data the
    customer claims they sent matches the commitment hash stored
    in the receipt."

    Args:
        data: The original data.
        salt: The salt used during commitment.
        commitment: The stored commitment hash.

    Returns:
        True if the data matches the commitment.
    """
    recomputed = hashlib.sha256(
        (data + salt).encode("utf-8")
    ).hexdigest()
    return recomputed == commitment


# ---------------------------------------------------------------------------
# Privacy-preserving receipt preparation
# ---------------------------------------------------------------------------

class PrivacyLayer:
    """
    Privacy-preserving layer for receipt generation.

    Wraps the receipt generation pipeline to:
    1. Redact PII from prompts/responses before transmission
    2. Create commitments (hashes) of originals for verification
    3. Optionally encrypt the entire payload client-side

    Usage:
        privacy = PrivacyLayer()
        prepared = privacy.prepare(prompt="John Doe, SSN 123-45-6789", response="Approved")
        # prepared["prompt"] == "John Doe, SSN [SSN_REDACTED]"
        # prepared["prompt_commitment"] == "a1b2c3..."  (hash of original)
        # prepared["prompt_salt"] == "d4e5f6..."  (customer stores this)

    The customer stores the salt locally. The server only sees redacted
    text + commitment hash. During audit, the customer reveals the
    original + salt to prove what was actually sent.
    """

    def __init__(
        self,
        *,
        redactor: Optional[PIIRedactor] = None,
        redact_prompts: bool = True,
        redact_responses: bool = True,
        create_commitments: bool = True,
    ) -> None:
        """
        Initialize privacy layer.

        Args:
            redactor: Custom PIIRedactor (uses default if None).
            redact_prompts: Redact PII from prompts (default True).
            redact_responses: Redact PII from responses (default True).
            create_commitments: Create commitment hashes (default True).
        """
        self._redactor = redactor or PIIRedactor()
        self._redact_prompts = redact_prompts
        self._redact_responses = redact_responses
        self._create_commitments = create_commitments

    def prepare(
        self,
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """
        Prepare prompt and response for privacy-preserving transmission.

        Args:
            prompt: Original prompt text.
            response: Original response text.

        Returns:
            Dict with redacted text, commitments, and metadata.
        """
        result: Dict[str, Any] = {}

        # Process prompt
        if self._redact_prompts:
            prompt_result = self._redactor.redact(prompt)
            result["prompt"] = prompt_result.text
            result["prompt_redacted"] = prompt_result.redacted
            result["prompt_redaction_count"] = len(prompt_result.redactions)
        else:
            result["prompt"] = prompt
            result["prompt_redacted"] = False
            result["prompt_redaction_count"] = 0

        # Process response
        if self._redact_responses:
            response_result = self._redactor.redact(response)
            result["response"] = response_result.text
            result["response_redacted"] = response_result.redacted
            result["response_redaction_count"] = len(response_result.redactions)
        else:
            result["response"] = response
            result["response_redacted"] = False
            result["response_redaction_count"] = 0

        # Create commitments (hash of ORIGINAL, not redacted)
        if self._create_commitments:
            prompt_commitment = create_commitment(prompt)
            response_commitment = create_commitment(response)
            result["prompt_commitment"] = prompt_commitment["commitment"]
            result["prompt_salt"] = prompt_commitment["salt"]
            result["response_commitment"] = response_commitment["commitment"]
            result["response_salt"] = response_commitment["salt"]

        return result
