"""
Veratum Verification Toolkit — Standalone receipt verification.

A zero-dependency library for independently verifying Veratum compliance receipts.
No API key required. No account required. No network calls.

Example:
    >>> from veratum_verify import ReceiptVerifier, verify_inclusion
    >>> verifier = ReceiptVerifier()
    >>> result = verifier.verify_receipt(receipt_dict)
    >>> if result.valid:
    ...     print("Receipt is valid!")
"""

from veratum_verify.core import (
    ReceiptVerifier,
    VerificationResult,
    ChainVerificationResult,
    verify_inclusion,
    verify_consistency,
    hash_leaf,
    hash_pair,
)

__version__ = "0.1.0"
__author__ = "Veratum Inc."
__license__ = "MIT"

__all__ = [
    "ReceiptVerifier",
    "VerificationResult",
    "ChainVerificationResult",
    "verify_inclusion",
    "verify_consistency",
    "hash_leaf",
    "hash_pair",
]
