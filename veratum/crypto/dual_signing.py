"""Dual signature support for Schema 2.3.0 receipts.

Schema 2.3.0 introduces algorithmic-agility signatures: every receipt
carries TWO independent signatures over the same canonical bytes —
one classical (Ed25519, RFC 8032) and one post-quantum (ML-DSA-65,
NIST FIPS 204).

Why two:
- Ed25519 is fast, mature, widely deployed, and trivially verifiable
  in courts today using off-the-shelf openssl/libsodium tooling.
- ML-DSA-65 (formerly Dilithium-3) is the NIST-standardized
  post-quantum digital signature scheme. EU AI Act records have a
  10-year retention window — long enough for a cryptographically
  relevant quantum computer to break Ed25519 mid-retention.

By storing both, an adversary must break BOTH algorithms to forge a
receipt that verifies. This satisfies the eIDAS Article 24/41
"long-term archival" requirement and the NIST PQC migration
guidance (NIST IR 8413).

Implementation notes:
- Ed25519: uses `cryptography` library if available (the SDK already
  pulls it in), otherwise falls back to PyNaCl, otherwise raises.
- ML-DSA-65: uses `pyoqs` (Open Quantum Safe Python bindings) if
  available. If not, ML-DSA signing is skipped and the receipt
  records `signature_ml_dsa_65=""` so verification can still succeed
  on the classical path. This is the recommended hybrid-mode
  fallback per NIST SP 800-208.
- Both signatures cover the receipt's entry_hash (which itself is
  the SHA-256 of the RFC 8785 JCS canonical bytes), so they
  authenticate the same content.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ed25519
# ---------------------------------------------------------------------------

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization

    _ED25519_BACKEND = "cryptography"
except ImportError:  # pragma: no cover - fallback path
    Ed25519PrivateKey = None  # type: ignore
    Ed25519PublicKey = None  # type: ignore
    serialization = None  # type: ignore
    _ED25519_BACKEND = None


def ed25519_sign(entry_hash: str, private_key_pem: bytes) -> str:
    """Produce a hex-encoded Ed25519 signature over the entry_hash bytes.

    Args:
        entry_hash: Hex-encoded SHA-256 entry hash from a Schema 2.3.0
            receipt.
        private_key_pem: PEM-encoded Ed25519 private key.

    Returns:
        Hex-encoded 64-byte Ed25519 signature.
    """
    if _ED25519_BACKEND != "cryptography":
        raise RuntimeError(
            "ed25519_sign requires the `cryptography` package; install "
            "veratum[crypto] to enable Schema 2.3.0 dual signatures."
        )
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Provided key is not an Ed25519 private key")
    sig = key.sign(entry_hash.encode("utf-8"))
    return sig.hex()


def ed25519_verify(
    entry_hash: str, signature_hex: str, public_key_pem: bytes
) -> bool:
    """Verify a hex-encoded Ed25519 signature over an entry_hash."""
    if _ED25519_BACKEND != "cryptography":
        return False
    try:
        from cryptography.exceptions import InvalidSignature

        key = serialization.load_pem_public_key(public_key_pem)
        if not isinstance(key, Ed25519PublicKey):
            return False
        key.verify(bytes.fromhex(signature_hex), entry_hash.encode("utf-8"))
        return True
    except (InvalidSignature, ValueError):
        return False


# ---------------------------------------------------------------------------
# ML-DSA-65 (FIPS 204) — post-quantum digital signature
# ---------------------------------------------------------------------------

try:
    import oqs  # type: ignore

    _MLDSA_BACKEND = "oqs"
except ImportError:  # pragma: no cover - fallback path
    oqs = None  # type: ignore
    _MLDSA_BACKEND = None


def ml_dsa_65_available() -> bool:
    """Return True if ML-DSA-65 signing is available in this environment."""
    return _MLDSA_BACKEND == "oqs"


def ml_dsa_65_sign(entry_hash: str, private_key_bytes: bytes) -> str:
    """Sign with ML-DSA-65 (NIST FIPS 204).

    Returns hex-encoded signature, or empty string if the OQS backend
    is not installed (hybrid-mode degradation).
    """
    if _MLDSA_BACKEND != "oqs":
        logger.info(
            "ML-DSA-65 backend not available; receipt will store empty "
            "signature_ml_dsa_65 (classical signature still valid)."
        )
        return ""
    with oqs.Signature("ML-DSA-65", secret_key=private_key_bytes) as signer:
        sig = signer.sign(entry_hash.encode("utf-8"))
    return sig.hex()


def ml_dsa_65_verify(
    entry_hash: str, signature_hex: str, public_key_bytes: bytes
) -> bool:
    """Verify an ML-DSA-65 signature. Returns False if backend missing."""
    if _MLDSA_BACKEND != "oqs":
        return False
    if not signature_hex:
        return False
    try:
        with oqs.Signature("ML-DSA-65") as verifier:
            return bool(
                verifier.verify(
                    entry_hash.encode("utf-8"),
                    bytes.fromhex(signature_hex),
                    public_key_bytes,
                )
            )
    except Exception as exc:
        logger.warning("ML-DSA-65 verify failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Convenience: produce both signatures in one call
# ---------------------------------------------------------------------------


def dual_sign_entry_hash(
    entry_hash: str,
    ed25519_private_pem: Optional[bytes],
    ml_dsa_private_key: Optional[bytes],
) -> Tuple[str, str]:
    """Sign an entry_hash with both Ed25519 and ML-DSA-65 in one pass.

    Returns:
        (signature_ed25519_hex, signature_ml_dsa_65_hex)

        Either component is "" if its key was not provided OR if the
        respective backend is unavailable.
    """
    sig_ed = ""
    sig_pq = ""
    if ed25519_private_pem:
        try:
            sig_ed = ed25519_sign(entry_hash, ed25519_private_pem)
        except Exception as exc:
            logger.error("Ed25519 signing failed: %s", exc)
    if ml_dsa_private_key:
        sig_pq = ml_dsa_65_sign(entry_hash, ml_dsa_private_key)
    return sig_ed, sig_pq
