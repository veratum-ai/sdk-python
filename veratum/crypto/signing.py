"""HMAC-signed receipts and sequence integrity.

Addresses two critical security concerns:
1. Hash chain is integrity-only, not authenticated — anyone who modifies
   a receipt can recompute the hash. HMAC adds authentication.
2. Missing receipts (deletions) go undetected if you only check adjacent
   prev_hash links.

Design decisions (researched from Certificate Transparency RFC 6962,
Amazon QLDB, AWS CloudTrail):

HMAC vs ECDSA per-receipt:
- HMAC-SHA256: ~microseconds, shared-key (both sides know key)
- ECDSA P-256: ~milliseconds, asymmetric (only signer knows private key)
- Decision: HMAC for per-receipt signing (performance), ECDSA for
  periodic signed checkpoints (non-repudiation). Same hybrid approach
  as Certificate Transparency (per-entry hashing + signed tree heads).

Sequence gap detection (from CloudTrail):
- CloudTrail publishes hourly digest files with hashes of all logs
- We do the same: periodic checkpoints that commit to a sequence range
- A checkpoint says "receipts 1-1000 exist with Merkle root X"
- If receipt 500 is deleted, the Merkle root won't match

Key rotation:
- HMAC keys rotate on a schedule (e.g., weekly)
- Each receipt includes the key_id it was signed with
- Old keys are kept for verification but not for new signing
"""

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .chain import jcs_canonicalize


def hmac_sign_receipt(
    receipt: Dict[str, Any],
    signing_key: bytes,
    key_id: str = "default",
) -> Dict[str, str]:
    """
    Compute HMAC-SHA256 signature for a receipt.

    Unlike plain SHA-256 hashing, HMAC proves that the receipt was
    signed by someone who possesses the signing key. An attacker
    who modifies a receipt cannot recompute the HMAC without the key.

    Args:
        receipt: Receipt dictionary (uses entry_hash as input).
        signing_key: HMAC signing key (bytes, minimum 32 bytes).
        key_id: Identifier for key rotation tracking.

    Returns:
        {
            "hmac_signature": hex-encoded HMAC-SHA256,
            "key_id": the key ID used,
            "algorithm": "hmac-sha256",
        }
    """
    if len(signing_key) < 32:
        raise ValueError("Signing key must be at least 32 bytes")

    # Sign the entry_hash (which covers all receipt content)
    entry_hash = receipt.get("entry_hash", "")
    if not entry_hash:
        raise ValueError("Receipt must have entry_hash before signing")

    signature = hmac.new(
        signing_key,
        entry_hash.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "hmac_signature": signature,
        "key_id": key_id,
        "algorithm": "hmac-sha256",
    }


def verify_hmac_signature(
    receipt: Dict[str, Any],
    signing_key: bytes,
    expected_signature: str,
) -> bool:
    """
    Verify HMAC signature on a receipt.

    Args:
        receipt: Receipt dictionary.
        signing_key: The signing key (must match the one used to sign).
        expected_signature: The stored HMAC signature to verify against.

    Returns:
        True if signature is valid.
    """
    entry_hash = receipt.get("entry_hash", "")
    if not entry_hash:
        return False

    computed = hmac.new(
        signing_key,
        entry_hash.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, expected_signature)


# ---------------------------------------------------------------------------
# Signed checkpoints (sequence gap detection)
# ---------------------------------------------------------------------------

class SequenceCheckpoint:
    """
    Periodic signed checkpoint for sequence gap detection.

    Inspired by Certificate Transparency Signed Tree Heads and
    AWS CloudTrail digest files. A checkpoint commits to:
    - A sequence range (first_seq → last_seq)
    - A running hash of all receipt entry_hashes in that range
    - A timestamp
    - An HMAC signature

    If any receipt in the range is deleted or modified after the
    checkpoint was created, the running hash won't match.

    Usage:
        cp = SequenceCheckpoint(signing_key=key)
        for receipt in receipts:
            cp.add(receipt)
        checkpoint = cp.finalize()
        # checkpoint["running_hash"] commits to all receipts
        # checkpoint["hmac_signature"] proves it wasn't forged
    """

    def __init__(
        self,
        signing_key: bytes,
        key_id: str = "default",
    ) -> None:
        """
        Initialize checkpoint builder.

        Args:
            signing_key: HMAC signing key.
            key_id: Key identifier for rotation.
        """
        if len(signing_key) < 32:
            raise ValueError("Signing key must be at least 32 bytes")

        self._signing_key = signing_key
        self._key_id = key_id
        self._running_hash = hashlib.sha256()
        self._count = 0
        self._first_seq: Optional[int] = None
        self._last_seq: Optional[int] = None
        self._first_hash: Optional[str] = None
        self._last_hash: Optional[str] = None

    def add(self, receipt: Dict[str, Any]) -> None:
        """
        Add a receipt to the checkpoint.

        Args:
            receipt: Receipt dictionary with entry_hash and sequence_no.
        """
        entry_hash = receipt.get("entry_hash", "")
        seq_no = receipt.get("sequence_no", 0)

        if not entry_hash:
            raise ValueError("Receipt must have entry_hash")

        # Update running hash (order-dependent — catches deletions AND reordering)
        self._running_hash.update(entry_hash.encode("utf-8"))
        self._count += 1

        if self._first_seq is None:
            self._first_seq = seq_no
            self._first_hash = entry_hash

        self._last_seq = seq_no
        self._last_hash = entry_hash

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the checkpoint and sign it.

        Returns:
            Signed checkpoint dictionary.
        """
        running_hash = self._running_hash.hexdigest()
        timestamp = time.time()

        # Build checkpoint payload
        checkpoint = {
            "type": "sequence_checkpoint",
            "first_sequence": self._first_seq or 0,
            "last_sequence": self._last_seq or 0,
            "first_entry_hash": self._first_hash or "",
            "last_entry_hash": self._last_hash or "",
            "receipt_count": self._count,
            "running_hash": running_hash,
            "timestamp": timestamp,
            "key_id": self._key_id,
        }

        # Sign the checkpoint
        sign_input = (
            f"{checkpoint['first_sequence']}:"
            f"{checkpoint['last_sequence']}:"
            f"{checkpoint['receipt_count']}:"
            f"{running_hash}:"
            f"{timestamp}"
        )

        checkpoint["hmac_signature"] = hmac.new(
            self._signing_key,
            sign_input.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        checkpoint["algorithm"] = "hmac-sha256"

        return checkpoint

    @property
    def count(self) -> int:
        """Number of receipts added so far."""
        return self._count


def verify_checkpoint(
    checkpoint: Dict[str, Any],
    receipts: List[Dict[str, Any]],
    signing_key: bytes,
) -> Dict[str, Any]:
    """
    Verify a signed checkpoint against a list of receipts.

    Checks:
    1. HMAC signature is valid (proves checkpoint wasn't forged)
    2. Running hash matches (proves no receipts were added/deleted/modified)
    3. Receipt count matches
    4. Sequence range matches

    Args:
        checkpoint: The signed checkpoint to verify.
        receipts: The receipts that should match this checkpoint.
        signing_key: The HMAC signing key.

    Returns:
        Verification result dict.
    """
    errors: List[str] = []

    # 1. Verify HMAC signature
    running_hash = checkpoint.get("running_hash", "")
    timestamp = checkpoint.get("timestamp", 0)
    first_seq = checkpoint.get("first_sequence", 0)
    last_seq = checkpoint.get("last_sequence", 0)
    receipt_count = checkpoint.get("receipt_count", 0)

    sign_input = (
        f"{first_seq}:{last_seq}:{receipt_count}:"
        f"{running_hash}:{timestamp}"
    )

    expected_sig = checkpoint.get("hmac_signature", "")
    computed_sig = hmac.new(
        signing_key,
        sign_input.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    signature_valid = hmac.compare_digest(computed_sig, expected_sig)
    if not signature_valid:
        errors.append("HMAC signature mismatch — checkpoint may be forged")

    # 2. Verify running hash
    running = hashlib.sha256()
    for r in receipts:
        entry_hash = r.get("entry_hash", "")
        running.update(entry_hash.encode("utf-8"))

    computed_running = running.hexdigest()
    hash_matches = computed_running == running_hash
    if not hash_matches:
        errors.append(
            f"Running hash mismatch — receipts modified/deleted/added "
            f"(expected {running_hash[:16]}..., got {computed_running[:16]}...)"
        )

    # 3. Verify count
    count_matches = len(receipts) == receipt_count
    if not count_matches:
        errors.append(
            f"Receipt count mismatch: checkpoint says {receipt_count}, "
            f"got {len(receipts)}"
        )

    # 4. Verify sequence range
    if receipts:
        actual_first = receipts[0].get("sequence_no", 0)
        actual_last = receipts[-1].get("sequence_no", 0)
        if actual_first != first_seq:
            errors.append(
                f"First sequence mismatch: expected {first_seq}, got {actual_first}"
            )
        if actual_last != last_seq:
            errors.append(
                f"Last sequence mismatch: expected {last_seq}, got {actual_last}"
            )

    return {
        "valid": len(errors) == 0,
        "signature_valid": signature_valid,
        "hash_matches": hash_matches,
        "count_matches": count_matches,
        "errors": errors,
    }
