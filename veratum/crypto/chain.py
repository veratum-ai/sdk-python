"""Hash chain integrity management for Veratum receipts.

Uses RFC 8785 JSON Canonicalization Scheme (JCS) for deterministic
serialization. This is legally required for litigation-grade evidence —
Python's json.dumps(sort_keys=True) does NOT satisfy RFC 8785 due to
key sorting differences (UTF-16 vs UTF-8) and number serialization.
"""

import hashlib
from typing import Any, Optional
from decimal import Decimal


# ============================================================================
# RFC 8785 JSON CANONICALIZATION SCHEME (JCS) — SDK Implementation
# ============================================================================

def jcs_canonicalize(obj: Any) -> bytes:
    """
    Serialize a Python object to canonical JSON per RFC 8785 (JCS).

    Returns:
        UTF-8 encoded canonical JSON bytes
    """
    return _jcs_serialize(obj).encode("utf-8")


def _jcs_serialize(obj: Any) -> str:
    """Internal recursive JCS serializer."""
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        import math
        if math.isnan(obj) or math.isinf(obj):
            return "null"
        if obj == 0.0:
            return "0"
        s = repr(obj)
        if "." in s and "e" not in s and "E" not in s:
            s = s.rstrip("0").rstrip(".")
        return s
    if isinstance(obj, str):
        return _jcs_serialize_string(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_jcs_serialize(item) for item in obj) + "]"
    if isinstance(obj, dict):
        sorted_keys = sorted(obj.keys(), key=_utf16_sort_key)
        pairs = ",".join(
            f"{_jcs_serialize_string(k)}:{_jcs_serialize(obj[k])}"
            for k in sorted_keys
        )
        return "{" + pairs + "}"
    if isinstance(obj, Decimal):
        return _jcs_serialize(float(obj))
    return _jcs_serialize_string(str(obj))


def _utf16_sort_key(s: str) -> list:
    """Sort key based on UTF-16 code unit order per RFC 8785 §3.2.3."""
    result = []
    for ch in s:
        cp = ord(ch)
        if cp >= 0x10000:
            cp -= 0x10000
            result.append(0xD800 + (cp >> 10))
            result.append(0xDC00 + (cp & 0x3FF))
        else:
            result.append(cp)
    return result


def _jcs_serialize_string(s: str) -> str:
    """Serialize string with minimal JSON escaping per RFC 8785."""
    result = ['"']
    for ch in s:
        cp = ord(ch)
        if ch == '"':
            result.append('\\"')
        elif ch == '\\':
            result.append('\\\\')
        elif ch == '\b':
            result.append('\\b')
        elif ch == '\f':
            result.append('\\f')
        elif ch == '\n':
            result.append('\\n')
        elif ch == '\r':
            result.append('\\r')
        elif ch == '\t':
            result.append('\\t')
        elif cp < 0x20:
            result.append(f"\\u{cp:04x}")
        else:
            result.append(ch)
    result.append('"')
    return "".join(result)


def jcs_hash(obj: Any) -> str:
    """Compute SHA-256 hash of JCS-canonicalized JSON. Returns hex digest."""
    return hashlib.sha256(jcs_canonicalize(obj)).hexdigest()


def jcs_hash_sha3(obj: Any) -> str:
    """Compute SHA3-256 hash of JCS-canonicalized JSON. Returns hex digest.

    Schema 2.3.0: provides a second, independent hash algorithm so that a
    future compromise of SHA-256 does not invalidate the evidence chain.
    SHA-3 uses the Keccak construction, making it structurally independent
    from SHA-2.
    """
    return hashlib.sha3_256(jcs_canonicalize(obj)).hexdigest()


# Fields excluded from hash computation. Both entry_hash and entry_hash_sha3
# are excluded because they are *computed over* the rest of the receipt.
# xrpl_tx_hash is excluded because anchoring happens *after* the hash is
# computed, and the signature/VC fields are attached post-hoc by the signer.
# (merkle_proof and rfc3161_token are set to deterministic placeholder values
#  before hashing and then overwritten later; they remain INSIDE the hash
#  via their placeholder, but are version-gated so old receipts still verify.)
_HASH_EXCLUDED_FIELDS = (
    "entry_hash",
    "entry_hash_sha3",
    "xrpl_tx_hash",
    "opentimestamps_proof",
    "rfc3161_token",
    "signature",
    "signature_ed25519",
    "signature_ml_dsa_65",
    "verifiable_credential",
)


class HashChain:
    """Manages cryptographic chain integrity for audit receipts."""

    def __init__(self) -> None:
        """Initialize the hash chain with genesis state."""
        self.sequence_no: int = 0
        self.prev_hash: str = "0" * 64
        self.last_entry_hash: Optional[str] = None

    def compute_entry_hash(self, receipt_dict: dict) -> str:
        """
        Compute SHA256 hash of RFC 8785 JCS-canonicalized receipt JSON.

        Uses RFC 8785 JSON Canonicalization Scheme for deterministic
        serialization that is legally defensible in EU courts.

        The hash excludes entry_hash, entry_hash_sha3, signature, and
        verifiable_credential fields to allow for subsequent signing and
        dual-hash storage.

        Args:
            receipt_dict: Receipt data dictionary

        Returns:
            Hex-encoded SHA256 hash
        """
        # Create canonical form by removing fields that shouldn't be hashed
        canonical = {
            k: v
            for k, v in receipt_dict.items()
            if k not in _HASH_EXCLUDED_FIELDS
        }

        # RFC 8785 JCS canonicalization (NOT json.dumps sort_keys)
        entry_hash = jcs_hash(canonical)
        return entry_hash

    def compute_entry_hash_sha3(self, receipt_dict: dict) -> str:
        """
        Compute SHA3-256 hash of RFC 8785 JCS-canonicalized receipt JSON.

        Schema 2.3.0 dual-hash companion to compute_entry_hash. Uses the
        same canonical form and exclusion set so the two hashes cover
        identical bytes, differing only in the hash algorithm.

        Args:
            receipt_dict: Receipt data dictionary

        Returns:
            Hex-encoded SHA3-256 hash
        """
        canonical = {
            k: v
            for k, v in receipt_dict.items()
            if k not in _HASH_EXCLUDED_FIELDS
        }
        return jcs_hash_sha3(canonical)

    def compute_dual_entry_hash(self, receipt_dict: dict) -> tuple:
        """
        Compute both SHA-256 and SHA3-256 entry hashes in one pass.

        Returns:
            Tuple of (sha256_hex, sha3_256_hex)
        """
        canonical = {
            k: v
            for k, v in receipt_dict.items()
            if k not in _HASH_EXCLUDED_FIELDS
        }
        canonical_bytes = jcs_canonicalize(canonical)
        return (
            hashlib.sha256(canonical_bytes).hexdigest(),
            hashlib.sha3_256(canonical_bytes).hexdigest(),
        )

    def advance_chain(self, receipt_dict: dict) -> None:
        """
        Advance the hash chain with a new receipt.

        Updates sequence number and establishes linkage to previous receipt
        via prev_hash pointing to previous entry_hash.

        Args:
            receipt_dict: Receipt dictionary (must have entry_hash set)
        """
        self.last_entry_hash = receipt_dict.get("entry_hash")
        self.sequence_no += 1
        self.prev_hash = self.last_entry_hash or "0" * 64

    def get_chain_state(self) -> dict:
        """
        Get current chain state.

        Returns:
            Dictionary with current sequence_no and prev_hash
        """
        return {"sequence_no": self.sequence_no, "prev_hash": self.prev_hash}

    def reset(self) -> None:
        """Reset chain to genesis state."""
        self.sequence_no = 0
        self.prev_hash = "0" * 64
        self.last_entry_hash = None
