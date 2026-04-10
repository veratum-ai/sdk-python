"""Receipt self-verification — independent verification without Veratum servers.

Addresses the critical customer concern: "What if Veratum shuts down?"

Every Veratum receipt is independently verifiable because:
1. The entry_hash is a SHA-256 of JCS-canonicalized receipt fields
2. The prev_hash links to the previous receipt (chain integrity)
3. The chain can be verified by anyone with the receipt sequence

No Veratum server, API key, or internet connection is needed. The
receipt IS the proof. This is the same principle behind certificate
transparency logs (RFC 9162).

Design decision: We chose JCS (RFC 8785) over json.dumps(sort_keys=True)
because JCS is a published standard with defined UTF-16 sort order and
number serialization rules, making it legally defensible in EU courts
under eIDAS Article 25 (electronic documents as evidence).
"""

from typing import Any, Dict, List, Optional, Tuple

from .chain import HashChain, jcs_canonicalize, jcs_hash


def verify_receipt(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify a single receipt's integrity independently.

    Recomputes the entry_hash from the receipt fields and compares
    it to the stored entry_hash. No server connection needed.

    Args:
        receipt: A Veratum receipt dictionary.

    Returns:
        Verification result dict:
        {
            "valid": bool,
            "entry_hash_match": bool,
            "computed_hash": str,
            "stored_hash": str,
            "errors": [str],
        }
    """
    errors: List[str] = []
    stored_hash = receipt.get("entry_hash", "")

    if not stored_hash:
        errors.append("Receipt missing entry_hash field")
        return {
            "valid": False,
            "entry_hash_match": False,
            "computed_hash": "",
            "stored_hash": "",
            "errors": errors,
        }

    # Recompute hash using same algorithm as HashChain.compute_entry_hash
    chain = HashChain()
    computed_hash = chain.compute_entry_hash(receipt)

    entry_hash_match = computed_hash == stored_hash
    if not entry_hash_match:
        errors.append(
            f"Hash mismatch: computed {computed_hash[:16]}... "
            f"!= stored {stored_hash[:16]}..."
        )

    # Check required fields exist
    required = ["timestamp", "schema_version", "sdk_version"]
    for field in required:
        if field not in receipt:
            errors.append(f"Missing required field: {field}")

    return {
        "valid": entry_hash_match and len(errors) == 0,
        "entry_hash_match": entry_hash_match,
        "computed_hash": computed_hash,
        "stored_hash": stored_hash,
        "errors": errors,
    }


def verify_chain(receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify an entire receipt chain's integrity.

    Checks:
    1. Each receipt's entry_hash is correct (content integrity)
    2. Each receipt's prev_hash matches the previous entry_hash (chain integrity)
    3. Sequence numbers are monotonically increasing
    4. No gaps in the sequence

    This is the full independent audit — a regulator or auditor can
    run this on exported receipts with zero Veratum infrastructure.

    Args:
        receipts: List of receipt dicts, ordered by sequence_no.

    Returns:
        Chain verification result:
        {
            "valid": bool,
            "total_receipts": int,
            "verified_count": int,
            "chain_intact": bool,
            "sequence_valid": bool,
            "errors": [{"index": int, "receipt_hash": str, "error": str}],
            "first_sequence": int,
            "last_sequence": int,
        }
    """
    if not receipts:
        return {
            "valid": True,
            "total_receipts": 0,
            "verified_count": 0,
            "chain_intact": True,
            "sequence_valid": True,
            "errors": [],
            "first_sequence": 0,
            "last_sequence": 0,
        }

    errors: List[Dict[str, Any]] = []
    verified_count = 0
    chain_intact = True
    sequence_valid = True
    prev_entry_hash: Optional[str] = None
    prev_seq: Optional[int] = None

    for i, receipt in enumerate(receipts):
        receipt_hash = receipt.get("entry_hash", "unknown")

        # 1. Verify individual receipt integrity
        result = verify_receipt(receipt)
        if result["valid"]:
            verified_count += 1
        else:
            for err in result["errors"]:
                errors.append({
                    "index": i,
                    "receipt_hash": receipt_hash,
                    "error": err,
                })

        # 2. Verify chain linkage (prev_hash → previous entry_hash)
        receipt_prev_hash = receipt.get("prev_hash", "")
        seq_no = receipt.get("sequence_no", 0)

        if i == 0:
            # First receipt: prev_hash should be genesis (all zeros)
            if receipt_prev_hash and receipt_prev_hash != "0" * 64:
                # Could be a partial chain export — warn but don't fail
                pass
        else:
            if prev_entry_hash and receipt_prev_hash != prev_entry_hash:
                chain_intact = False
                errors.append({
                    "index": i,
                    "receipt_hash": receipt_hash,
                    "error": (
                        f"Chain break: prev_hash {receipt_prev_hash[:16]}... "
                        f"!= previous entry_hash {prev_entry_hash[:16]}..."
                    ),
                })

        # 3. Verify sequence monotonicity
        if prev_seq is not None:
            if seq_no <= prev_seq:
                sequence_valid = False
                errors.append({
                    "index": i,
                    "receipt_hash": receipt_hash,
                    "error": f"Sequence not monotonic: {seq_no} <= {prev_seq}",
                })
            elif seq_no != prev_seq + 1:
                sequence_valid = False
                errors.append({
                    "index": i,
                    "receipt_hash": receipt_hash,
                    "error": f"Sequence gap: expected {prev_seq + 1}, got {seq_no}",
                })

        prev_entry_hash = receipt.get("entry_hash")
        prev_seq = seq_no

    first_seq = receipts[0].get("sequence_no", 0) if receipts else 0
    last_seq = receipts[-1].get("sequence_no", 0) if receipts else 0

    return {
        "valid": len(errors) == 0,
        "total_receipts": len(receipts),
        "verified_count": verified_count,
        "chain_intact": chain_intact,
        "sequence_valid": sequence_valid,
        "errors": errors,
        "first_sequence": first_seq,
        "last_sequence": last_seq,
    }


def export_verification_report(
    receipts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a human-readable verification report for auditors.

    Suitable for inclusion in compliance documentation, audit reports,
    or legal proceedings. Contains all information needed to independently
    reproduce the verification.

    Args:
        receipts: List of receipt dicts.

    Returns:
        Verification report dict with summary and detailed results.
    """
    chain_result = verify_chain(receipts)

    return {
        "report_type": "veratum_independent_verification",
        "verification_method": "SHA-256 over RFC 8785 JCS-canonicalized JSON",
        "requires_veratum_server": False,
        "summary": {
            "verdict": "PASS" if chain_result["valid"] else "FAIL",
            "total_receipts": chain_result["total_receipts"],
            "verified_receipts": chain_result["verified_count"],
            "chain_integrity": "intact" if chain_result["chain_intact"] else "broken",
            "sequence_integrity": "valid" if chain_result["sequence_valid"] else "gaps_detected",
        },
        "chain_verification": chain_result,
        "reproducibility_note": (
            "This verification can be reproduced by any party with access to "
            "the receipt data. The algorithm is: (1) Remove entry_hash, "
            "signature, and verifiable_credential fields, "
            "(2) Serialize remaining fields using RFC 8785 JSON Canonicalization "
            "Scheme, (3) Compute SHA-256 hash, (4) Compare to stored entry_hash. "
            "No Veratum API access, keys, or servers are required."
        ),
    }
