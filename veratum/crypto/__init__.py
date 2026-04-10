"""Cryptographic primitives — hashing, signing, Merkle trees, verification.

Modules:
    chain         — SHA-256 hash chaining with JCS canonicalization (RFC 8785)
    merkle        — Merkle tree batch anchoring and inclusion proofs
    signing       — HMAC receipt signing and sequence checkpoints
    transparency  — Certificate Transparency-style append-only logs (RFC 9162)
    verify        — Receipt and chain integrity verification

Note: ``zk`` (zero-knowledge proof generation) has been moved to
``veratum.future`` until product-market fit is established.
"""

from .chain import HashChain, jcs_canonicalize, jcs_hash
from .merkle import MerkleTree, BatchAnchor, verify_proof
from .signing import hmac_sign_receipt, verify_hmac_signature, SequenceCheckpoint, verify_checkpoint
from .transparency import TransparencyLog, WitnessRegistry, verify_inclusion, verify_consistency, hash_receipt
from .verify import verify_receipt, verify_chain, export_verification_report

__all__ = [
    "HashChain", "jcs_canonicalize", "jcs_hash",
    "MerkleTree", "BatchAnchor", "verify_proof",
    "hmac_sign_receipt", "verify_hmac_signature", "SequenceCheckpoint", "verify_checkpoint",
    "TransparencyLog", "WitnessRegistry", "verify_inclusion", "verify_consistency", "hash_receipt",
    "verify_receipt", "verify_chain", "export_verification_report",
]
