# Veratum Verification Toolkit

**Independent verification of Veratum compliance receipts — no account required, no servers needed, pure cryptography.**

The Veratum Verification Toolkit is a standalone, zero-dependency Python library for verifying compliance audit receipts. It proves that Veratum's audit logs are append-only and tamper-evident, using RFC 9162-compatible Merkle tree cryptography. If Veratum disappears tomorrow, your evidence is still verifiable.

## Why This Matters

Veratum generates compliance receipts for every AI decision made under its oversight. These receipts are cryptographically signed and linked into an append-only transparency log. But what if you need to prove compliance to regulators *without* calling Veratum's API? What if Veratum's servers go down? This toolkit answers that by letting anyone verify receipts offline using only the receipt and a proof.

## Install

```bash
pip install veratum-verify
```

## Quick Start

```python
from veratum_verify import ReceiptVerifier

# Load a receipt (e.g., from a file or database)
receipt = {
    "receipt_id": "audit-001",
    "timestamp": 1704067200,
    "credential": {"audit_id": "audit-001", "decision": "approved"},
    "receipt_hash": "a1b2c3...",
    "signature": "d4e5f6...",
}

# Verify the receipt
verifier = ReceiptVerifier()
result = verifier.verify_receipt(receipt)

if result.valid:
    print("Receipt is valid!")
else:
    print("Receipt failed validation:")
    for error in result.errors:
        print(f"  - {error}")
```

## CLI Usage

Verify a receipt from the command line:

```bash
# Verify a single receipt
veratum-verify receipt receipt.json

# Verify a chain of receipts
veratum-verify chain receipts.json

# Verify an inclusion proof (receipt exists in the log)
veratum-verify inclusion --leaf <LEAF_HASH> --proof proof.json --root <ROOT_HASH> --size <TREE_SIZE> --leaf-index <INDEX>

# Verify a consistency proof (log didn't shrink)
veratum-verify consistency --old-size 100 --new-size 150 --old-root <OLD_ROOT> --new-root <NEW_ROOT> --proof proof.json
```

Example:

```bash
$ veratum-verify receipt compliance-audit-001.json
```

Output:

```
Veratum Receipt Verification
------
Receipt ID: audit-001

✓ fields_present          [PASS]
✓ hash_matches           [PASS]
✓ timestamp_valid        [PASS]
✓ signature_valid        [PASS]
✓ credential_valid       [PASS]

✓ Receipt is valid
```

## API Reference

### ReceiptVerifier

```python
from veratum_verify import ReceiptVerifier

verifier = ReceiptVerifier(hmac_secret="veratum-default-secret")
```

#### `verify_receipt(receipt: dict) -> VerificationResult`

Verify a single receipt's integrity. Checks:
- All required fields present (receipt_id, timestamp, credential, receipt_hash, signature)
- Hash matches content
- Signature is valid (HMAC-SHA256)
- Timestamp is reasonable (between 1970 and 2100)
- Credential is non-empty

```python
result = verifier.verify_receipt(receipt)
assert result.valid
assert result.checks["signature_valid"]
for error in result.errors:
    print(error)
```

#### `verify_chain(receipts: list) -> ChainVerificationResult`

Verify a chain of receipts for sequential integrity. Each receipt's `prev_hash` must match the hash of the previous receipt.

```python
result = verifier.verify_chain([receipt1, receipt2, receipt3])
assert result.valid
assert result.chain_length == 3
if result.breaks:
    print(f"Chain breaks at indices: {result.breaks}")
```

#### `verify_inclusion(leaf_hash: str, proof: list, tree_size: int, root_hash: str, leaf_index: int) -> bool`

Verify a Merkle inclusion proof (RFC 9162). Proves that a specific receipt exists at a given position in the transparency log.

```python
valid = verifier.verify_inclusion(
    leaf_hash="a1b2c3...",
    proof=["sibling_hash_1", "sibling_hash_2"],
    tree_size=1000,
    root_hash="root_hash...",
    leaf_index=42,
)
assert valid
```

#### `verify_consistency(old_size: int, new_size: int, old_root: str, new_root: str, proof: list) -> bool`

Verify a Merkle consistency proof (RFC 9162). Proves that the new tree contains all entries from the old tree (no deletions).

```python
valid = verifier.verify_consistency(
    old_size=100,
    new_size=150,
    old_root="old_root_hash...",
    new_root="new_root_hash...",
    proof=[...],
)
assert valid
```

### Standalone Functions

```python
from veratum_verify import verify_inclusion, verify_consistency, hash_leaf, hash_pair

# Hash functions (RFC 9162)
leaf_hash = hash_leaf({"key": "value"})  # 0x00-prefixed SHA256
parent_hash = hash_pair(left_hash, right_hash)  # 0x01-prefixed SHA256

# Verification functions
assert verify_inclusion(leaf_hash, proof, tree_size, root, leaf_index)
assert verify_consistency(old_size, new_size, old_root, new_root, proof)
```

### VerificationResult

```python
@dataclass
class VerificationResult:
    valid: bool  # Overall result
    checks: dict  # {check_name: passed_bool}
    errors: list  # Error messages
    receipt_id: str  # ID of receipt being verified

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        ...
```

### ChainVerificationResult

```python
@dataclass
class ChainVerificationResult:
    valid: bool  # Overall result
    chain_length: int  # Number of receipts
    breaks: list  # Indices where chain breaks
    errors: list  # Error messages

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        ...
```

## How It Works

### Receipt Verification

A Veratum receipt is a JSON object with:
- `receipt_id`: Unique identifier
- `timestamp`: Unix timestamp of the audit
- `credential`: The audit data (dict)
- `receipt_hash`: SHA256 of the receipt (without hash and signature)
- `signature`: HMAC-SHA256 signature over the receipt

The toolkit verifies:
1. **Hash integrity**: Recomputes the SHA256 and compares
2. **Signature validity**: Recomputes the HMAC-SHA256 using the secret key
3. **Field presence**: All required fields are present
4. **Timestamp validity**: Timestamp is in a reasonable range
5. **Credential validity**: Credential is non-empty

### Chain Verification

Receipts can be linked into a chain by including the previous receipt's hash:

```json
{
  "receipt_id": "audit-002",
  "prev_hash": "a1b2c3...",
  ...
}
```

The toolkit verifies each receipt's `prev_hash` matches the computed hash of the previous receipt, proving they form an unbroken chain.

### Inclusion Proofs

Veratum's transparency log is a Merkle tree. An inclusion proof proves that a specific receipt exists at a given position in the tree without needing the entire tree. The toolkit uses RFC 9162 Merkle tree math to reconstruct the root hash and verify it matches.

### Consistency Proofs

A consistency proof proves that a new version of the log is an extension of an old version (no deletions, no reordering). The toolkit verifies this by checking that the proof contains enough information to reconstruct both tree states.

## Cryptography Details

- **Hashing**: SHA-256 (FIPS 180-4)
- **Signatures**: HMAC-SHA256 with domain separation
- **Merkle tree**: RFC 9162 (Certificate Transparency)
- **Domain separation**: 0x00 prefix for leaves, 0x01 prefix for internal nodes

All cryptography uses only the Python standard library (hashlib, hmac). No external dependencies.

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

The test suite includes 40+ tests covering:
- Hash function correctness
- Receipt verification (valid, tampered, missing fields)
- Chain verification (valid chains, breaks, reordering)
- Inclusion proof verification (valid proofs, invalid proofs, edge cases)
- Consistency proof verification (valid, invalid, tree growth/shrinkage)
- CLI functionality

## License

MIT. Copyright 2026 Veratum Inc.

## Contributing

Contributions welcome. Please file issues or open pull requests on GitHub.

## FAQ

**Q: Can I verify receipts without the secret key?**

A: If you're verifying receipts signed by Veratum, you don't need the secret key. The toolkit can verify any receipt that has a valid signature. However, if you're running your own log with a custom secret key, provide it to the ReceiptVerifier:

```python
verifier = ReceiptVerifier(hmac_secret="your-secret-key")
```

**Q: What if the log operator tampers with receipts?**

A: The Merkle tree structure and cryptographic hashing make tampering detectable. If a receipt is modified, its hash changes, and the inclusion/consistency proofs will fail. Additionally, third-party witnesses can cosign tree heads to provide independent verification.

**Q: Is this production-ready?**

A: This is a reference implementation suitable for integration into compliance systems. For production use, consider:
- Storing receipts in a database with integrity checks
- Using third-party witnesses to cosign tree heads
- Implementing periodic consistency checks
- Archiving old receipts for long-term retention

**Q: Why no external dependencies?**

A: Cryptographic verification should be auditable and portable. Using only the Python standard library makes the code easier to audit, reduces supply chain risk, and ensures it works in any Python environment.

## See Also

- [RFC 9162 - Certificate Transparency](https://datatracker.ietf.org/doc/html/rfc9162)
- [Veratum Documentation](https://docs.veratum.ai/)
- [Veratum SDK](https://github.com/veratum-ai/veratum-v2)
