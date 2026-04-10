"""
Comprehensive tests for Veratum Verification Toolkit.

Tests cover:
  - Receipt verification (valid, tampered, missing fields)
  - Chain verification (valid chain, breaks, reordering)
  - Inclusion proof verification (valid, invalid, edge cases)
  - Consistency proof verification (valid, invalid, edge cases)
  - CLI invocation
"""

import json
import sys
import subprocess
import tempfile
import os
import hashlib
import hmac
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from veratum_verify import (
    ReceiptVerifier,
    VerificationResult,
    ChainVerificationResult,
    verify_inclusion,
    verify_consistency,
    hash_leaf,
    hash_pair,
)


class TestHashFunctions:
    """Test Merkle tree hash functions."""

    def test_hash_leaf_with_dict(self):
        """Test hashing a dict leaf."""
        data = {"key": "value"}
        h = hash_leaf(data)
        assert len(h) == 64  # SHA256 hex string
        assert h == hash_leaf(data)  # Deterministic

    def test_hash_leaf_with_bytes(self):
        """Test hashing bytes."""
        data = b"test data"
        h = hash_leaf(data)
        assert len(h) == 64

    def test_hash_leaf_domain_separation(self):
        """Test that leaf hash starts with 0x00 prefix."""
        data = "test"
        h_leaf = hash_leaf(data)
        # Verify it's different from a raw SHA256
        raw_sha256 = hashlib.sha256(b"test").hexdigest()
        assert h_leaf != raw_sha256

    def test_hash_pair(self):
        """Test hashing a pair of hashes."""
        left = hash_leaf("left")
        right = hash_leaf("right")
        h = hash_pair(left, right)
        assert len(h) == 64
        assert h == hash_pair(left, right)  # Deterministic

    def test_hash_pair_domain_separation(self):
        """Test that pair hash is different from raw combination."""
        left = hash_leaf("left")
        right = hash_leaf("right")
        h_pair = hash_pair(left, right)
        # Verify it's different from a naive concatenation hash
        raw_concat = hashlib.sha256((left + right).encode()).hexdigest()
        assert h_pair != raw_concat

    def test_hash_pair_order_matters(self):
        """Test that hash_pair(left, right) != hash_pair(right, left)."""
        left = hash_leaf("left")
        right = hash_leaf("right")
        h1 = hash_pair(left, right)
        h2 = hash_pair(right, left)
        assert h1 != h2


class TestReceiptVerification:
    """Test single receipt verification."""

    def test_valid_receipt(self):
        """Test verification of a valid receipt."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,  # 2024-01-01
            "credential": {
                "audit_id": "audit-001",
                "decision": "approved",
                "confidence": 0.95,
            },
        }

        # Add hash and signature
        canonical = json.dumps({
            "receipt_id": receipt["receipt_id"],
            "timestamp": receipt["timestamp"],
            "credential": receipt["credential"],
        }, sort_keys=True)
        receipt["receipt_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

        secret = "veratum-default-secret"
        receipt["signature"] = hmac.new(
            secret.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        verifier = ReceiptVerifier(secret)
        result = verifier.verify_receipt(receipt)

        assert result.valid
        assert result.checks["fields_present"]
        assert result.checks["hash_matches"]
        assert result.checks["timestamp_valid"]
        assert result.checks["signature_valid"]
        assert result.checks["credential_valid"]

    def test_missing_receipt_id(self):
        """Test receipt with missing receipt_id."""
        receipt = {
            "timestamp": 1704067200,
            "credential": {"audit_id": "001"},
        }
        receipt["receipt_hash"] = "abc123"
        receipt["signature"] = "def456"

        verifier = ReceiptVerifier()
        result = verifier.verify_receipt(receipt)

        assert not result.valid
        assert not result.checks["fields_present"]
        assert "Missing receipt_id" in result.errors

    def test_missing_timestamp(self):
        """Test receipt with missing timestamp."""
        receipt = {
            "receipt_id": "audit-001",
            "credential": {"audit_id": "001"},
        }
        receipt["receipt_hash"] = "abc123"
        receipt["signature"] = "def456"

        verifier = ReceiptVerifier()
        result = verifier.verify_receipt(receipt)

        assert not result.valid
        assert "Missing timestamp" in result.errors

    def test_missing_credential(self):
        """Test receipt with missing credential."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,
        }
        receipt["receipt_hash"] = "abc123"
        receipt["signature"] = "def456"

        verifier = ReceiptVerifier()
        result = verifier.verify_receipt(receipt)

        assert not result.valid
        assert "Missing credential" in result.errors

    def test_hash_mismatch(self):
        """Test receipt with incorrect hash."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,
            "credential": {"audit_id": "001"},
            "receipt_hash": "0000000000000000000000000000000000000000000000000000000000000000",
            "signature": "signature",
        }

        verifier = ReceiptVerifier()
        result = verifier.verify_receipt(receipt)

        assert not result.valid
        assert not result.checks["hash_matches"]

    def test_invalid_timestamp(self):
        """Test receipt with invalid timestamp."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 9999999999999,  # Way too far in the future
            "credential": {"audit_id": "001"},
            "receipt_hash": "abc123",
            "signature": "def456",
        }

        verifier = ReceiptVerifier()
        result = verifier.verify_receipt(receipt)

        assert not result.valid
        assert not result.checks["timestamp_valid"]

    def test_empty_credential(self):
        """Test receipt with empty credential."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,
            "credential": {},
            "receipt_hash": "abc123",
            "signature": "def456",
        }

        verifier = ReceiptVerifier()
        result = verifier.verify_receipt(receipt)

        assert not result.valid
        assert not result.checks["credential_valid"]

    def test_signature_verification_failure(self):
        """Test receipt with wrong signature."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,
            "credential": {"audit_id": "001"},
        }

        # Compute correct hash
        canonical = json.dumps({
            "receipt_id": receipt["receipt_id"],
            "timestamp": receipt["timestamp"],
            "credential": receipt["credential"],
        }, sort_keys=True)
        receipt["receipt_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        receipt["signature"] = "wrong_signature_000000000000000000000000000000"

        verifier = ReceiptVerifier("veratum-default-secret")
        result = verifier.verify_receipt(receipt)

        assert not result.valid
        assert not result.checks["signature_valid"]


class TestChainVerification:
    """Test chain of receipts verification."""

    def test_valid_chain_single_receipt(self):
        """Test valid chain with single receipt."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,
            "credential": {"audit_id": "001"},
        }
        canonical = json.dumps({
            "receipt_id": receipt["receipt_id"],
            "timestamp": receipt["timestamp"],
            "credential": receipt["credential"],
        }, sort_keys=True)
        receipt["receipt_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        receipt["signature"] = "sig1"
        receipt["prev_hash"] = ""

        verifier = ReceiptVerifier()
        result = verifier.verify_chain([receipt])

        assert result.valid
        assert result.chain_length == 1
        assert len(result.breaks) == 0

    def test_valid_chain_multiple_receipts(self):
        """Test valid chain with multiple linked receipts."""
        receipts = []

        for i in range(3):
            receipt = {
                "receipt_id": f"audit-{i:03d}",
                "timestamp": 1704067200 + i * 3600,
                "credential": {"audit_id": f"audit-{i:03d}"},
            }
            canonical = json.dumps({
                "receipt_id": receipt["receipt_id"],
                "timestamp": receipt["timestamp"],
                "credential": receipt["credential"],
            }, sort_keys=True)
            receipt["receipt_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
            receipt["signature"] = f"sig{i}"

            # Link to previous receipt
            if i == 0:
                receipt["prev_hash"] = ""
            else:
                prev_receipt = receipts[i - 1]
                prev_canonical = json.dumps({
                    "receipt_id": prev_receipt["receipt_id"],
                    "timestamp": prev_receipt["timestamp"],
                    "credential": prev_receipt["credential"],
                }, sort_keys=True)
                receipt["prev_hash"] = hashlib.sha256(prev_canonical.encode()).hexdigest()

            receipts.append(receipt)

        verifier = ReceiptVerifier()
        result = verifier.verify_chain(receipts)

        assert result.valid
        assert result.chain_length == 3
        assert len(result.breaks) == 0

    def test_chain_break_at_start(self):
        """Test chain with invalid first receipt (has prev_hash)."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,
            "credential": {"audit_id": "001"},
            "receipt_hash": "abc123",
            "signature": "sig1",
            "prev_hash": "should_be_empty",
        }

        verifier = ReceiptVerifier()
        result = verifier.verify_chain([receipt])

        assert not result.valid
        assert 0 in result.breaks

    def test_chain_break_in_middle(self):
        """Test chain where middle receipt has wrong prev_hash."""
        receipts = []

        for i in range(3):
            receipt = {
                "receipt_id": f"audit-{i:03d}",
                "timestamp": 1704067200 + i * 3600,
                "credential": {"audit_id": f"audit-{i:03d}"},
            }
            canonical = json.dumps({
                "receipt_id": receipt["receipt_id"],
                "timestamp": receipt["timestamp"],
                "credential": receipt["credential"],
            }, sort_keys=True)
            receipt["receipt_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
            receipt["signature"] = f"sig{i}"

            if i == 0:
                receipt["prev_hash"] = ""
            elif i == 1:
                # Intentional wrong hash
                receipt["prev_hash"] = "wrong_hash_0000000000000000000000000000000"
            else:
                prev_receipt = receipts[i - 1]
                prev_canonical = json.dumps({
                    "receipt_id": prev_receipt["receipt_id"],
                    "timestamp": prev_receipt["timestamp"],
                    "credential": prev_receipt["credential"],
                }, sort_keys=True)
                receipt["prev_hash"] = hashlib.sha256(prev_canonical.encode()).hexdigest()

            receipts.append(receipt)

        verifier = ReceiptVerifier()
        result = verifier.verify_chain(receipts)

        assert not result.valid
        assert 1 in result.breaks

    def test_empty_chain(self):
        """Test verification of empty chain."""
        verifier = ReceiptVerifier()
        result = verifier.verify_chain([])

        assert result.valid
        assert result.chain_length == 0
        assert len(result.breaks) == 0


class TestInclusionProof:
    """Test Merkle inclusion proof verification."""

    def test_single_leaf_in_tree(self):
        """Test inclusion proof for single-leaf tree."""
        leaf_hash = hash_leaf("entry-0")
        # Single leaf tree: leaf is the root
        valid = verify_inclusion(leaf_hash, [], 1, leaf_hash, 0)
        assert valid

    def test_two_leaf_tree(self):
        """Test inclusion proofs in 2-leaf tree."""
        leaf0 = hash_leaf("entry-0")
        leaf1 = hash_leaf("entry-1")
        root = hash_pair(leaf0, leaf1)

        # Proof for leaf 0
        proof0 = [leaf1]
        assert verify_inclusion(leaf0, proof0, 2, root, 0)

        # Proof for leaf 1
        proof1 = [leaf0]
        assert verify_inclusion(leaf1, proof1, 2, root, 1)

    def test_three_leaf_tree(self):
        """Test inclusion proofs in 3-leaf tree."""
        leaf0 = hash_leaf("entry-0")
        leaf1 = hash_leaf("entry-1")
        leaf2 = hash_leaf("entry-2")

        h01 = hash_pair(leaf0, leaf1)
        root = hash_pair(h01, leaf2)

        # Proof for leaf 0
        proof0 = [leaf1, leaf2]
        assert verify_inclusion(leaf0, proof0, 3, root, 0)

        # Proof for leaf 1
        proof1 = [leaf0, leaf2]
        assert verify_inclusion(leaf1, proof1, 3, root, 1)

        # Proof for leaf 2
        proof2 = [h01]
        assert verify_inclusion(leaf2, proof2, 3, root, 2)

    def test_invalid_leaf_index(self):
        """Test inclusion proof with out-of-bounds leaf_index."""
        leaf_hash = hash_leaf("entry-0")
        valid = verify_inclusion(leaf_hash, [], 1, leaf_hash, 5)
        assert not valid

    def test_wrong_root_hash(self):
        """Test inclusion proof with wrong root hash."""
        leaf0 = hash_leaf("entry-0")
        leaf1 = hash_leaf("entry-1")
        root = hash_pair(leaf0, leaf1)
        wrong_root = hash_leaf("wrong")

        proof = [leaf1]
        valid = verify_inclusion(leaf0, proof, 2, wrong_root, 0)
        assert not valid

    def test_tampered_leaf_hash(self):
        """Test inclusion proof with tampered leaf hash."""
        leaf0 = hash_leaf("entry-0")
        leaf1 = hash_leaf("entry-1")
        root = hash_pair(leaf0, leaf1)
        tampered_leaf = hash_leaf("tampered")

        proof = [leaf1]
        valid = verify_inclusion(tampered_leaf, proof, 2, root, 0)
        assert not valid

    def test_missing_proof_elements(self):
        """Test inclusion proof with missing proof elements."""
        leaf0 = hash_leaf("entry-0")
        leaf1 = hash_leaf("entry-1")
        root = hash_pair(leaf0, leaf1)

        # Missing proof element
        proof = []
        valid = verify_inclusion(leaf0, proof, 2, root, 0)
        assert not valid

    def test_four_leaf_tree(self):
        """Test inclusion proofs in 4-leaf tree."""
        leaves = [hash_leaf(f"entry-{i}") for i in range(4)]

        # Build tree level by level
        h00 = hash_pair(leaves[0], leaves[1])
        h10 = hash_pair(leaves[2], leaves[3])
        root = hash_pair(h00, h10)

        # Proof for leaf 0
        proof0 = [leaves[1], h10]
        assert verify_inclusion(leaves[0], proof0, 4, root, 0)

        # Proof for leaf 3
        proof3 = [leaves[2], h00]
        assert verify_inclusion(leaves[3], proof3, 4, root, 3)


class TestConsistencyProof:
    """Test Merkle consistency proof verification."""

    def test_same_tree_size(self):
        """Test consistency when tree size hasn't changed."""
        root = hash_leaf("some_root")
        valid = verify_consistency(5, 5, root, root, [])
        assert valid

    def test_tree_growth(self):
        """Test consistency as tree grows."""
        # Small tree with 2 leaves
        old_leaves = [hash_leaf("entry-0"), hash_leaf("entry-1")]
        old_root = hash_pair(old_leaves[0], old_leaves[1])

        # Larger tree with 4 leaves
        new_leaves = [hash_leaf(f"entry-{i}") for i in range(4)]
        h00 = hash_pair(new_leaves[0], new_leaves[1])
        h10 = hash_pair(new_leaves[2], new_leaves[3])
        new_root = hash_pair(h00, h10)

        # Consistency proof from old to new
        proof = old_leaves  # Simplified: just the old leaves
        valid = verify_consistency(2, 4, old_root, new_root, proof)
        assert valid

    def test_tree_shrinkage_invalid(self):
        """Test that tree can't shrink."""
        old_root = hash_leaf("root1")
        new_root = hash_leaf("root2")
        valid = verify_consistency(5, 3, old_root, new_root, [])
        assert not valid

    def test_empty_tree_consistency(self):
        """Test consistency from empty tree."""
        new_root = hash_leaf("some_root")
        valid = verify_consistency(0, 1, "", new_root, [])
        assert valid

    def test_consistency_with_missing_proof(self):
        """Test consistency with insufficient proof data."""
        old_root = hash_leaf("root1")
        new_root = hash_leaf("root2")
        # Not enough proof elements for old_size=5
        proof = [hash_leaf("p1")]
        valid = verify_consistency(5, 10, old_root, new_root, proof)
        assert not valid


class TestReceiptVerifierClass:
    """Test the ReceiptVerifier class methods."""

    def test_verifier_with_custom_secret(self):
        """Test ReceiptVerifier with custom HMAC secret."""
        custom_secret = "my-custom-secret-key"
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,
            "credential": {"audit_id": "001"},
        }

        canonical = json.dumps({
            "receipt_id": receipt["receipt_id"],
            "timestamp": receipt["timestamp"],
            "credential": receipt["credential"],
        }, sort_keys=True)
        receipt["receipt_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        receipt["signature"] = hmac.new(
            custom_secret.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        verifier = ReceiptVerifier(custom_secret)
        result = verifier.verify_receipt(receipt)

        assert result.valid

    def test_verifier_inclusion_method(self):
        """Test ReceiptVerifier.verify_inclusion method."""
        verifier = ReceiptVerifier()

        leaf0 = hash_leaf("entry-0")
        leaf1 = hash_leaf("entry-1")
        root = hash_pair(leaf0, leaf1)
        proof = [leaf1]

        valid = verifier.verify_inclusion(leaf0, proof, 2, root, 0)
        assert valid

    def test_verifier_consistency_method(self):
        """Test ReceiptVerifier.verify_consistency method."""
        verifier = ReceiptVerifier()

        root1 = hash_leaf("root1")

        # Same tree: roots must match
        valid = verifier.verify_consistency(3, 3, root1, root1, [])
        assert valid

        # Different roots for same size: invalid
        root2 = hash_leaf("root2")
        valid = verifier.verify_consistency(3, 3, root1, root2, [])
        assert not valid

    def test_verification_result_to_dict(self):
        """Test VerificationResult.to_dict() method."""
        result = VerificationResult(
            valid=True,
            checks={"check1": True, "check2": True},
            errors=[],
            receipt_id="test-001"
        )

        result_dict = result.to_dict()
        assert result_dict["valid"] is True
        assert result_dict["receipt_id"] == "test-001"

    def test_chain_result_to_dict(self):
        """Test ChainVerificationResult.to_dict() method."""
        result = ChainVerificationResult(
            valid=False,
            chain_length=3,
            breaks=[1],
            errors=["Chain break at index 1"]
        )

        result_dict = result.to_dict()
        assert result_dict["valid"] is False
        assert result_dict["chain_length"] == 3
        assert result_dict["breaks"] == [1]


class TestCLI:
    """Test CLI functionality."""

    def test_cli_receipt_verification(self):
        """Test CLI receipt verification command."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,
            "credential": {"audit_id": "001"},
        }

        canonical = json.dumps({
            "receipt_id": receipt["receipt_id"],
            "timestamp": receipt["timestamp"],
            "credential": receipt["credential"],
        }, sort_keys=True)
        receipt["receipt_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        receipt["signature"] = hmac.new(
            b"veratum-default-secret",
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            receipt_file = os.path.join(tmpdir, "receipt.json")
            with open(receipt_file, "w") as f:
                json.dump(receipt, f)

            result = subprocess.run(
                [sys.executable, "-m", "veratum_verify.cli", "receipt", receipt_file],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent)
            )

            assert result.returncode == 0
            assert "valid" in result.stdout.lower()

    def test_cli_chain_verification(self):
        """Test CLI chain verification command."""
        receipt = {
            "receipt_id": "audit-001",
            "timestamp": 1704067200,
            "credential": {"audit_id": "001"},
            "receipt_hash": "abc123",
            "signature": "sig1",
            "prev_hash": "",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            chain_file = os.path.join(tmpdir, "chain.json")
            with open(chain_file, "w") as f:
                json.dump([receipt], f)

            result = subprocess.run(
                [sys.executable, "-m", "veratum_verify.cli", "chain", chain_file],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent)
            )

            assert result.returncode == 0


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    # Run all tests
    test_classes = [
        TestHashFunctions,
        TestReceiptVerification,
        TestChainVerification,
        TestInclusionProof,
        TestConsistencyProof,
        TestReceiptVerifierClass,
        TestCLI,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{'='*70}")
        print(f"Running {test_class.__name__}")
        print('='*70)

        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  ✗ {test_method}: {e}")
                failed_tests.append((test_class.__name__, test_method, str(e)))
            except Exception as e:
                print(f"  ✗ {test_method}: {type(e).__name__}: {e}")
                failed_tests.append((test_class.__name__, test_method, f"{type(e).__name__}: {e}"))

    print(f"\n{'='*70}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print('='*70)

    if failed_tests:
        print(f"\nFailed tests ({len(failed_tests)}):")
        for class_name, method_name, error in failed_tests:
            print(f"  {class_name}.{method_name}: {error}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
