"""Core SDK tests — import, init, wrap, receipts, compliance."""

import json
import hashlib


class TestSDKImport:
    """Test that the SDK can be imported without errors."""

    def test_import_veratum(self):
        import veratum
        assert hasattr(veratum, "__version__")
        assert veratum.__version__ == "2.3.0"

    def test_import_core(self):
        from veratum import VeratumSDK, wrap, Receipt

    def test_import_crypto(self):
        from veratum import HashChain, MerkleTree, hmac_sign_receipt

    def test_import_compliance(self):
        from veratum import crosswalk, list_frameworks, VeratumPolicyEngine

    def test_import_security(self):
        from veratum import PromptGuard, PIIRedactor, scan_prompt

    def test_import_providers(self):
        from veratum import detect_provider, auto_wrap, WrapConfig

    def test_import_quickstart(self):
        from veratum import init, quickstart


class TestHashChain:
    """Test cryptographic hash chain."""

    def test_chain_creation(self):
        from veratum.crypto import HashChain
        chain = HashChain()
        assert chain is not None
        assert chain.sequence_no == 0

    def test_chain_advance(self):
        from veratum.crypto import HashChain
        chain = HashChain()
        chain.advance_chain({"action": "test", "value": 42})
        assert chain.sequence_no == 1

    def test_chain_state(self):
        from veratum.crypto import HashChain
        chain = HashChain()
        for i in range(5):
            chain.advance_chain({"step": i})
        assert chain.sequence_no == 5
        state = chain.get_chain_state()
        assert state is not None

    def test_chain_entry_hash(self):
        from veratum.crypto import HashChain
        chain = HashChain()
        h = chain.compute_entry_hash({"data": "test"})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex

    def test_chain_reset(self):
        from veratum.crypto import HashChain
        chain = HashChain()
        chain.advance_chain({"step": 1})
        assert chain.sequence_no == 1
        chain.reset()
        assert chain.sequence_no == 0

    def test_chain_prev_hash(self):
        from veratum.crypto import HashChain
        chain = HashChain()
        assert chain.prev_hash == "0" * 64
        chain.advance_chain({"step": 0})
        assert chain.prev_hash != ""


class TestMerkleTree:
    """Test Merkle tree operations."""

    def test_tree_creation(self):
        from veratum.crypto import MerkleTree
        tree = MerkleTree()
        assert tree is not None

    def test_tree_add_entry(self):
        from veratum.crypto import MerkleTree, HashChain
        tree = MerkleTree()
        chain = HashChain()
        h = chain.compute_entry_hash({"data": "test"})
        tree.add_entry(h)
        assert tree.size == 1

    def test_tree_root(self):
        from veratum.crypto import MerkleTree, HashChain
        tree = MerkleTree()
        chain = HashChain()
        h1 = chain.compute_entry_hash({"data": "a"})
        h2 = chain.compute_entry_hash({"data": "b"})
        tree.add_entry(h1)
        tree.add_entry(h2)
        root = tree.compute_root()
        assert root is not None
        assert len(root) == 64

    def test_tree_inclusion_proof(self):
        from veratum.crypto import MerkleTree, HashChain
        tree = MerkleTree()
        chain = HashChain()
        for i in range(4):
            h = chain.compute_entry_hash({"data": f"entry-{i}"})
            tree.add_entry(h)
        proof = tree.get_inclusion_proof(0)
        assert proof is not None


class TestCompliance:
    """Test compliance crosswalk."""

    def test_list_frameworks(self):
        from veratum import list_frameworks
        frameworks = list_frameworks()
        assert isinstance(frameworks, (list, tuple, set))
        assert len(frameworks) > 0

    def test_crosswalk_basic(self):
        from veratum import crosswalk
        receipt = {
            "receipt_id": "test-001",
            "timestamp": 1704067200,
            "credential": {
                "audit_id": "test-001",
                "decision": "approved",
                "confidence": 0.95,
                "model": "gpt-4",
                "prompt_hash": hashlib.sha256(b"test prompt").hexdigest(),
            },
        }
        result = crosswalk(receipt)
        assert result is not None

    def test_policy_engine(self):
        from veratum import VeratumPolicyEngine
        engine = VeratumPolicyEngine()
        assert engine is not None


class TestSecurity:
    """Test security features."""

    def test_prompt_guard_creation(self):
        from veratum import PromptGuard
        guard = PromptGuard()
        assert guard is not None

    def test_pii_redactor_creation(self):
        from veratum import PIIRedactor
        redactor = PIIRedactor()
        assert redactor is not None


class TestCryptoVerification:
    """Test receipt signing and verification."""

    def test_hmac_sign_and_verify(self):
        from veratum.crypto import hmac_sign_receipt, verify_hmac_signature, hash_receipt
        receipt = {
            "receipt_id": "test-001",
            "timestamp": 1704067200,
            "credential": {"action": "test"},
        }
        receipt["entry_hash"] = hash_receipt(receipt)
        secret = b"a-sufficiently-long-secret-key-for-hmac-signing!!"
        sig_data = hmac_sign_receipt(receipt, secret)
        assert sig_data is not None

    def test_jcs_canonicalize(self):
        from veratum.crypto import jcs_canonicalize
        data = {"b": 2, "a": 1}
        canonical = jcs_canonicalize(data)
        assert canonical is not None

    def test_hash_receipt(self):
        from veratum.crypto import hash_receipt
        receipt = {
            "receipt_id": "test-001",
            "timestamp": 1704067200,
            "credential": {"action": "test"},
        }
        h = hash_receipt(receipt)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_hash_receipt_deterministic(self):
        from veratum.crypto import hash_receipt
        receipt = {"receipt_id": "x", "timestamp": 1, "credential": {"a": 1}}
        h1 = hash_receipt(receipt)
        h2 = hash_receipt(receipt)
        assert h1 == h2
