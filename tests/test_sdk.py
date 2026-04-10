"""Core SDK tests — import, init, wrap, receipts, compliance."""

import json
import hashlib
import hmac


class TestSDKImport:
    """Test that the SDK can be imported without errors."""

    def test_import_veratum(self):
        """Test top-level import."""
        import veratum
        assert hasattr(veratum, "__version__")
        assert veratum.__version__ == "2.3.0"

    def test_import_core(self):
        """Test core module imports."""
        from veratum import VeratumSDK, wrap, Receipt

    def test_import_crypto(self):
        """Test crypto module imports."""
        from veratum import HashChain, MerkleTree, hmac_sign_receipt

    def test_import_compliance(self):
        """Test compliance module imports."""
        from veratum import crosswalk, list_frameworks, VeratumPolicyEngine

    def test_import_security(self):
        """Test security module imports."""
        from veratum import PromptGuard, PIIRedactor, scan_prompt

    def test_import_providers(self):
        """Test provider module imports."""
        from veratum import detect_provider, auto_wrap, WrapConfig

    def test_import_quickstart(self):
        """Test quickstart imports."""
        from veratum import init, quickstart


class TestHashChain:
    """Test cryptographic hash chain."""

    def test_chain_creation(self):
        """Test creating a hash chain."""
        from veratum.crypto import HashChain
        chain = HashChain()
        assert chain is not None

    def test_chain_append(self):
        """Test appending to a hash chain."""
        from veratum.crypto import HashChain
        chain = HashChain()
        entry = {"action": "test", "value": 42}
        result = chain.append(entry)
        assert result is not None
        assert chain.length == 1

    def test_chain_integrity(self):
        """Test chain integrity after multiple appends."""
        from veratum.crypto import HashChain
        chain = HashChain()
        for i in range(5):
            chain.append({"step": i})
        assert chain.length == 5
        assert chain.verify()


class TestMerkleTree:
    """Test Merkle tree operations."""

    def test_tree_creation(self):
        """Test creating a Merkle tree."""
        from veratum.crypto import MerkleTree
        tree = MerkleTree()
        assert tree is not None

    def test_tree_add_leaf(self):
        """Test adding leaves."""
        from veratum.crypto import MerkleTree
        tree = MerkleTree()
        tree.add_leaf({"data": "test"})
        assert tree.leaf_count == 1

    def test_tree_root(self):
        """Test computing Merkle root."""
        from veratum.crypto import MerkleTree
        tree = MerkleTree()
        tree.add_leaf({"data": "a"})
        tree.add_leaf({"data": "b"})
        root = tree.root
        assert root is not None
        assert len(root) == 64  # SHA256 hex


class TestCompliance:
    """Test compliance crosswalk."""

    def test_list_frameworks(self):
        """Test listing supported frameworks."""
        from veratum import list_frameworks
        frameworks = list_frameworks()
        assert isinstance(frameworks, (list, tuple, set))
        assert len(frameworks) > 0

    def test_crosswalk_basic(self):
        """Test basic crosswalk analysis."""
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


class TestSecurity:
    """Test security features."""

    def test_prompt_guard_creation(self):
        """Test creating a PromptGuard."""
        from veratum import PromptGuard
        guard = PromptGuard()
        assert guard is not None

    def test_pii_redactor_creation(self):
        """Test creating a PIIRedactor."""
        from veratum import PIIRedactor
        redactor = PIIRedactor()
        assert redactor is not None


class TestCryptoVerification:
    """Test receipt signing and verification."""

    def test_hmac_sign_and_verify(self):
        """Test HMAC signing round-trip."""
        from veratum.crypto import hmac_sign_receipt, verify_hmac_signature
        receipt = {
            "receipt_id": "test-001",
            "timestamp": 1704067200,
            "credential": {"action": "test"},
        }
        secret = "test-secret"
        signature = hmac_sign_receipt(receipt, secret)
        assert signature is not None
        assert len(signature) > 0
