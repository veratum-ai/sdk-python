"""Merkle tree batch anchoring for efficient proof aggregation.

Implements RFC 6962 (Certificate Transparency) style Merkle trees
for batching multiple audit receipts into a single anchored proof.

Benefits:
- Batch N receipts into a single anchored root hash (cost reduction)
- Each receipt gets an individual inclusion proof (O(log N))
- Verifiable without downloading the entire batch
- Compliant with ISO/IEC 27037:2012 (digital evidence integrity)

Compliance:
- RFC 6962 §2.1 (Merkle Tree Hash computation)
- ISO/IEC 27037:2012 (forensic evidence handling)
- eIDAS Art.41 (qualified timestamp applies to tree root)
- ETSI TS 119 312 (SHA-256 as approved hash algorithm)
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> bytes:
    """SHA-256 hash (returns raw bytes)."""
    return hashlib.sha256(data).digest()


def _leaf_hash(entry: bytes) -> bytes:
    """
    RFC 6962 §2.1 leaf hash: SHA-256(0x00 || entry).

    The 0x00 prefix distinguishes leaf nodes from internal nodes,
    preventing second-preimage attacks.
    """
    return _sha256(b"\x00" + entry)


def _node_hash(left: bytes, right: bytes) -> bytes:
    """
    RFC 6962 §2.1 internal node hash: SHA-256(0x01 || left || right).

    The 0x01 prefix distinguishes internal nodes from leaf nodes.
    """
    return _sha256(b"\x01" + left + right)


# ---------------------------------------------------------------------------
# Merkle tree construction
# ---------------------------------------------------------------------------

class MerkleTree:
    """
    RFC 6962 compliant Merkle tree for batch proof aggregation.

    Usage:
        tree = MerkleTree()
        for receipt in receipts:
            tree.add_entry(receipt["entry_hash"])
        root = tree.compute_root()
        proof = tree.get_inclusion_proof(index)
    """

    def __init__(self) -> None:
        """Initialize empty Merkle tree."""
        self._entries: List[bytes] = []
        self._leaves: List[bytes] = []
        self._root: Optional[bytes] = None
        self._levels: List[List[bytes]] = []

    @property
    def size(self) -> int:
        """Number of entries in the tree."""
        return len(self._entries)

    @property
    def root_hex(self) -> str:
        """Root hash as hex string. Computes if not already computed."""
        if self._root is None:
            self.compute_root()
        return self._root.hex() if self._root else ""

    def add_entry(self, entry_hash: str) -> int:
        """
        Add an entry to the tree.

        Args:
            entry_hash: Hex-encoded hash of the receipt entry.

        Returns:
            Index of the entry in the tree (0-based).
        """
        entry_bytes = bytes.fromhex(entry_hash)
        self._entries.append(entry_bytes)
        self._leaves.append(_leaf_hash(entry_bytes))
        self._root = None  # Invalidate cached root
        self._levels = []
        return len(self._entries) - 1

    def compute_root(self) -> str:
        """
        Compute the Merkle tree root hash.

        Uses RFC 6962 §2.1 algorithm:
        - Empty tree: SHA-256 of empty string
        - Single leaf: leaf hash
        - Multiple: recursive pairing with node hashes

        Returns:
            Hex-encoded root hash.
        """
        if not self._leaves:
            self._root = _sha256(b"")
            self._levels = [[self._root]]
            return self._root.hex()

        if len(self._leaves) == 1:
            self._root = self._leaves[0]
            self._levels = [self._leaves[:]]
            return self._root.hex()

        # Build tree bottom-up
        current_level = self._leaves[:]
        self._levels = [current_level[:]]

        while len(current_level) > 1:
            next_level: List[bytes] = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    next_level.append(_node_hash(current_level[i], current_level[i + 1]))
                else:
                    # Odd node: promote to next level (RFC 6962 convention)
                    next_level.append(current_level[i])
            current_level = next_level
            self._levels.append(current_level[:])

        self._root = current_level[0]
        return self._root.hex()

    def get_inclusion_proof(self, index: int) -> Dict[str, Any]:
        """
        Generate an inclusion proof for the entry at the given index.

        The proof allows verification that a specific entry is included
        in the tree without needing the full dataset.

        Proof verification:
        1. Start with leaf_hash(entry)
        2. For each (hash, direction) in path:
           - If direction == "left":  current = node_hash(hash, current)
           - If direction == "right": current = node_hash(current, hash)
        3. Result should equal root_hash

        Args:
            index: 0-based index of the entry.

        Returns:
            Dict with proof data:
            {
                "index": int,
                "entry_hash": str,
                "leaf_hash": str,
                "root_hash": str,
                "tree_size": int,
                "path": [{"hash": str, "direction": "left"|"right"}, ...]
            }

        Raises:
            IndexError: If index is out of bounds.
            RuntimeError: If tree has not been computed.
        """
        if index < 0 or index >= len(self._entries):
            raise IndexError(f"Index {index} out of range [0, {len(self._entries)})")

        if not self._levels:
            self.compute_root()

        path: List[Dict[str, str]] = []
        idx = index

        for level_idx in range(len(self._levels) - 1):
            level = self._levels[level_idx]
            # Determine sibling
            if idx % 2 == 0:
                # We're the left child — sibling is on the right
                if idx + 1 < len(level):
                    path.append({
                        "hash": level[idx + 1].hex(),
                        "direction": "right",
                    })
                # If no sibling (odd last node), no proof step needed
            else:
                # We're the right child — sibling is on the left
                path.append({
                    "hash": level[idx - 1].hex(),
                    "direction": "left",
                })
            idx //= 2

        return {
            "index": index,
            "entry_hash": self._entries[index].hex(),
            "leaf_hash": self._leaves[index].hex(),
            "root_hash": self.root_hex,
            "tree_size": len(self._entries),
            "path": path,
        }

    def verify_inclusion_proof(self, proof: Dict[str, Any]) -> bool:
        """
        Verify an inclusion proof against this tree's root.

        Args:
            proof: Proof dict from get_inclusion_proof().

        Returns:
            True if the proof is valid.
        """
        return verify_proof(proof)


def verify_proof(proof: Dict[str, Any]) -> bool:
    """
    Standalone verification of a Merkle inclusion proof.

    Can be run independently of the tree that generated it —
    only needs the proof dict and the claimed root hash.

    Args:
        proof: Inclusion proof dict with leaf_hash, root_hash, path.

    Returns:
        True if the proof is valid.
    """
    current = bytes.fromhex(proof["leaf_hash"])
    root_hash = bytes.fromhex(proof["root_hash"])

    for step in proof["path"]:
        sibling = bytes.fromhex(step["hash"])
        if step["direction"] == "right":
            current = _node_hash(current, sibling)
        else:  # direction == "left"
            current = _node_hash(sibling, current)

    return current == root_hash


# ---------------------------------------------------------------------------
# Batch anchoring — ties tree root to external anchors
# ---------------------------------------------------------------------------

class BatchAnchor:
    """
    Batches multiple receipt entry_hashes into a Merkle tree and
    produces an anchor record with a root hash and inclusion proofs.

    Schema 2.3.0: supports both a size-based batch window
    (``max_batch_size``) and a time-based batch window
    (``max_window_seconds``). The batch is considered "due" when *either*
    threshold is reached. This lets low-volume tenants still get frequent
    anchoring without waiting to fill a size-based bucket.

    Usage:
        anchor = BatchAnchor(max_batch_size=256, max_window_seconds=3600)
        for receipt in receipts:
            anchor.add(receipt["entry_hash"])
            if anchor.is_due:
                record = anchor.seal()
                anchor = BatchAnchor(max_batch_size=256, max_window_seconds=3600)
    """

    def __init__(
        self,
        max_batch_size: int = 256,
        max_window_seconds: Optional[float] = None,
    ) -> None:
        """
        Initialize batch anchor.

        Args:
            max_batch_size: Maximum entries per batch (default 256).
            max_window_seconds: Maximum wall-clock age of the batch before
                it becomes due for sealing, even if under max_batch_size.
                ``None`` (default) disables the time window — batches
                only seal when full. Recommended values:
                - 60s for real-time compliance (EU AI Act Art.12)
                - 3600s (1h) for standard batch anchoring
                - 86400s (24h) for cost-optimized low-volume tenants
        """
        import time as _time_mod
        self._time_mod = _time_mod
        self._tree = MerkleTree()
        self._max_batch_size = max_batch_size
        self._max_window_seconds = max_window_seconds
        self._opened_at = _time_mod.monotonic()
        self._sealed = False
        self._entry_hashes: List[str] = []

    @property
    def is_full(self) -> bool:
        """Check if batch has reached max size."""
        return self._tree.size >= self._max_batch_size

    @property
    def age_seconds(self) -> float:
        """Wall-clock age of the batch (since opened), in seconds."""
        return self._time_mod.monotonic() - self._opened_at

    @property
    def is_window_expired(self) -> bool:
        """True if the batch has exceeded its configured time window."""
        if self._max_window_seconds is None:
            return False
        if self._tree.size == 0:
            return False  # never seal an empty batch on a timer
        return self.age_seconds >= self._max_window_seconds

    @property
    def is_due(self) -> bool:
        """True if the batch should be sealed (size OR time window reached)."""
        return self.is_full or self.is_window_expired

    @property
    def size(self) -> int:
        """Current batch size."""
        return self._tree.size

    def add(self, entry_hash: str) -> int:
        """
        Add an entry hash to the batch.

        Args:
            entry_hash: Hex-encoded SHA-256 hash of a receipt.

        Returns:
            Index within the batch.

        Raises:
            RuntimeError: If batch is sealed or full.
        """
        if self._sealed:
            raise RuntimeError("Batch is sealed — create a new BatchAnchor")
        if self.is_full:
            raise RuntimeError(
                f"Batch is full ({self._max_batch_size} entries) — seal and start new batch"
            )
        self._entry_hashes.append(entry_hash)
        return self._tree.add_entry(entry_hash)

    def seal(self) -> Dict[str, Any]:
        """
        Seal the batch and produce the anchor record.

        After sealing, no more entries can be added. The root hash
        can be stored as the batch integrity proof.

        Returns:
            Anchor record:
            {
                "root_hash": str,
                "tree_size": int,
                "entry_hashes": [str, ...],
                "sealed": True,
            }
        """
        self._sealed = True
        root = self._tree.compute_root()
        return {
            "root_hash": root,
            "tree_size": self._tree.size,
            "entry_hashes": self._entry_hashes[:],
            "sealed": True,
            "sealed_reason": "full" if self.is_full else ("window_expired" if self.is_window_expired else "manual"),
            "batch_age_seconds": round(self.age_seconds, 3),
            "max_batch_size": self._max_batch_size,
            "max_window_seconds": self._max_window_seconds,
        }

    def get_proof(self, index: int) -> Dict[str, Any]:
        """
        Get inclusion proof for an entry.

        Args:
            index: 0-based index of the entry.

        Returns:
            Inclusion proof dict.
        """
        if not self._sealed:
            # Compute root if not sealed yet (for preview purposes)
            self._tree.compute_root()
        return self._tree.get_inclusion_proof(index)

    def get_proof_by_hash(self, entry_hash: str) -> Dict[str, Any]:
        """
        Get inclusion proof by entry hash.

        Args:
            entry_hash: The entry hash to find.

        Returns:
            Inclusion proof dict.

        Raises:
            ValueError: If entry hash not found in batch.
        """
        try:
            index = self._entry_hashes.index(entry_hash)
        except ValueError:
            raise ValueError(f"Entry hash {entry_hash[:16]}... not found in batch")
        return self.get_proof(index)

    # ------------------------------------------------------------------
    # External anchoring — Bitcoin (primary) + XRPL (secondary, optional)
    # ------------------------------------------------------------------

    def anchor_root(
        self,
        merkle_root_hex: Optional[str] = None,
        include_xrpl: bool = True,
    ) -> Dict[str, Any]:
        """Anchor the Merkle root to Bitcoin (primary) and XRPL (secondary).

        Two independent public ledgers receive the root hash. If one
        fails, the other still anchors. Receipt creation never blocks
        on either — both are best-effort and return structured error
        dicts rather than raising.

        Args:
            merkle_root_hex: Optional override. Defaults to the tree's
                computed root. Accepts a precomputed root hash so
                callers can re-anchor an existing batch record.
            include_xrpl: If True (default), also attempt XRPL
                anchoring via the optional ``xrpl_anchor`` module. If
                the module isn't available the XRPL result is simply
                marked unavailable — not an error.

        Returns:
            dict with shape::

                {
                  "merkle_root": <hex>,
                  "anchors": {
                    "bitcoin": { ...anchor_hash() result... },
                    "xrpl":    { "anchored": bool, "error": str | None, ... }
                  },
                  "primary_anchor": "bitcoin",
                  "secondary_anchor": "xrpl",
                }
        """
        root_hex = merkle_root_hex or self._tree.root_hex or self._tree.compute_root()

        results: Dict[str, Any] = {}

        # Primary: Bitcoin via OpenTimestamps
        try:
            from .bitcoin_anchor import anchor_hash as bitcoin_anchor_hash

            btc_result = bitcoin_anchor_hash(root_hex)
            results["bitcoin"] = btc_result
            if btc_result.get("anchored"):
                logger.info(
                    "Bitcoin anchor submitted (%d calendars). Confirmation in ~10 min.",
                    len(btc_result.get("calendars_succeeded") or []),
                )
            else:
                logger.warning(
                    "Bitcoin anchor did not succeed: %s",
                    btc_result.get("error"),
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Bitcoin anchor crashed unexpectedly: %s", exc)
            results["bitcoin"] = {
                "anchored": False,
                "error": f"unexpected exception: {type(exc).__name__}: {exc}",
            }

        # Secondary: XRPL — optional, only if the module exists.
        if include_xrpl:
            try:
                from . import xrpl_anchor  # type: ignore

                anchor_fn = getattr(xrpl_anchor, "anchor_root_hash", None)
                if anchor_fn is None:
                    results["xrpl"] = {
                        "anchored": False,
                        "error": "xrpl_anchor.anchor_root_hash not found",
                    }
                else:
                    results["xrpl"] = anchor_fn(root_hex)
            except ImportError:
                results["xrpl"] = {
                    "anchored": False,
                    "error": "xrpl_anchor module not installed (optional dependency)",
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("XRPL anchor failed (non-critical): %s", exc)
                results["xrpl"] = {
                    "anchored": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        return {
            "merkle_root": root_hex,
            "anchors": results,
            "primary_anchor": "bitcoin",
            "secondary_anchor": "xrpl" if include_xrpl else None,
        }
