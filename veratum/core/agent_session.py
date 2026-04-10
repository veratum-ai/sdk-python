"""Agent session context manager (Schema 2.3.0).

An AgentSession ties a series of AI decisions together with:
  - a single session_id (UUIDv7) so the dashboard can render an
    end-to-end timeline,
  - an authorization_envelope_id so every receipt declares the
    authority under which it was produced,
  - a parent_receipt_id pointer that lets agents-calling-agents
    form a directed acyclic graph rather than a flat list,
  - a SessionMerkleTree (closed when the session ends) that produces
    a single root hash committing to every decision the agent made
    in that session.

Usage:

    from veratum.core.agent_session import AgentSession

    with AgentSession(
        veratum,
        agent_id="ops-bot-1",
        envelope_id="018e...",
    ) as session:
        receipt_a = session.record_decision(prompt=..., response=...)
        receipt_b = session.record_decision(
            prompt=..., response=..., parent=receipt_a
        )
    # On exit: session.close() is called, computing the Merkle root
    # over [receipt_a, receipt_b, ...] and emitting a final
    # session_close receipt that anchors them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .receipt import generate_uuidv7

logger = logging.getLogger(__name__)


class SessionMerkleTree:
    """Builds an RFC 6962 Merkle tree from a session's receipts.

    Wraps the SDK's existing veratum.crypto.merkle.MerkleTree.
    """

    def __init__(self) -> None:
        from ..crypto.merkle import MerkleTree

        self._tree = MerkleTree()

    def add(self, entry_hash_hex: str) -> None:
        try:
            self._tree.add_entry(entry_hash_hex)
        except ValueError:
            logger.warning("SessionMerkleTree skipping non-hex leaf %s", entry_hash_hex)

    def root(self) -> str:
        """Return the hex-encoded RFC 6962 Merkle root, or empty string."""
        if self._tree.size == 0:
            return ""
        try:
            return self._tree.compute_root()
        except Exception as exc:
            logger.error("Merkle root computation failed: %s", exc)
            return ""

    def __len__(self) -> int:
        return self._tree.size


class AgentSession:
    """Context manager for grouped agent decisions with cross-receipt linkage.

    Construct with a Veratum SDK instance (anything that exposes a
    `record_decision(...)` method that returns a receipt dict). The
    session decorates each call by injecting:
      - session_id
      - authorization_envelope_id
      - parent_receipt_id
      - capture_method="sdk"

    On close it computes the Merkle root over all session receipts
    and (best-effort) records a final ``session_close`` receipt with
    the root in metadata.
    """

    def __init__(
        self,
        sdk: Any,
        agent_id: str,
        envelope_id: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> None:
        self.sdk = sdk
        self.agent_id = agent_id
        self.envelope_id = envelope_id
        self.purpose = purpose
        self.session_id = generate_uuidv7()
        self.tree = SessionMerkleTree()
        self.receipts: List[Dict[str, Any]] = []
        self._closed = False

    def __enter__(self) -> "AgentSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            self.close()

    def record_decision(
        self,
        prompt: Any,
        response: Any,
        parent: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Record a decision under this session and link it to a parent."""
        if self._closed:
            raise RuntimeError("Cannot record decisions on a closed session")

        metadata = dict(kwargs.pop("metadata", None) or {})
        metadata.setdefault("agent_id", self.agent_id)
        if self.purpose:
            metadata.setdefault("purpose", self.purpose)

        kwargs.update(
            {
                "metadata": metadata,
                "session_id": self.session_id,
                "authorization_envelope_id": self.envelope_id,
                "parent_receipt_id": (parent or {}).get("receipt_id"),
                "capture_method": "sdk",
            }
        )

        # The underlying SDK's record_decision is the canonical entry
        # point. We don't enforce that all of these kwargs are accepted
        # — only the ones that map to receipt fields will land.
        try:
            receipt = self.sdk.record_decision(prompt=prompt, response=response, **kwargs)
        except TypeError:
            # SDK shim that doesn't accept extra kwargs — fall back
            # gracefully and inject the session fields post-hoc.
            receipt = self.sdk.record_decision(prompt=prompt, response=response)
            for key in (
                "session_id",
                "authorization_envelope_id",
                "parent_receipt_id",
                "capture_method",
            ):
                if kwargs.get(key) is not None:
                    receipt[key] = kwargs[key]

        if isinstance(receipt, dict):
            self.receipts.append(receipt)
            entry_hash = receipt.get("entry_hash")
            if entry_hash:
                self.tree.add(entry_hash)
        return receipt

    def close(self) -> Dict[str, Any]:
        """Finalize the session and emit a session_close summary receipt.

        Returns the summary dict (not necessarily a full receipt — falls
        back to a plain dict if the SDK can't accept synthetic decisions).
        """
        self._closed = True
        merkle_root = self.tree.root()
        summary: Dict[str, Any] = {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "envelope_id": self.envelope_id,
            "purpose": self.purpose,
            "decisions": len(self.receipts),
            "merkle_root": merkle_root,
        }
        try:
            close_receipt = self.sdk.record_decision(
                prompt=f"<session_close:{self.session_id}>",
                response="",
                decision_type="session_close",
                metadata={
                    "agent_id": self.agent_id,
                    "session_id": self.session_id,
                    "session_merkle_root": merkle_root,
                    "session_decisions_count": len(self.receipts),
                },
            )
            if isinstance(close_receipt, dict):
                summary["close_receipt_id"] = close_receipt.get("receipt_id")
        except Exception as exc:  # pragma: no cover
            logger.warning("session_close receipt emission failed: %s", exc)
        return summary
