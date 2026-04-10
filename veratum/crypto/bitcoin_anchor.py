"""
Bitcoin anchoring via the OpenTimestamps protocol.

Public calendar servers, no authentication required, no payment, no
accounts. Free open infrastructure.

Two-stage process
-----------------
  Stage 1: Submit a SHA-256 hash to one or more OpenTimestamps calendars
           → receive an immediate ``.ots`` proof (milliseconds).
  Stage 2: Bitcoin confirms the calendar's aggregation transaction
           (~10 minutes, 1 block) → the ``.ots`` proof is *upgraded*
           to include the Bitcoin block header attestation.

The Stage 1 proof is sufficient for compliance logging and receipt
creation. The Stage 2 upgrade produces a proof verifiable by any
Bitcoin node, block explorer, or the reference OpenTimestamps client,
forever, without any dependency on Veratum's infrastructure.

Public calendars used:
  * ``https://alice.btc.calendar.opentimestamps.org``
  * ``https://bob.btc.calendar.opentimestamps.org``
  * ``https://finney.calendar.eternitywall.com``

Contract:
  * ``anchor_hash`` and ``upgrade_proof`` NEVER raise on failure — a
    total calendar outage returns a structured error dict so receipt
    creation continues unimpeded.
  * Both functions are safe to call from inside a Lambda handler or
    from any synchronous code path.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public OpenTimestamps calendars — no auth, no payment, no API keys.
# Order matters: alice is the primary, bob is the hot standby, and
# finney (Eternity Wall) is the independent third operator.
# ---------------------------------------------------------------------------
_CALENDAR_URLS: List[str] = [
    "https://alice.btc.calendar.opentimestamps.org",
    "https://bob.btc.calendar.opentimestamps.org",
    "https://finney.calendar.eternitywall.com",
]


def _calendar_urls() -> List[str]:
    """Return a copy of the public calendar URL list.

    Kept as a function so tests can monkey-patch a single call site
    rather than reaching into a module-level list.
    """
    return list(_CALENDAR_URLS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def anchor_hash(data_hex: str) -> Dict[str, Any]:
    """Submit a SHA-256 hash to public OpenTimestamps calendars.

    Args:
        data_hex: 64-character hex string — the SHA-256 digest of the
            content being timestamped (typically a Merkle tree root).

    Returns:
        A dict with the following keys, ALWAYS populated:

        ``anchored``
            Bool — True iff at least one calendar accepted the submission.
        ``ots_proof_base64``
            Base64-encoded ``.ots`` file bytes, or None on failure.
        ``calendars_contacted``
            List of all calendar URLs the client attempted to reach.
        ``calendars_succeeded``
            List of calendar URLs that accepted the submission.
        ``calendars_failed``
            List of calendar URLs that failed (network error, HTTP error, …).
        ``submitted_at``
            ISO 8601 UTC timestamp string.
        ``bitcoin_confirmed``
            Always False at Stage 1. Becomes True after ``upgrade_proof``
            runs and Bitcoin has confirmed the calendar's aggregation tx.
        ``bitcoin_block_height``
            None at Stage 1. Populated by ``upgrade_proof`` after
            Bitcoin confirmation.
        ``bitcoin_txid``
            Reserved for forward compat. The OpenTimestamps block header
            attestation embeds the block height, not the txid — callers
            can resolve the txid externally via a block explorer.
        ``error``
            None on success, a human-readable string on failure.

    Raises:
        Never. All failures are reported via the ``error`` field.
    """
    # Validate hex input shape — this *can* raise ValueError, which is
    # caught by the outer try/except and reported via the error field.
    try:
        data_bytes = bytes.fromhex(data_hex)
    except (ValueError, TypeError) as exc:
        return _error_result(
            f"invalid hex input: {type(exc).__name__}: {exc}",
        )
    if len(data_bytes) != 32:
        return _error_result(
            f"expected 32-byte SHA-256 hash, got {len(data_bytes)} bytes",
        )

    try:
        from opentimestamps.core.timestamp import Timestamp  # type: ignore
        # RemoteCalendar has lived in multiple locations across OTS client
        # versions (0.5 → `opentimestamps.calendar`, 0.7+ →
        # `opentimestamps.client.calendar`). Try both.
        try:
            from opentimestamps.client.calendar import RemoteCalendar  # type: ignore
        except ImportError:  # pragma: no cover - old client layout
            from opentimestamps.calendar import RemoteCalendar  # type: ignore
        import opentimestamps.core.serialize as ots_serialize  # type: ignore
        import io
    except ImportError:
        logger.error(
            "opentimestamps-client not installed. Add to requirements.txt "
            "or pyproject.toml and rebuild the Lambda layer."
        )
        return _error_result("opentimestamps-client not installed")

    timestamp = Timestamp(data_bytes)

    calendars = _calendar_urls()
    succeeded: List[str] = []
    failed: List[str] = []

    for cal_url in calendars:
        try:
            cal = RemoteCalendar(cal_url)
            cal.submit(timestamp)
            succeeded.append(cal_url)
            logger.info("OpenTimestamps: submitted to %s", cal_url)
        except Exception as exc:  # noqa: BLE001 - we want ANY failure contained
            failed.append(cal_url)
            logger.warning(
                "OpenTimestamps calendar failed (%s): %s", cal_url, exc
            )

    if not succeeded:
        return _error_result(
            f"All {len(calendars)} calendars unreachable",
            calendars_contacted=calendars,
            calendars_failed=failed,
        )

    # Serialize the .ots proof to bytes for storage in S3.
    ots_b64: Optional[str]
    try:
        buf = io.BytesIO()
        ctx = ots_serialize.StreamSerializationContext(buf)
        timestamp.serialize(ctx)
        ots_bytes = buf.getvalue()
        ots_b64 = base64.b64encode(ots_bytes).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to serialize .ots proof: %s", exc)
        ots_b64 = None

    return {
        "anchored": True,
        "ots_proof_base64": ots_b64,
        "calendars_contacted": calendars,
        "calendars_succeeded": succeeded,
        "calendars_failed": failed,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "bitcoin_confirmed": False,
        "bitcoin_block_height": None,
        "bitcoin_txid": None,
        "error": None,
    }


def upgrade_proof(ots_proof_base64: str) -> Dict[str, Any]:
    """Attempt to upgrade an existing ``.ots`` proof to include Bitcoin confirmation.

    Intended to be called from a background job roughly 10-20 minutes
    after ``anchor_hash``. Once Bitcoin confirms the aggregation
    transaction (typically after 1 block), the calendar exposes the
    completed path and the proof can be upgraded to embed the
    ``BitcoinBlockHeaderAttestation``.

    Args:
        ots_proof_base64: Base64-encoded ``.ots`` file bytes returned
            by ``anchor_hash``.

    Returns:
        dict with keys:

        ``upgraded``
            Bool — True iff the proof was successfully upgraded.
        ``ots_proof_base64``
            The updated proof bytes (base64), or the original if the
            upgrade path isn't available yet.
        ``bitcoin_confirmed``
            True iff a ``BitcoinBlockHeaderAttestation`` is now present.
        ``bitcoin_block_height``
            Bitcoin block height, or None if not yet confirmed.
        ``bitcoin_txid``
            Reserved — the block header attestation does not include
            the tx hash directly.
        ``error``
            None on success, string describing why upgrade failed.

    Raises:
        Never.
    """
    try:
        from opentimestamps.core.timestamp import Timestamp  # type: ignore
        try:
            from opentimestamps.client.calendar import RemoteCalendar  # type: ignore
        except ImportError:  # pragma: no cover
            from opentimestamps.calendar import RemoteCalendar  # type: ignore
        from opentimestamps.core.notary import (  # type: ignore
            BitcoinBlockHeaderAttestation,
        )
        import opentimestamps.core.serialize as ots_serialize  # type: ignore
        import io
    except ImportError:
        return {
            "upgraded": False,
            "ots_proof_base64": ots_proof_base64,
            "bitcoin_confirmed": False,
            "bitcoin_block_height": None,
            "bitcoin_txid": None,
            "error": "opentimestamps-client not installed",
        }

    # Decode & parse the incoming proof.
    try:
        ots_bytes = base64.b64decode(ots_proof_base64)
        buf = io.BytesIO(ots_bytes)
        ctx = ots_serialize.StreamDeserializationContext(buf)
        # Timestamp.deserialize expects (ctx, msg); for a free-standing
        # timestamp the root message is what we seeded with. Callers of
        # upgrade_proof start from the output of anchor_hash(), so the
        # top-level timestamp starts with the original SHA-256 bytes
        # we handed in. The OTS library exposes a higher-level
        # DetachedTimestampFile helper when we want to parse a bare
        # .ots file from disk, but we want to round-trip the in-memory
        # Timestamp object here.
        try:
            from opentimestamps.core.timestamp import DetachedTimestampFile  # type: ignore
            buf.seek(0)
            ctx = ots_serialize.StreamDeserializationContext(buf)
            detached = DetachedTimestampFile.deserialize(ctx)
            timestamp = detached.timestamp
        except Exception:
            # Fall back to plain Timestamp round-trip (used if the caller
            # serialized only the bare Timestamp, not the detached wrapper).
            buf.seek(0)
            ctx = ots_serialize.StreamDeserializationContext(buf)
            timestamp = Timestamp.deserialize(ctx, b"\x00" * 32)
            detached = None  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        return {
            "upgraded": False,
            "ots_proof_base64": ots_proof_base64,
            "bitcoin_confirmed": False,
            "bitcoin_block_height": None,
            "bitcoin_txid": None,
            "error": f"failed to deserialize .ots proof: {exc}",
        }

    upgraded = False
    for cal_url in _calendar_urls():
        try:
            cal = RemoteCalendar(cal_url)
            # Older clients call it ``upgrade_timestamp``; newer call
            # it ``upgrade``. Try both.
            upgrade_fn = getattr(cal, "upgrade_timestamp", None) or getattr(
                cal, "upgrade", None
            )
            if upgrade_fn is None:
                continue
            upgrade_fn(timestamp)
            upgraded = True
        except Exception:
            # A calendar that doesn't yet have the Bitcoin path is not
            # an error — just try the next one.
            continue

    # Check if Bitcoin attestation is now present in the timestamp tree.
    bitcoin_confirmed = False
    block_height: Optional[int] = None
    txid: Optional[str] = None

    try:
        for msg, attestation in timestamp.all_attestations():
            if isinstance(attestation, BitcoinBlockHeaderAttestation):
                bitcoin_confirmed = True
                block_height = attestation.height
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not check Bitcoin attestation: %s", exc)

    # Re-serialize the (possibly updated) proof for storage.
    try:
        buf = io.BytesIO()
        ctx = ots_serialize.StreamSerializationContext(buf)
        if detached is not None:  # type: ignore[has-type]
            detached.serialize(ctx)  # type: ignore[has-type]
        else:
            timestamp.serialize(ctx)
        updated_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        updated_b64 = ots_proof_base64  # fall back to the input unchanged

    return {
        "upgraded": upgraded,
        "ots_proof_base64": updated_b64,
        "bitcoin_confirmed": bitcoin_confirmed,
        "bitcoin_block_height": block_height,
        "bitcoin_txid": txid,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(
    error: str,
    calendars_contacted: Optional[List[str]] = None,
    calendars_failed: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a uniform error-shape dict matching ``anchor_hash`` contract."""
    return {
        "anchored": False,
        "ots_proof_base64": None,
        "calendars_contacted": calendars_contacted or _calendar_urls(),
        "calendars_succeeded": [],
        "calendars_failed": calendars_failed or [],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "bitcoin_confirmed": False,
        "bitcoin_block_height": None,
        "bitcoin_txid": None,
        "error": error,
    }


__all__ = ["anchor_hash", "upgrade_proof"]
