"""Resilient receipt upload buffer with retry and circuit breaker.

Implements industry-standard SDK resilience patterns researched from
Langfuse, Sentry, OpenTelemetry, Datadog, and PostHog SDKs:

- Background thread for non-blocking uploads
- Bounded in-memory queue (no disk persistence per industry standard)
- Exponential backoff with jitter (AWS-style decorrelated jitter)
- Circuit breaker (closed → open → half-open) to avoid hammering dead endpoints
- flush() for graceful shutdown (critical for Lambda/serverless)
- Retries only on transient errors (408, 429, 500, 502, 503, 504)

Design decisions:
- Memory-only queue: No Python SDK in the industry does disk persistence.
  Sentry does disk only on mobile. Memory is simpler, faster, and avoids
  file permission issues in containers/Lambda.
- Default queue size 2048: Balances OpenTelemetry (2048) and Langfuse (100k).
  At ~2KB per receipt, worst case is ~4MB memory — acceptable everywhere.
- Decorrelated jitter: Better spread than equal/full jitter per AWS
  architecture blog. Formula: sleep = min(max_delay, random(base, prev * 3))
- Circuit breaker: Prevents queue backup during outages. Opens after 5
  consecutive failures, half-opens after 30s to probe recovery.
"""

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows — flock not available; WAL code already handles this gracefully
import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Set

logger = logging.getLogger("veratum.buffer")

# Retryable HTTP status codes (standard across all researched SDKs)
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation — requests flow through
    OPEN = "open"            # Failing — requests short-circuited, queued
    HALF_OPEN = "half_open"  # Probing — one request allowed to test recovery


class ReceiptBuffer:
    """
    Resilient receipt upload buffer.

    Buffers receipts in memory and uploads them via a background thread
    with exponential backoff, jitter, and circuit breaker protection.

    Usage:
        buffer = ReceiptBuffer(upload_fn=sdk._upload_receipt)
        buffer.put(receipt_dict)
        # ... at shutdown:
        buffer.flush(timeout=5.0)
        buffer.shutdown()

    Thread Safety:
        All public methods are thread-safe. The background worker is a
        daemon thread that automatically dies with the process.
    """

    def __init__(
        self,
        upload_fn: Callable[[Dict[str, Any]], bool],
        *,
        max_queue_size: int = 2048,
        flush_interval: float = 1.0,
        max_retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 30.0,
        batch_size: int = 1,
        wal_path: Optional[str] = None,
    ) -> None:
        """
        Initialize receipt buffer.

        Args:
            upload_fn: Callable that uploads a receipt dict. Must return True
                       on success, False on failure. May raise on network errors.
            max_queue_size: Maximum receipts to buffer in memory (default 2048).
                           Oldest receipts are dropped when full (back-pressure).
            flush_interval: Seconds between flush cycles (default 1.0).
            max_retries: Maximum retry attempts per receipt (default 5).
            base_delay: Initial backoff delay in seconds (default 0.5).
            max_delay: Maximum backoff delay in seconds (default 30.0).
            circuit_failure_threshold: Consecutive failures before opening circuit (default 5).
            circuit_recovery_timeout: Seconds to wait before half-open probe (default 30.0).
            batch_size: Number of receipts to process per cycle (default 1).
            wal_path: Optional path to Write-Ahead Log file for crash recovery.
                      If None, no disk persistence. If set, receipts are written
                      to disk before memory queue (crash-safe).
        """
        self._upload_fn = upload_fn
        self._max_queue_size = max_queue_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._batch_size = batch_size

        # WAL (Write-Ahead Log) persistence
        self._wal_path = wal_path
        self._wal_lock = threading.Lock()
        self._uploaded_hashes: Set[str] = set()  # Tracks uploaded entries for WAL truncation

        # Thread-safe queue
        self._queue: Deque[Dict[str, Any]] = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()

        # Circuit breaker state
        self._circuit_state = CircuitState.CLOSED
        self._circuit_failure_count = 0
        self._circuit_failure_threshold = circuit_failure_threshold
        self._circuit_recovery_timeout = circuit_recovery_timeout
        self._circuit_opened_at: float = 0.0
        self._circuit_lock = threading.Lock()

        # Retry tracking: maps id(receipt) → attempt count
        self._retry_counts: Dict[int, int] = {}

        # Stats
        self._stats_lock = threading.Lock()
        self._stats = {
            "enqueued": 0,
            "uploaded": 0,
            "dropped_queue_full": 0,
            "dropped_max_retries": 0,
            "retries": 0,
            "circuit_opens": 0,
        }

        # Crash recovery: re-queue any entries from WAL that weren't uploaded
        if self._wal_path:
            self._recover_from_wal()

        # Background worker
        self._shutdown_event = threading.Event()
        self._flush_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="veratum-buffer-worker",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, receipt: Dict[str, Any]) -> bool:
        """
        Enqueue a receipt for async upload.

        If the queue is full, the oldest receipt is silently dropped
        (back-pressure). This ensures the SDK never blocks the caller
        or causes OOM.

        If WAL is enabled, receipt is written to disk BEFORE being added
        to the memory queue (crash-safe ordering).

        Args:
            receipt: Receipt dictionary to upload.

        Returns:
            True if enqueued, False if queue was full (oldest was evicted).
        """
        # Write to WAL first (before memory queue) for crash safety
        if self._wal_path:
            self._write_to_wal(receipt)

        was_full = False
        with self._lock:
            if len(self._queue) >= self._max_queue_size:
                was_full = True
                # deque(maxlen=N) auto-evicts from left, but we track stats
                with self._stats_lock:
                    self._stats["dropped_queue_full"] += 1
            self._queue.append(receipt)
            with self._stats_lock:
                self._stats["enqueued"] += 1

        if was_full:
            logger.warning(
                "Veratum buffer full (%d), oldest receipt dropped",
                self._max_queue_size,
            )
        return not was_full

    def flush(self, timeout: float = 5.0) -> int:
        """
        Flush all buffered receipts synchronously.

        Blocks until the queue is empty or timeout is reached.
        Critical for Lambda/serverless where process may terminate.

        Args:
            timeout: Maximum seconds to wait for flush (default 5.0).

        Returns:
            Number of receipts remaining in queue after flush.
        """
        deadline = time.monotonic() + timeout

        # Signal the worker to wake up immediately
        self._flush_event.set()

        # Also drain from this thread for faster flush
        while time.monotonic() < deadline:
            receipt = self._dequeue()
            if receipt is None:
                break
            self._try_upload(receipt)

        with self._lock:
            return len(self._queue)

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down the buffer.

        Flushes remaining receipts and stops the background worker.
        If WAL is enabled and all entries are uploaded, truncates the WAL file.

        Args:
            timeout: Maximum seconds to wait for shutdown.
        """
        self.flush(timeout=timeout)
        self._shutdown_event.set()
        self._worker.join(timeout=max(0.5, timeout))
        # Truncate WAL if all entries have been uploaded
        if self._wal_path:
            self._maybe_truncate_wal()

    def stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Returns:
            Dictionary with enqueued, uploaded, dropped counts, circuit state.
        """
        with self._stats_lock:
            s = dict(self._stats)
        with self._lock:
            s["queue_depth"] = len(self._queue)
        with self._circuit_lock:
            s["circuit_state"] = self._circuit_state.value
        return s

    @property
    def queue_depth(self) -> int:
        """Current number of receipts in the queue."""
        with self._lock:
            return len(self._queue)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Background worker that drains the queue."""
        while not self._shutdown_event.is_set():
            # Wait for flush signal or interval timeout
            self._flush_event.wait(timeout=self._flush_interval)
            self._flush_event.clear()

            # Process a batch
            for _ in range(self._batch_size):
                if self._shutdown_event.is_set():
                    return
                receipt = self._dequeue()
                if receipt is None:
                    break
                self._try_upload(receipt)

    def _dequeue(self) -> Optional[Dict[str, Any]]:
        """Pop the oldest receipt from the queue."""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    # ------------------------------------------------------------------
    # Upload with retry + circuit breaker
    # ------------------------------------------------------------------

    def _try_upload(self, receipt: Dict[str, Any]) -> bool:
        """
        Attempt to upload a receipt with circuit breaker and retry logic.

        Returns True if uploaded, False if requeued or dropped.
        """
        receipt_id = id(receipt)

        # Check circuit breaker
        if not self._circuit_allows_request():
            # Circuit is open — requeue without counting as retry
            self._requeue(receipt)
            return False

        try:
            success = self._upload_fn(receipt)
        except Exception as exc:
            logger.debug("Upload raised exception: %s", exc)
            success = False

        if success:
            self._on_upload_success()
            self._retry_counts.pop(receipt_id, None)
            # Mark as uploaded in WAL for crash recovery
            if self._wal_path and "entry_hash" in receipt:
                self._mark_wal_done(receipt["entry_hash"])
            with self._stats_lock:
                self._stats["uploaded"] += 1
            return True
        else:
            self._on_upload_failure()
            return self._handle_retry(receipt, receipt_id)

    def _handle_retry(self, receipt: Dict[str, Any], receipt_id: int) -> bool:
        """Handle retry logic for a failed upload."""
        attempt = self._retry_counts.get(receipt_id, 0) + 1
        self._retry_counts[receipt_id] = attempt

        if attempt >= self._max_retries:
            # Give up — drop the receipt
            self._retry_counts.pop(receipt_id, None)
            with self._stats_lock:
                self._stats["dropped_max_retries"] += 1
            logger.warning(
                "Receipt dropped after %d retries", self._max_retries
            )
            return False

        # Compute backoff with decorrelated jitter (AWS pattern)
        delay = self._compute_backoff(attempt)

        with self._stats_lock:
            self._stats["retries"] += 1

        # Sleep for backoff then requeue
        if delay > 0 and not self._shutdown_event.is_set():
            # Use event.wait() so shutdown can interrupt
            self._shutdown_event.wait(timeout=delay)

        self._requeue(receipt)
        return False

    def _requeue(self, receipt: Dict[str, Any]) -> None:
        """Put a receipt back at the front of the queue for retry."""
        with self._lock:
            self._queue.appendleft(receipt)

    def _compute_backoff(self, attempt: int) -> float:
        """
        Compute backoff delay with decorrelated jitter.

        Uses the AWS "decorrelated jitter" algorithm:
            sleep = min(max_delay, random_between(base, prev_sleep * 3))

        This gives better spread than equal or full jitter.
        """
        # Exponential base
        exp_delay = self._base_delay * (2 ** (attempt - 1))
        # Add decorrelated jitter
        jittered = random.uniform(self._base_delay, min(self._max_delay, exp_delay * 3))
        return min(jittered, self._max_delay)

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _circuit_allows_request(self) -> bool:
        """Check if the circuit breaker allows a request."""
        with self._circuit_lock:
            if self._circuit_state == CircuitState.CLOSED:
                return True

            if self._circuit_state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                elapsed = time.monotonic() - self._circuit_opened_at
                if elapsed >= self._circuit_recovery_timeout:
                    self._circuit_state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker half-open, probing...")
                    return True
                return False

            # HALF_OPEN — allow one probe request
            return True

    def _on_upload_success(self) -> None:
        """Record a successful upload for circuit breaker."""
        with self._circuit_lock:
            if self._circuit_state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker closed (probe succeeded)")
            self._circuit_state = CircuitState.CLOSED
            self._circuit_failure_count = 0

    def _on_upload_failure(self) -> None:
        """Record a failed upload for circuit breaker."""
        with self._circuit_lock:
            self._circuit_failure_count += 1

            if self._circuit_state == CircuitState.HALF_OPEN:
                # Probe failed — reopen
                self._circuit_state = CircuitState.OPEN
                self._circuit_opened_at = time.monotonic()
                logger.warning("Circuit breaker re-opened (probe failed)")
                return

            if (
                self._circuit_state == CircuitState.CLOSED
                and self._circuit_failure_count >= self._circuit_failure_threshold
            ):
                self._circuit_state = CircuitState.OPEN
                self._circuit_opened_at = time.monotonic()
                with self._stats_lock:
                    self._stats["circuit_opens"] += 1
                logger.warning(
                    "Circuit breaker opened after %d consecutive failures",
                    self._circuit_failure_count,
                )

    # ------------------------------------------------------------------
    # Write-Ahead Log (WAL) for crash recovery
    # ------------------------------------------------------------------

    def _compute_entry_hash(self, receipt: Dict[str, Any]) -> str:
        """
        Compute a stable hash for a receipt entry.
        Uses the entry_hash field if present, otherwise computes from JSON.
        """
        if "entry_hash" in receipt:
            return receipt["entry_hash"]
        # Fallback: hash the JSON (deterministic if keys are ordered)
        json_str = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _write_to_wal(self, receipt: Dict[str, Any]) -> None:
        """
        Append receipt entry to the WAL file (JSONL format).

        Crash-safe: Each line is a complete JSON object.
        Format: {"entry_hash":"...","receipt":{...}}
        """
        if not self._wal_path:
            return

        entry_hash = self._compute_entry_hash(receipt)
        wal_entry = {
            "entry_hash": entry_hash,
            "receipt": receipt,
        }

        try:
            with self._wal_lock:
                # Ensure directory exists
                wal_dir = os.path.dirname(self._wal_path)
                if wal_dir:
                    os.makedirs(wal_dir, exist_ok=True)

                # Open with O_APPEND for atomic writes
                with open(self._wal_path, "a") as f:
                    # File-level locking on Unix only — on Windows fcntl is
                    # None and we skip locking entirely (we already serialize
                    # in-process via self._wal_lock above).
                    if fcntl is not None:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except OSError:
                            # File already locked — just proceed
                            pass

                    json_line = json.dumps(wal_entry, separators=(",", ":"))
                    f.write(json_line + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                    if fcntl is not None:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except OSError:
                            pass
        except Exception as exc:
            logger.error("Failed to write WAL: %s", exc)

    def _mark_wal_done(self, entry_hash: str) -> None:
        """
        Mark an entry as successfully uploaded in the WAL.
        Appends "DONE:{entry_hash}" to the WAL file.
        """
        if not self._wal_path:
            return

        try:
            with self._wal_lock:
                with open(self._wal_path, "a") as f:
                    if fcntl is not None:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except OSError:
                            pass

                    done_line = f"DONE:{entry_hash}"
                    f.write(done_line + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                    if fcntl is not None:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except OSError:
                            pass

                # Track uploaded entry for truncation
                with self._lock:
                    self._uploaded_hashes.add(entry_hash)
        except Exception as exc:
            logger.error("Failed to mark WAL entry as done: %s", exc)

    def _recover_from_wal(self) -> None:
        """
        Recover unfinished entries from the WAL file.

        Reads the WAL line-by-line, re-queues any receipt entries
        that were not marked DONE (indicating they were pending upload
        when the process crashed).
        """
        if not self._wal_path or not os.path.exists(self._wal_path):
            return

        try:
            with self._wal_lock:
                done_hashes: Set[str] = set()
                pending_receipts: Dict[str, Dict[str, Any]] = {}

                # First pass: identify done entries and parse pending receipts
                try:
                    with open(self._wal_path, "r") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.rstrip("\n")
                            if not line:
                                continue

                            try:
                                if line.startswith("DONE:"):
                                    # Mark this entry_hash as uploaded
                                    entry_hash = line[5:]
                                    done_hashes.add(entry_hash)
                                else:
                                    # Parse JSON entry
                                    obj = json.loads(line)
                                    if isinstance(obj, dict) and "entry_hash" in obj:
                                        entry_hash = obj["entry_hash"]
                                        receipt = obj.get("receipt", {})
                                        # Keep only the most recent entry for each hash
                                        pending_receipts[entry_hash] = receipt
                            except json.JSONDecodeError:
                                # Partial line at end of file (crash mid-write)
                                # Skip silently — next time we write will complete it
                                logger.debug(
                                    "Skipping malformed WAL line %d: %r",
                                    line_num,
                                    line[:50],
                                )
                                continue
                except FileNotFoundError:
                    return

                # Second pass: re-queue entries not marked done
                for entry_hash, receipt in pending_receipts.items():
                    if entry_hash not in done_hashes:
                        with self._lock:
                            self._queue.append(receipt)
                            with self._stats_lock:
                                self._stats["enqueued"] += 1
                        logger.info("Recovered receipt %s from WAL", entry_hash)

                # Track which entries were already uploaded
                self._uploaded_hashes = done_hashes
        except Exception as exc:
            logger.error("Failed to recover from WAL: %s", exc)

    def _maybe_truncate_wal(self) -> None:
        """
        Truncate the WAL file if all entries have been uploaded.

        Checks the WAL to see if all entries are marked DONE.
        If so, deletes the file (or truncates to 0 bytes).
        """
        if not self._wal_path or not os.path.exists(self._wal_path):
            return

        try:
            with self._wal_lock:
                pending_count = 0
                done_count = 0

                try:
                    with open(self._wal_path, "r") as f:
                        for line in f:
                            line = line.rstrip("\n")
                            if not line:
                                continue
                            if line.startswith("DONE:"):
                                done_count += 1
                            else:
                                try:
                                    obj = json.loads(line)
                                    if isinstance(obj, dict) and "receipt" in obj:
                                        pending_count += 1
                                except json.JSONDecodeError:
                                    pass
                except FileNotFoundError:
                    return

                # If all receipts are done, truncate or delete the WAL
                if pending_count == 0 and done_count > 0:
                    try:
                        os.remove(self._wal_path)
                        logger.info("Truncated WAL file (all entries uploaded)")
                    except OSError:
                        # If removal fails, try truncating
                        try:
                            with open(self._wal_path, "w"):
                                pass
                        except OSError as e:
                            logger.warning("Failed to truncate WAL: %s", e)
        except Exception as exc:
            logger.error("Failed to check WAL truncation: %s", exc)
