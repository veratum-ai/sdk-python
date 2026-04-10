"""
Veratum Evidence Layer for OpenAI Agents SDK.

Drop-in compliance for OpenAI's Agents framework.
Creates immutable evidence receipts for every agent run, tool call,
and handoff in an OpenAI Agents workflow.

Example usage:
    from agents import Agent, Runner
    from veratum.integrations.openai_agents_plugin import VeratumTracingProcessor

    processor = VeratumTracingProcessor()

    agent = Agent(name="assistant", instructions="You are helpful.")
    result = Runner.run_sync(agent, "What is 2+2?", run_config=RunConfig(
        tracing_processors=[processor]
    ))

    # Every agent action now has a Veratum receipt
    print(processor.get_receipts())

Or with the convenience wrapper:
    from veratum.integrations.openai_agents_plugin import enable_veratum

    processor = enable_veratum()
    # Pass processor to RunConfig(tracing_processors=[processor])
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import OpenAI Agents tracing interface
_AGENTS_AVAILABLE = False
_BASE_PROCESSOR = object

try:
    from agents.tracing import TracingProcessor
    _BASE_PROCESSOR = TracingProcessor
    _AGENTS_AVAILABLE = True
except ImportError:
    try:
        from openai.agents.tracing import TracingProcessor
        _BASE_PROCESSOR = TracingProcessor
        _AGENTS_AVAILABLE = True
    except ImportError:
        pass


class VeratumTracingProcessor(_BASE_PROCESSOR):
    """
    OpenAI Agents tracing processor that creates Veratum evidence receipts.

    Hooks into the OpenAI Agents SDK tracing system to capture every
    agent run, LLM generation, tool call, and handoff with cryptographic
    evidence receipts.

    Features:
    - Automatic receipt generation for all agent spans
    - LLM generation evidence with model and token tracking
    - Tool call evidence with input/output capture
    - Agent handoff tracking
    - Guardrail execution evidence
    - Non-blocking async upload
    - Thread-safe receipt storage

    Args:
        api_key: Veratum API key for receipt upload
        endpoint: Veratum API endpoint URL
        metadata: Default metadata added to all receipts
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Veratum tracing processor."""
        if _AGENTS_AVAILABLE and hasattr(_BASE_PROCESSOR, '__init__'):
            super().__init__()

        self.api_key = api_key or self._get_api_key_from_env()
        self.endpoint = endpoint or "https://api.veratum.ai"
        self.default_metadata = metadata or {}

        # Try to import EvidenceEngine
        self._engine = None
        try:
            from veratum.core.evidence import EvidenceEngine
            self._engine = EvidenceEngine(api_key=api_key, endpoint=endpoint)
            logger.info("EvidenceEngine initialized for OpenAI Agents processor")
        except (ImportError, Exception) as e:
            logger.warning(f"EvidenceEngine not available ({e})")

        # Receipt storage
        self._receipts: List[Dict[str, Any]] = []
        self._receipt_lock = threading.Lock()
        self._upload_threads: List[threading.Thread] = []

        # Active span tracking
        self._active_spans: Dict[str, Dict[str, Any]] = {}
        self._spans_lock = threading.Lock()

        logger.info("VeratumTracingProcessor initialized for OpenAI Agents")

    @staticmethod
    def _get_api_key_from_env() -> Optional[str]:
        """Get Veratum API key from environment."""
        import os
        return os.environ.get("VERATUM_API_KEY")

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _create_receipt(
        self,
        span_id: str,
        span_type: str,
        span_data: Dict[str, Any],
        status: str = "SUCCESS",
    ) -> Dict[str, Any]:
        """Create an evidence receipt from span data."""
        receipt_id = f"receipt_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        inputs = span_data.get("inputs", "")
        outputs = span_data.get("outputs", "")
        model = span_data.get("model", "unknown")
        provider = "openai"

        receipt = {
            "receipt_id": receipt_id,
            "timestamp": timestamp,
            "span_id": span_id,
            "span_type": span_type,
            "framework": "openai_agents",
            "model": model,
            "provider": provider,
            "status": status,
            "latency_ms": span_data.get("latency_ms", 0),
            "input_hash": self._compute_hash(inputs),
            "output_hash": self._compute_hash(outputs),
            "error": span_data.get("error"),
            "agent_name": span_data.get("agent_name"),
            "tool_name": span_data.get("tool_name"),
            "handoff_to": span_data.get("handoff_to"),
            "token_usage": span_data.get("token_usage"),
            "metadata": {**self.default_metadata},
        }

        # Use EvidenceEngine for chained receipts
        if self._engine and span_type in ("generation", "agent"):
            try:
                evidence = self._engine.create_evidence(
                    request={"prompt": str(inputs)},
                    response={"text": str(outputs)},
                    provider=provider,
                    model=model,
                    metadata=self.default_metadata,
                )
                receipt["entry_hash"] = evidence.get("entry_hash")
                receipt["prev_hash"] = evidence.get("prev_hash")
                receipt["sequence_no"] = evidence.get("sequence_no")
            except Exception as e:
                logger.debug(f"EvidenceEngine creation failed: {e}")

        with self._receipt_lock:
            self._receipts.append(receipt)

        self._upload_async(receipt)
        return receipt

    def _upload_async(self, receipt: Dict[str, Any]) -> None:
        """Upload receipt in background thread."""
        def upload():
            self._upload_receipt(receipt)

        thread = threading.Thread(target=upload, daemon=True)
        self._upload_threads.append(thread)
        thread.start()
        self._upload_threads = [t for t in self._upload_threads if t.is_alive()]

    def _upload_receipt(self, receipt: Dict[str, Any]) -> None:
        """Upload receipt to Veratum API."""
        if self._engine:
            try:
                self._engine.upload_evidence(receipt)
                return
            except Exception:
                pass

        if not self.api_key:
            return

        try:
            from urllib.request import Request, urlopen
            payload = json.dumps(receipt, default=str).encode("utf-8")
            req = Request(
                f"{self.endpoint}/v2/evidence/upload",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(req, timeout=5) as resp:
                pass
        except Exception as e:
            logger.warning(f"Upload failed for {receipt['receipt_id']}: {e}")

    # ─── TracingProcessor Interface ─────────────────────────────────

    def on_trace_start(self, trace: Any) -> None:
        """Called when a new trace (full agent run) starts."""
        trace_id = getattr(trace, 'trace_id', str(uuid.uuid4()))
        with self._spans_lock:
            self._active_spans[str(trace_id)] = {
                "type": "trace",
                "start_time": time.time(),
                "trace_id": str(trace_id),
                "name": getattr(trace, 'name', None),
            }
        logger.debug(f"Trace started: {trace_id}")

    def on_trace_end(self, trace: Any) -> None:
        """Called when a trace completes."""
        trace_id = str(getattr(trace, 'trace_id', ''))
        with self._spans_lock:
            span_data = self._active_spans.pop(trace_id, None)

        if span_data:
            span_data["latency_ms"] = (time.time() - span_data["start_time"]) * 1000
            span_data["outputs"] = "trace_complete"

            receipt = self._create_receipt(
                span_id=trace_id,
                span_type="trace",
                span_data=span_data,
                status="SUCCESS",
            )
            logger.info(f"Trace receipt {receipt['receipt_id']} | {span_data['latency_ms']:.0f}ms")

    def on_span_start(self, span: Any) -> None:
        """
        Called when a span starts within a trace.

        Spans can be: agent, generation, tool, handoff, guardrail, etc.
        """
        span_id = str(getattr(span, 'span_id', uuid.uuid4()))
        span_type = getattr(span, 'span_type', 'unknown')
        if hasattr(span, 'type'):
            span_type = span.type

        span_data = {
            "type": str(span_type),
            "start_time": time.time(),
            "span_id": span_id,
            "parent_id": str(getattr(span, 'parent_id', None)),
        }

        # Extract span-specific data
        span_info = getattr(span, 'span_data', None) or span

        if str(span_type) == "agent":
            span_data["agent_name"] = getattr(span_info, 'name', None)
            span_data["inputs"] = getattr(span_info, 'input', '')

        elif str(span_type) == "generation":
            span_data["model"] = getattr(span_info, 'model', 'unknown')
            span_data["inputs"] = getattr(span_info, 'input', '')

        elif str(span_type) == "function":
            span_data["tool_name"] = getattr(span_info, 'name', None)
            span_data["inputs"] = getattr(span_info, 'input', '')

        elif str(span_type) == "handoff":
            span_data["handoff_to"] = getattr(span_info, 'to_agent', None)
            span_data["inputs"] = getattr(span_info, 'input', '')

        elif str(span_type) == "guardrail":
            span_data["inputs"] = getattr(span_info, 'input', '')

        with self._spans_lock:
            self._active_spans[span_id] = span_data

    def on_span_end(self, span: Any) -> None:
        """Called when a span ends."""
        span_id = str(getattr(span, 'span_id', ''))

        with self._spans_lock:
            span_data = self._active_spans.pop(span_id, None)

        if not span_data:
            return

        span_data["latency_ms"] = (time.time() - span_data["start_time"]) * 1000

        # Extract outputs
        span_info = getattr(span, 'span_data', None) or span
        span_type = span_data.get("type", "unknown")

        if span_type == "generation":
            span_data["outputs"] = getattr(span_info, 'output', '')
            # Extract token usage
            usage = getattr(span_info, 'usage', None)
            if usage:
                span_data["token_usage"] = {
                    "input_tokens": getattr(usage, 'input_tokens',
                                           getattr(usage, 'prompt_tokens', None)),
                    "output_tokens": getattr(usage, 'output_tokens',
                                            getattr(usage, 'completion_tokens', None)),
                }
        elif span_type == "function":
            span_data["outputs"] = getattr(span_info, 'output', '')
        elif span_type == "agent":
            span_data["outputs"] = getattr(span_info, 'output', '')
        elif span_type == "handoff":
            span_data["outputs"] = getattr(span_info, 'to_agent', '')
        elif span_type == "guardrail":
            span_data["outputs"] = getattr(span_info, 'output', '')
            triggered = getattr(span_info, 'triggered', False)
            if triggered:
                span_data["metadata"] = {**self.default_metadata, "guardrail_triggered": True}

        # Check for errors
        error = getattr(span_info, 'error', None)
        status = "FAILED" if error else "SUCCESS"
        if error:
            span_data["error"] = str(error)

        receipt = self._create_receipt(
            span_id=span_id,
            span_type=span_type,
            span_data=span_data,
            status=status,
        )

        logger.debug(
            f"Span receipt {receipt['receipt_id']} | "
            f"type={span_type} | {span_data['latency_ms']:.0f}ms"
        )

    # ─── Public API ─────────────────────────────────────────────────

    def get_receipts(self) -> List[Dict[str, Any]]:
        """Get all captured receipts."""
        with self._receipt_lock:
            return self._receipts.copy()

    def get_receipt_count(self) -> int:
        """Get the number of captured receipts."""
        return len(self._receipts)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of evidence captured."""
        with self._receipt_lock:
            receipts = self._receipts.copy()

        by_type = {}
        models_used = set()

        for r in receipts:
            span_type = r.get("span_type", "unknown")
            by_type[span_type] = by_type.get(span_type, 0) + 1
            if r.get("model") and r["model"] != "unknown":
                models_used.add(r["model"])

        return {
            "total_receipts": len(receipts),
            "by_span_type": by_type,
            "models_used": list(models_used),
        }

    def wait_for_uploads(self, timeout: float = 30.0) -> bool:
        """Wait for pending uploads."""
        start = time.time()
        for thread in self._upload_threads:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                return False
            thread.join(timeout=remaining)
            if thread.is_alive():
                return False
        return True


def enable_veratum(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> VeratumTracingProcessor:
    """
    Create a Veratum tracing processor for OpenAI Agents.

    One-liner to add compliance evidence to any OpenAI Agents workflow.

    Usage:
        from veratum.integrations.openai_agents_plugin import enable_veratum
        from agents import Agent, Runner, RunConfig

        processor = enable_veratum()

        agent = Agent(name="assistant", instructions="You are helpful.")
        result = Runner.run_sync(
            agent,
            "What is 2+2?",
            run_config=RunConfig(tracing_processors=[processor])
        )

        print(processor.get_receipts())

    Args:
        api_key: Veratum API key
        endpoint: Veratum API endpoint
        metadata: Default metadata for all receipts

    Returns:
        VeratumTracingProcessor instance
    """
    return VeratumTracingProcessor(
        api_key=api_key,
        endpoint=endpoint,
        metadata=metadata,
    )


__all__ = [
    "VeratumTracingProcessor",
    "enable_veratum",
]
