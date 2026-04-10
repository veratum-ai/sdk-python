"""
Veratum Evidence Layer for Haystack.

Drop-in compliance for Haystack pipelines.
Creates immutable evidence receipts for every component run in a
Haystack pipeline — generators, retrievers, converters, and custom components.

Example usage:
    from haystack import Pipeline
    from haystack.components.generators import OpenAIGenerator
    from veratum.integrations.haystack_plugin import VeratumTracer, enable_veratum

    tracer = enable_veratum()

    pipe = Pipeline()
    pipe.add_component("llm", OpenAIGenerator(model="gpt-4o"))
    pipe.tracing.tracer = tracer

    result = pipe.run({"llm": {"prompt": "What is 2+2?"}})

    # Every component run now has a Veratum receipt
    print(tracer.get_receipts())

Or wrap individual components:
    from veratum.integrations.haystack_plugin import wrap_component

    generator = OpenAIGenerator(model="gpt-4o")
    wrapped = wrap_component(generator)
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Try to import Haystack tracer interface
_HAYSTACK_AVAILABLE = False
_BASE_TRACER = object

try:
    from haystack.tracing import Tracer, Span
    _BASE_TRACER = Tracer
    _HAYSTACK_AVAILABLE = True
except ImportError:
    try:
        from haystack import tracing
        _BASE_TRACER = tracing.Tracer
        _HAYSTACK_AVAILABLE = True
    except (ImportError, AttributeError):
        pass


class VeratumSpan:
    """
    A span representing a single operation within a Haystack pipeline.

    Implements the Haystack Span interface for compatibility.
    """

    def __init__(
        self,
        operation_name: str,
        parent_span: Optional["VeratumSpan"] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        self.operation_name = operation_name
        self.parent_span = parent_span
        self.span_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self._tags: Dict[str, Any] = tags or {}
        self._content: Dict[str, Any] = {}

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the span."""
        self._tags[key] = value

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set multiple tags."""
        self._tags.update(tags)

    def get_tag(self, key: str) -> Optional[Any]:
        """Get a tag value."""
        return self._tags.get(key)

    def set_content_tag(self, key: str, value: Any) -> None:
        """Set a content tag (for input/output data)."""
        self._content[key] = value

    def raw_span(self) -> "VeratumSpan":
        """Return the raw span object."""
        return self

    def finish(self) -> None:
        """Mark the span as finished."""
        self.end_time = time.time()

    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_ms": self.latency_ms,
            "tags": self._tags,
            "content": self._content,
            "parent_span_id": self.parent_span.span_id if self.parent_span else None,
        }


class VeratumTracer(_BASE_TRACER):
    """
    Haystack Tracer implementation that creates Veratum evidence receipts.

    Hooks into Haystack's tracing system to capture every component
    execution with cryptographic evidence receipts.

    Features:
    - Automatic receipt generation for all pipeline components
    - Generator/LLM call evidence with model tracking
    - Retriever evidence for RAG audit trails
    - Component input/output hashing
    - Non-blocking async upload
    - Thread-safe receipt storage
    - Pipeline-level evidence summary

    Args:
        api_key: Veratum API key for receipt upload
        endpoint: Veratum API endpoint URL
        capture_retrievers: Capture retriever evidence (default: True)
        capture_generators: Capture generator/LLM evidence (default: True)
        capture_all: Capture all component types (default: True)
        metadata: Default metadata added to all receipts
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        capture_retrievers: bool = True,
        capture_generators: bool = True,
        capture_all: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Veratum Haystack tracer."""
        self.api_key = api_key or self._get_api_key_from_env()
        self.endpoint = endpoint or "https://api.veratum.ai"
        self.capture_retrievers = capture_retrievers
        self.capture_generators = capture_generators
        self.capture_all = capture_all
        self.default_metadata = metadata or {}

        # Try to import EvidenceEngine
        self._engine = None
        try:
            from veratum.core.evidence import EvidenceEngine
            self._engine = EvidenceEngine(api_key=api_key, endpoint=endpoint)
            logger.info("EvidenceEngine initialized for Haystack tracer")
        except (ImportError, Exception) as e:
            logger.warning(f"EvidenceEngine not available ({e})")

        # Receipt storage
        self._receipts: List[Dict[str, Any]] = []
        self._receipt_lock = threading.Lock()
        self._upload_threads: List[threading.Thread] = []

        # Current span context
        self._current_span: Optional[VeratumSpan] = None

        logger.info("VeratumTracer initialized for Haystack pipelines")

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

    @contextmanager
    def trace(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Any] = None,
    ) -> Generator[VeratumSpan, None, None]:
        """
        Create a trace span for a pipeline operation.

        This is the main Haystack Tracer interface method.

        Args:
            operation_name: Name of the operation (component name)
            tags: Initial tags for the span
            parent_span: Parent span for nested operations

        Yields:
            VeratumSpan instance
        """
        span = VeratumSpan(
            operation_name=operation_name,
            parent_span=parent_span or self._current_span,
            tags=tags or {},
        )

        # Track current span for nesting
        previous_span = self._current_span
        self._current_span = span

        try:
            yield span
        except Exception as e:
            span.set_tag("error", True)
            span.set_tag("error.message", str(e))
            raise
        finally:
            span.finish()
            self._current_span = previous_span
            self._process_span(span)

    def _process_span(self, span: VeratumSpan) -> None:
        """Process a completed span and create a receipt."""
        operation = span.operation_name
        tags = span._tags
        content = span._content

        # Determine component type from operation name and tags
        component_type = tags.get("haystack.component.type", "unknown")
        is_generator = "generator" in component_type.lower() or "generator" in operation.lower()
        is_retriever = "retriever" in component_type.lower() or "retriever" in operation.lower()

        # Filter based on configuration
        if not self.capture_all:
            if is_generator and not self.capture_generators:
                return
            if is_retriever and not self.capture_retrievers:
                return

        # Extract inputs/outputs from content tags
        inputs = {}
        outputs = {}
        for key, value in content.items():
            if "input" in key.lower():
                inputs[key] = value
            elif "output" in key.lower():
                outputs[key] = value

        # Also check tags for component info
        model = tags.get("haystack.component.model", "unknown")
        if model == "unknown" and is_generator:
            # Try to detect from component type
            for key, value in tags.items():
                if "model" in key.lower() and isinstance(value, str):
                    model = value
                    break

        # Detect provider
        provider = self._detect_provider(model, component_type)

        # Build receipt
        receipt = self._create_receipt(
            span=span,
            component_type=component_type,
            is_generator=is_generator,
            is_retriever=is_retriever,
            model=model,
            provider=provider,
            inputs=inputs,
            outputs=outputs,
        )

        logger.debug(
            f"Haystack receipt {receipt['receipt_id']} | "
            f"component={operation} | {span.latency_ms:.0f}ms"
        )

    def _create_receipt(
        self,
        span: VeratumSpan,
        component_type: str,
        is_generator: bool,
        is_retriever: bool,
        model: str,
        provider: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a Veratum receipt from a Haystack span."""
        receipt_id = f"receipt_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        error = span._tags.get("error.message")
        status = "FAILED" if span._tags.get("error") else "SUCCESS"

        receipt = {
            "receipt_id": receipt_id,
            "timestamp": timestamp,
            "span_id": span.span_id,
            "operation_name": span.operation_name,
            "component_type": component_type,
            "framework": "haystack",
            "model": model,
            "provider": provider,
            "is_generator": is_generator,
            "is_retriever": is_retriever,
            "status": status,
            "latency_ms": span.latency_ms,
            "input_hash": self._compute_hash(inputs),
            "output_hash": self._compute_hash(outputs),
            "error": error,
            "parent_span_id": span.parent_span.span_id if span.parent_span else None,
            "metadata": {**self.default_metadata},
        }

        # Use EvidenceEngine for chained receipts on LLM calls
        if self._engine and is_generator:
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

    @staticmethod
    def _detect_provider(model: str, component_type: str) -> str:
        """Detect provider from model name or component type."""
        combined = f"{model} {component_type}".lower()
        if any(x in combined for x in ["openai", "gpt-", "o1-", "o3-"]):
            return "openai"
        if "anthropic" in combined or "claude" in combined:
            return "anthropic"
        if "google" in combined or "gemini" in combined:
            return "google"
        if "cohere" in combined or "command" in combined:
            return "cohere"
        if "hugging" in combined or "hf" in combined:
            return "huggingface"
        if "ollama" in combined:
            return "ollama"
        return "unknown"

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

    # ─── Public API ─────────────────────────────────────────────────

    def current_span(self) -> Optional[VeratumSpan]:
        """Get the current active span."""
        return self._current_span

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

        by_component = {}
        generators = 0
        retrievers = 0

        for r in receipts:
            comp = r.get("component_type", "unknown")
            by_component[comp] = by_component.get(comp, 0) + 1
            if r.get("is_generator"):
                generators += 1
            if r.get("is_retriever"):
                retrievers += 1

        return {
            "total_receipts": len(receipts),
            "by_component_type": by_component,
            "generator_calls": generators,
            "retriever_calls": retrievers,
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
    capture_all: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> VeratumTracer:
    """
    Create a Veratum tracer for Haystack pipelines.

    One-liner to add compliance evidence to any Haystack pipeline.

    Usage:
        from veratum.integrations.haystack_plugin import enable_veratum

        tracer = enable_veratum()

        # Set as the pipeline tracer
        from haystack import tracing
        tracing.tracer.actual_tracer = tracer

        # Or set on individual pipelines
        pipe.tracing.tracer = tracer

    Args:
        api_key: Veratum API key
        endpoint: Veratum API endpoint
        capture_all: Capture all component types (default: True)
        metadata: Default metadata for all receipts

    Returns:
        VeratumTracer instance
    """
    return VeratumTracer(
        api_key=api_key,
        endpoint=endpoint,
        capture_all=capture_all,
        metadata=metadata,
    )


def wrap_component(component: Any, tracer: Optional[VeratumTracer] = None) -> Any:
    """
    Wrap a Haystack component to capture evidence.

    Wraps the component's run method to create evidence receipts.
    Useful for capturing evidence from individual components
    outside of a full pipeline.

    Usage:
        from haystack.components.generators import OpenAIGenerator
        from veratum.integrations.haystack_plugin import wrap_component

        generator = OpenAIGenerator(model="gpt-4o")
        wrapped = wrap_component(generator)

        result = wrapped.run(prompt="What is 2+2?")

    Args:
        component: Haystack component instance
        tracer: Optional VeratumTracer (creates one if not provided)

    Returns:
        Wrapped component with evidence capture
    """
    if tracer is None:
        tracer = VeratumTracer()

    original_run = component.run

    def wrapped_run(*args, **kwargs):
        component_name = type(component).__name__
        with tracer.trace(
            operation_name=component_name,
            tags={"haystack.component.type": component_name},
        ) as span:
            # Capture inputs
            span.set_content_tag("input", kwargs or args)

            result = original_run(*args, **kwargs)

            # Capture outputs
            span.set_content_tag("output", result)

            return result

    component.run = wrapped_run
    component._veratum_tracer = tracer
    return component


__all__ = [
    "VeratumTracer",
    "VeratumSpan",
    "enable_veratum",
    "wrap_component",
]
