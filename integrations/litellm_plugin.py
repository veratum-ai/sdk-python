"""
Veratum Evidence Layer for LiteLLM.

Drop-in compliance for 140+ LLM providers via LiteLLM's callback system.
Creates immutable evidence receipts for every LLM call automatically.

Example usage:
    import litellm
    from veratum.integrations.litellm_plugin import VeratumLiteLLMCallback

    litellm.callbacks = [VeratumLiteLLMCallback()]

    # Every LLM call now gets a Veratum receipt
    response = litellm.completion(model="gpt-4o", messages=[...])

Or use the convenience function:
    from veratum.integrations.litellm_plugin import enable_veratum

    enable_veratum(api_key="your-key")
    # Veratum is now enabled for all LiteLLM calls
"""

import logging
import time
import threading
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import CustomLogger from litellm for proper inheritance
_CUSTOM_LOGGER_BASE = None
try:
    from litellm.integrations.custom_logger import CustomLogger
    _CUSTOM_LOGGER_BASE = CustomLogger
except (ImportError, AttributeError):
    # CustomLogger not available, will use object as base
    _CUSTOM_LOGGER_BASE = object


class ProviderType(str, Enum):
    """Enumeration of LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    AZURE = "azure"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"
    TOGETHER = "together"
    REPLICATE = "replicate"
    PALM = "palm"
    HUGGING_FACE = "hugging_face"
    ALEPH_ALPHA = "aleph_alpha"
    BASETEN = "baseten"
    PETALS = "petals"
    VLLM = "vllm"
    UNDEFINED = "undefined"


@dataclass
class VeratumReceipt:
    """
    Immutable evidence receipt for an LLM interaction.

    Attributes:
        receipt_id: Unique identifier for this receipt
        timestamp: ISO 8601 timestamp of the call
        model: Model identifier used
        provider: LLM provider name
        status: 'SUCCESS' or 'FAILED'
        latency_ms: Call duration in milliseconds
        input_tokens: Token count for input (if available)
        output_tokens: Token count for output (if available)
        request_hash: SHA-256 hash of the request for integrity
        response_hash: SHA-256 hash of the response for integrity
        metadata: Additional context (temperature, top_p, etc.)
    """
    receipt_id: str
    timestamp: str
    model: str
    provider: str
    status: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    request_hash: Optional[str] = None
    response_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary format."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert receipt to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class VeratumLiteLLMCallback(_CUSTOM_LOGGER_BASE):
    """
    LiteLLM callback integration for Veratum evidence layer.

    Subclasses litellm.integrations.custom_logger.CustomLogger (if available)
    and hooks into every LLM call to create immutable evidence receipts automatically.

    Features:
    - Non-blocking async evidence creation
    - Support for 140+ LLM providers
    - Automatic token counting and latency tracking
    - Thread-safe receipt storage and upload
    - Graceful fallback for missing EvidenceEngine
    - Comprehensive error handling and logging

    Args:
        api_key: Veratum API key for receipt upload (optional)
        endpoint: Veratum API endpoint URL (optional)
        queue_receipts: If True, batch receipts before uploading (default: True)
        queue_size: Number of receipts to batch before upload (default: 10)
        upload_timeout: Seconds to wait for upload before continuing (default: 2)
        log_level: Python logging level (default: logging.INFO)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        queue_receipts: bool = True,
        queue_size: int = 10,
        upload_timeout: float = 2.0,
        log_level: int = logging.INFO,
    ):
        """Initialize the Veratum LiteLLM callback."""
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.veratum.ai"
        self.queue_receipts = queue_receipts
        self.queue_size = queue_size
        self.upload_timeout = upload_timeout

        # Initialize logger
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - veratum.litellm - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Try to import EvidenceEngine, fall back to direct API calls
        self._engine = None
        try:
            from veratum.core.evidence import EvidenceEngine
            self._engine = EvidenceEngine(api_key=api_key, endpoint=endpoint)
            logger.info("EvidenceEngine initialized for receipt upload")
        except (ImportError, Exception) as e:
            logger.warning(
                f"EvidenceEngine not available ({e}), falling back to direct API calls"
            )

        # Receipt queue and thread lock for thread safety
        self._receipt_queue: List[VeratumReceipt] = []
        self._queue_lock = threading.Lock()
        self._upload_threads: List[threading.Thread] = []  # Track all upload threads

    def _generate_receipt_id(self) -> str:
        """Generate a unique receipt ID."""
        import uuid
        return f"receipt_{uuid.uuid4().hex[:12]}"

    def _extract_provider(self, model: str) -> str:
        """
        Extract provider from model string.

        LiteLLM model format: "provider/model-name" or "provider::model-name"
        Falls back to parsing common provider prefixes.
        """
        if "/" in model:
            return model.split("/")[0]
        if "::" in model:
            return model.split("::")[0]

        # Common provider prefixes
        prefixes = {
            "gpt-": "openai",
            "claude-": "anthropic",
            "gemini-": "google",
            "command-": "cohere",
            "bison-": "palm",
            "bedrock-": "bedrock",
            "ollama-": "ollama",
        }

        for prefix, provider in prefixes.items():
            if model.startswith(prefix):
                return provider

        return "undefined"

    def _compute_hash(self, data: Union[str, Dict, List]) -> str:
        """Compute SHA-256 hash of data for integrity verification."""
        import hashlib

        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def _extract_token_counts(
        self, kwargs: Dict[str, Any], response_obj: Optional[Any]
    ) -> tuple:
        """
        Extract input and output token counts.

        LiteLLM normalizes to OpenAI format, but also preserves provider-specific
        usage information.
        """
        input_tokens = None
        output_tokens = None

        # Try to get from response usage (OpenAI format)
        if response_obj and hasattr(response_obj, "usage"):
            usage = response_obj.usage
            if hasattr(usage, "prompt_tokens"):
                input_tokens = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                output_tokens = usage.completion_tokens
        # Fallback to raw response dict
        elif isinstance(response_obj, dict) and "usage" in response_obj:
            usage = response_obj["usage"]
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", output_tokens)

        return input_tokens, output_tokens

    def _create_receipt(
        self,
        model: str,
        kwargs: Dict[str, Any],
        response_obj: Optional[Any],
        start_time: float,
        end_time: float,
        status: str = "SUCCESS",
        error: Optional[Exception] = None,
    ) -> VeratumReceipt:
        """
        Create an immutable evidence receipt.

        Args:
            model: Model identifier
            kwargs: LiteLLM kwargs dict (contains request info)
            response_obj: Response object from LLM
            start_time: Request start time (unix timestamp)
            end_time: Request end time (unix timestamp)
            status: 'SUCCESS' or 'FAILED'
            error: Exception object if status is FAILED

        Returns:
            VeratumReceipt object
        """
        receipt_id = self._generate_receipt_id()
        latency_ms = (end_time - start_time) * 1000
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Extract provider
        provider = self._extract_provider(model)

        # Extract token counts
        input_tokens, output_tokens = self._extract_token_counts(kwargs, response_obj)

        # Compute hashes for integrity
        request_hash = None
        response_hash = None
        try:
            # Extract request data (messages, parameters)
            request_data = {
                "model": model,
                "messages": kwargs.get("messages", []),
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "max_tokens": kwargs.get("max_tokens"),
            }
            request_hash = self._compute_hash(request_data)

            if response_obj:
                response_hash = self._compute_hash(response_obj)
        except Exception as e:
            logger.debug(f"Failed to compute hashes: {e}")

        # Build metadata
        metadata = {
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "max_tokens": kwargs.get("max_tokens"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty"),
        }

        # Add error info if failed
        if error:
            metadata["error_type"] = type(error).__name__
            metadata["error_message"] = str(error)

        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        receipt = VeratumReceipt(
            receipt_id=receipt_id,
            timestamp=timestamp,
            model=model,
            provider=provider,
            status=status,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_hash=request_hash,
            response_hash=response_hash,
            metadata=metadata or None,
        )

        return receipt

    def _upload_receipt(self, receipt: VeratumReceipt) -> bool:
        """
        Upload receipt to Veratum backend.

        Returns:
            True if upload successful, False otherwise
        """
        try:
            if self._engine:
                # Use EvidenceEngine if available
                self._engine.upload_evidence(receipt.to_dict())
                logger.debug(f"Receipt {receipt.receipt_id} uploaded via EvidenceEngine")
                return True
            else:
                # Fall back to direct API call
                self._upload_via_api(receipt)
                logger.debug(f"Receipt {receipt.receipt_id} uploaded via API")
                return True
        except Exception as e:
            logger.warning(f"Failed to upload receipt {receipt.receipt_id}: {e}")
            return False

    def _upload_via_api(self, receipt: VeratumReceipt) -> None:
        """
        Upload receipt directly via HTTP API (fallback).

        Requires api_key to be set.
        """
        if not self.api_key:
            logger.warning("No API key configured for Veratum receipt upload")
            return

        try:
            import requests
        except ImportError:
            logger.warning("requests library not available for API upload")
            return

        try:
            response = requests.post(
                f"{self.endpoint}/evidence/receipt",
                json=receipt.to_dict(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.upload_timeout,
            )
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"API upload failed: {e}")

    def _queue_receipt_for_upload(self, receipt: VeratumReceipt) -> None:
        """Add receipt to queue and flush if threshold reached."""
        if not self.queue_receipts:
            self._upload_receipt(receipt)
            return

        with self._queue_lock:
            self._receipt_queue.append(receipt)
            if len(self._receipt_queue) >= self.queue_size:
                self._flush_receipt_queue()

    def _flush_receipt_queue(self) -> None:
        """Upload all queued receipts (must be called with lock held)."""
        if not self._receipt_queue:
            return

        # Upload in background thread
        receipts_to_upload = self._receipt_queue.copy()
        self._receipt_queue.clear()

        def upload_batch():
            for receipt in receipts_to_upload:
                self._upload_receipt(receipt)

        upload_thread = threading.Thread(target=upload_batch, daemon=True)
        self._upload_threads.append(upload_thread)
        upload_thread.start()

        # Clean up finished threads periodically (keep list bounded)
        self._upload_threads = [t for t in self._upload_threads if t.is_alive()]

    # LiteLLM Callback Interface Methods

    def log_pre_api_call(
        self,
        model: str,
        messages: List[Dict],
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Called before the API call is made.

        Can be used for request validation, logging, etc.
        """
        logger.debug(
            f"LiteLLM pre-call: model={model}, messages={len(messages)}, "
            f"temperature={kwargs.get('temperature')}"
        )

    def log_post_api_call(
        self,
        kwargs: Dict[str, Any],
        response_obj: Optional[Any],
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Called after the API call completes (success or failure).

        Used for general logging/analytics.
        """
        model = kwargs.get("model", "unknown")
        latency = (end_time - start_time) * 1000
        logger.debug(f"LiteLLM post-call: model={model}, latency={latency:.0f}ms")

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Optional[Any],
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Called when an LLM call succeeds.

        Creates and uploads a SUCCESS receipt to Veratum.
        Logs errors with full traceback for debugging.
        """
        try:
            model = kwargs.get("model", "unknown")
            receipt = self._create_receipt(
                model=model,
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                status="SUCCESS",
            )
            logger.info(
                f"Created success receipt {receipt.receipt_id} for {model} "
                f"({receipt.latency_ms:.0f}ms)"
            )
            self._queue_receipt_for_upload(receipt)
        except Exception as e:
            logger.error(
                f"Failed to create success receipt: {e}",
                exc_info=True,
                stack_info=True
            )
            raise

    def log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Optional[Any],
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Called when an LLM call fails.

        Creates and uploads a FAILED receipt to Veratum.
        Failed AI calls are also evidence.
        Logs errors with full traceback for debugging.
        """
        try:
            model = kwargs.get("model", "unknown")
            error = None
            if isinstance(response_obj, Exception):
                error = response_obj

            receipt = self._create_receipt(
                model=model,
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                status="FAILED",
                error=error,
            )
            logger.warning(
                f"Created failure receipt {receipt.receipt_id} for {model} "
                f"({receipt.latency_ms:.0f}ms)"
            )
            self._queue_receipt_for_upload(receipt)
        except Exception as e:
            logger.error(
                f"Failed to create failure receipt: {e}",
                exc_info=True,
                stack_info=True
            )
            raise

    # Async versions for compatibility with litellm async callbacks

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Optional[Any],
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Async version of log_success_event.

        Runs sync version in executor to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.log_success_event, kwargs, response_obj, start_time, end_time
        )

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Optional[Any],
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Async version of log_failure_event.

        Runs sync version in executor to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.log_failure_event, kwargs, response_obj, start_time, end_time
        )


def enable_veratum(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    queue_receipts: bool = True,
    queue_size: int = 10,
    **kwargs,
) -> VeratumLiteLLMCallback:
    """
    One-liner to enable Veratum evidence on all LiteLLM calls.

    This is the easiest way to add Veratum evidence layer to your LiteLLM setup.

    Args:
        api_key: Veratum API key (optional if using EvidenceEngine)
        endpoint: Veratum API endpoint (optional, defaults to https://api.veratum.ai)
        queue_receipts: Batch receipts for more efficient uploading (default: True)
        queue_size: Number of receipts per batch (default: 10)
        **kwargs: Additional arguments passed to VeratumLiteLLMCallback

    Returns:
        VeratumLiteLLMCallback instance that was registered with litellm

    Example:
        >>> from veratum.integrations.litellm_plugin import enable_veratum
        >>> enable_veratum(api_key="your-key")
        >>> import litellm
        >>> response = litellm.completion(model="gpt-4o", messages=[...])
        >>> # Every call now creates a Veratum receipt automatically
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "LiteLLM is required. Install with: pip install litellm"
        )

    callback = VeratumLiteLLMCallback(
        api_key=api_key,
        endpoint=endpoint,
        queue_receipts=queue_receipts,
        queue_size=queue_size,
        **kwargs,
    )

    # Register callback with litellm
    if not hasattr(litellm, "callbacks"):
        litellm.callbacks = []
    litellm.callbacks.append(callback)

    logger.info(
        "Veratum evidence layer enabled for LiteLLM "
        f"(endpoint={endpoint or 'default'})"
    )

    return callback


__all__ = [
    "VeratumLiteLLMCallback",
    "VeratumReceipt",
    "ProviderType",
    "enable_veratum",
]
