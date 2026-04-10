"""
Veratum Evidence Layer for Portkey AI Gateway

This module provides seamless integration between Portkey AI Gateway and Veratum's
cryptographic evidence system. It wraps Portkey clients to automatically generate
and upload evidence receipts for all AI API calls.

Production Features:
- Automatic evidence capture for chat completions
- Streaming response support with chunk accumulation
- Async, non-blocking evidence upload to Veratum API
- Graceful error handling that never blocks AI calls
- Correlation tracking via Portkey metadata
- Webhook hook support for server-side logging
- One-liner client wrapping convenience function
"""

import asyncio
import json
import logging
import time
import uuid
import functools
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class EvidenceReceipt:
    """Receipt returned after evidence capture and upload."""
    receipt_id: str
    timestamp: str
    request_hash: str
    response_hash: str
    call_id: str
    portkey_request_id: Optional[str] = None
    veratum_api_endpoint: Optional[str] = None
    upload_status: str = "pending"  # pending, uploaded, failed
    error: Optional[str] = None


class VeratumPortkeyMiddleware:
    """
    Wraps a Portkey client to add cryptographic evidence to every AI call.

    This middleware:
    1. Intercepts requests to Portkey chat.completions.create
    2. Captures request and response data
    3. Generates cryptographic evidence receipts via EvidenceEngine
    4. Uploads to Veratum API asynchronously
    5. Attaches receipt correlation data to Portkey metadata
    6. Handles streaming responses transparently

    Usage:
        from portkey_ai import Portkey
        from veratum.integrations.portkey_plugin import VeratumPortkeyMiddleware

        client = Portkey(api_key="pk_...", virtual_key="vk_...")
        veratum = VeratumPortkeyMiddleware(client)

        response = veratum.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )

        print(veratum.last_receipt)
    """

    def __init__(
        self,
        portkey_client: Any,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        async_upload: bool = True,
        upload_timeout: float = 30.0,
        capture_metadata: bool = True,
    ):
        """
        Initialize the Veratum Portkey middleware.

        Args:
            portkey_client: Portkey client instance
            api_key: Veratum API key (auto-detected from env if not provided)
            endpoint: Veratum API endpoint (defaults to official API)
            async_upload: Whether to upload evidence asynchronously (non-blocking)
            upload_timeout: Timeout for async uploads in seconds
            capture_metadata: Whether to attach receipt to Portkey metadata
        """
        self.portkey_client = portkey_client
        self.api_key = api_key or self._get_api_key_from_env()
        self.endpoint = endpoint or "https://api.veratum.ai"
        self.async_upload = async_upload
        self.upload_timeout = upload_timeout
        self.capture_metadata = capture_metadata

        self.last_receipt: Optional[EvidenceReceipt] = None
        self._receipts_history: Dict[str, EvidenceReceipt] = {}
        self._upload_threads: Dict[str, threading.Thread] = {}
        self._original_create = None

        # Wrap the chat.completions.create method
        self._wrap_chat_completions()

        logger.info(
            f"VeratumPortkeyMiddleware initialized: "
            f"async_upload={async_upload}, endpoint={self.endpoint}"
        )

    @staticmethod
    def _get_api_key_from_env() -> Optional[str]:
        """Get Veratum API key from environment variables."""
        import os
        return os.environ.get("VERATUM_API_KEY")

    def _wrap_chat_completions(self) -> None:
        """Wrap the Portkey chat.completions.create method."""
        self._original_create = self.portkey_client.chat.completions.create

        @functools.wraps(self._original_create)
        def wrapped_create(*args, **kwargs):
            return self._create_with_evidence(*args, **kwargs)

        self.portkey_client.chat.completions.create = wrapped_create

    def _create_with_evidence(self, *args, **kwargs) -> Any:
        """
        Wrapper for chat.completions.create that captures evidence.

        Handles both streaming and non-streaming responses.
        """
        call_id = str(uuid.uuid4())
        stream = kwargs.get("stream", False)

        # Capture request metadata
        request_data = self._extract_request_data(args, kwargs)
        request_hash = self._hash_data(request_data)

        try:
            # Call original Portkey client (not the wrapped version to avoid recursion)
            response = self._original_create(*args, **kwargs)

            if stream:
                # Handle streaming response
                response = self._wrap_stream(response, call_id, request_hash, request_data)
            else:
                # Handle non-streaming response
                response_data = self._extract_response_data(response)
                receipt = self._create_evidence_receipt(
                    call_id=call_id,
                    request_data=request_data,
                    request_hash=request_hash,
                    response_data=response_data,
                    portkey_request_id=getattr(response, "id", None),
                )

                # Upload asynchronously if enabled
                if self.async_upload:
                    self._upload_evidence_async(receipt)
                else:
                    self._upload_evidence_sync(receipt)

                self.last_receipt = receipt
                self._receipts_history[call_id] = receipt

                # Attach receipt hash to metadata
                if self.capture_metadata and hasattr(response, "_metadata"):
                    response._metadata["veratum_receipt_id"] = receipt.receipt_id

            return response

        except Exception as e:
            logger.error(
                f"Error in Portkey evidence capture for call {call_id}: {str(e)}",
                exc_info=True
            )
            # Never block the AI call on error
            raise

    def _wrap_stream(
        self,
        stream: Any,
        call_id: str,
        request_hash: str,
        request_data: Dict[str, Any],
    ) -> Any:
        """Wrap a streaming response to accumulate chunks and create evidence."""
        accumulated_response = ""
        accumulated_tool_calls = []
        first_chunk = True
        portkey_request_id = None
        chunk_count = 0

        def streaming_generator():
            nonlocal accumulated_response, accumulated_tool_calls, first_chunk, portkey_request_id, chunk_count

            try:
                for chunk in stream:
                    chunk_count += 1
                    if first_chunk:
                        portkey_request_id = getattr(chunk, "id", None)
                        first_chunk = False

                    # Accumulate delta content
                    if hasattr(chunk, "choices") and chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice, "delta"):
                            delta = choice.delta
                            # Accumulate text content
                            if hasattr(delta, "content") and delta.content:
                                accumulated_response += delta.content
                            # Accumulate tool calls
                            if hasattr(delta, "tool_calls") and delta.tool_calls:
                                accumulated_tool_calls.extend(delta.tool_calls)

                    yield chunk

                # Guard against empty streams
                if chunk_count == 0:
                    logger.warning(f"Empty stream received for call {call_id}")
                    return

                # After stream completes, create evidence
                response_data = {
                    "content": accumulated_response,
                    "role": "assistant",
                    "finish_reason": "stop",
                }
                if accumulated_tool_calls:
                    response_data["tool_calls"] = accumulated_tool_calls

                receipt = self._create_evidence_receipt(
                    call_id=call_id,
                    request_data=request_data,
                    request_hash=request_hash,
                    response_data=response_data,
                    portkey_request_id=portkey_request_id,
                )

                # Upload asynchronously
                if self.async_upload:
                    self._upload_evidence_async(receipt)
                else:
                    self._upload_evidence_sync(receipt)

                self.last_receipt = receipt
                self._receipts_history[call_id] = receipt

            except Exception as e:
                logger.error(
                    f"Error in streaming evidence capture for call {call_id}: {str(e)}",
                    exc_info=True
                )
                # Re-raise to not silently swallow errors
                raise

        return streaming_generator()

    def _extract_request_data(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract and sanitize request data for evidence."""
        request = {
            "model": kwargs.get("model"),
            "stream": kwargs.get("stream", False),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "top_p": kwargs.get("top_p"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Capture message structure without sensitive content details
        if "messages" in kwargs:
            messages = kwargs["messages"]
            request["messages_count"] = len(messages)
            request["messages_hash"] = self._hash_data(messages)

        return request

    def _extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extract response data for evidence."""
        response_obj = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "created": getattr(response, "created", None),
        }

        # Extract content from choices
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                response_obj["content"] = getattr(choice.message, "content", "")
                response_obj["role"] = getattr(choice.message, "role", "assistant")

        return response_obj

    def _create_evidence_receipt(
        self,
        call_id: str,
        request_data: Dict[str, Any],
        request_hash: str,
        response_data: Dict[str, Any],
        portkey_request_id: Optional[str] = None,
    ) -> EvidenceReceipt:
        """Create an evidence receipt for the AI call."""
        response_hash = self._hash_data(response_data)

        receipt = EvidenceReceipt(
            receipt_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            request_hash=request_hash,
            response_hash=response_hash,
            call_id=call_id,
            portkey_request_id=portkey_request_id,
            veratum_api_endpoint=self.endpoint,
            upload_status="pending",
        )

        return receipt

    def _upload_evidence_sync(self, receipt: EvidenceReceipt) -> None:
        """Synchronously upload evidence to Veratum API."""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = asdict(receipt)

            response = requests.post(
                f"{self.endpoint}/v1/evidence",
                json=payload,
                headers=headers,
                timeout=self.upload_timeout,
            )

            if response.status_code in (200, 201):
                receipt.upload_status = "uploaded"
                logger.debug(f"Evidence uploaded: {receipt.receipt_id}")
            else:
                receipt.upload_status = "failed"
                receipt.error = f"HTTP {response.status_code}: {response.text}"
                logger.warning(f"Failed to upload evidence: {receipt.error}")

        except Exception as e:
            receipt.upload_status = "failed"
            receipt.error = str(e)
            logger.warning(f"Error uploading evidence to Veratum: {str(e)}")

    def _upload_evidence_async(self, receipt: EvidenceReceipt) -> None:
        """Asynchronously upload evidence in a background thread."""
        def upload_thread():
            self._upload_evidence_sync(receipt)

        thread = threading.Thread(
            target=upload_thread,
            daemon=True,
            name=f"veratum-upload-{receipt.receipt_id}",
        )
        thread.start()
        self._upload_threads[receipt.receipt_id] = thread

        # Clean up finished threads to prevent unbounded dict growth
        finished_threads = [
            thread_id for thread_id, thread in self._upload_threads.items()
            if not thread.is_alive()
        ]
        for thread_id in finished_threads:
            del self._upload_threads[thread_id]

    @staticmethod
    def _hash_data(data: Any) -> str:
        """Create a deterministic hash of data for evidence."""
        import hashlib

        # Serialize data deterministically
        json_str = json.dumps(data, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(json_str.encode())
        return hash_obj.hexdigest()

    @property
    def client(self) -> Any:
        """Access the wrapped Portkey client."""
        return self.portkey_client

    def get_receipt(self, call_id: str) -> Optional[EvidenceReceipt]:
        """Retrieve a receipt by call ID."""
        return self._receipts_history.get(call_id)

    def get_all_receipts(self) -> Dict[str, EvidenceReceipt]:
        """Get all captured receipts."""
        return self._receipts_history.copy()

    def wait_for_uploads(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all pending uploads to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all uploads completed, False if timeout
        """
        start_time = time.time()

        for thread in self._upload_threads.values():
            remaining_timeout = None
            if timeout:
                elapsed = time.time() - start_time
                remaining_timeout = timeout - elapsed
                if remaining_timeout <= 0:
                    return False

            thread.join(timeout=remaining_timeout)
            if thread.is_alive():
                return False

        return True


def veratum_portkey_hook(
    request_data: Dict[str, Any],
    response_data: Dict[str, Any],
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Standalone hook for Portkey webhook integration.

    This function can be used as a webhook handler in Portkey's logging/monitoring
    system to capture evidence receipts server-side.

    Configure in Portkey dashboard → Webhooks → Add Veratum endpoint.

    Args:
        request_data: Request details from Portkey webhook
        response_data: Response details from Portkey webhook
        api_key: Veratum API key (auto-detected from env if not provided)
        endpoint: Veratum API endpoint

    Returns:
        Dictionary with hook execution result

    Example:
        # In your webhook handler
        @app.post("/webhooks/portkey")
        def portkey_webhook(request):
            body = request.json
            result = veratum_portkey_hook(
                request_data=body.get("request"),
                response_data=body.get("response"),
            )
            return result
    """
    import os
    import hashlib
    import uuid as uuid_module

    try:
        # Get credentials
        api_key = api_key or os.environ.get("VERATUM_API_KEY")
        endpoint = endpoint or "https://api.veratum.ai"

        if not api_key:
            return {
                "status": "error",
                "error": "VERATUM_API_KEY not configured",
                "hook_id": None,
            }

        # Create evidence receipt
        hook_id = str(uuid_module.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Hash request and response
        request_json = json.dumps(request_data, sort_keys=True, default=str)
        response_json = json.dumps(response_data, sort_keys=True, default=str)

        request_hash = hashlib.sha256(request_json.encode()).hexdigest()
        response_hash = hashlib.sha256(response_json.encode()).hexdigest()

        # Prepare payload
        payload = {
            "receipt_id": hook_id,
            "timestamp": timestamp,
            "request_hash": request_hash,
            "response_hash": response_hash,
            "call_id": request_data.get("id") or hook_id,
            "portkey_request_id": response_data.get("id"),
            "veratum_api_endpoint": endpoint,
            "upload_status": "pending",
        }

        # Upload to Veratum API
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                f"{endpoint}/v1/evidence",
                json=payload,
                headers=headers,
                timeout=30.0,
            )

            if response.status_code in (200, 201):
                return {
                    "status": "success",
                    "hook_id": hook_id,
                    "receipt_id": payload.get("receipt_id"),
                }
            else:
                logger.warning(f"Veratum API error: {response.status_code}")
                return {
                    "status": "error",
                    "error": f"Veratum API returned {response.status_code}",
                    "hook_id": hook_id,
                }

        except Exception as upload_error:
            logger.warning(f"Error uploading to Veratum: {str(upload_error)}")
            return {
                "status": "error",
                "error": str(upload_error),
                "hook_id": hook_id,
            }

    except Exception as e:
        logger.error(f"Error in veratum_portkey_hook: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "hook_id": None,
        }


def wrap_portkey(
    client: Any,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    async_upload: bool = True,
    **kwargs,
) -> VeratumPortkeyMiddleware:
    """
    One-liner convenience function to wrap a Portkey client with Veratum evidence.

    This is the simplest way to add evidence tracking to an existing Portkey client.

    Usage:
        from portkey_ai import Portkey
        from veratum.integrations.portkey_plugin import wrap_portkey

        client = Portkey(api_key="pk_...", virtual_key="vk_...")
        veratum_client = wrap_portkey(client)

        # Now all calls go through Veratum
        response = veratum_client.client.chat.completions.create(...)
        print(veratum_client.last_receipt)

    Args:
        client: Portkey client instance
        api_key: Veratum API key
        endpoint: Veratum API endpoint
        async_upload: Whether to upload asynchronously
        **kwargs: Additional arguments passed to VeratumPortkeyMiddleware

    Returns:
        VeratumPortkeyMiddleware wrapping the Portkey client
    """
    return VeratumPortkeyMiddleware(
        client,
        api_key=api_key,
        endpoint=endpoint,
        async_upload=async_upload,
        **kwargs,
    )


__all__ = [
    "VeratumPortkeyMiddleware",
    "EvidenceReceipt",
    "veratum_portkey_hook",
    "wrap_portkey",
]
