"""Veratum SDK for AI auditability and accountability."""

import functools
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, TypeVar

import requests

from .buffer import ReceiptBuffer
from ..crypto.chain import HashChain
from .receipt import Receipt
from .tiers import AuditLevel, apply_preset, get_audit_level
from ..providers import auto_wrap, detect_provider, WrapConfig, UnsupportedProviderError

logger = logging.getLogger("veratum.sdk")

T = TypeVar("T")


class VeratumSDK:
    """
    Veratum SDK for auditing and monitoring AI model interactions.

    Intercepts API calls to LLM providers, captures prompts and responses,
    generates audit receipts with full chain integrity, and uploads them
    to the Veratum endpoint for compliance and auditability.

    Example:
        >>> from veratum import VeratumSDK
        >>> import anthropic
        >>>
        >>> sdk = VeratumSDK(
        ...     endpoint="https://api.veratum.ai/v1",
        ...     api_key="vsk_...",
        ...     vertical="hiring"
        ... )
        >>> client = anthropic.Anthropic(api_key="sk_...")
        >>> wrapped_client = sdk.wrap(client)
        >>> response = wrapped_client.messages.create(
        ...     model="claude-3-opus-20250219",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        vertical: Optional[str] = None,
        timeout: float = 30.0,
        auto_timestamp: bool = True,
        *,
        buffered: bool = True,
        buffer_size: int = 2048,
        flush_interval: float = 1.0,
        max_retries: int = 5,
        audit_level: Optional[AuditLevel] = None,
        zk_prover: Optional[Any] = None,
        policy_engine: Optional[Any] = None,
    ) -> None:
        """
        Initialize Veratum SDK.

        Args:
            endpoint: Veratum API endpoint (default: auto-loads from VERATUM_ENDPOINT env var,
                     or 'https://api.veratum.ai/v1' if not set)
            api_key: Veratum API key (default: auto-loads from VERATUM_API_KEY env var,
                    required if not provided)
            vertical: Industry vertical for classification (default: auto-loads from VERATUM_VERTICAL,
                     or 'general' if not set)
            timeout: Request timeout in seconds (default: 30)
            auto_timestamp: Automatically request timestamp after receipt upload (default: True)
            buffered: Enable async upload buffer with retry (default: True).
                      Set False for synchronous uploads (testing, debugging).
            buffer_size: Maximum receipts to buffer in memory (default: 2048).
            flush_interval: Seconds between background flush cycles (default: 1.0).
            max_retries: Maximum retry attempts per receipt (default: 5).
            audit_level: Force audit level (default: auto-detect from vertical).
            zk_prover: Optional VeratumZKProver instance for Zero-Knowledge proof
                       generation. When set, ZK proofs are generated asynchronously
                       in a background thread after each decision is captured.
                       Install with: pip install veratum[zk]
            policy_engine: Optional VeratumPolicyEngine instance for real-time
                          decision prevention. When set, every AI decision is
                          evaluated against configured policies before taking
                          effect. Blocked decisions create "BLOCKED" receipts.

        Raises:
            ValueError: If api_key is not provided and VERATUM_API_KEY env var is not set
        """
        # Auto-load endpoint from environment or use default
        if endpoint is None:
            endpoint = os.environ.get("VERATUM_ENDPOINT", "https://api.veratum.ai/v1")

        # Auto-load api_key from environment (required)
        if api_key is None:
            api_key = os.environ.get("VERATUM_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key must be provided as argument or set via VERATUM_API_KEY environment variable"
                )

        # Auto-load vertical from environment or use default
        if vertical is None:
            vertical = os.environ.get("VERATUM_VERTICAL", "general")

        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("endpoint must be a non-empty string")
        allow_http = os.environ.get("VERATUM_ALLOW_HTTP", "0") == "1"
        if not endpoint.startswith("https://") and not allow_http:
            raise ValueError(
                "endpoint must use HTTPS (got http://). "
                "Veratum SDK requires TLS to protect API keys and receipt data in transit. "
                "Set VERATUM_ALLOW_HTTP=1 for local development only."
            )
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string")

        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.vertical = vertical
        self.timeout = timeout
        self.auto_timestamp = auto_timestamp
        self.audit_level = audit_level or get_audit_level(vertical)
        self.buffered = buffered

        self._hash_chain = HashChain()
        self._receipt_generator = Receipt(self._hash_chain)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "veratum-sdk/2.0.0",
            }
        )

        # Initialize upload buffer (async retry with circuit breaker)
        self._buffer: Optional[ReceiptBuffer] = None
        if buffered:
            self._buffer = ReceiptBuffer(
                upload_fn=self._upload_receipt_sync,
                max_queue_size=buffer_size,
                flush_interval=flush_interval,
                max_retries=max_retries,
            )

        # ZK proof support (optional)
        self._zk_prover = zk_prover
        self._zk_pending: Dict[str, Dict[str, Any]] = {}  # entry_hash -> input/output data

        # Policy prevention engine (optional)
        self._policy_engine = policy_engine

    def wrap(self, client: Any, provider: Optional[str] = None) -> Any:
        """
        Wrap an AI client to intercept and audit API calls.

        Auto-detects the LLM provider (OpenAI, Anthropic, Google, Mistral, Cohere, AWS Bedrock)
        and applies the appropriate wrapper. Monkey-patches the client's methods to:
        1. Capture prompt and response
        2. Generate audit receipt with chain integrity
        3. Upload receipt to Veratum endpoint
        4. Return original response transparently

        Supports:
        - OpenAI (openai.OpenAI, openai.AsyncOpenAI)
        - Anthropic (anthropic.Anthropic, anthropic.AsyncAnthropic)
        - Google GenerativeAI (google.generativeai.GenerativeModel)
        - Mistral AI (mistralai.Mistral)
        - Cohere (cohere.Client)
        - AWS Bedrock (boto3 bedrock-runtime client)

        Args:
            client: LLM client instance (e.g., anthropic.Anthropic, openai.OpenAI)
            provider: Optional provider name to skip auto-detection.
                     If not provided, will auto-detect from client class.

        Returns:
            Wrapped client with same interface as original

        Raises:
            UnsupportedProviderError: If provider cannot be detected or is unsupported
        """
        # Detect provider if not explicitly provided
        if provider is None:
            provider = detect_provider(client)

        if provider == "unknown":
            raise UnsupportedProviderError(
                f"Unable to detect LLM provider for client {client.__class__.__name__} "
                f"from module {client.__class__.__module__}. "
                f"Supported providers: openai, anthropic, google, mistral, cohere, bedrock"
            )

        # Create receipt callback that will be invoked by the provider wrapper
        def receipt_callback(
            model: str,
            prompt: Optional[str],
            response: Optional[str],
            tokens_in: int,
            tokens_out: int,
            provider_name: str,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            """Callback invoked by provider wrapper to generate and upload receipt."""
            self._create_and_upload_receipt(
                prompt=prompt or "",
                response=response or "",
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=metadata.get("duration_ms", 0) if metadata else 0,
                metadata=metadata,
                provider=provider_name,
            )

        # Apply provider-specific wrapper
        wrap_config = WrapConfig(
            capture_prompts=True,
            capture_responses=True,
            max_prompt_length=10000,
            max_response_length=10000,
            metadata={},
        )

        try:
            auto_wrap(client, receipt_callback, wrap_config)
        except Exception as e:
            self._log_error(f"Failed to wrap client: {str(e)}")
            raise

        logger.info(f"Successfully wrapped {provider} client for auditing")
        return client

    def _extract_prompt_from_messages(self, messages: Any) -> str:
        """
        Extract prompt text from messages structure.

        Args:
            messages: Messages list or structure

        Returns:
            Concatenated prompt text
        """
        if not messages:
            return ""

        try:
            if isinstance(messages, list):
                prompt_parts = []
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        content = msg["content"]
                        if isinstance(content, str):
                            prompt_parts.append(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    prompt_parts.append(item["text"])
                return "\n".join(prompt_parts)
        except Exception:
            pass

        return str(messages)[:1000]  # Fallback: truncate to 1000 chars

    def _extract_response_text(self, response: Any) -> str:
        """
        Extract response text from response object.

        Args:
            response: Response object from API

        Returns:
            Response text
        """
        try:
            # Anthropic format: response.content[0].text
            if hasattr(response, "content") and isinstance(response.content, list):
                if len(response.content) > 0:
                    content = response.content[0]
                    if hasattr(content, "text"):
                        return content.text
            # Fallback: check for text attribute
            if hasattr(response, "text"):
                return response.text
        except Exception:
            pass

        return str(response)[:1000]  # Fallback: truncate to 1000 chars

    def _create_and_upload_receipt(
        self,
        prompt: str,
        response: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
        provider: str = "anthropic",
    ) -> bool:
        """
        Generate and upload audit receipt.

        Args:
            prompt: Input prompt text
            response: Model response text
            model: Model identifier
            tokens_in: Input tokens
            tokens_out: Output tokens
            duration_ms: Request duration in milliseconds
            metadata: Additional metadata
            provider: LLM provider name (default: anthropic)

        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Add ZK status to metadata if ZK is enabled
            extra_metadata = {
                **(metadata or {}),
                "duration_ms": duration_ms,
            }

            # Generate receipt with all Article 12 / ISO 24970 fields
            receipt = self._receipt_generator.generate(
                prompt=prompt,
                response=response,
                model=model,
                provider=provider,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                decision_type="content_generation",
                vertical=self.vertical,
                metadata=extra_metadata,
            )

            # If ZK enabled, set zk_status to pending
            if self._zk_prover:
                receipt["zk_status"] = "pending"
                receipt["zk_framework"] = "ezkl"

            # Upload receipt
            success = self._upload_receipt(receipt)

            # Request timestamp if auto_timestamp is enabled
            if success and self.auto_timestamp:
                entry_hash = receipt.get("entry_hash")
                if entry_hash:
                    self.request_timestamp(entry_hash)

            # Trigger async ZK proof generation if enabled
            if success and self._zk_prover:
                entry_hash = receipt.get("entry_hash")
                if entry_hash:
                    self._zk_pending[entry_hash] = True
                    self._generate_zk_proof_async(
                        entry_hash=entry_hash,
                        input_data={"input_data": [[hash(prompt) % 1000 / 1000]]},
                        output_data={"output_data": [[hash(response) % 1000 / 1000]]},
                    )

            return success

        except Exception as e:
            self._log_error(f"Receipt creation failed: {str(e)}")
            return False

    def _upload_receipt(self, receipt: Dict[str, Any]) -> bool:
        """
        Upload receipt — routes through buffer if enabled, else sync.

        Args:
            receipt: Receipt dictionary

        Returns:
            True if enqueued/uploaded successfully, False otherwise
        """
        if self._buffer is not None:
            return self._buffer.put(receipt)
        return self._upload_receipt_sync(receipt)

    def _upload_receipt_sync(self, receipt: Dict[str, Any]) -> bool:
        """
        Upload receipt to Veratum endpoint synchronously.

        Args:
            receipt: Receipt dictionary

        Returns:
            True if upload successful, False otherwise
        """
        try:
            url = f"{self.endpoint}/receipts"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = self._session.post(
                url,
                json=receipt,
                headers=headers,
                timeout=self.timeout,
            )

            # Accept 2xx status codes
            if 200 <= response.status_code < 300:
                return True
            else:
                self._log_error(
                    f"Receipt upload failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return False

        except requests.Timeout:
            self._log_error("Receipt upload timeout")
            return False
        except Exception as e:
            self._log_error(f"Receipt upload error: {str(e)}")
            return False

    def flush(self, timeout: float = 5.0) -> int:
        """
        Flush buffered receipts synchronously.

        Critical for Lambda/serverless — call before handler returns.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            Number of receipts remaining in buffer.
        """
        if self._buffer is not None:
            return self._buffer.flush(timeout=timeout)
        return 0

    def buffer_stats(self) -> Dict[str, Any]:
        """Get upload buffer statistics."""
        if self._buffer is not None:
            return self._buffer.stats()
        return {"buffered": False}

    def _log_error(self, message: str) -> None:
        """
        Log error message.

        Args:
            message: Error message
        """
        # Silent logging by default (can be extended with proper logging)
        print(f"[Veratum] {message}")

    def get_chain_state(self) -> Dict[str, Any]:
        """
        Get current hash chain state.

        Returns:
            Dictionary with sequence_no and prev_hash
        """
        return self._hash_chain.get_chain_state()

    def reset_chain(self) -> None:
        """Reset hash chain to genesis state."""
        self._hash_chain.reset()

    def request_timestamp(self, entry_hash: str) -> Dict[str, Any]:
        """
        Request an RFC 3161 timestamp for an entry hash.

        Args:
            entry_hash: The entry hash to timestamp

        Returns:
            Timestamp token dictionary
        """
        try:
            url = f"{self.endpoint}/timestamps"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = self._session.post(
                url,
                json={"entry_hash": entry_hash},
                headers=headers,
                timeout=self.timeout,
            )

            if 200 <= response.status_code < 300:
                return response.json()
            else:
                self._log_error(
                    f"Timestamp request failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return {}

        except requests.Timeout:
            self._log_error("Timestamp request timeout")
            return {}
        except Exception as e:
            self._log_error(f"Timestamp request error: {str(e)}")
            return {}

    def submit_for_review(
        self,
        entry_hash: str,
        review_type: str = "optional",
        priority: str = "medium",
    ) -> Dict[str, Any]:
        """
        Submit an entry for human review.

        Args:
            entry_hash: The entry hash to review
            review_type: Type of review ('optional', 'mandatory', 'escalation')
            priority: Priority level ('low', 'medium', 'high')

        Returns:
            Review submission response
        """
        try:
            url = f"{self.endpoint}/reviews"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = self._session.post(
                url,
                json={
                    "entry_hash": entry_hash,
                    "review_type": review_type,
                    "priority": priority,
                },
                headers=headers,
                timeout=self.timeout,
            )

            if 200 <= response.status_code < 300:
                return response.json()
            else:
                self._log_error(
                    f"Review submission failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return {}

        except requests.Timeout:
            self._log_error("Review submission timeout")
            return {}
        except Exception as e:
            self._log_error(f"Review submission error: {str(e)}")
            return {}

    def get_pending_reviews(self) -> list:
        """
        Get all pending reviews.

        Returns:
            List of pending review objects
        """
        try:
            url = f"{self.endpoint}/reviews"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = self._session.get(
                url,
                headers=headers,
                timeout=self.timeout,
            )

            if 200 <= response.status_code < 300:
                data = response.json()
                return data.get("reviews", []) if isinstance(data, dict) else data
            else:
                self._log_error(
                    f"Get pending reviews failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return []

        except requests.Timeout:
            self._log_error("Get pending reviews timeout")
            return []
        except Exception as e:
            self._log_error(f"Get pending reviews error: {str(e)}")
            return []

    def submit_review_decision(
        self,
        review_id: str,
        decision: str,
        reviewer_id: str,
        reason: str = "",
        override_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Submit a decision on a pending review.

        Args:
            review_id: The review ID
            decision: Decision ('approved', 'rejected', 'escalated')
            reviewer_id: ID of the reviewer making the decision
            reason: Optional reason for the decision
            override_score: Optional score override (0-1)

        Returns:
            Decision submission response
        """
        try:
            url = f"{self.endpoint}/reviews/{review_id}/decision"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload: Dict[str, Any] = {
                "decision": decision,
                "reviewer_id": reviewer_id,
            }

            if reason:
                payload["reason"] = reason
            if override_score is not None:
                payload["override_score"] = override_score

            response = self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            if 200 <= response.status_code < 300:
                return response.json()
            else:
                self._log_error(
                    f"Review decision submission failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return {}

        except requests.Timeout:
            self._log_error("Review decision submission timeout")
            return {}
        except Exception as e:
            self._log_error(f"Review decision submission error: {str(e)}")
            return {}

    def _generate_zk_proof_async(
        self, entry_hash: str, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> None:
        """
        Generate ZK proof in a background thread and update the receipt.

        The proof is generated locally — input data never leaves the machine.
        Only the proof hash is sent to Veratum via PATCH.
        """
        if not self._zk_prover:
            return

        def _prove() -> None:
            try:
                proof = self._zk_prover.prove(
                    input_data=input_data,
                    output_data=output_data,
                )
                # Submit proof hash to Veratum (proof stays local)
                zk_fields = proof.to_receipt_fields()
                zk_fields["entry_hash"] = entry_hash
                self._session.patch(
                    f"{self.endpoint}/receipts/zk-update",
                    json=zk_fields,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout,
                )
                logger.info("ZK proof submitted for %s", entry_hash[:16])
            except Exception as e:
                logger.error("ZK proof generation failed for %s: %s", entry_hash[:16], e)
                # Mark as failed
                try:
                    self._session.patch(
                        f"{self.endpoint}/receipts/zk-update",
                        json={"entry_hash": entry_hash, "zk_status": "failed"},
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=10.0,
                    )
                except Exception:
                    pass
            finally:
                self._zk_pending.pop(entry_hash, None)

        thread = threading.Thread(target=_prove, daemon=True, name=f"veratum-zk-{entry_hash[:8]}")
        thread.start()

    def evaluate_decision(
        self,
        decision: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Evaluate a decision against active policies.

        Use this for manual policy checking before acting on an AI decision.
        If the policy engine is not configured, returns None.

        Args:
            decision: Decision to evaluate (score, outcome, decision_type, etc.)
            context: Additional context (vertical, jurisdiction, etc.)

        Returns:
            PolicyResult if engine is configured, None otherwise.

        Raises:
            PolicyViolationError: If decision is blocked (only in strict mode).
        """
        if not self._policy_engine:
            return None

        result = self._policy_engine.evaluate(decision, context)

        if not result.allowed:
            # Create a blocked receipt
            self._create_blocked_receipt(decision, context or {}, result)

        return result

    def _create_blocked_receipt(
        self,
        decision: Dict[str, Any],
        context: Dict[str, Any],
        policy_result: Any,
    ) -> None:
        """Create a receipt for a blocked decision."""
        try:
            import hashlib
            import json as _json

            blocked_data = {
                "decision": decision,
                "policy_result": policy_result.result,
                "blocked_reason": policy_result.blocked_reason,
            }
            input_hash = hashlib.sha256(
                _json.dumps(decision, sort_keys=True, default=str).encode()
            ).hexdigest()

            receipt = self._receipt_generator.generate(
                prompt=f"BLOCKED: {policy_result.blocked_reason}",
                response="Decision prevented by policy engine",
                model=decision.get("model", "unknown"),
                provider="policy_engine",
                tokens_in=0,
                tokens_out=0,
                decision_type="BLOCKED",
                vertical=context.get("vertical", self.vertical),
                metadata={
                    "blocked_reason": policy_result.blocked_reason,
                    "original_decision_type": decision.get("decision_type"),
                    "original_outcome": decision.get("outcome"),
                    "risk_score": policy_result.risk_score,
                },
            )

            # Add policy evaluation
            receipt["policy_evaluation"] = policy_result.to_receipt_fields()["policy_evaluation"]
            receipt["human_review_state"] = "required"
            receipt["override_reason"] = policy_result.blocked_reason

            self._upload_receipt(receipt)
            logger.info(
                "BLOCKED receipt created: %s — %s",
                receipt.get("entry_hash", "?")[:16],
                policy_result.blocked_reason,
            )

        except Exception as e:
            logger.error("Failed to create blocked receipt: %s", e)

    @property
    def policy_enabled(self) -> bool:
        """Whether policy prevention is enabled."""
        return self._policy_engine is not None

    @property
    def zk_enabled(self) -> bool:
        """Whether ZK proof generation is enabled."""
        return self._zk_prover is not None

    @property
    def zk_pending_count(self) -> int:
        """Number of ZK proofs currently being generated."""
        return len(self._zk_pending)

    def close(self) -> None:
        """Close SDK resources and flush buffered receipts."""
        if self._buffer is not None:
            self._buffer.shutdown(timeout=5.0)
        self._session.close()

    def __enter__(self) -> "VeratumSDK":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


# Module-level convenience function
_default_sdk: Optional[VeratumSDK] = None


def wrap(client: Any, **kwargs: Any) -> Any:
    """Wrap an AI client with Veratum auditing. Auto-configures from environment variables.

    This is the simplest way to use Veratum:
        import veratum
        client = veratum.wrap(your_llm_client)

    Set VERATUM_API_KEY environment variable before use.
    Optional: VERATUM_ENDPOINT, VERATUM_VERTICAL

    Args:
        client: LLM client instance (e.g., anthropic.Anthropic)
        **kwargs: Additional arguments to pass to VeratumSDK (e.g., audit_level, timeout)

    Returns:
        Wrapped client with same interface as original

    Raises:
        ValueError: If VERATUM_API_KEY environment variable is not set
    """
    global _default_sdk
    if _default_sdk is None:
        _default_sdk = VeratumSDK(**kwargs)
    return _default_sdk.wrap(client)
