"""Auto-instrumentation for LLM providers.

Provides @veratum.audit() decorator and provider-agnostic wrapping for:
- Anthropic (messages.create)
- OpenAI (chat.completions.create)
- AWS Bedrock (invoke_model)
- Generic callable (any function returning text)

Compliance: EU AI Act Art.12 (logging), Art.14 (human oversight hooks),
Colorado SB24-205 s5 (impact assessment data capture),
GDPR Art.30 (processing records).
"""

import functools
import time
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger("veratum.instrument")

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_BEDROCK = "bedrock"
PROVIDER_GENERIC = "generic"


def _detect_provider(client: Any) -> str:
    """Detect LLM provider from client object."""
    module = type(client).__module__ or ""
    cls_name = type(client).__name__ or ""

    if "anthropic" in module.lower() or "Anthropic" in cls_name:
        return PROVIDER_ANTHROPIC
    if "openai" in module.lower() or "OpenAI" in cls_name:
        return PROVIDER_OPENAI
    if "botocore" in module.lower() or "bedrock" in cls_name.lower():
        return PROVIDER_BEDROCK
    return PROVIDER_GENERIC


# ---------------------------------------------------------------------------
# Response extractors — one per provider
# ---------------------------------------------------------------------------

def _extract_anthropic_response(response: Any) -> Dict[str, Any]:
    """Extract text, tokens from Anthropic response."""
    text = ""
    tokens_in = 0
    tokens_out = 0
    model = "unknown"

    try:
        if hasattr(response, "content") and isinstance(response.content, list):
            parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            text = "\n".join(parts)
        if hasattr(response, "usage"):
            tokens_in = getattr(response.usage, "input_tokens", 0)
            tokens_out = getattr(response.usage, "output_tokens", 0)
        if hasattr(response, "model"):
            model = response.model
    except Exception:
        text = str(response)[:2000]

    return {
        "text": text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model": model,
    }


def _extract_openai_response(response: Any) -> Dict[str, Any]:
    """Extract text, tokens from OpenAI ChatCompletion response."""
    text = ""
    tokens_in = 0
    tokens_out = 0
    model = "unknown"

    try:
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            text = getattr(msg, "content", "") or ""
        if hasattr(response, "usage") and response.usage:
            tokens_in = getattr(response.usage, "prompt_tokens", 0)
            tokens_out = getattr(response.usage, "completion_tokens", 0)
        if hasattr(response, "model"):
            model = response.model
    except Exception:
        text = str(response)[:2000]

    return {
        "text": text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model": model,
    }


def _extract_bedrock_response(response: Any, model_id: str = "unknown") -> Dict[str, Any]:
    """Extract text, tokens from Bedrock invoke_model response."""
    import json as _json

    text = ""
    tokens_in = 0
    tokens_out = 0

    try:
        if isinstance(response, dict):
            body = response.get("body")
            if hasattr(body, "read"):
                raw = body.read()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                parsed = _json.loads(raw)
            elif isinstance(body, str):
                parsed = _json.loads(body)
            else:
                parsed = body or {}

            # Anthropic Bedrock format
            if "content" in parsed and isinstance(parsed["content"], list):
                parts = [b.get("text", "") for b in parsed["content"] if isinstance(b, dict)]
                text = "\n".join(parts)
            # Amazon Titan format
            elif "results" in parsed:
                text = parsed["results"][0].get("outputText", "")
            # Cohere format
            elif "generations" in parsed:
                text = parsed["generations"][0].get("text", "")
            # AI21 format
            elif "completions" in parsed:
                text = parsed["completions"][0].get("data", {}).get("text", "")
            else:
                text = str(parsed)[:2000]

            # Token usage (Anthropic Bedrock)
            usage = parsed.get("usage", {})
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
    except Exception:
        text = str(response)[:2000]

    return {
        "text": text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model": model_id,
    }


# ---------------------------------------------------------------------------
# Prompt extractors
# ---------------------------------------------------------------------------

def _extract_prompt_from_messages(messages: Any) -> str:
    """Extract prompt text from a messages list (Anthropic/OpenAI format)."""
    if not messages:
        return ""
    if not isinstance(messages, list):
        return str(messages)[:2000]

    parts: List[str] = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            content = msg["content"]
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
    return "\n".join(parts) if parts else str(messages)[:2000]


def _extract_prompt_from_bedrock_body(body: Any) -> str:
    """Extract prompt from Bedrock invoke_model body."""
    import json as _json
    try:
        if isinstance(body, (str, bytes)):
            parsed = _json.loads(body)
        else:
            parsed = body
        # Anthropic format
        if "messages" in parsed:
            return _extract_prompt_from_messages(parsed["messages"])
        # Titan/Cohere/AI21
        if "inputText" in parsed:
            return parsed["inputText"]
        if "prompt" in parsed:
            return parsed["prompt"]
        return str(parsed)[:2000]
    except Exception:
        return str(body)[:2000]


# ---------------------------------------------------------------------------
# Instrument class — the core engine
# ---------------------------------------------------------------------------

class Instrument:
    """
    Auto-instrumentation engine for LLM providers.

    Transparently intercepts LLM calls, generates audit receipts, and
    uploads them to the Veratum endpoint. Zero code changes required
    beyond wrapping the client.

    Usage:
        instrument = Instrument(sdk)
        client = instrument.wrap(anthropic.Anthropic())
        # All calls to client.messages.create() are now audited.

    Or with the decorator:
        @instrument.audit(model="gpt-4", provider="openai")
        def my_llm_call(prompt):
            return openai_client.chat.completions.create(...)
    """

    def __init__(
        self,
        sdk: Any,
        *,
        decision_type: str = "content_generation",
        vertical: Optional[str] = None,
        default_metadata: Optional[Dict[str, Any]] = None,
        compliance_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize instrumentation engine.

        Args:
            sdk: VeratumSDK instance (provides endpoint, auth, chain)
            decision_type: Default decision type for receipts
            vertical: Industry vertical override (falls back to sdk.vertical)
            default_metadata: Default metadata merged into every receipt
            compliance_fields: Default compliance fields (e.g., jurisdiction tags,
                               data_processing_basis) merged into every receipt
        """
        self._sdk = sdk
        self._decision_type = decision_type
        self._vertical = vertical or getattr(sdk, "vertical", "general")
        self._default_metadata = default_metadata or {}
        self._compliance_fields = compliance_fields or {}

    # ------------------------------------------------------------------
    # wrap() — provider-aware monkey-patching
    # ------------------------------------------------------------------

    def wrap(self, client: Any, *, provider: Optional[str] = None) -> Any:
        """
        Wrap an LLM client for automatic auditing.

        Detects the provider and monkey-patches the appropriate method:
        - Anthropic: client.messages.create
        - OpenAI:    client.chat.completions.create
        - Bedrock:   client.invoke_model

        Args:
            client: LLM client instance
            provider: Force provider detection (optional)

        Returns:
            The same client object with auditing enabled
        """
        detected = provider or _detect_provider(client)

        if detected == PROVIDER_ANTHROPIC:
            return self._wrap_anthropic(client)
        elif detected == PROVIDER_OPENAI:
            return self._wrap_openai(client)
        elif detected == PROVIDER_BEDROCK:
            return self._wrap_bedrock(client)
        else:
            logger.warning(
                "Unknown provider %r — use @instrument.audit() decorator instead",
                detected,
            )
            return client

    def _wrap_anthropic(self, client: Any) -> Any:
        """Wrap Anthropic client.messages.create."""
        original = client.messages.create

        @functools.wraps(original)
        def instrumented(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            prompt = _extract_prompt_from_messages(kwargs.get("messages", []))
            model = kwargs.get("model", "unknown")

            response = original(*args, **kwargs)

            duration_ms = int((time.monotonic() - start) * 1000)
            extracted = _extract_anthropic_response(response)
            self._emit_receipt(
                prompt=prompt,
                response_text=extracted["text"],
                model=extracted.get("model", model),
                provider=PROVIDER_ANTHROPIC,
                tokens_in=extracted["tokens_in"],
                tokens_out=extracted["tokens_out"],
                duration_ms=duration_ms,
            )
            return response

        client.messages.create = instrumented
        return client

    def _wrap_openai(self, client: Any) -> Any:
        """Wrap OpenAI client.chat.completions.create."""
        original = client.chat.completions.create

        @functools.wraps(original)
        def instrumented(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            prompt = _extract_prompt_from_messages(kwargs.get("messages", []))
            model = kwargs.get("model", "unknown")

            response = original(*args, **kwargs)

            duration_ms = int((time.monotonic() - start) * 1000)
            extracted = _extract_openai_response(response)
            self._emit_receipt(
                prompt=prompt,
                response_text=extracted["text"],
                model=extracted.get("model", model),
                provider=PROVIDER_OPENAI,
                tokens_in=extracted["tokens_in"],
                tokens_out=extracted["tokens_out"],
                duration_ms=duration_ms,
            )
            return response

        client.chat.completions.create = instrumented
        return client

    def _wrap_bedrock(self, client: Any) -> Any:
        """Wrap Bedrock runtime client.invoke_model."""
        original = client.invoke_model

        @functools.wraps(original)
        def instrumented(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            model_id = kwargs.get("modelId", "unknown")
            body = kwargs.get("body", "")
            prompt = _extract_prompt_from_bedrock_body(body)

            response = original(*args, **kwargs)

            duration_ms = int((time.monotonic() - start) * 1000)
            extracted = _extract_bedrock_response(response, model_id)
            self._emit_receipt(
                prompt=prompt,
                response_text=extracted["text"],
                model=extracted.get("model", model_id),
                provider=PROVIDER_BEDROCK,
                tokens_in=extracted["tokens_in"],
                tokens_out=extracted["tokens_out"],
                duration_ms=duration_ms,
            )
            return response

        client.invoke_model = instrumented
        return client

    # ------------------------------------------------------------------
    # audit() — decorator for generic functions
    # ------------------------------------------------------------------

    def audit(
        self,
        *,
        model: str = "unknown",
        provider: str = "generic",
        extract_prompt: Optional[Callable[..., str]] = None,
        extract_response: Optional[Callable[..., str]] = None,
    ) -> Callable[[F], F]:
        """
        Decorator for auditing any function that calls an LLM.

        Usage:
            @instrument.audit(model="gpt-4", provider="openai")
            def classify(text):
                resp = openai.chat.completions.create(...)
                return resp.choices[0].message.content

        Args:
            model: Model identifier
            provider: Provider name
            extract_prompt: Optional callable(args, kwargs) -> str for prompt extraction
            extract_response: Optional callable(result) -> str for response extraction

        Returns:
            Decorated function with auditing
        """
        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()

                # Extract prompt
                if extract_prompt:
                    prompt = extract_prompt(*args, **kwargs)
                elif args:
                    prompt = str(args[0])[:2000]
                else:
                    prompt = str(kwargs)[:2000]

                # Call original function
                result = fn(*args, **kwargs)

                duration_ms = int((time.monotonic() - start) * 1000)

                # Extract response
                if extract_response:
                    response_text = extract_response(result)
                elif isinstance(result, str):
                    response_text = result
                else:
                    response_text = str(result)[:2000]

                self._emit_receipt(
                    prompt=prompt,
                    response_text=response_text,
                    model=model,
                    provider=provider,
                    tokens_in=0,
                    tokens_out=0,
                    duration_ms=duration_ms,
                )
                return result

            return wrapper  # type: ignore[return-value]
        return decorator

    # ------------------------------------------------------------------
    # Receipt emission
    # ------------------------------------------------------------------

    def _emit_receipt(
        self,
        prompt: str,
        response_text: str,
        model: str,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: int,
    ) -> None:
        """Generate receipt and upload via SDK (fire-and-forget on error)."""
        try:
            # Merge compliance fields into receipt kwargs
            receipt_kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "response": response_text,
                "model": model,
                "provider": provider,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "decision_type": self._decision_type,
                "vertical": self._vertical,
                "metadata": {
                    **self._default_metadata,
                    "duration_ms": duration_ms,
                    "instrumentation": "auto",
                },
            }

            # Merge any default compliance fields
            for key, value in self._compliance_fields.items():
                if key not in receipt_kwargs:
                    receipt_kwargs[key] = value

            receipt = self._sdk._receipt_generator.generate(**receipt_kwargs)

            # Upload
            self._sdk._upload_receipt(receipt)

            # Auto-timestamp if enabled
            if getattr(self._sdk, "auto_timestamp", False):
                entry_hash = receipt.get("entry_hash")
                if entry_hash:
                    self._sdk.request_timestamp(entry_hash)

        except Exception as exc:
            logger.error("Veratum receipt emission failed: %s", exc, exc_info=True)
