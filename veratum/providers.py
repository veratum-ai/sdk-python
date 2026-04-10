"""Multi-provider LLM client auto-detection and wrapping.

Supports auto-detection and wrapping for major LLM providers:
- OpenAI (chat.completions.create)
- Anthropic (messages.create)
- Google GenerativeAI (generate_content)
- Mistral AI (chat.complete)
- Cohere (generate)
- AWS Bedrock (invoke_model, converse)

Each provider wrapper patches the client transparently. Responses are returned
unchanged — the audit layer is invisible. Receipt generation is handled via
callback, allowing integration with any receipt backend.
"""

import functools
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("veratum.providers")


class UnsupportedProviderError(Exception):
    """Raised when client provider cannot be auto-detected or is unsupported."""

    pass


@dataclass
class WrapConfig:
    """Configuration for provider wrapping behavior."""

    capture_prompts: bool = True  # Store prompt text in receipt
    capture_responses: bool = True  # Store response text in receipt
    redact_pii: bool = False  # Run PII redaction before storing (future)
    run_prompt_guard: bool = False  # Scan for injection before LLM call (future)
    track_cost: bool = False  # Calculate and track cost
    max_prompt_length: int = 10000  # Truncate long prompts in receipt
    max_response_length: int = 10000  # Truncate long responses
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra metadata per receipt


def detect_provider(client: Any) -> str:
    """
    Auto-detect LLM provider by inspecting client class names, modules, and attributes.

    Supports:
    - openai.OpenAI / openai.AsyncOpenAI → "openai"
    - anthropic.Anthropic / anthropic.AsyncAnthropic → "anthropic"
    - google.generativeai.GenerativeModel → "google"
    - mistralai.Mistral → "mistral"
    - cohere.Client → "cohere"
    - boto3 bedrock-runtime client → "bedrock"

    Args:
        client: LLM client instance

    Returns:
        Provider name as string. Returns "unknown" if unrecognized.
    """
    if client is None:
        return "unknown"

    client_class = client.__class__
    class_name = client_class.__name__
    module_name = client_class.__module__

    # OpenAI detection
    if "openai" in module_name.lower():
        if class_name in ("OpenAI", "AsyncOpenAI"):
            return "openai"

    # Anthropic detection
    if "anthropic" in module_name.lower():
        if class_name in ("Anthropic", "AsyncAnthropic"):
            return "anthropic"

    # Google GenerativeAI detection
    if "google" in module_name.lower() and "generativeai" in module_name.lower():
        if class_name == "GenerativeModel":
            return "google"

    # Mistral detection
    if "mistralai" in module_name.lower():
        if class_name == "Mistral":
            return "mistral"

    # Cohere detection
    if "cohere" in module_name.lower():
        if class_name == "Client" or class_name == "CohereClient":
            return "cohere"

    # AWS Bedrock detection (boto3 service client)
    if "bedrock" in module_name.lower() or class_name == "BedrockRuntime":
        return "bedrock"

    # Check for boto3 bedrock-runtime attributes
    if hasattr(client, "invoke_model") and hasattr(client, "_service_model"):
        service_model = getattr(client, "_service_model", None)
        if service_model is not None:
            service_name = getattr(service_model, "service_name", "")
            if isinstance(service_name, str) and "bedrock" in service_name.lower():
                return "bedrock"

    return "unknown"


def _truncate(text: Optional[str], max_length: int) -> Optional[str]:
    """Truncate text to max_length if needed."""
    if text is None:
        return None
    if len(text) > max_length:
        return text[:max_length] + f"... [truncated {len(text) - max_length} chars]"
    return text


def _generate_receipt(
    receipt_fn: Callable,
    model: str,
    prompt: Optional[str],
    response: Optional[str],
    tokens_in: int,
    tokens_out: int,
    duration_ms: int,
    provider: str,
    config: WrapConfig,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Helper to generate and upload a receipt via callback.

    Args:
        receipt_fn: Callback to generate/upload receipt
        model: Model identifier
        prompt: Input prompt (truncated if needed)
        response: Response text (truncated if needed)
        tokens_in: Input tokens
        tokens_out: Output tokens
        duration_ms: Request duration in milliseconds
        provider: Provider name
        config: Wrap configuration
        metadata: Additional metadata
    """
    try:
        # Apply truncation if configured
        prompt_to_store = (
            _truncate(prompt, config.max_prompt_length) if config.capture_prompts else None
        )
        response_to_store = (
            _truncate(response, config.max_response_length)
            if config.capture_responses
            else None
        )

        # Merge metadata
        extra_metadata = {**(metadata or {}), "duration_ms": duration_ms, **config.metadata}

        # Call the receipt callback
        receipt_fn(
            model=model,
            prompt=prompt_to_store,
            response=response_to_store,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            provider=provider,
            metadata=extra_metadata,
        )
    except Exception as e:
        logger.error(f"Receipt generation failed for {provider}: {str(e)}")


def wrap_openai(
    client: Any, receipt_fn: Callable, config: Optional[WrapConfig] = None
) -> None:
    """
    Wrap OpenAI client to intercept and audit chat completions.

    Patches:
    - client.chat.completions.create (sync)
    - client.chat.completions.acreate (async, if available)

    Extracts: model, messages→prompt, choices[0].message.content→response,
    usage.prompt_tokens, usage.completion_tokens

    Args:
        client: openai.OpenAI or openai.AsyncOpenAI instance
        receipt_fn: Callback(model, prompt, response, tokens_in, tokens_out, provider, metadata)
        config: Wrap configuration
    """
    if config is None:
        config = WrapConfig()

    try:
        # Wrap sync create method
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            original_create = client.chat.completions.create

            @functools.wraps(original_create)
            def wrapped_create(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                model = kwargs.get("model", "unknown")
                messages = kwargs.get("messages", [])

                # Extract prompt from messages
                prompt_text = ""
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
                    prompt_text = "\n".join(prompt_parts)

                try:
                    response = original_create(*args, **kwargs)

                    # Extract response
                    response_text = ""
                    if (
                        hasattr(response, "choices")
                        and len(response.choices) > 0
                        and hasattr(response.choices[0], "message")
                    ):
                        response_text = response.choices[0].message.content or ""

                    # Extract token usage
                    tokens_in = 0
                    tokens_out = 0
                    if hasattr(response, "usage"):
                        tokens_in = getattr(response.usage, "prompt_tokens", 0)
                        tokens_out = getattr(response.usage, "completion_tokens", 0)

                    duration_ms = int((time.time() - start_time) * 1000)

                    _generate_receipt(
                        receipt_fn=receipt_fn,
                        model=model,
                        prompt=prompt_text,
                        response=response_text,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        duration_ms=duration_ms,
                        provider="openai",
                        config=config,
                    )

                    return response
                except Exception as e:
                    logger.error(f"OpenAI wrap error: {str(e)}")
                    return original_create(*args, **kwargs)

            client.chat.completions.create = wrapped_create

        # Wrap async create method if available
        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "acreate")
        ):
            original_acreate = client.chat.completions.acreate

            @functools.wraps(original_acreate)
            async def wrapped_acreate(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                model = kwargs.get("model", "unknown")
                messages = kwargs.get("messages", [])

                # Extract prompt
                prompt_text = ""
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
                    prompt_text = "\n".join(prompt_parts)

                try:
                    response = await original_acreate(*args, **kwargs)

                    # Extract response
                    response_text = ""
                    if (
                        hasattr(response, "choices")
                        and len(response.choices) > 0
                        and hasattr(response.choices[0], "message")
                    ):
                        response_text = response.choices[0].message.content or ""

                    # Extract token usage
                    tokens_in = 0
                    tokens_out = 0
                    if hasattr(response, "usage"):
                        tokens_in = getattr(response.usage, "prompt_tokens", 0)
                        tokens_out = getattr(response.usage, "completion_tokens", 0)

                    duration_ms = int((time.time() - start_time) * 1000)

                    _generate_receipt(
                        receipt_fn=receipt_fn,
                        model=model,
                        prompt=prompt_text,
                        response=response_text,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        duration_ms=duration_ms,
                        provider="openai",
                        config=config,
                    )

                    return response
                except Exception as e:
                    logger.error(f"OpenAI async wrap error: {str(e)}")
                    return await original_acreate(*args, **kwargs)

            client.chat.completions.acreate = wrapped_acreate

    except Exception as e:
        logger.error(f"Failed to wrap OpenAI client: {str(e)}")


def wrap_anthropic(
    client: Any, receipt_fn: Callable, config: Optional[WrapConfig] = None
) -> None:
    """
    Wrap Anthropic client to intercept and audit messages.

    Patches: client.messages.create

    Extracts: model, messages→prompt, content[0].text→response,
    usage.input_tokens, usage.output_tokens

    Args:
        client: anthropic.Anthropic or anthropic.AsyncAnthropic instance
        receipt_fn: Callback(model, prompt, response, tokens_in, tokens_out, provider, metadata)
        config: Wrap configuration
    """
    if config is None:
        config = WrapConfig()

    try:
        if hasattr(client, "messages"):
            original_create = client.messages.create

            @functools.wraps(original_create)
            def wrapped_create(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                model = kwargs.get("model", "unknown")
                messages = kwargs.get("messages", [])

                # Extract prompt
                prompt_text = ""
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
                    prompt_text = "\n".join(prompt_parts)

                try:
                    response = original_create(*args, **kwargs)

                    # Extract response
                    response_text = ""
                    if hasattr(response, "content") and isinstance(response.content, list):
                        if len(response.content) > 0 and hasattr(response.content[0], "text"):
                            response_text = response.content[0].text

                    # Extract token usage
                    tokens_in = 0
                    tokens_out = 0
                    if hasattr(response, "usage"):
                        tokens_in = getattr(response.usage, "input_tokens", 0)
                        tokens_out = getattr(response.usage, "output_tokens", 0)

                    duration_ms = int((time.time() - start_time) * 1000)

                    _generate_receipt(
                        receipt_fn=receipt_fn,
                        model=model,
                        prompt=prompt_text,
                        response=response_text,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        duration_ms=duration_ms,
                        provider="anthropic",
                        config=config,
                    )

                    return response
                except Exception as e:
                    logger.error(f"Anthropic wrap error: {str(e)}")
                    return original_create(*args, **kwargs)

            client.messages.create = wrapped_create

    except Exception as e:
        logger.error(f"Failed to wrap Anthropic client: {str(e)}")


def wrap_google(
    client: Any, receipt_fn: Callable, config: Optional[WrapConfig] = None
) -> None:
    """
    Wrap Google GenerativeAI client to intercept and audit generations.

    Patches: model.generate_content

    Extracts: model_name, prompt text, response.text, usage_metadata

    Args:
        client: google.generativeai.GenerativeModel instance
        receipt_fn: Callback(model, prompt, response, tokens_in, tokens_out, provider, metadata)
        config: Wrap configuration
    """
    if config is None:
        config = WrapConfig()

    try:
        if hasattr(client, "generate_content"):
            original_generate = client.generate_content

            @functools.wraps(original_generate)
            def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                # Extract model name
                model = getattr(client, "model_name", "google-unknown")

                # Extract prompt from content
                prompt_text = ""
                if len(args) > 0:
                    content = args[0]
                    if isinstance(content, str):
                        prompt_text = content
                    elif isinstance(content, list):
                        prompt_parts = []
                        for item in content:
                            if isinstance(item, str):
                                prompt_parts.append(item)
                            elif isinstance(item, dict) and "text" in item:
                                prompt_parts.append(item["text"])
                        prompt_text = "\n".join(prompt_parts)

                try:
                    response = original_generate(*args, **kwargs)

                    # Extract response text
                    response_text = ""
                    if hasattr(response, "text"):
                        response_text = response.text

                    # Extract token usage
                    tokens_in = 0
                    tokens_out = 0
                    if hasattr(response, "usage_metadata"):
                        usage = response.usage_metadata
                        tokens_in = getattr(usage, "prompt_token_count", 0)
                        tokens_out = getattr(usage, "candidates_token_count", 0)

                    duration_ms = int((time.time() - start_time) * 1000)

                    _generate_receipt(
                        receipt_fn=receipt_fn,
                        model=model,
                        prompt=prompt_text,
                        response=response_text,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        duration_ms=duration_ms,
                        provider="google",
                        config=config,
                    )

                    return response
                except Exception as e:
                    logger.error(f"Google wrap error: {str(e)}")
                    return original_generate(*args, **kwargs)

            client.generate_content = wrapped_generate

    except Exception as e:
        logger.error(f"Failed to wrap Google GenerativeAI client: {str(e)}")


def wrap_mistral(
    client: Any, receipt_fn: Callable, config: Optional[WrapConfig] = None
) -> None:
    """
    Wrap Mistral AI client to intercept and audit chat completions.

    Patches: client.chat.complete

    Extracts: model, messages, choices[0].message.content, usage

    Args:
        client: mistralai.Mistral instance
        receipt_fn: Callback(model, prompt, response, tokens_in, tokens_out, provider, metadata)
        config: Wrap configuration
    """
    if config is None:
        config = WrapConfig()

    try:
        if hasattr(client, "chat") and hasattr(client.chat, "complete"):
            original_complete = client.chat.complete

            @functools.wraps(original_complete)
            def wrapped_complete(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                model = kwargs.get("model", "unknown")
                messages = kwargs.get("messages", [])

                # Extract prompt
                prompt_text = ""
                if isinstance(messages, list):
                    prompt_parts = []
                    for msg in messages:
                        if isinstance(msg, dict) and "content" in msg:
                            content = msg["content"]
                            if isinstance(content, str):
                                prompt_parts.append(content)
                        elif hasattr(msg, "content"):
                            prompt_parts.append(msg.content)
                    prompt_text = "\n".join(prompt_parts)

                try:
                    response = original_complete(*args, **kwargs)

                    # Extract response
                    response_text = ""
                    if (
                        hasattr(response, "choices")
                        and len(response.choices) > 0
                        and hasattr(response.choices[0], "message")
                    ):
                        response_text = response.choices[0].message.content or ""

                    # Extract token usage
                    tokens_in = 0
                    tokens_out = 0
                    if hasattr(response, "usage"):
                        tokens_in = getattr(response.usage, "prompt_tokens", 0)
                        tokens_out = getattr(response.usage, "completion_tokens", 0)

                    duration_ms = int((time.time() - start_time) * 1000)

                    _generate_receipt(
                        receipt_fn=receipt_fn,
                        model=model,
                        prompt=prompt_text,
                        response=response_text,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        duration_ms=duration_ms,
                        provider="mistral",
                        config=config,
                    )

                    return response
                except Exception as e:
                    logger.error(f"Mistral wrap error: {str(e)}")
                    return original_complete(*args, **kwargs)

            client.chat.complete = wrapped_complete

    except Exception as e:
        logger.error(f"Failed to wrap Mistral client: {str(e)}")


def wrap_cohere(
    client: Any, receipt_fn: Callable, config: Optional[WrapConfig] = None
) -> None:
    """
    Wrap Cohere client to intercept and audit text generation.

    Patches: client.generate

    Extracts: model (or "cohere"), prompt, generations[0].text, token usage

    Args:
        client: cohere.Client instance
        receipt_fn: Callback(model, prompt, response, tokens_in, tokens_out, provider, metadata)
        config: Wrap configuration
    """
    if config is None:
        config = WrapConfig()

    try:
        if hasattr(client, "generate"):
            original_generate = client.generate

            @functools.wraps(original_generate)
            def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                # Extract model
                model = kwargs.get("model", "cohere-default")
                prompt_text = kwargs.get("prompt", "")

                try:
                    response = original_generate(*args, **kwargs)

                    # Extract response
                    response_text = ""
                    if (
                        hasattr(response, "generations")
                        and len(response.generations) > 0
                        and hasattr(response.generations[0], "text")
                    ):
                        response_text = response.generations[0].text

                    # Extract token usage (Cohere provides limited token info)
                    tokens_in = 0
                    tokens_out = 0
                    if hasattr(response, "tokens"):
                        tokens_info = response.tokens
                        if hasattr(tokens_info, "input_tokens"):
                            tokens_in = tokens_info.input_tokens
                        if hasattr(tokens_info, "output_tokens"):
                            tokens_out = tokens_info.output_tokens

                    duration_ms = int((time.time() - start_time) * 1000)

                    _generate_receipt(
                        receipt_fn=receipt_fn,
                        model=model,
                        prompt=prompt_text,
                        response=response_text,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        duration_ms=duration_ms,
                        provider="cohere",
                        config=config,
                    )

                    return response
                except Exception as e:
                    logger.error(f"Cohere wrap error: {str(e)}")
                    return original_generate(*args, **kwargs)

            client.generate = wrapped_generate

    except Exception as e:
        logger.error(f"Failed to wrap Cohere client: {str(e)}")


def wrap_bedrock(
    client: Any, receipt_fn: Callable, config: Optional[WrapConfig] = None
) -> None:
    """
    Wrap AWS Bedrock runtime client to intercept and audit model invocations.

    Patches: client.invoke_model or client.converse

    Extracts: model ID, request body (prompt), response body, usage tokens

    Args:
        client: boto3 bedrock-runtime client instance
        receipt_fn: Callback(model, prompt, response, tokens_in, tokens_out, provider, metadata)
        config: Wrap configuration
    """
    if config is None:
        config = WrapConfig()

    try:
        # Wrap invoke_model if available
        if hasattr(client, "invoke_model"):
            original_invoke = client.invoke_model

            @functools.wraps(original_invoke)
            def wrapped_invoke(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                # Extract model ID
                model = kwargs.get("modelId", "bedrock-unknown")

                # Extract prompt/body from request
                prompt_text = ""
                body = kwargs.get("body", "")
                if isinstance(body, (str, bytes)):
                    prompt_text = body if isinstance(body, str) else body.decode("utf-8", errors="ignore")

                try:
                    response = original_invoke(*args, **kwargs)

                    # Extract response body
                    response_text = ""
                    if hasattr(response, "body"):
                        response_body = response.body.read()
                        response_text = response_body.decode("utf-8", errors="ignore") if isinstance(response_body, bytes) else response_body

                    # Token usage (varies by model/provider)
                    tokens_in = 0
                    tokens_out = 0

                    duration_ms = int((time.time() - start_time) * 1000)

                    _generate_receipt(
                        receipt_fn=receipt_fn,
                        model=model,
                        prompt=prompt_text,
                        response=response_text,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        duration_ms=duration_ms,
                        provider="bedrock",
                        config=config,
                    )

                    return response
                except Exception as e:
                    logger.error(f"Bedrock invoke_model wrap error: {str(e)}")
                    return original_invoke(*args, **kwargs)

            client.invoke_model = wrapped_invoke

        # Wrap converse if available
        if hasattr(client, "converse"):
            original_converse = client.converse

            @functools.wraps(original_converse)
            def wrapped_converse(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                model = kwargs.get("modelId", "bedrock-unknown")
                messages = kwargs.get("messages", [])

                # Extract prompt from messages
                prompt_text = ""
                if isinstance(messages, list):
                    prompt_parts = []
                    for msg in messages:
                        if isinstance(msg, dict) and "content" in msg:
                            content = msg["content"]
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        prompt_parts.append(item["text"])
                    prompt_text = "\n".join(prompt_parts)

                try:
                    response = original_converse(*args, **kwargs)

                    # Extract response
                    response_text = ""
                    if hasattr(response, "output") and hasattr(response.output, "message"):
                        msg = response.output.message
                        if hasattr(msg, "content") and isinstance(msg.content, list):
                            content_parts = []
                            for item in msg.content:
                                if isinstance(item, dict) and "text" in item:
                                    content_parts.append(item["text"])
                            response_text = "\n".join(content_parts)

                    # Extract token usage
                    tokens_in = 0
                    tokens_out = 0
                    if hasattr(response, "usage"):
                        tokens_in = getattr(response.usage, "inputTokens", 0)
                        tokens_out = getattr(response.usage, "outputTokens", 0)

                    duration_ms = int((time.time() - start_time) * 1000)

                    _generate_receipt(
                        receipt_fn=receipt_fn,
                        model=model,
                        prompt=prompt_text,
                        response=response_text,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        duration_ms=duration_ms,
                        provider="bedrock",
                        config=config,
                    )

                    return response
                except Exception as e:
                    logger.error(f"Bedrock converse wrap error: {str(e)}")
                    return original_converse(*args, **kwargs)

            client.converse = wrapped_converse

    except Exception as e:
        logger.error(f"Failed to wrap Bedrock client: {str(e)}")


def auto_wrap(
    client: Any,
    receipt_fn: Callable,
    config: Optional[WrapConfig] = None,
) -> str:
    """
    Auto-detect LLM provider and apply correct wrapper.

    Wraps the client transparently — responses are returned unchanged.
    The audit layer is invisible.

    Args:
        client: LLM client instance
        receipt_fn: Callback to generate/upload receipt
        config: Wrap configuration (default: auto-create with defaults)

    Returns:
        Provider name that was detected and wrapped

    Raises:
        UnsupportedProviderError: If provider is unknown and cannot be wrapped
    """
    if config is None:
        config = WrapConfig()

    provider = detect_provider(client)

    if provider == "openai":
        wrap_openai(client, receipt_fn, config)
    elif provider == "anthropic":
        wrap_anthropic(client, receipt_fn, config)
    elif provider == "google":
        wrap_google(client, receipt_fn, config)
    elif provider == "mistral":
        wrap_mistral(client, receipt_fn, config)
    elif provider == "cohere":
        wrap_cohere(client, receipt_fn, config)
    elif provider == "bedrock":
        wrap_bedrock(client, receipt_fn, config)
    elif provider == "unknown":
        raise UnsupportedProviderError(
            f"Unable to detect LLM provider for client {client.__class__.__name__} "
            f"from module {client.__class__.__module__}. "
            f"Supported providers: openai, anthropic, google, mistral, cohere, bedrock"
        )
    else:
        raise UnsupportedProviderError(f"Unsupported provider: {provider}")

    logger.info(f"Successfully wrapped {provider} client")
    return provider
