"""
Veratum Evidence Layer for LangChain.

Drop-in compliance for any LangChain chain, agent, or tool call.
Creates immutable evidence receipts for every LLM interaction automatically.

Example usage:
    from langchain_openai import ChatOpenAI
    from veratum.integrations.langchain_callback import VeratumCallbackHandler

    handler = VeratumCallbackHandler()
    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

    # Every LLM call now gets a Veratum receipt
    response = llm.invoke("What is 2+2?")

Or use the convenience function:
    from veratum.integrations.langchain_callback import enable_veratum

    handler = enable_veratum(api_key="your-key")
    # Pass handler to any chain, agent, or LLM
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Try to import LangChain base callback handler
_LANGCHAIN_AVAILABLE = False
_BASE_HANDLER = object

try:
    from langchain_core.callbacks import BaseCallbackHandler
    _BASE_HANDLER = BaseCallbackHandler
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        _BASE_HANDLER = BaseCallbackHandler
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        pass


class VeratumCallbackHandler(_BASE_HANDLER):
    """
    LangChain callback handler that creates Veratum evidence receipts.

    Hooks into LangChain's callback system to capture every LLM call,
    chain run, tool invocation, and agent action with cryptographic
    evidence receipts.

    Features:
    - Automatic receipt generation for all LLM calls
    - Chain and agent run tracking with parent-child linkage
    - Tool call evidence capture
    - Non-blocking async upload to Veratum API
    - Thread-safe receipt queue with batch upload
    - Graceful fallback when EvidenceEngine not available
    - Token usage tracking
    - Error evidence (failed calls are also evidence)

    Args:
        api_key: Veratum API key for receipt upload
        endpoint: Veratum API endpoint URL
        queue_receipts: Batch receipts before uploading (default: True)
        queue_size: Number of receipts per batch (default: 10)
        capture_chains: Also capture chain start/end events (default: True)
        capture_tools: Also capture tool calls (default: True)
        capture_agents: Also capture agent actions (default: True)
        metadata: Default metadata added to all receipts
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        queue_receipts: bool = True,
        queue_size: int = 10,
        capture_chains: bool = True,
        capture_tools: bool = True,
        capture_agents: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Veratum LangChain callback handler."""
        if _LANGCHAIN_AVAILABLE and hasattr(_BASE_HANDLER, '__init__'):
            super().__init__()

        self.api_key = api_key or self._get_api_key_from_env()
        self.endpoint = endpoint or "https://api.veratum.ai"
        self.queue_receipts = queue_receipts
        self.queue_size = queue_size
        self.capture_chains = capture_chains
        self.capture_tools = capture_tools
        self.capture_agents = capture_agents
        self.default_metadata = metadata or {}

        # Try to import EvidenceEngine
        self._engine = None
        try:
            from veratum.core.evidence import EvidenceEngine
            self._engine = EvidenceEngine(api_key=api_key, endpoint=endpoint)
            logger.info("EvidenceEngine initialized for LangChain callback")
        except (ImportError, Exception) as e:
            logger.warning(
                f"EvidenceEngine not available ({e}), using direct API upload"
            )

        # Receipt queue and thread safety
        self._receipt_queue: List[Dict[str, Any]] = []
        self._queue_lock = threading.Lock()
        self._upload_threads: List[threading.Thread] = []

        # Track active runs for timing and parent-child linkage
        self._active_runs: Dict[str, Dict[str, Any]] = {}
        self._runs_lock = threading.Lock()

        # Public access to receipts
        self.receipts: List[Dict[str, Any]] = []

        logger.info(
            f"VeratumCallbackHandler initialized "
            f"(chains={capture_chains}, tools={capture_tools}, agents={capture_agents})"
        )

    @staticmethod
    def _get_api_key_from_env() -> Optional[str]:
        """Get Veratum API key from environment variables."""
        import os
        return os.environ.get("VERATUM_API_KEY")

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _start_run(
        self,
        run_id: str,
        run_type: str,
        inputs: Any = None,
        parent_run_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Track the start of a run."""
        with self._runs_lock:
            self._active_runs[run_id] = {
                "run_type": run_type,
                "start_time": time.time(),
                "inputs": inputs,
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                **kwargs,
            }

    def _end_run(
        self,
        run_id: str,
        outputs: Any = None,
        error: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Complete a run and return the run data."""
        with self._runs_lock:
            run_data = self._active_runs.pop(run_id, None)

        if run_data:
            run_data["end_time"] = time.time()
            run_data["outputs"] = outputs
            run_data["error"] = error
            run_data["latency_ms"] = (run_data["end_time"] - run_data["start_time"]) * 1000
        return run_data

    def _create_receipt(
        self,
        run_id: str,
        run_data: Dict[str, Any],
        status: str = "SUCCESS",
    ) -> Dict[str, Any]:
        """Create a Veratum evidence receipt from run data."""
        receipt_id = f"receipt_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract model info
        model = run_data.get("model", "unknown")
        provider = self._detect_provider(model)
        serialization = run_data.get("serialized", {})
        if isinstance(serialization, dict):
            model = serialization.get("kwargs", {}).get("model_name", model)
            model = serialization.get("kwargs", {}).get("model", model)

        # Build input/output hashes
        input_hash = self._compute_hash(run_data.get("inputs", ""))
        output_hash = self._compute_hash(run_data.get("outputs", ""))

        # Extract token usage if available
        token_usage = {}
        outputs = run_data.get("outputs")
        if isinstance(outputs, dict):
            # LangChain LLMResult format
            llm_output = outputs.get("llm_output", {})
            if isinstance(llm_output, dict):
                usage = llm_output.get("token_usage", {})
                token_usage = {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }

        receipt = {
            "receipt_id": receipt_id,
            "timestamp": timestamp,
            "run_id": run_id,
            "run_type": run_data.get("run_type", "unknown"),
            "parent_run_id": run_data.get("parent_run_id"),
            "model": model,
            "provider": provider,
            "status": status,
            "latency_ms": run_data.get("latency_ms", 0),
            "input_hash": input_hash,
            "output_hash": output_hash,
            "error": run_data.get("error"),
            "metadata": {
                **self.default_metadata,
                **token_usage,
            },
        }

        # If EvidenceEngine available, create a proper chained receipt
        if self._engine:
            try:
                prompt = self._extract_prompt(run_data.get("inputs"))
                response = self._extract_response(run_data.get("outputs"))

                evidence_receipt = self._engine.create_evidence(
                    request={"prompt": prompt, "messages": run_data.get("inputs")},
                    response={"text": response},
                    provider=provider,
                    model=model,
                    metadata=self.default_metadata,
                )
                # Merge chain info into our receipt
                receipt["entry_hash"] = evidence_receipt.get("entry_hash")
                receipt["prev_hash"] = evidence_receipt.get("prev_hash")
                receipt["sequence_no"] = evidence_receipt.get("sequence_no")
                receipt["schema_version"] = evidence_receipt.get("schema_version")
            except Exception as e:
                logger.debug(f"EvidenceEngine receipt creation failed, using basic receipt: {e}")

        self.receipts.append(receipt)
        return receipt

    def _extract_prompt(self, inputs: Any) -> str:
        """Extract prompt text from LangChain inputs."""
        if isinstance(inputs, str):
            return inputs
        if isinstance(inputs, dict):
            # ChatModel: inputs = {"messages": [...]}
            if "messages" in inputs:
                messages = inputs["messages"]
                if isinstance(messages, list):
                    parts = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
                        elif hasattr(msg, 'content'):
                            role = getattr(msg, 'type', getattr(msg, 'role', 'user'))
                            parts.append(f"{role}: {msg.content}")
                    return "\n".join(parts)
            # LLM: inputs = {"prompts": ["..."]}
            if "prompts" in inputs:
                return "\n".join(inputs["prompts"])
            # Simple: inputs = {"input": "..."}
            for key in ["input", "query", "question", "prompt"]:
                if key in inputs:
                    return str(inputs[key])
        if isinstance(inputs, list):
            return "\n".join(str(item) for item in inputs)
        return str(inputs)

    def _extract_response(self, outputs: Any) -> str:
        """Extract response text from LangChain outputs."""
        if isinstance(outputs, str):
            return outputs
        if isinstance(outputs, dict):
            # LLMResult: {"generations": [[{"text": "..."}]]}
            generations = outputs.get("generations", [])
            if generations and isinstance(generations, list):
                first_gen = generations[0]
                if isinstance(first_gen, list) and first_gen:
                    gen = first_gen[0]
                    if isinstance(gen, dict):
                        return gen.get("text", "")
                    if hasattr(gen, 'text'):
                        return gen.text
            # Simple: {"output": "..."}
            for key in ["output", "result", "answer", "text", "response"]:
                if key in outputs:
                    return str(outputs[key])
        if hasattr(outputs, 'content'):
            return outputs.content
        return str(outputs)

    @staticmethod
    def _detect_provider(model: str) -> str:
        """Detect provider from model name."""
        model_lower = str(model).lower()
        if any(x in model_lower for x in ["gpt-", "o1-", "o3-", "davinci", "turbo"]):
            return "openai"
        if "claude" in model_lower:
            return "anthropic"
        if "gemini" in model_lower or "palm" in model_lower:
            return "google"
        if "command" in model_lower:
            return "cohere"
        if "llama" in model_lower or "mistral" in model_lower or "mixtral" in model_lower:
            return "open_source"
        if "bedrock" in model_lower:
            return "bedrock"
        return "unknown"

    def _queue_receipt(self, receipt: Dict[str, Any]) -> None:
        """Add receipt to upload queue."""
        if not self.queue_receipts:
            self._upload_receipt(receipt)
            return

        with self._queue_lock:
            self._receipt_queue.append(receipt)
            if len(self._receipt_queue) >= self.queue_size:
                self._flush_queue()

    def _flush_queue(self) -> None:
        """Upload queued receipts in background thread."""
        if not self._receipt_queue:
            return

        batch = self._receipt_queue.copy()
        self._receipt_queue.clear()

        def upload_batch():
            for receipt in batch:
                self._upload_receipt(receipt)

        thread = threading.Thread(target=upload_batch, daemon=True)
        self._upload_threads.append(thread)
        thread.start()
        self._upload_threads = [t for t in self._upload_threads if t.is_alive()]

    def _upload_receipt(self, receipt: Dict[str, Any]) -> None:
        """Upload a single receipt to Veratum API."""
        if self._engine:
            try:
                self._engine.upload_evidence(receipt)
                logger.debug(f"Receipt {receipt['receipt_id']} uploaded via EvidenceEngine")
                return
            except Exception as e:
                logger.warning(f"EvidenceEngine upload failed: {e}")

        # Fallback to direct API
        if not self.api_key:
            logger.debug("No API key, skipping upload (receipt stored locally)")
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
                logger.debug(f"Receipt {receipt['receipt_id']} uploaded via API")
        except Exception as e:
            logger.warning(f"API upload failed for {receipt['receipt_id']}: {e}")

    # ─── LangChain Callback Interface ───────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts running."""
        self._start_run(
            run_id=str(run_id),
            run_type="llm",
            inputs={"prompts": prompts},
            parent_run_id=parent_run_id,
            serialized=serialized,
            tags=tags,
            run_metadata=metadata,
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chat model starts running."""
        # Convert message objects to serializable format
        serializable_messages = []
        for message_batch in messages:
            batch = []
            for msg in message_batch:
                if hasattr(msg, 'content'):
                    batch.append({
                        "role": getattr(msg, 'type', getattr(msg, 'role', 'user')),
                        "content": msg.content,
                    })
                elif isinstance(msg, dict):
                    batch.append(msg)
                else:
                    batch.append({"role": "user", "content": str(msg)})
            serializable_messages.append(batch)

        self._start_run(
            run_id=str(run_id),
            run_type="chat_model",
            inputs={"messages": serializable_messages},
            parent_run_id=parent_run_id,
            serialized=serialized,
            tags=tags,
            run_metadata=metadata,
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM finishes."""
        # Convert LLMResult to dict
        outputs = {}
        if hasattr(response, 'generations'):
            outputs["generations"] = [
                [
                    {"text": gen.text, "message": getattr(gen, 'message', None)}
                    for gen in gen_list
                ]
                for gen_list in response.generations
            ]
        if hasattr(response, 'llm_output') and response.llm_output:
            outputs["llm_output"] = response.llm_output

        run_data = self._end_run(str(run_id), outputs=outputs)
        if run_data:
            receipt = self._create_receipt(str(run_id), run_data, status="SUCCESS")
            self._queue_receipt(receipt)
            logger.info(
                f"LLM receipt {receipt['receipt_id']} | "
                f"model={receipt['model']} | {receipt['latency_ms']:.0f}ms"
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM errors. Failed calls are also evidence."""
        run_data = self._end_run(str(run_id), error=str(error))
        if run_data:
            receipt = self._create_receipt(str(run_id), run_data, status="FAILED")
            self._queue_receipt(receipt)
            logger.warning(
                f"LLM error receipt {receipt['receipt_id']} | error={str(error)[:100]}"
            )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts running."""
        if self.capture_chains:
            self._start_run(
                run_id=str(run_id),
                run_type="chain",
                inputs=inputs,
                parent_run_id=parent_run_id,
                serialized=serialized,
                tags=tags,
                run_metadata=metadata,
            )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain finishes."""
        if self.capture_chains:
            run_data = self._end_run(str(run_id), outputs=outputs)
            if run_data:
                receipt = self._create_receipt(str(run_id), run_data, status="SUCCESS")
                self._queue_receipt(receipt)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors."""
        if self.capture_chains:
            run_data = self._end_run(str(run_id), error=str(error))
            if run_data:
                receipt = self._create_receipt(str(run_id), run_data, status="FAILED")
                self._queue_receipt(receipt)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts running."""
        if self.capture_tools:
            self._start_run(
                run_id=str(run_id),
                run_type="tool",
                inputs={"input": input_str},
                parent_run_id=parent_run_id,
                serialized=serialized,
                tags=tags,
                run_metadata=metadata,
            )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes."""
        if self.capture_tools:
            run_data = self._end_run(str(run_id), outputs={"output": str(output)})
            if run_data:
                receipt = self._create_receipt(str(run_id), run_data, status="SUCCESS")
                self._queue_receipt(receipt)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        if self.capture_tools:
            run_data = self._end_run(str(run_id), error=str(error))
            if run_data:
                receipt = self._create_receipt(str(run_id), run_data, status="FAILED")
                self._queue_receipt(receipt)

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent takes an action."""
        if self.capture_agents:
            action_data = {
                "tool": getattr(action, 'tool', str(action)),
                "tool_input": getattr(action, 'tool_input', None),
                "log": getattr(action, 'log', None),
            }
            self._start_run(
                run_id=f"{run_id}_action_{uuid.uuid4().hex[:6]}",
                run_type="agent_action",
                inputs=action_data,
                parent_run_id=run_id,
            )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent finishes."""
        if self.capture_agents:
            output = {
                "output": getattr(finish, 'return_values', str(finish)),
                "log": getattr(finish, 'log', None),
            }
            # Find and close any open agent action runs
            with self._runs_lock:
                agent_runs = [
                    rid for rid, data in self._active_runs.items()
                    if data.get("parent_run_id") == str(run_id)
                    and data.get("run_type") == "agent_action"
                ]
            for agent_run_id in agent_runs:
                run_data = self._end_run(agent_run_id, outputs=output)
                if run_data:
                    receipt = self._create_receipt(agent_run_id, run_data, status="SUCCESS")
                    self._queue_receipt(receipt)

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Called on arbitrary text. No-op for evidence purposes."""
        pass

    def on_retry(self, *args: Any, **kwargs: Any) -> None:
        """Called on retry. No-op for evidence purposes."""
        pass

    # ─── Public API ─────────────────────────────────────────────────

    def flush(self) -> None:
        """Force upload all queued receipts."""
        with self._queue_lock:
            self._flush_queue()

    def wait_for_uploads(self, timeout: float = 30.0) -> bool:
        """
        Wait for all pending uploads to complete.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if all uploads completed, False if timeout.
        """
        self.flush()
        start = time.time()
        for thread in self._upload_threads:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                return False
            thread.join(timeout=remaining)
            if thread.is_alive():
                return False
        return True

    def get_receipts(self) -> List[Dict[str, Any]]:
        """Get all captured receipts."""
        return self.receipts.copy()

    def get_receipt_count(self) -> int:
        """Get the number of captured receipts."""
        return len(self.receipts)


def enable_veratum(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    queue_receipts: bool = True,
    queue_size: int = 10,
    capture_chains: bool = True,
    capture_tools: bool = True,
    capture_agents: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> "VeratumCallbackHandler":
    """
    Create a Veratum callback handler for LangChain.

    One-liner to get compliance evidence on any LangChain component.

    Usage:
        from veratum.integrations.langchain_callback import enable_veratum

        handler = enable_veratum()

        # Use with any LangChain LLM
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

        # Use with any chain
        chain = prompt | llm
        result = chain.invoke({"input": "hello"}, config={"callbacks": [handler]})

        # Use with agents
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])

    Args:
        api_key: Veratum API key
        endpoint: Veratum API endpoint
        queue_receipts: Batch receipts for efficient upload (default: True)
        queue_size: Receipts per batch (default: 10)
        capture_chains: Capture chain events (default: True)
        capture_tools: Capture tool events (default: True)
        capture_agents: Capture agent events (default: True)
        metadata: Default metadata for all receipts

    Returns:
        VeratumCallbackHandler instance to pass to LangChain components
    """
    return VeratumCallbackHandler(
        api_key=api_key,
        endpoint=endpoint,
        queue_receipts=queue_receipts,
        queue_size=queue_size,
        capture_chains=capture_chains,
        capture_tools=capture_tools,
        capture_agents=capture_agents,
        metadata=metadata,
    )


__all__ = [
    "VeratumCallbackHandler",
    "enable_veratum",
]
