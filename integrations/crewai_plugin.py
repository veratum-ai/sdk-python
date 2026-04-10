"""
Veratum Evidence Layer for CrewAI.

Drop-in compliance for CrewAI agent crews.
Creates immutable evidence receipts for every task, tool call, and LLM
interaction within a CrewAI workflow.

Example usage:
    from crewai import Agent, Task, Crew
    from veratum.integrations.crewai_plugin import VeratumCrewAIHandler, enable_veratum

    handler = enable_veratum()

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        step_callback=handler.step_callback,
        task_callback=handler.task_callback,
    )

    result = crew.kickoff()

    # Every task and step now has a Veratum receipt
    print(handler.get_receipts())
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class VeratumCrewAIHandler:
    """
    CrewAI integration for Veratum evidence layer.

    Provides callback functions for CrewAI's step_callback and task_callback
    hooks, creating immutable evidence receipts for every agent action.

    Features:
    - Task-level evidence receipts with agent attribution
    - Step-level receipts for individual LLM calls and tool uses
    - Agent metadata capture (role, goal, backstory)
    - Non-blocking async upload to Veratum API
    - Thread-safe receipt queue
    - Full crew execution summary

    Args:
        api_key: Veratum API key for receipt upload
        endpoint: Veratum API endpoint URL
        capture_steps: Capture step-level events (default: True)
        capture_tasks: Capture task-level events (default: True)
        metadata: Default metadata added to all receipts
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        capture_steps: bool = True,
        capture_tasks: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Veratum CrewAI handler."""
        self.api_key = api_key or self._get_api_key_from_env()
        self.endpoint = endpoint or "https://api.veratum.ai"
        self.capture_steps = capture_steps
        self.capture_tasks = capture_tasks
        self.default_metadata = metadata or {}

        # Try to import EvidenceEngine
        self._engine = None
        try:
            from veratum.core.evidence import EvidenceEngine
            self._engine = EvidenceEngine(api_key=api_key, endpoint=endpoint)
            logger.info("EvidenceEngine initialized for CrewAI handler")
        except (ImportError, Exception) as e:
            logger.warning(
                f"EvidenceEngine not available ({e}), using direct API upload"
            )

        # Receipt storage and thread safety
        self._receipts: List[Dict[str, Any]] = []
        self._receipt_lock = threading.Lock()
        self._upload_threads: List[threading.Thread] = []

        # Task timing tracker
        self._task_timers: Dict[str, float] = {}

        logger.info(
            f"VeratumCrewAIHandler initialized "
            f"(steps={capture_steps}, tasks={capture_tasks})"
        )

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
        event_type: str,
        inputs: Any,
        outputs: Any,
        agent_info: Optional[Dict[str, Any]] = None,
        task_info: Optional[Dict[str, Any]] = None,
        status: str = "SUCCESS",
        latency_ms: float = 0,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an evidence receipt."""
        receipt_id = f"receipt_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        receipt = {
            "receipt_id": receipt_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "framework": "crewai",
            "status": status,
            "latency_ms": latency_ms,
            "input_hash": self._compute_hash(inputs),
            "output_hash": self._compute_hash(outputs),
            "error": error,
            "agent": agent_info,
            "task": task_info,
            "metadata": {**self.default_metadata},
        }

        # Use EvidenceEngine if available
        if self._engine and event_type in ("task_completion", "llm_call"):
            try:
                prompt = str(inputs) if not isinstance(inputs, str) else inputs
                response = str(outputs) if not isinstance(outputs, str) else outputs
                model = "unknown"
                provider = "crewai"

                if agent_info:
                    model = agent_info.get("llm", "unknown")
                    provider = self._detect_provider(model)

                evidence = self._engine.create_evidence(
                    request={"prompt": prompt},
                    response={"text": response},
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

        # Upload async
        self._upload_async(receipt)
        return receipt

    @staticmethod
    def _detect_provider(model: str) -> str:
        """Detect provider from model name."""
        model_lower = str(model).lower()
        if any(x in model_lower for x in ["gpt-", "o1-", "o3-"]):
            return "openai"
        if "claude" in model_lower:
            return "anthropic"
        if "gemini" in model_lower:
            return "google"
        if "command" in model_lower:
            return "cohere"
        return "unknown"

    def _extract_agent_info(self, agent: Any) -> Dict[str, Any]:
        """Extract metadata from a CrewAI Agent object."""
        if agent is None:
            return {}
        return {
            "role": getattr(agent, 'role', 'unknown'),
            "goal": getattr(agent, 'goal', None),
            "backstory": getattr(agent, 'backstory', None),
            "llm": str(getattr(agent, 'llm', 'unknown')),
            "tools": [str(t) for t in getattr(agent, 'tools', [])],
            "allow_delegation": getattr(agent, 'allow_delegation', None),
            "verbose": getattr(agent, 'verbose', None),
        }

    def _extract_task_info(self, task: Any) -> Dict[str, Any]:
        """Extract metadata from a CrewAI Task object."""
        if task is None:
            return {}
        return {
            "description": getattr(task, 'description', 'unknown'),
            "expected_output": getattr(task, 'expected_output', None),
            "agent_role": getattr(getattr(task, 'agent', None), 'role', None),
        }

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
                logger.debug(f"Receipt {receipt['receipt_id']} uploaded")
                return
            except Exception as e:
                logger.warning(f"EvidenceEngine upload failed: {e}")

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
                logger.debug(f"Receipt {receipt['receipt_id']} uploaded via API")
        except Exception as e:
            logger.warning(f"Upload failed for {receipt['receipt_id']}: {e}")

    # ─── CrewAI Callback Functions ──────────────────────────────────

    def step_callback(self, step_output: Any) -> None:
        """
        CrewAI step callback — called after each agent step.

        Pass this to Crew(step_callback=handler.step_callback)

        Captures individual agent actions: LLM calls, tool uses,
        delegation decisions.

        Args:
            step_output: CrewAI StepOutput or AgentAction object
        """
        if not self.capture_steps:
            return

        try:
            # Extract step details based on CrewAI output type
            step_type = "agent_step"
            inputs = ""
            outputs = ""
            agent_info = {}

            if hasattr(step_output, 'tool'):
                # Tool use step
                step_type = "tool_call"
                inputs = {
                    "tool": getattr(step_output, 'tool', 'unknown'),
                    "tool_input": getattr(step_output, 'tool_input', ''),
                }
                outputs = getattr(step_output, 'result', '')
            elif hasattr(step_output, 'text'):
                # LLM output step
                step_type = "llm_call"
                inputs = getattr(step_output, 'prompt', '')
                outputs = getattr(step_output, 'text', '')
            elif hasattr(step_output, 'output'):
                # Generic step output
                inputs = getattr(step_output, 'input', '')
                outputs = getattr(step_output, 'output', '')
            else:
                # Fallback — capture whatever we can
                inputs = str(step_output)
                outputs = str(step_output)

            receipt = self._create_receipt(
                event_type=step_type,
                inputs=inputs,
                outputs=outputs,
                agent_info=agent_info,
                status="SUCCESS",
            )

            logger.debug(f"Step receipt {receipt['receipt_id']} | type={step_type}")

        except Exception as e:
            logger.error(f"Error in step_callback: {e}", exc_info=True)

    def task_callback(self, task_output: Any) -> None:
        """
        CrewAI task callback — called after each task completes.

        Pass this to Crew(task_callback=handler.task_callback)

        Captures task-level evidence: the full task description,
        which agent executed it, and the complete output.

        Args:
            task_output: CrewAI TaskOutput object
        """
        if not self.capture_tasks:
            return

        try:
            # Extract task details
            description = getattr(task_output, 'description', '')
            raw_output = getattr(task_output, 'raw', '')
            pydantic_output = getattr(task_output, 'pydantic', None)
            json_output = getattr(task_output, 'json_dict', None)
            agent = getattr(task_output, 'agent', None)

            task_info = {
                "description": description,
                "has_pydantic_output": pydantic_output is not None,
                "has_json_output": json_output is not None,
            }

            agent_info = {}
            if agent:
                agent_info = {
                    "role": getattr(agent, 'role', 'unknown') if hasattr(agent, 'role') else str(agent),
                }

            outputs = raw_output or str(pydantic_output) or str(json_output) or ""

            receipt = self._create_receipt(
                event_type="task_completion",
                inputs=description,
                outputs=outputs,
                agent_info=agent_info,
                task_info=task_info,
                status="SUCCESS",
            )

            logger.info(
                f"Task receipt {receipt['receipt_id']} | "
                f"agent={agent_info.get('role', 'unknown')}"
            )

        except Exception as e:
            logger.error(f"Error in task_callback: {e}", exc_info=True)

    # ─── Decorator Interface ────────────────────────────────────────

    def track_crew(self, crew: Any) -> Any:
        """
        Attach Veratum evidence to a CrewAI Crew instance.

        Alternative to passing callbacks manually. Modifies the crew
        in-place to add step and task callbacks.

        Usage:
            crew = Crew(agents=[...], tasks=[...])
            handler.track_crew(crew)
            result = crew.kickoff()

        Args:
            crew: CrewAI Crew instance

        Returns:
            The same crew instance (modified in-place)
        """
        if self.capture_steps:
            crew.step_callback = self.step_callback
        if self.capture_tasks:
            crew.task_callback = self.task_callback

        logger.info("Veratum evidence tracking attached to CrewAI crew")
        return crew

    # ─── Public API ─────────────────────────────────────────────────

    def get_receipts(self) -> List[Dict[str, Any]]:
        """Get all captured receipts."""
        with self._receipt_lock:
            return self._receipts.copy()

    def get_receipt_count(self) -> int:
        """Get the number of captured receipts."""
        return len(self._receipts)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all evidence captured.

        Returns:
            Summary dict with counts by event type, agents, and status.
        """
        with self._receipt_lock:
            receipts = self._receipts.copy()

        by_type = {}
        by_agent = {}
        by_status = {"SUCCESS": 0, "FAILED": 0}

        for r in receipts:
            event_type = r.get("event_type", "unknown")
            by_type[event_type] = by_type.get(event_type, 0) + 1

            agent = r.get("agent", {})
            if agent:
                role = agent.get("role", "unknown")
                by_agent[role] = by_agent.get(role, 0) + 1

            status = r.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_receipts": len(receipts),
            "by_event_type": by_type,
            "by_agent": by_agent,
            "by_status": by_status,
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
    capture_steps: bool = True,
    capture_tasks: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> VeratumCrewAIHandler:
    """
    Create a Veratum handler for CrewAI.

    One-liner to add compliance evidence to any CrewAI crew.

    Usage:
        from veratum.integrations.crewai_plugin import enable_veratum

        handler = enable_veratum()

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            step_callback=handler.step_callback,
            task_callback=handler.task_callback,
        )

        # Or use track_crew:
        handler.track_crew(crew)

    Args:
        api_key: Veratum API key
        endpoint: Veratum API endpoint
        capture_steps: Capture step events (default: True)
        capture_tasks: Capture task events (default: True)
        metadata: Default metadata for all receipts

    Returns:
        VeratumCrewAIHandler instance
    """
    return VeratumCrewAIHandler(
        api_key=api_key,
        endpoint=endpoint,
        capture_steps=capture_steps,
        capture_tasks=capture_tasks,
        metadata=metadata,
    )


__all__ = [
    "VeratumCrewAIHandler",
    "enable_veratum",
]
