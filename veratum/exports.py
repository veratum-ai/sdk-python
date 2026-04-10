"""
Veratum Export and Webhook System

Enterprise customers need to send Veratum data to existing tools:
- SIEM systems (Splunk, Datadog)
- Compliance platforms (ServiceNow, Jira)
- Data lakes (S3, BigQuery)
- Notification systems (Slack, PagerDuty)

This module provides flexible, extensible export destinations and an event
emission system with thread-safe async delivery.

Example Usage:
    >>> from veratum.exports import ExportManager, SlackDestination, FileDestination
    >>>
    >>> manager = ExportManager()
    >>> manager.add(
    ...     SlackDestination(webhook_url="https://hooks.slack.com/services/..."),
    ...     events=["threat_blocked", "budget_exceeded"]
    ... )
    >>> manager.add(
    ...     FileDestination(path="/var/log/veratum/events.jsonl"),
    ...     events="*"  # All events
    ... )
    >>>
    >>> event = ExportEvent(
    ...     event_type="threat_blocked",
    ...     severity="critical",
    ...     data={"threat": "prompt_injection", "user": "user-123", "model": "gpt-4"}
    ... )
    >>> manager.emit(event)  # Non-blocking

Supported Event Types:
    - "receipt": Proof of processing
    - "threat_blocked": Security threat detected and blocked
    - "cost_anomaly": Unusual spending patterns
    - "compliance_gap": Policy violation
    - "pii_detected": Sensitive data detected
    - "shadow_ai_found": Unauthorized AI tool
    - "budget_exceeded": Cost limit breached
    - "dpia_generated": Data Protection Impact Assessment created
"""

import json
import threading
import queue
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable, List, Set, Union, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import urllib.request
import urllib.error
import io


logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EventType(str, Enum):
    """Supported event types."""
    RECEIPT = "receipt"
    THREAT_BLOCKED = "threat_blocked"
    COST_ANOMALY = "cost_anomaly"
    COMPLIANCE_GAP = "compliance_gap"
    PII_DETECTED = "pii_detected"
    SHADOW_AI_FOUND = "shadow_ai_found"
    BUDGET_EXCEEDED = "budget_exceeded"
    DPIA_GENERATED = "dpia_generated"


@dataclass
class ExportEvent:
    """
    Represents a Veratum event for export.

    Attributes:
        event_type: Type of event (see EventType enum)
        severity: Severity level (info, warning, critical)
        data: Event payload data
        timestamp: ISO 8601 timestamp (auto-generated if not provided)
        source: Source component (prompt_guard, cost_tracker, shadow_ai, etc.)
    """
    event_type: str
    severity: str = Severity.INFO.value
    data: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    source: str = "veratum"

    def __post_init__(self):
        """Validate event data."""
        valid_severities = {s.value for s in Severity}
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity: {self.severity}. Must be one of {valid_severities}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


class ExportDestination(ABC):
    """
    Base class for export destinations.

    All export destinations must implement send() and ideally send_batch()
    for efficient bulk delivery.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        min_severity: str = Severity.INFO.value,
        transform_fn: Optional[Callable[[ExportEvent], ExportEvent]] = None,
    ):
        """
        Initialize destination.

        Args:
            name: Display name for this destination
            min_severity: Minimum severity to send (info, warning, critical)
            transform_fn: Optional function to transform event before sending
        """
        self.name = name or self.__class__.__name__
        self.min_severity = min_severity
        self.transform_fn = transform_fn
        self._severity_order = {
            Severity.INFO.value: 0,
            Severity.WARNING.value: 1,
            Severity.CRITICAL.value: 2,
        }

    def _should_send(self, event: ExportEvent) -> bool:
        """Check if event meets severity threshold."""
        event_level = self._severity_order.get(event.severity, 0)
        min_level = self._severity_order.get(self.min_severity, 0)
        return event_level >= min_level

    def _transform(self, event: ExportEvent) -> ExportEvent:
        """Apply transformation if provided."""
        if self.transform_fn:
            return self.transform_fn(event)
        return event

    @abstractmethod
    def send(self, event: ExportEvent) -> bool:
        """
        Send a single event.

        Args:
            event: The event to send

        Returns:
            True if successful, False otherwise
        """
        pass

    def send_batch(self, events: List[ExportEvent]) -> int:
        """
        Send multiple events. Default implementation sends individually.

        Args:
            events: List of events to send

        Returns:
            Number of successfully sent events
        """
        count = 0
        for event in events:
            if self.send(event):
                count += 1
        return count


class WebhookDestination(ExportDestination):
    """
    Generic HTTP webhook destination.

    Posts events to any HTTP endpoint with configurable headers, auth, and retry.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[tuple] = None,
        retry_count: int = 3,
        timeout: float = 10.0,
        filter_events: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize webhook destination.

        Args:
            url: Target webhook URL
            headers: Optional HTTP headers
            auth: Optional (username, password) tuple for basic auth
            retry_count: Number of retry attempts
            timeout: Request timeout in seconds
            filter_events: List of event types to send (None = all)
            **kwargs: Passed to parent (name, min_severity, transform_fn)
        """
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}
        self.auth = auth
        self.retry_count = retry_count
        self.timeout = timeout
        self.filter_events = set(filter_events) if filter_events else None

    def _should_send(self, event: ExportEvent) -> bool:
        """Check severity and event type filter."""
        if not super()._should_send(event):
            return False
        if self.filter_events and event.event_type not in self.filter_events:
            return False
        return True

    def send(self, event: ExportEvent) -> bool:
        """Send event via HTTP POST."""
        if not self._should_send(event):
            return True  # Filtered out, not an error

        event = self._transform(event)
        payload = event.to_json().encode('utf-8')

        headers = self.headers.copy()
        headers['Content-Type'] = 'application/json'
        headers['Content-Length'] = str(len(payload))

        req = urllib.request.Request(
            self.url,
            data=payload,
            headers=headers,
            method='POST'
        )

        if self.auth:
            import base64
            credentials = base64.b64encode(
                f"{self.auth[0]}:{self.auth[1]}".encode()
            ).decode()
            req.add_header('Authorization', f'Basic {credentials}')

        for attempt in range(self.retry_count):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if 200 <= response.status < 300:
                        logger.debug(f"Webhook sent to {self.name}: {event.event_type}")
                        return True
            except urllib.error.HTTPError as e:
                logger.warning(
                    f"Webhook {self.name} failed (attempt {attempt + 1}/{self.retry_count}): "
                    f"{e.code} {e.reason}"
                )
            except urllib.error.URLError as e:
                logger.warning(
                    f"Webhook {self.name} network error (attempt {attempt + 1}/{self.retry_count}): "
                    f"{e.reason}"
                )
            except Exception as e:
                logger.error(f"Webhook {self.name} error: {e}")

            if attempt < self.retry_count - 1:
                time.sleep(0.5 * (2 ** attempt))  # Exponential backoff

        return False


class SlackDestination(ExportDestination):
    """
    Slack webhook destination with formatted messages.

    Sends richly formatted messages with severity indicators and structured blocks.
    """

    SEVERITY_EMOJI = {
        Severity.CRITICAL.value: "🔴",
        Severity.WARNING.value: "🟡",
        Severity.INFO.value: "🟢",
    }

    SEVERITY_COLOR = {
        Severity.CRITICAL.value: "#FF0000",
        Severity.WARNING.value: "#FFA500",
        Severity.INFO.value: "#00AA00",
    }

    def __init__(self, webhook_url: str, **kwargs):
        """
        Initialize Slack destination.

        Args:
            webhook_url: Slack incoming webhook URL
            **kwargs: Passed to parent (name, min_severity, transform_fn)
        """
        super().__init__(**kwargs)
        self.webhook = WebhookDestination(
            url=webhook_url,
            name=self.name,
            min_severity=self.min_severity,
        )

    def _format_slack_message(self, event: ExportEvent) -> Dict[str, Any]:
        """Format event as Slack message block."""
        emoji = self.SEVERITY_EMOJI.get(event.severity, "⚪")
        color = self.SEVERITY_COLOR.get(event.severity, "#808080")

        # Build fields from event data
        fields = [
            {"type": "mrkdwn", "text": f"*Event Type:*\n{event.event_type}"},
            {"type": "mrkdwn", "text": f"*Severity:*\n{emoji} {event.severity}"},
            {"type": "mrkdwn", "text": f"*Source:*\n{event.source}"},
            {"type": "mrkdwn", "text": f"*Time:*\n{event.timestamp}"},
        ]

        # Add custom data fields
        if event.data:
            for key, value in list(event.data.items())[:4]:  # Limit to 4 fields
                value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                fields.append({"type": "mrkdwn", "text": f"*{key}:*\n{value_str}"})

        return {
            "attachments": [
                {
                    "fallback": f"{emoji} {event.event_type}: {event.severity}",
                    "color": color,
                    "title": f"{emoji} {event.event_type}",
                    "blocks": [
                        {
                            "type": "section",
                            "fields": fields,
                        }
                    ],
                    "ts": int(datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')).timestamp()),
                }
            ]
        }

    def send(self, event: ExportEvent) -> bool:
        """Send formatted event to Slack."""
        if not self._should_send(event):
            return True

        event = self._transform(event)

        try:
            payload = json.dumps(self._format_slack_message(event)).encode('utf-8')
            req = urllib.request.Request(
                self.webhook.url,
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=self.webhook.timeout) as response:
                if 200 <= response.status < 300:
                    logger.debug(f"Slack message sent: {event.event_type}")
                    return True
        except Exception as e:
            logger.error(f"Slack send failed: {e}")

        return False


class SIEMDestination(ExportDestination):
    """
    SIEM destination for Splunk, Datadog, etc.

    Outputs events in CEF (Common Event Format) or JSON format.
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        format: str = "json",
        **kwargs
    ):
        """
        Initialize SIEM destination.

        Args:
            endpoint: SIEM endpoint URL
            token: Authentication token/API key
            format: Output format ("json" or "cef")
            **kwargs: Passed to parent
        """
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.token = token
        self.format = format.lower()
        if self.format not in ("json", "cef"):
            raise ValueError("Format must be 'json' or 'cef'")

    def _to_cef(self, event: ExportEvent) -> str:
        """Convert event to CEF format."""
        # CEF:0|Veratum|Veratum|1.0|event_type|Event|severity|extensions
        severity_value = {"info": 3, "warning": 6, "critical": 9}.get(event.severity, 5)

        extensions = [
            f"src={event.source}",
            f"rt={event.timestamp}",
        ]

        # Add event data as extensions
        for key, value in event.data.items():
            safe_key = key.replace(' ', '_').replace('=', '_')
            safe_value = str(value).replace('=', '_').replace('\n', '\\n')[:512]
            extensions.append(f"{safe_key}={safe_value}")

        ext_str = " ".join(extensions)
        return f"CEF:0|Veratum|Veratum|1.0|{event.event_type}|Veratum Event|{severity_value}|{ext_str}"

    def send(self, event: ExportEvent) -> bool:
        """Send event to SIEM."""
        if not self._should_send(event):
            return True

        event = self._transform(event)

        if self.format == "cef":
            payload = self._to_cef(event).encode('utf-8')
        else:
            payload = event.to_json().encode('utf-8')

        headers = {
            'Content-Type': 'application/json' if self.format == "json" else 'text/plain',
            'Authorization': f'Bearer {self.token}',
            'Content-Length': str(len(payload)),
        }

        try:
            req = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if 200 <= response.status < 300:
                    logger.debug(f"SIEM event sent: {event.event_type}")
                    return True
        except Exception as e:
            logger.error(f"SIEM send failed: {e}")

        return False


class FileDestination(ExportDestination):
    """
    File-based destination for audit trails and local logging.

    Appends events to JSONL file with optional rotation.
    """

    def __init__(
        self,
        path: str,
        rotate_mb: int = 100,
        **kwargs
    ):
        """
        Initialize file destination.

        Args:
            path: Path to JSONL log file
            rotate_mb: Rotate file when it exceeds this size (0 = no rotation)
            **kwargs: Passed to parent
        """
        super().__init__(**kwargs)
        self.path = path
        self.rotate_mb = rotate_mb
        self._lock = threading.Lock()

    def _rotate_if_needed(self):
        """Rotate file if it exceeds size limit."""
        if self.rotate_mb <= 0:
            return

        try:
            import os
            if os.path.exists(self.path):
                size_mb = os.path.getsize(self.path) / (1024 * 1024)
                if size_mb > self.rotate_mb:
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    rotated = f"{self.path}.{timestamp}"
                    os.rename(self.path, rotated)
                    logger.info(f"Rotated log file to {rotated}")
        except Exception as e:
            logger.error(f"File rotation failed: {e}")

    def send(self, event: ExportEvent) -> bool:
        """Append event to file."""
        if not self._should_send(event):
            return True

        event = self._transform(event)

        with self._lock:
            try:
                self._rotate_if_needed()

                # Ensure directory exists
                import os
                os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)

                with open(self.path, 'a') as f:
                    f.write(event.to_json() + '\n')

                logger.debug(f"File event logged: {event.event_type}")
                return True
            except Exception as e:
                logger.error(f"File write failed: {e}")
                return False


class S3Destination(ExportDestination):
    """
    AWS S3 destination for data lake ingestion.

    Requires boto3 (optional dependency).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "events/",
        region: str = "us-east-1",
        **kwargs
    ):
        """
        Initialize S3 destination.

        Args:
            bucket: S3 bucket name
            prefix: Object key prefix (e.g., "events/")
            region: AWS region
            **kwargs: Passed to parent
        """
        super().__init__(**kwargs)
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/'
        self.region = region
        self._s3 = None
        self._lock = threading.Lock()
        self._init_s3()

    def _init_s3(self):
        """Lazy-load boto3."""
        try:
            import boto3
            self._s3 = boto3.client('s3', region_name=self.region)
        except ImportError:
            logger.warning("boto3 not installed; S3 exports will fail")

    def send(self, event: ExportEvent) -> bool:
        """Upload event to S3."""
        if not self._should_send(event):
            return True

        if not self._s3:
            logger.error("S3 not available; install boto3")
            return False

        event = self._transform(event)

        # Use timestamp + event type for key
        timestamp = event.timestamp.replace(':', '').replace('.', '_')
        key = f"{self.prefix}{event.event_type}/{timestamp}_{event.source}.json"

        try:
            self._s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=event.to_json(),
                ContentType='application/json',
            )
            logger.debug(f"S3 event uploaded: {key}")
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False


class CallbackDestination(ExportDestination):
    """
    Custom callback destination for user-defined handlers.
    """

    def __init__(
        self,
        fn: Callable[[ExportEvent], bool],
        **kwargs
    ):
        """
        Initialize callback destination.

        Args:
            fn: Function that takes ExportEvent and returns bool (success)
            **kwargs: Passed to parent
        """
        super().__init__(**kwargs)
        self.fn = fn

    def send(self, event: ExportEvent) -> bool:
        """Call user function with event."""
        if not self._should_send(event):
            return True

        event = self._transform(event)

        try:
            return bool(self.fn(event))
        except Exception as e:
            logger.error(f"Callback failed: {e}")
            return False


class ExportManager:
    """
    Central event emission and routing manager.

    Handles thread-safe async delivery to multiple destinations with
    filtering, transformation, and error handling.
    """

    def __init__(self, num_workers: int = 1, queue_size: int = 1000):
        """
        Initialize export manager.

        Args:
            num_workers: Number of background worker threads
            queue_size: Maximum queue size before blocking
        """
        self._destinations: Dict[str, tuple[ExportDestination, Set[str]]] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._workers = []

        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"VeratumExportWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)

    def _worker_loop(self):
        """Background worker that processes queued events."""
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=1)
                if event is None:  # Shutdown signal
                    break

                self._dispatch(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _dispatch(self, event: ExportEvent):
        """Send event to matching destinations."""
        for dest_name, (dest, event_types) in self._destinations.items():
            # Check event type filter
            if "*" not in event_types and event.event_type not in event_types:
                continue

            try:
                dest.send(event)
            except Exception as e:
                logger.error(f"Dispatch to {dest_name} failed: {e}")

    def add(
        self,
        destination: ExportDestination,
        events: Union[str, List[str]] = "*",
        name: Optional[str] = None,
    ) -> str:
        """
        Register a destination.

        Args:
            destination: ExportDestination instance
            events: Event types to send ("*" for all, or list of event types)
            name: Display name (defaults to destination.name)

        Returns:
            Destination name/ID
        """
        dest_name = name or destination.name

        if isinstance(events, str):
            event_types = {"*"} if events == "*" else set()
        else:
            event_types = set(events)

        self._destinations[dest_name] = (destination, event_types)
        logger.info(f"Registered destination: {dest_name} for events {event_types}")
        return dest_name

    def remove(self, name: str) -> bool:
        """
        Unregister a destination.

        Args:
            name: Destination name

        Returns:
            True if found and removed
        """
        if name in self._destinations:
            del self._destinations[name]
            logger.info(f"Removed destination: {name}")
            return True
        return False

    def emit(self, event: ExportEvent) -> None:
        """
        Emit an event for async processing.

        Non-blocking; queues event for background delivery.

        Args:
            event: Event to emit
        """
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.error("Export queue full; event dropped")

    def emit_sync(self, event: ExportEvent) -> Dict[str, bool]:
        """
        Emit an event synchronously.

        Blocks until all destinations have processed the event.

        Args:
            event: Event to emit

        Returns:
            Dict mapping destination names to success (True/False)
        """
        results = {}
        for dest_name, (dest, event_types) in self._destinations.items():
            if "*" not in event_types and event.event_type not in event_types:
                results[dest_name] = True  # Filtered
                continue

            try:
                results[dest_name] = dest.send(event)
            except Exception as e:
                logger.error(f"Sync emit to {dest_name} failed: {e}")
                results[dest_name] = False

        return results

    def flush(self, timeout: float = 5.0) -> None:
        """
        Wait for all queued events to be processed.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start = time.time()
        while not self._queue.empty() and (time.time() - start) < timeout:
            time.sleep(0.1)

    def stop(self, timeout: float = 5.0) -> None:
        """
        Shutdown the manager and wait for workers.

        Args:
            timeout: Maximum time to wait for workers
        """
        self.flush(timeout=timeout)
        self._stop_event.set()

        for worker in self._workers:
            worker.join(timeout=timeout)

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, *args):
        """Context manager cleanup."""
        self.stop()
