"""
Shadow AI discovery and inventory management.

Detects unauthorized/unmonitored AI model usage across an organization.
Shadow AI is one of the biggest enterprise compliance risks — employees
using ChatGPT, Claude, Gemini, or other AI services without governance.

Zenity charges $150K+ for shadow AI discovery. Veratum provides it
as a module that integrates with your existing network monitoring,
proxy logs, or DNS logs.

Detection methods:
1. DNS/HTTP endpoint matching — identify calls to known AI API endpoints
2. Receipt gap analysis — find models in network logs not in Veratum receipts
3. Token pattern detection — identify API key patterns in outbound traffic
4. User agent fingerprinting — detect AI SDK user agents

Output: Shadow AI inventory with risk assessment per discovery,
       mapped to EU AI Act Article 26 (deployer obligations) and
       organizational AI governance requirements.

Example:
    >>> from veratum.shadow_ai import ShadowAIDetector
    >>>
    >>> detector = ShadowAIDetector()
    >>>
    >>> # Feed network/proxy logs
    >>> detector.ingest_dns_logs([
    ...     {"domain": "api.openai.com", "source_ip": "10.0.1.50", "timestamp": "..."},
    ...     {"domain": "api.anthropic.com", "source_ip": "10.0.1.72", "timestamp": "..."},
    ... ])
    >>>
    >>> # Compare with registered AI usage
    >>> detector.set_registered_models(["gpt-4o"])  # Only GPT-4o is sanctioned
    >>>
    >>> inventory = detector.get_inventory()
    >>> print(inventory.unregistered)  # [ShadowAIInstance(provider="anthropic", ...)]
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("veratum.shadow_ai")


# ---------------------------------------------------------------------------
# Known AI API endpoints and patterns
# ---------------------------------------------------------------------------

AI_ENDPOINTS: Dict[str, Dict[str, Any]] = {
    # OpenAI
    "api.openai.com": {
        "provider": "openai",
        "service": "OpenAI API",
        "risk": "high",
        "models": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o1", "o3", "dall-e"],
    },
    "chat.openai.com": {
        "provider": "openai",
        "service": "ChatGPT (consumer)",
        "risk": "critical",
        "models": ["chatgpt"],
    },
    "chatgpt.com": {
        "provider": "openai",
        "service": "ChatGPT (consumer)",
        "risk": "critical",
        "models": ["chatgpt"],
    },
    # Anthropic
    "api.anthropic.com": {
        "provider": "anthropic",
        "service": "Anthropic API",
        "risk": "high",
        "models": ["claude-opus-4", "claude-sonnet-4", "claude-haiku-3.5"],
    },
    "claude.ai": {
        "provider": "anthropic",
        "service": "Claude (consumer)",
        "risk": "critical",
        "models": ["claude"],
    },
    # Google
    "generativelanguage.googleapis.com": {
        "provider": "google",
        "service": "Gemini API",
        "risk": "high",
        "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
    },
    "gemini.google.com": {
        "provider": "google",
        "service": "Gemini (consumer)",
        "risk": "critical",
        "models": ["gemini"],
    },
    "aistudio.google.com": {
        "provider": "google",
        "service": "Google AI Studio",
        "risk": "high",
        "models": ["gemini"],
    },
    # Mistral
    "api.mistral.ai": {
        "provider": "mistral",
        "service": "Mistral API",
        "risk": "high",
        "models": ["mistral-large", "mistral-medium", "codestral"],
    },
    "chat.mistral.ai": {
        "provider": "mistral",
        "service": "Le Chat (consumer)",
        "risk": "critical",
        "models": ["mistral"],
    },
    # Cohere
    "api.cohere.ai": {
        "provider": "cohere",
        "service": "Cohere API",
        "risk": "high",
        "models": ["command-r-plus", "command-r"],
    },
    # Hugging Face
    "api-inference.huggingface.co": {
        "provider": "huggingface",
        "service": "Hugging Face Inference API",
        "risk": "medium",
        "models": ["various"],
    },
    # Replicate
    "api.replicate.com": {
        "provider": "replicate",
        "service": "Replicate API",
        "risk": "medium",
        "models": ["various"],
    },
    # Together AI
    "api.together.xyz": {
        "provider": "together",
        "service": "Together AI API",
        "risk": "medium",
        "models": ["various"],
    },
    # Groq
    "api.groq.com": {
        "provider": "groq",
        "service": "Groq API",
        "risk": "medium",
        "models": ["llama", "mixtral"],
    },
    # Perplexity
    "api.perplexity.ai": {
        "provider": "perplexity",
        "service": "Perplexity API",
        "risk": "medium",
        "models": ["sonar"],
    },
    "perplexity.ai": {
        "provider": "perplexity",
        "service": "Perplexity (consumer)",
        "risk": "critical",
        "models": ["perplexity"],
    },
    # DeepSeek
    "api.deepseek.com": {
        "provider": "deepseek",
        "service": "DeepSeek API",
        "risk": "critical",  # China-based, data sovereignty concern
        "models": ["deepseek-v3", "deepseek-r1"],
    },
    "chat.deepseek.com": {
        "provider": "deepseek",
        "service": "DeepSeek Chat (consumer)",
        "risk": "critical",
        "models": ["deepseek"],
    },
    # AI gateways / proxies
    "api.helicone.ai": {
        "provider": "helicone",
        "service": "Helicone Gateway",
        "risk": "medium",
        "models": ["proxy"],
    },
    "api.portkey.ai": {
        "provider": "portkey",
        "service": "Portkey Gateway",
        "risk": "medium",
        "models": ["proxy"],
    },
    # Azure OpenAI (pattern match)
    "openai.azure.com": {
        "provider": "azure_openai",
        "service": "Azure OpenAI Service",
        "risk": "medium",
        "models": ["gpt-4o", "gpt-4"],
    },
    # AWS Bedrock
    "bedrock-runtime.*.amazonaws.com": {
        "provider": "aws_bedrock",
        "service": "AWS Bedrock",
        "risk": "medium",
        "models": ["claude", "llama", "titan"],
    },
}

# API key patterns for different providers
API_KEY_PATTERNS: Dict[str, str] = {
    "openai": r"sk-[a-zA-Z0-9]{20,}",
    "anthropic": r"sk-ant-[a-zA-Z0-9-]{20,}",
    "google": r"AIza[0-9A-Za-z_-]{35}",
    "huggingface": r"hf_[a-zA-Z0-9]{20,}",
    "cohere": r"[a-zA-Z0-9]{40}",  # Generic but paired with domain
    "mistral": r"[a-zA-Z0-9]{32}",
    "replicate": r"r8_[a-zA-Z0-9]{37}",
}

# User agent patterns
AI_USER_AGENTS: List[Tuple[str, str]] = [
    (r"openai-python", "OpenAI Python SDK"),
    (r"anthropic-python", "Anthropic Python SDK"),
    (r"anthropic-typescript", "Anthropic TypeScript SDK"),
    (r"google-generativeai", "Google GenAI SDK"),
    (r"langchain", "LangChain framework"),
    (r"llama[-_]index", "LlamaIndex framework"),
    (r"crewai", "CrewAI framework"),
    (r"haystack", "Haystack framework"),
    (r"litellm", "LiteLLM proxy"),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShadowAIInstance:
    """A single discovered shadow AI usage."""
    provider: str
    service: str
    risk: str  # "low", "medium", "high", "critical"
    domain: str
    first_seen: str = ""
    last_seen: str = ""
    request_count: int = 0
    source_ips: List[str] = field(default_factory=list)
    users: List[str] = field(default_factory=list)
    is_registered: bool = False
    is_consumer: bool = False  # Consumer app vs API
    data_residency: str = ""  # Country where data is sent
    models_detected: List[str] = field(default_factory=list)
    api_keys_detected: int = 0
    compliance_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowAIInventory:
    """Complete shadow AI inventory for the organization."""
    scan_timestamp: str = ""
    total_discoveries: int = 0
    registered: List[ShadowAIInstance] = field(default_factory=list)
    unregistered: List[ShadowAIInstance] = field(default_factory=list)
    by_risk: Dict[str, int] = field(default_factory=dict)
    by_provider: Dict[str, int] = field(default_factory=dict)
    consumer_apps_detected: int = 0
    api_usage_detected: int = 0
    compliance_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["registered"] = [i.to_dict() for i in self.registered]
        d["unregistered"] = [i.to_dict() for i in self.unregistered]
        return d


# ---------------------------------------------------------------------------
# Shadow AI Detector
# ---------------------------------------------------------------------------

class ShadowAIDetector:
    """
    Shadow AI discovery engine.

    Ingests network data (DNS logs, HTTP logs, proxy logs) and
    identifies unauthorized AI service usage. Compares against
    a registry of sanctioned AI tools to flag shadow usage.

    Usage:
        detector = ShadowAIDetector()
        detector.set_registered_models(["gpt-4o", "claude-sonnet-4"])
        detector.ingest_dns_logs(dns_entries)
        detector.ingest_http_logs(http_entries)
        inventory = detector.get_inventory()
    """

    def __init__(
        self,
        *,
        registered_providers: Optional[Set[str]] = None,
        registered_models: Optional[Set[str]] = None,
        registered_domains: Optional[Set[str]] = None,
    ):
        self.registered_providers = registered_providers or set()
        self.registered_models = registered_models or set()
        self.registered_domains = registered_domains or set()

        # Discovery state
        self._discoveries: Dict[str, ShadowAIInstance] = {}
        self._compiled_endpoints: Dict[str, Dict[str, Any]] = {}

        # Compile endpoint patterns
        for domain, info in AI_ENDPOINTS.items():
            if "*" in domain:
                # Wildcard domain — compile as regex
                pattern = domain.replace(".", r"\.").replace("*", r"[a-z0-9-]+")
                self._compiled_endpoints[pattern] = {**info, "pattern": True, "domain": domain}
            else:
                self._compiled_endpoints[domain] = {**info, "pattern": False, "domain": domain}

        self._ua_compiled = [
            (re.compile(p, re.IGNORECASE), desc) for p, desc in AI_USER_AGENTS
        ]
        self._key_compiled = {
            provider: re.compile(pattern)
            for provider, pattern in API_KEY_PATTERNS.items()
        }

    def set_registered_models(self, models: List[str]) -> None:
        """Set the list of sanctioned/registered AI models."""
        self.registered_models = set(models)

    def set_registered_providers(self, providers: List[str]) -> None:
        """Set the list of sanctioned AI providers."""
        self.registered_providers = set(providers)

    def set_registered_domains(self, domains: List[str]) -> None:
        """Set domains that are sanctioned for AI usage."""
        self.registered_domains = set(domains)

    def ingest_dns_logs(self, entries: List[Dict[str, Any]]) -> int:
        """
        Ingest DNS query logs.

        Expected fields per entry:
            - domain (str): Queried domain
            - source_ip (str, optional): Source IP
            - timestamp (str, optional): Query time
            - user (str, optional): User identifier

        Returns:
            Number of AI-related domains detected.
        """
        detected = 0
        for entry in entries:
            domain = entry.get("domain", "").lower().strip(".")
            match = self._match_domain(domain)
            if match:
                detected += 1
                self._record_discovery(
                    domain=domain,
                    endpoint_info=match,
                    source_ip=entry.get("source_ip"),
                    timestamp=entry.get("timestamp"),
                    user=entry.get("user"),
                )
        return detected

    def ingest_http_logs(self, entries: List[Dict[str, Any]]) -> int:
        """
        Ingest HTTP/HTTPS proxy logs.

        Expected fields per entry:
            - url or domain (str): Request URL or domain
            - source_ip (str, optional): Source IP
            - user_agent (str, optional): HTTP User-Agent
            - timestamp (str, optional): Request time
            - user (str, optional): Authenticated user
            - headers (dict, optional): HTTP headers

        Returns:
            Number of AI-related requests detected.
        """
        detected = 0
        for entry in entries:
            url = entry.get("url", "")
            domain = entry.get("domain", "")

            # Extract domain from URL if needed
            if url and not domain:
                import re as _re
                m = _re.match(r"https?://([^/]+)", url)
                if m:
                    domain = m.group(1).lower()

            match = self._match_domain(domain)
            if match:
                detected += 1
                discovery = self._record_discovery(
                    domain=domain,
                    endpoint_info=match,
                    source_ip=entry.get("source_ip"),
                    timestamp=entry.get("timestamp"),
                    user=entry.get("user"),
                )

                # Check user agent
                ua = entry.get("user_agent", "")
                if ua:
                    for pattern, desc in self._ua_compiled:
                        if pattern.search(ua):
                            if desc not in (discovery.models_detected or []):
                                discovery.models_detected.append(f"sdk:{desc}")

                # Check for API keys in headers
                headers = entry.get("headers", {})
                auth = headers.get("Authorization", "") or headers.get("authorization", "")
                if auth:
                    for provider, pattern in self._key_compiled.items():
                        if pattern.search(auth):
                            discovery.api_keys_detected += 1

        return detected

    def ingest_receipts(self, receipts: List[Dict[str, Any]]) -> None:
        """
        Ingest Veratum receipts to build registered model inventory.

        Automatically adds models and providers found in receipts
        to the registered list.
        """
        for r in receipts:
            model = r.get("model")
            provider = r.get("provider")
            if model:
                self.registered_models.add(model)
            if provider:
                self.registered_providers.add(provider)

    def get_inventory(self) -> ShadowAIInventory:
        """
        Get the complete shadow AI inventory.

        Returns:
            ShadowAIInventory with registered and unregistered AI usage.
        """
        registered = []
        unregistered = []
        by_risk: Dict[str, int] = defaultdict(int)
        by_provider: Dict[str, int] = defaultdict(int)
        consumer_count = 0
        api_count = 0

        for key, instance in self._discoveries.items():
            # Determine if registered
            is_registered = (
                instance.provider in self.registered_providers
                or instance.domain in self.registered_domains
                or any(m in self.registered_models for m in instance.models_detected)
            )
            instance.is_registered = is_registered

            # Detect consumer vs API
            is_consumer = any(
                x in instance.service.lower() for x in ("consumer", "chat")
            )
            instance.is_consumer = is_consumer

            # Compliance issues
            issues = self._assess_compliance_issues(instance)
            instance.compliance_issues = issues

            if is_registered:
                registered.append(instance)
            else:
                unregistered.append(instance)

            by_risk[instance.risk] += 1
            by_provider[instance.provider] += 1
            if is_consumer:
                consumer_count += 1
            else:
                api_count += 1

        # Sort unregistered by risk (critical first)
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        unregistered.sort(key=lambda x: risk_order.get(x.risk, 9))

        # Compliance summary
        compliance = {
            "eu_ai_act_art26": {
                "status": "non_compliant" if unregistered else "compliant",
                "description": "All AI systems must be registered and monitored (deployer obligations)",
                "unregistered_count": len(unregistered),
            },
            "shadow_ai_risk": {
                "level": "critical" if any(u.risk == "critical" for u in unregistered) else
                         "high" if any(u.risk == "high" for u in unregistered) else
                         "medium" if unregistered else "low",
                "consumer_apps": consumer_count,
                "api_usage": api_count,
            },
            "data_residency": {
                "providers_outside_eu": [
                    u.provider for u in unregistered
                    if u.provider in ("deepseek",)  # Known China-hosted
                ],
            },
        }

        return ShadowAIInventory(
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
            total_discoveries=len(self._discoveries),
            registered=registered,
            unregistered=unregistered,
            by_risk=dict(by_risk),
            by_provider=dict(by_provider),
            consumer_apps_detected=consumer_count,
            api_usage_detected=api_count,
            compliance_summary=compliance,
        )

    # --- Internal methods ---------------------------------------------------

    def _match_domain(self, domain: str) -> Optional[Dict[str, Any]]:
        """Match a domain against known AI endpoints."""
        domain = domain.lower().strip(".")

        # Direct match first
        if domain in self._compiled_endpoints:
            info = self._compiled_endpoints[domain]
            if not info.get("pattern"):
                return info

        # Pattern match (wildcards like *.amazonaws.com)
        for pattern, info in self._compiled_endpoints.items():
            if info.get("pattern"):
                if re.match(pattern, domain):
                    return info

        # Subdomain match (e.g., us-east-1.api.openai.com → api.openai.com)
        parts = domain.split(".")
        for i in range(len(parts)):
            subdomain = ".".join(parts[i:])
            if subdomain in self._compiled_endpoints:
                info = self._compiled_endpoints[subdomain]
                if not info.get("pattern"):
                    return info

        return None

    def _record_discovery(
        self,
        domain: str,
        endpoint_info: Dict[str, Any],
        source_ip: Optional[str] = None,
        timestamp: Optional[str] = None,
        user: Optional[str] = None,
    ) -> ShadowAIInstance:
        """Record a shadow AI discovery."""
        key = f"{endpoint_info['provider']}:{endpoint_info['service']}"

        if key not in self._discoveries:
            self._discoveries[key] = ShadowAIInstance(
                provider=endpoint_info["provider"],
                service=endpoint_info["service"],
                risk=endpoint_info["risk"],
                domain=domain,
                first_seen=timestamp or datetime.now(timezone.utc).isoformat(),
                last_seen=timestamp or datetime.now(timezone.utc).isoformat(),
                request_count=0,
                models_detected=endpoint_info.get("models", []).copy(),
            )

        instance = self._discoveries[key]
        instance.request_count += 1
        if timestamp:
            instance.last_seen = max(instance.last_seen, timestamp)

        if source_ip and source_ip not in instance.source_ips:
            instance.source_ips.append(source_ip)
        if user and user not in instance.users:
            instance.users.append(user)

        return instance

    @staticmethod
    def _assess_compliance_issues(instance: ShadowAIInstance) -> List[str]:
        """Assess compliance issues for a shadow AI instance."""
        issues = []

        if not instance.is_registered:
            issues.append(
                "EU AI Act Art. 26: Unregistered AI system — deployers must "
                "maintain inventory of all AI systems in use"
            )

        if instance.is_consumer:
            issues.append(
                "Consumer AI apps may send corporate data to third-party "
                "servers without DPA or adequate safeguards"
            )
            issues.append(
                "GDPR Art. 28: No data processing agreement in place "
                "for consumer AI service"
            )

        if instance.provider == "deepseek":
            issues.append(
                "Data residency: Provider based in China — potential "
                "conflict with EU data adequacy requirements"
            )

        if instance.risk == "critical":
            issues.append(
                "CRITICAL: Immediate action required — high-risk "
                "unmonitored AI usage detected"
            )

        return issues


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def scan_domains(domains: List[str]) -> List[Dict[str, Any]]:
    """
    Quick scan a list of domains for AI endpoints.

    Usage:
        from veratum.shadow_ai import scan_domains
        results = scan_domains(["api.openai.com", "google.com", "api.anthropic.com"])
        # Returns only AI-related domains with provider info
    """
    detector = ShadowAIDetector()
    results = []
    for domain in domains:
        match = detector._match_domain(domain)
        if match:
            results.append({
                "domain": domain,
                "provider": match["provider"],
                "service": match["service"],
                "risk": match["risk"],
            })
    return results


__all__ = [
    "ShadowAIDetector",
    "ShadowAIInstance",
    "ShadowAIInventory",
    "scan_domains",
    "AI_ENDPOINTS",
]
