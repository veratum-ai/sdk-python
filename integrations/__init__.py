"""
Veratum integrations package.

Drop-in evidence layer for popular LLM platforms and frameworks.

Supported integrations:
- LangChain: Callback handler for chains, agents, and tools
- CrewAI: Step and task callbacks for multi-agent crews
- OpenAI Agents: Tracing processor for agent runs and tool calls
- Haystack: Tracer for pipeline component execution
- LiteLLM: Callback for 140+ LLM providers
- Portkey: Middleware for Portkey AI Gateway

Usage:
    # LangChain
    from veratum.integrations.langchain_callback import enable_veratum
    handler = enable_veratum()

    # CrewAI
    from veratum.integrations.crewai_plugin import enable_veratum
    handler = enable_veratum()

    # OpenAI Agents
    from veratum.integrations.openai_agents_plugin import enable_veratum
    processor = enable_veratum()

    # Haystack
    from veratum.integrations.haystack_plugin import enable_veratum
    tracer = enable_veratum()

    # LiteLLM
    from veratum.integrations.litellm_plugin import enable_veratum
    callback = enable_veratum()

    # Portkey
    from veratum.integrations.portkey_plugin import wrap_portkey
    middleware = wrap_portkey(client)
"""

__version__ = "2.0.0"

# Lazy imports — each integration has its own dependency requirements
# Import individually to avoid requiring all framework dependencies

__all__ = [
    "langchain_callback",
    "crewai_plugin",
    "openai_agents_plugin",
    "haystack_plugin",
    "litellm_plugin",
    "portkey_plugin",
]
