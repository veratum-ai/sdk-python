"""
Veratum integrations package.

Provides drop-in evidence layer for popular LLM platforms and frameworks.
This package contains plugins for:
- LangChain (chains, agents, tools)
- CrewAI (multi-agent crews)
- OpenAI Agents SDK (agent runs, tool calls)
- Haystack (pipeline components)
- LiteLLM (140+ LLM providers)
- Portkey AI Gateway
- Model Context Protocol (MCP)
"""

__version__ = "2.1.0"
__all__ = [
    "langchain_callback",
    "crewai_plugin",
    "openai_agents_plugin",
    "haystack_plugin",
    "litellm_plugin",
    "portkey_plugin",
]
