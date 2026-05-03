"""Helpers for discovering callable Home Assistant conversation agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .const import DOMAIN

LLM_AGENT_DOMAINS = {
    "openai_conversation",
    "google_generative_ai_conversation",
    "ollama",
    "anthropic",
    "llama_conversation",
    "chatgpt",
    "extended_openai_conversation",
}


@dataclass(slots=True)
class ConversationAgentDescriptor:
    """A callable conversation agent surfaced by Home Assistant."""

    agent_id: str
    label: str
    domain: str | None


def get_registered_conversation_agents(hass: Any) -> list[ConversationAgentDescriptor]:
    """Return callable conversation agents currently registered with Home Assistant."""
    try:
        from homeassistant.components.conversation.agent_manager import get_agent_manager
    except Exception:
        return []

    descriptors: list[ConversationAgentDescriptor] = []
    seen: set[str] = set()
    manager = get_agent_manager(hass)

    for info in manager.async_get_agent_info():
        agent_id = str(info.id)
        if agent_id in seen:
            continue

        domain: str | None = None
        if "." not in agent_id:
            entry = hass.config_entries.async_get_entry(agent_id)
            if entry is None:
                continue
            domain = entry.domain
            if domain == DOMAIN:
                continue

        seen.add(agent_id)
        descriptors.append(
            ConversationAgentDescriptor(
                agent_id=agent_id,
                label=str(info.name),
                domain=domain,
            )
        )

    descriptors.sort(key=lambda item: item.label.lower())
    return descriptors


def get_registered_llm_agents(hass: Any) -> list[ConversationAgentDescriptor]:
    """Return callable LLM-capable conversation agents currently registered."""
    agents: list[ConversationAgentDescriptor] = []
    for descriptor in get_registered_conversation_agents(hass):
        domain = descriptor.domain or ""
        if domain in LLM_AGENT_DOMAINS or "conversation" in domain:
            agents.append(descriptor)
    return agents
