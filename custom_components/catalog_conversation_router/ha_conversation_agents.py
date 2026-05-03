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
    descriptors: list[ConversationAgentDescriptor] = []
    seen: set[str] = set()

    _append_manager_agents(hass, descriptors, seen)
    _append_conversation_entities(hass, descriptors, seen)
    _append_conversation_subentry_entries(hass, descriptors, seen)
    _append_config_entry_fallback(hass, descriptors, seen)

    descriptors.sort(key=lambda item: item.label.lower())
    return descriptors


def get_registered_llm_agents(hass: Any) -> list[ConversationAgentDescriptor]:
    """Return callable LLM-capable conversation agents currently registered."""
    agents: list[ConversationAgentDescriptor] = []
    for descriptor in get_registered_conversation_agents(hass):
        domain = (descriptor.domain or "").lower()
        label = descriptor.label.lower()
        if (
            domain in LLM_AGENT_DOMAINS
            or "conversation" in domain
            or any(token in label for token in ("openai", "ollama", "anthropic", "llama", "qwen"))
        ):
            agents.append(descriptor)
    return agents


def _append_manager_agents(
    hass: Any,
    descriptors: list[ConversationAgentDescriptor],
    seen: set[str],
) -> None:
    """Append agents exposed through HA's conversation agent manager."""
    try:
        from homeassistant.components.conversation.agent_manager import get_agent_manager
    except Exception:
        return

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


def _append_conversation_entities(
    hass: Any,
    descriptors: list[ConversationAgentDescriptor],
    seen: set[str],
) -> None:
    """Append conversation agents exposed as conversation entities."""
    try:
        from homeassistant.components import conversation
    except Exception:
        return

    component = hass.data.get(getattr(conversation, "DATA_COMPONENT", None))
    if component is None:
        return

    for entity in getattr(component, "entities", []):
        agent_id = getattr(entity, "entity_id", None)
        if not agent_id:
            continue
        agent_id = str(agent_id)
        if agent_id in seen:
            continue

        config_entry_id = getattr(entity, "config_entry_id", None)
        domain: str | None = None
        if config_entry_id:
            entry = hass.config_entries.async_get_entry(config_entry_id)
            if entry is not None:
                domain = entry.domain
                if domain == DOMAIN:
                    continue

        if domain is None:
            platform = getattr(entity, "platform", None)
            platform_name = getattr(platform, "platform_name", None)
            domain = str(platform_name) if platform_name else None
            if domain == DOMAIN:
                continue

        seen.add(agent_id)
        descriptors.append(
            ConversationAgentDescriptor(
                agent_id=agent_id,
                label=str(getattr(entity, "name", None) or agent_id),
                domain=domain,
            )
        )


def _append_config_entry_fallback(
    hass: Any,
    descriptors: list[ConversationAgentDescriptor],
    seen: set[str],
) -> None:
    """Fallback to likely conversation config entries when no runtime agents are discoverable."""
    for entry in hass.config_entries.async_entries():
        state_name = getattr(getattr(entry, "state", None), "name", "")
        if state_name and state_name != "LOADED":
            continue
        domain = entry.domain
        if domain == DOMAIN:
            continue
        if domain not in LLM_AGENT_DOMAINS and "conversation" not in domain:
            continue
        agent_id = entry.entry_id
        if agent_id in seen:
            continue
        seen.add(agent_id)
        descriptors.append(
            ConversationAgentDescriptor(
                agent_id=agent_id,
                label=entry.title or domain.replace("_", " ").title(),
                domain=domain,
            )
        )


def _append_conversation_subentry_entries(
    hass: Any,
    descriptors: list[ConversationAgentDescriptor],
    seen: set[str],
) -> None:
    """Append conversation-capable config entries that expose conversation subentries."""
    for entry in hass.config_entries.async_entries():
        state_name = getattr(getattr(entry, "state", None), "name", "")
        if state_name and state_name != "LOADED":
            continue
        domain = entry.domain
        if domain == DOMAIN:
            continue
        subentries = getattr(entry, "subentries", None)
        if not subentries:
            continue
        has_conversation_subentry = any(
            getattr(subentry, "subentry_type", None) == "conversation"
            for subentry in subentries.values()
        )
        if not has_conversation_subentry:
            continue
        agent_id = entry.entry_id
        if agent_id in seen:
            continue
        seen.add(agent_id)
        descriptors.append(
            ConversationAgentDescriptor(
                agent_id=agent_id,
                label=entry.title or domain.replace("_", " ").title(),
                domain=domain,
            )
        )
