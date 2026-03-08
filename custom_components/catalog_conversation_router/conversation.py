"""Custom conversation agent entity for Home Assistant."""

from __future__ import annotations

import logging
from typing import Any

from .models import RouterResult

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - exercised in HA runtime
    from homeassistant.components.conversation import (
        AbstractConversationAgent,
        ConversationInput,
        ConversationResult,
        async_set_agent,
        async_unset_agent,
    )
except Exception:  # pragma: no cover - tests without HA
    AbstractConversationAgent = object  # type: ignore[assignment]
    ConversationInput = Any  # type: ignore[assignment]
    ConversationResult = Any  # type: ignore[assignment]

    def async_set_agent(hass, config_entry, agent):  # type: ignore[no-redef]
        return None

    def async_unset_agent(hass, config_entry):  # type: ignore[no-redef]
        return None


class CatalogRouterConversationAgent(AbstractConversationAgent):
    """Conversation agent that delegates through AgentRouter."""

    def __init__(self, router, language: str) -> None:
        self._router = router
        self._language = language

    @property
    def attribution(self) -> str | None:
        """Agent attribution string."""
        return "Catalog Conversation Router"

    @property
    def supported_languages(self) -> list[str] | str:
        """Language support."""
        return [self._language]

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a conversation input."""
        result: RouterResult = await self._router.async_route(
            text=user_input.text,
            language=user_input.language or self._language,
            conversation_id=user_input.conversation_id,
            context=user_input.context,
            dry_run=False,
        )

        if result.trace:
            _LOGGER.debug("Resolution trace: %s", result.trace.as_dict())

        if result.outcome.response is not None:
            return result.outcome.response

        # Fallback-only edge case where adapter failed to return a native response object.
        from homeassistant.helpers.intent import IntentResponse

        intent_response = IntentResponse(language=user_input.language or self._language)
        intent_response.async_set_speech(result.outcome.response_text or "I could not process that request.")
        return ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
            continue_conversation=False,
        )


async def async_register_agent(hass, entry, agent: CatalogRouterConversationAgent) -> None:
    """Register the custom conversation agent with HA."""
    async_set_agent(hass, entry, agent)


async def async_unregister_agent(hass, entry) -> None:
    """Unregister custom conversation agent."""
    async_unset_agent(hass, entry)
