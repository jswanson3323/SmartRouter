"""Custom conversation agent entity for Home Assistant."""

from __future__ import annotations

import logging
from typing import Any

from .models import RouterResult

_LOGGER = logging.getLogger(__name__)

_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - exercised in HA runtime
    from homeassistant.components.conversation import (
        AbstractConversationAgent,
        ConversationInput,
        ConversationResult,
        async_set_agent,
        async_unset_agent,
    )
    from homeassistant.helpers.intent import IntentResponse
    _CONVERSATION_API_AVAILABLE = True
except Exception as err:  # pragma: no cover - tests without HA
    _CONVERSATION_API_AVAILABLE = False
    _IMPORT_ERROR = err
    AbstractConversationAgent = object  # type: ignore[assignment]
    ConversationInput = Any  # type: ignore[assignment]
    ConversationResult = Any  # type: ignore[assignment]
    IntentResponse = Any  # type: ignore[assignment]


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
            device_id=getattr(user_input, "device_id", None),
            satellite_id=getattr(user_input, "satellite_id", None),
        )

        if result.trace:
            _LOGGER.debug("Resolution trace: %s", result.trace.as_dict())

        if result.outcome.response is not None:
            return result.outcome.response

        # Fallback-only edge case where adapter failed to return a native response object.
        if not _CONVERSATION_API_AVAILABLE:  # pragma: no cover
            raise RuntimeError("Conversation API unavailable for fallback response")
        intent_response = IntentResponse(language=user_input.language or self._language)
        intent_response.async_set_speech(result.outcome.response_text or "I could not process that request.")
        return ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
            continue_conversation=False,
        )


async def async_register_agent(hass, entry, agent: CatalogRouterConversationAgent) -> None:
    """Register the custom conversation agent with HA."""
    if not _CONVERSATION_API_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Failed to load Home Assistant conversation API") from _IMPORT_ERROR
    async_set_agent(hass, entry, agent)


async def async_unregister_agent(hass, entry) -> None:
    """Unregister custom conversation agent."""
    if not _CONVERSATION_API_AVAILABLE:  # pragma: no cover
        return
    async_unset_agent(hass, entry)
