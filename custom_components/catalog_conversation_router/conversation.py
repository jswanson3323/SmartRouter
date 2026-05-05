"""Custom conversation agent entity for Home Assistant."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from time import perf_counter
from typing import Any

from .models import LocalAgentOutcome, RouterResult, StreamingFallbackRequest

_LOGGER = logging.getLogger(__name__)

_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - exercised in HA runtime
    from homeassistant.components.conversation import (
        AbstractConversationAgent,
        AssistantContent,
        ConversationInput,
        ConversationResult,
        async_get_result_from_chat_log,
        async_set_agent,
        async_unset_agent,
    )
    from homeassistant.helpers.intent import IntentResponse

    try:
        from homeassistant.components.conversation.agent_manager import async_get_agent
        from homeassistant.components.conversation.chat_log import async_get_chat_log
        from homeassistant.components.conversation.entity import ConversationEntity
        from homeassistant.helpers import chat_session

        _STREAMING_CONVERSATION_API_AVAILABLE = True
    except Exception:
        class _StreamingConversationEntityFallback:
            """Fallback when the streaming entity API is unavailable."""

        ConversationEntity = _StreamingConversationEntityFallback  # type: ignore[assignment]
        _STREAMING_CONVERSATION_API_AVAILABLE = False

    _CONVERSATION_API_AVAILABLE = True
except Exception as err:  # pragma: no cover - tests without HA
    _CONVERSATION_API_AVAILABLE = False
    _STREAMING_CONVERSATION_API_AVAILABLE = False
    _IMPORT_ERROR = err

    class _AbstractConversationAgent:
        """Fallback abstract conversation agent stub."""

    class _ConversationEntity:
        """Fallback conversation entity stub."""

    AbstractConversationAgent = _AbstractConversationAgent  # type: ignore[assignment]
    ConversationEntity = _ConversationEntity  # type: ignore[assignment]
    ConversationInput = Any  # type: ignore[assignment]
    ConversationResult = Any  # type: ignore[assignment]
    IntentResponse = Any  # type: ignore[assignment]


class CatalogRouterConversationAgent(ConversationEntity, AbstractConversationAgent):
    """Conversation agent that delegates through AgentRouter."""

    _attr_supports_streaming = True

    def __init__(self, router, language: str, entry_id: str) -> None:
        self._router = router
        self._language = language
        self._hass = getattr(router, "_hass", None)
        self._entry_id = entry_id
        self._attr_name = "Catalog Conversation Router"
        self._attr_unique_id = entry_id
        self.entity_id = f"conversation.catalog_conversation_router_{entry_id.lower()}"

    @property
    def id(self) -> str:
        """Stable conversation entity id for Assist pipeline selection."""
        return self.entity_id

    @property
    def unique_id(self) -> str:
        """Stable unique id for this virtual conversation entity."""
        return self._entry_id

    @property
    def attribution(self) -> str | None:
        """Agent attribution string."""
        return "Catalog Conversation Router"

    @property
    def supported_languages(self) -> list[str] | str:
        """Language support."""
        return [self._language]

    async def internal_async_process(
        self,
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Use HA's native ConversationEntity processing when available."""
        if _STREAMING_CONVERSATION_API_AVAILABLE:
            return await super().internal_async_process(user_input)
        return await self.async_process(user_input)

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a conversation input with blocking compatibility behavior."""
        result = await self._route_request(
            user_input,
            allow_streaming_llm_fallback=False,
        )
        return self._finalize_non_streaming_response(user_input, result)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log,
    ) -> ConversationResult:
        """Process a conversation input with HA chat-log streaming when supported."""
        result = await self._route_request(
            user_input,
            allow_streaming_llm_fallback=True,
        )

        if result.streaming_request is None:
            return self._finalize_non_streaming_response(user_input, result)

        if not _STREAMING_CONVERSATION_API_AVAILABLE:
            result.trace.llm_fallback_stream_supported = False
            result.trace.llm_fallback_stream_used = False
            result.trace.llm_fallback_stream_fallback_reason = "streaming_api_unavailable"
            blocking_result = await self._route_request(
                user_input,
                allow_streaming_llm_fallback=False,
            )
            return self._finalize_non_streaming_response(user_input, blocking_result)

        stream_started = perf_counter()
        stream_outcome = await self._async_execute_streaming_fallback(
            user_input=user_input,
            chat_log=chat_log,
            request=result.streaming_request,
            trace=result.trace,
        )
        result.trace.llm_fallback_duration_ms = round(
            (perf_counter() - stream_started) * 1000,
            3,
        )
        self._router.finalize_streaming_fallback(
            conversation_id=user_input.conversation_id,
            outcome=stream_outcome,
            agent_id=result.streaming_request.llm_agent_id,
        )
        _LOGGER.debug("Post-stream resolution trace: %s", result.trace.as_dict())
        return async_get_result_from_chat_log(user_input, chat_log)

    async def _route_request(
        self,
        user_input: ConversationInput,
        *,
        allow_streaming_llm_fallback: bool,
    ) -> RouterResult:
        _LOGGER.debug(
            "CONVERSATION ENTRY: text=%r language=%r conversation_id=%r device_id=%r satellite_id=%r extra_system_prompt=%r",
            user_input.text,
            user_input.language,
            user_input.conversation_id,
            getattr(user_input, "device_id", None),
            getattr(user_input, "satellite_id", None),
            getattr(user_input, "extra_system_prompt", None),
        )
        result: RouterResult = await self._router.async_route(
            text=user_input.text,
            language=user_input.language or self._language,
            conversation_id=user_input.conversation_id,
            context=user_input.context,
            dry_run=False,
            allow_streaming_llm_fallback=allow_streaming_llm_fallback,
            device_id=getattr(user_input, "device_id", None),
            satellite_id=getattr(user_input, "satellite_id", None),
            extra_system_prompt=getattr(user_input, "extra_system_prompt", None),
        )
        if result.trace:
            _LOGGER.debug("Resolution trace: %s", result.trace.as_dict())
        return result

    async def _async_execute_streaming_fallback(
        self,
        *,
        user_input: ConversationInput,
        chat_log,
        request: StreamingFallbackRequest,
        trace,
    ) -> LocalAgentOutcome:
        if not _STREAMING_CONVERSATION_API_AVAILABLE:
            trace.llm_fallback_stream_supported = False
            trace.llm_fallback_stream_used = False
            trace.llm_fallback_stream_fallback_reason = "streaming_api_unavailable"
            return await self._router._agent_adapter.async_process(  # noqa: SLF001
                agent_id=request.llm_agent_id,
                text=request.utterance,
                language=request.language,
                conversation_id=request.conversation_id,
                context=request.context,
                device_id=request.device_id,
                satellite_id=request.satellite_id,
                extra_system_prompt=request.extra_system_prompt,
            )

        downstream_agent = async_get_agent(self._hass, request.llm_agent_id)
        if downstream_agent is None:
            trace.llm_fallback_stream_supported = False
            trace.llm_fallback_stream_used = False
            trace.llm_fallback_stream_fallback_reason = "agent_not_found"
            return await self._router._agent_adapter.async_process(  # noqa: SLF001
                agent_id=request.llm_agent_id,
                text=request.utterance,
                language=request.language,
                conversation_id=request.conversation_id,
                context=request.context,
                device_id=request.device_id,
                satellite_id=request.satellite_id,
                extra_system_prompt=request.extra_system_prompt,
            )

        if (
            not isinstance(downstream_agent, ConversationEntity)
            or not getattr(downstream_agent, "supports_streaming", False)
            or not hasattr(downstream_agent, "_async_handle_message")
        ):
            trace.llm_fallback_stream_supported = False
            trace.llm_fallback_stream_used = False
            trace.llm_fallback_stream_fallback_reason = "downstream_agent_not_stream_capable"
            return await self._router._agent_adapter.async_process(  # noqa: SLF001
                agent_id=request.llm_agent_id,
                text=request.utterance,
                language=request.language,
                conversation_id=request.conversation_id,
                context=request.context,
                device_id=request.device_id,
                satellite_id=request.satellite_id,
                extra_system_prompt=request.extra_system_prompt,
            )

        trace.llm_fallback_stream_supported = True

        delta_queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()
        stream_done = object()
        chunk_count = 0
        content_parts: list[str] = []

        def _delta_listener(_downstream_chat_log, delta: dict[str, Any]) -> None:
            nonlocal chunk_count
            chunk_count += 1
            if delta.get("content"):
                content_parts.append(str(delta["content"]))
            delta_queue.put_nowait(dict(delta))

        async def _delta_stream() -> AsyncGenerator[dict[str, Any], None]:
            while True:
                item = await delta_queue.get()
                if item is stream_done:
                    return
                yield item

        async def _run_downstream():
            downstream_input = ConversationInput(
                text=request.utterance,
                context=request.context,
                conversation_id=request.conversation_id,
                device_id=request.device_id,
                satellite_id=request.satellite_id,
                language=request.language,
                agent_id=request.llm_agent_id,
                extra_system_prompt=request.extra_system_prompt,
            )
            with chat_session.async_get_chat_session(
                self._hass,
                request.conversation_id,
            ) as session:
                with async_get_chat_log(
                    self._hass,
                    session,
                    downstream_input,
                    chat_log_delta_listener=_delta_listener,
                ) as downstream_chat_log:
                    return await downstream_agent._async_handle_message(  # type: ignore[attr-defined]
                        downstream_input,
                        downstream_chat_log,
                    )

        downstream_task = self._hass.async_create_task(
            _run_downstream(),
            name="catalog_router_streaming_llm_fallback",
        )
        downstream_result = None
        stream_error: Exception | None = None

        async def _await_downstream():
            try:
                return await downstream_task
            finally:
                await delta_queue.put(stream_done)

        await_task = self._hass.async_create_task(
            _await_downstream(),
            name="catalog_router_streaming_llm_fallback_waiter",
        )
        try:
            async for _ in chat_log.async_add_delta_content_stream(
                getattr(user_input, "agent_id", None) or request.llm_agent_id,
                _delta_stream(),
            ):
                pass
            downstream_result = await await_task
        except Exception as err:  # pragma: no cover - runtime safety
            stream_error = err
            if not await_task.done():
                await delta_queue.put(stream_done)
                await await_task

        if stream_error is not None:
            trace.llm_fallback_stream_used = chunk_count > 0
            trace.llm_fallback_stream_chunk_count = chunk_count
            trace.llm_fallback_stream_fallback_reason = type(stream_error).__name__
            if chunk_count == 0:
                return await self._router._agent_adapter.async_process(  # noqa: SLF001
                    agent_id=request.llm_agent_id,
                    text=request.utterance,
                    language=request.language,
                    conversation_id=request.conversation_id,
                    context=request.context,
                    device_id=request.device_id,
                    satellite_id=request.satellite_id,
                    extra_system_prompt=request.extra_system_prompt,
                )
            partial_text = "".join(content_parts).strip()
            if partial_text:
                chat_log.async_add_assistant_content_without_tools(
                    AssistantContent(
                        agent_id=getattr(user_input, "agent_id", None)
                        or request.llm_agent_id,
                        content=partial_text,
                    )
                )
            return LocalAgentOutcome(
                success=bool(partial_text),
                response=None,
                response_text=partial_text or str(stream_error),
                failure_category=None if partial_text else None,
                raw=None,
                conversation_id=request.conversation_id,
                continue_conversation=False,
            )

        outcome = self._router._agent_adapter.outcome_from_response(  # noqa: SLF001
            agent_id=request.llm_agent_id,
            response=downstream_result,
            device_id=request.device_id,
            satellite_id=request.satellite_id,
        )
        if not outcome.response_text and content_parts:
            outcome.response_text = "".join(content_parts).strip()
        trace.llm_fallback_stream_used = chunk_count > 0
        trace.llm_fallback_stream_chunk_count = chunk_count
        if chunk_count == 0:
            trace.llm_fallback_stream_fallback_reason = "no_deltas_emitted"
        return outcome

    def _finalize_non_streaming_response(
        self,
        user_input: ConversationInput,
        result: RouterResult,
    ) -> ConversationResult:
        if result.outcome.response is not None:
            return self._wrap_outcome_as_result(user_input, result.outcome)

        if not _CONVERSATION_API_AVAILABLE:  # pragma: no cover
            raise RuntimeError("Conversation API unavailable for fallback response")
        intent_response = IntentResponse(language=user_input.language or self._language)
        intent_response.async_set_speech(
            result.outcome.response_text or "I could not process that request."
        )
        return ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
            continue_conversation=False,
        )

    def _wrap_outcome_as_result(
        self,
        user_input: ConversationInput,
        outcome: LocalAgentOutcome,
    ) -> ConversationResult:
        response = outcome.response
        processed_locally = outcome.processed_locally is True
        continue_conversation = outcome.continue_conversation is True

        if response is not None and hasattr(response, "response") and hasattr(
            response,
            "conversation_id",
        ):
            try:
                setattr(response, "conversation_id", user_input.conversation_id)
            except Exception:
                pass
            try:
                setattr(response, "continue_conversation", continue_conversation)
            except Exception:
                pass
            if processed_locally:
                try:
                    setattr(response, "processed_locally", True)
                except Exception:
                    pass
            return response

        if not _CONVERSATION_API_AVAILABLE:  # pragma: no cover
            return response

        try:
            return ConversationResult(
                response=response,
                conversation_id=user_input.conversation_id,
                continue_conversation=continue_conversation,
                processed_locally=processed_locally,
            )
        except TypeError:
            wrapped = ConversationResult(
                response=response,
                conversation_id=user_input.conversation_id,
                continue_conversation=continue_conversation,
            )
            if processed_locally:
                try:
                    setattr(wrapped, "processed_locally", True)
                except Exception:
                    pass
            return wrapped


class CatalogRouterLegacyAgentAlias(AbstractConversationAgent):
    """Legacy manager-style alias for backward compatibility with old pipeline ids."""

    def __init__(
        self,
        *,
        legacy_agent_id: str,
        entity_agent: CatalogRouterConversationAgent,
        language: str,
    ) -> None:
        self._legacy_agent_id = legacy_agent_id
        self._entity_agent = entity_agent
        self._language = language

    @property
    def attribution(self) -> str | None:
        """Agent attribution string."""
        return "Catalog Conversation Router"

    @property
    def supported_languages(self) -> list[str] | str:
        """Language support."""
        return [self._language]

    @property
    def id(self) -> str:
        """Legacy callable agent id."""
        return self._legacy_agent_id

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Delegate legacy manager calls through the streaming-capable entity path."""
        return await self._entity_agent.internal_async_process(user_input)


async def async_setup_entry(hass, entry, async_add_entities) -> None:
    """Set up the router as a real Home Assistant conversation entity."""
    runtime = hass.data["catalog_conversation_router"][entry.entry_id]
    async_add_entities([runtime.conversation_agent])


async def async_register_agent_alias(hass, entry, agent: AbstractConversationAgent) -> None:
    """Register a legacy manager alias for backwards compatibility."""
    if not _CONVERSATION_API_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Failed to load Home Assistant conversation API") from _IMPORT_ERROR
    async_set_agent(hass, entry, agent)


async def async_unregister_agent_alias(hass, entry) -> None:
    """Unregister the legacy manager alias."""
    if not _CONVERSATION_API_AVAILABLE:  # pragma: no cover
        return
    async_unset_agent(hass, entry)
