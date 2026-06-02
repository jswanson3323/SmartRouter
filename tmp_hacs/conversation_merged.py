"""Defines the various LLM Backend Agents"""
from __future__ import annotations

from typing import Literal, List, Tuple, Any
import logging

from homeassistant.components.conversation import (
    ConversationInput,
    ConversationResult,
    ConversationEntity,
)
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from custom_components.llama_conversation.utils import MalformedToolCallException

from .entity import LocalLLMEntity, LocalLLMClient, LocalLLMConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_PROMPT,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


def _normalize_extra_system_prompt(user_input: ConversationInput) -> str | None:
    """Return a cleaned upstream extra prompt."""
    extra_prompt = getattr(user_input, "extra_system_prompt", None)
    if extra_prompt is None:
        return None
    normalized = str(extra_prompt).strip()
    return normalized or None


def _build_prompt_cache_key(
    raw_prompt: str,
    runtime_options: dict[str, Any],
    extra_system_prompt: str | None,
) -> tuple[Any, ...]:
    """Build a stable key for deciding when to rebuild the system prompt."""
    return (
        raw_prompt,
        extra_system_prompt,
        runtime_options.get(CONF_CHAT_MODEL),
        runtime_options.get(CONF_LLM_HASS_API),
    )


def _compose_system_prompt(base_prompt: str, extra_system_prompt: str | None) -> str:
    """Combine the base and upstream prompts without adding volatility."""
    if extra_system_prompt is None:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n{extra_system_prompt}"


def _build_message_history(
    chat_log: conversation.ChatLog,
    *,
    remember_conversation: bool,
) -> list[Any]:
    """Return the message history to send to the backend."""
    if remember_conversation:
        return chat_log.content[:]

    message_history: list[Any] = []
    if not chat_log.content:
        return message_history

    first_message = chat_log.content[0]
    if getattr(first_message, "role", None) == "system":
        message_history.append(first_message)

    latest_user_message = next(
        (
            message
            for message in reversed(chat_log.content)
            if getattr(message, "role", None) == "user"
        ),
        None,
    )
    if latest_user_message is not None and (
        not message_history or latest_user_message is not message_history[-1]
    ):
        message_history.append(latest_user_message)

    return message_history


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LocalLLMConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> bool:
    """Set up Local LLM Conversation from a config entry."""

    for subentry in entry.subentries.values():
        if subentry.subentry_type != conversation.DOMAIN:
            continue

        if CONF_CHAT_MODEL not in subentry.data:
            _LOGGER.warning(
                "Conversation subentry %s missing required config key %s. "
                "You must delete the model and re-create it.",
                subentry.subentry_id,
                CONF_CHAT_MODEL,
            )
            continue

        agent_entity = LocalLLMAgent(hass, entry, subentry, entry.runtime_data)

        await entry.runtime_data._async_load_model(dict(subentry.data))

        async_add_entities(
            [agent_entity],
            config_subentry_id=subentry.subentry_id,
        )

    return True


class LocalLLMAgent(ConversationEntity, AbstractConversationAgent, LocalLLMEntity):
    """Base Local LLM conversation agent."""

    _attr_supports_streaming = True

    @property
    def supports_streaming(self) -> bool:
        """Return if this conversation agent supports streaming."""
        return True

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        subentry: ConfigSubentry,
        client: LocalLLMClient,
    ) -> None:
        super().__init__(hass, entry, subentry, client)

        if subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> ConversationResult:
        """Process a sentence using HA's pipeline-owned chat log."""

        raw_prompt = self.runtime_options.get(CONF_PROMPT, DEFAULT_PROMPT)
        extra_system_prompt = _normalize_extra_system_prompt(user_input)
        refresh_system_prompt = self.runtime_options.get(
            CONF_REFRESH_SYSTEM_PROMPT,
            DEFAULT_REFRESH_SYSTEM_PROMPT,
        )
        remember_conversation = self.runtime_options.get(
            CONF_REMEMBER_CONVERSATION,
            DEFAULT_REMEMBER_CONVERSATION,
        )
        remember_num_interactions = self.runtime_options.get(
            CONF_REMEMBER_NUM_INTERACTIONS,
            DEFAULT_REMEMBER_NUM_INTERACTIONS,
        )
        max_tool_call_iterations = self.runtime_options.get(
            CONF_MAX_TOOL_CALL_ITERATIONS,
            DEFAULT_MAX_TOOL_CALL_ITERATIONS,
        )

        llm_api: llm.APIInstance | None = None

        if self.runtime_options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass,
                    self.runtime_options[CONF_LLM_HASS_API],
                    llm_context=user_input.as_llm_context(DOMAIN),
                )
            except HomeAssistantError as err:
                _LOGGER.error("Error getting LLM API: %s", err)

                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Error preparing LLM API: {err}",
                )

                return ConversationResult(
                    response=intent_response,
                    conversation_id=user_input.conversation_id,
                )

        # Ensure this chat log has the LLM API instance for tool handling.
        chat_log.llm_api = llm_api
        previous_extra_system_prompt = getattr(chat_log, "extra_system_prompt", None)
        chat_log.extra_system_prompt = extra_system_prompt
        message_history = _build_message_history(
            chat_log,
            remember_conversation=remember_conversation,
        )

        if (
            remember_conversation
            and
            remember_num_interactions
            and len(message_history) > (remember_num_interactions * 2) + 1
        ):
            new_message_history = [message_history[0]]
            new_message_history.extend(
                message_history[1:][-(remember_num_interactions * 2):]
            )
            message_history = new_message_history

        prompt_cache_key = _build_prompt_cache_key(
            raw_prompt,
            self.runtime_options,
            extra_system_prompt,
        )
        current_system_prompt = (
            message_history[0]
            if message_history and getattr(message_history[0], "role", None) == "system"
            else None
        )
        cached_prompt_key = getattr(chat_log, "_local_llm_prompt_cache_key", None)
        should_refresh_prompt = (
            current_system_prompt is None
            or cached_prompt_key != prompt_cache_key
            or (
                refresh_system_prompt
                and extra_system_prompt != previous_extra_system_prompt
            )
        )

        if should_refresh_prompt:
            try:
                prompt_source = _compose_system_prompt(raw_prompt, extra_system_prompt)
                system_prompt = conversation.SystemContent(
                    content=self.client._generate_system_prompt(
                        prompt_source,
                        llm_api,
                        self.runtime_options,
                    )
                )
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)

                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )

                return ConversationResult(
                    response=intent_response,
                    conversation_id=user_input.conversation_id,
                )

            if current_system_prompt is None:
                message_history.insert(0, system_prompt)
            else:
                message_history[0] = system_prompt
            setattr(chat_log, "_local_llm_prompt_cache_key", prompt_cache_key)

        tool_calls: List[Tuple[llm.ToolInput, Any]] = []

        # If max_tool_call_iterations is 0, generate the response and any tool
        # call in one pass. Otherwise, allow tool-call follow-up iterations.
        for idx in range(max(1, max_tool_call_iterations)):
            _LOGGER.debug(
                "Generating response for user_input.text=%r, iteration %s/%s",
                user_input.text,
                idx + 1,
                max_tool_call_iterations,
            )

            try:
                generation_result = await self.client._async_generate(
                    message_history,
                    user_input.agent_id,
                    chat_log,
                    self.runtime_options,
                )
            except Exception as err:
                _LOGGER.exception("There was a problem talking to the backend")

                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                    f"Sorry, there was a problem talking to the backend: {repr(err)}",
                )

                return ConversationResult(
                    response=intent_response,
                    conversation_id=user_input.conversation_id,
                )

            last_generation_had_tool_calls = False

            while True:
                try:
                    message = await anext(generation_result)

                    # Keep local history only. Do not manually add streaming
                    # deltas here. The client/_async_generate path owns the
                    # chat_log deltas shown by HA Debug Assistant.
                    message_history.append(message)

                    _LOGGER.debug("Added message to history: %s", message)

                    if message.role == "assistant":
                        if message.tool_calls and len(message.tool_calls) > 0:
                            last_generation_had_tool_calls = True
                        else:
                            last_generation_had_tool_calls = False

                except StopAsyncIteration:
                    break

                except MalformedToolCallException as err:
                    message_history.extend(err.as_tool_messages())
                    last_generation_had_tool_calls = True
                    _LOGGER.debug("Malformed tool call produced", exc_info=err)

                except Exception as err:
                    _LOGGER.exception("There was a problem talking to the backend")

                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                        f"Sorry, there was a problem talking to the backend: {repr(err)}",
                    )

                    return ConversationResult(
                        response=intent_response,
                        conversation_id=user_input.conversation_id,
                    )

            if not last_generation_had_tool_calls:
                break

            if idx == max_tool_call_iterations - 1 and max_tool_call_iterations > 0:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                    "Sorry, I ran out of attempts to handle your request",
                )

                return ConversationResult(
                    response=intent_response,
                    conversation_id=user_input.conversation_id,
                )

        if len(tool_calls) > 0:
            intent_response = intent.IntentResponse(language=user_input.language)

            str_tools = [
                f"{input.tool_name}({', '.join(str(x) for x in input.tool_args.values())})"
                for input, response in tool_calls
            ]
            tools_str = "\n".join(str_tools)

            intent_response.async_set_card(
                title="Changes",
                content=f"Ran the following tools:\n{tools_str}",
            )

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
