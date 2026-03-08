"""Config flow for Catalog Conversation Router."""

from __future__ import annotations

import json
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_LANGUAGE
from homeassistant.helpers import selector

from .const import (
    CONF_AMBIGUITY_GAP,
    CONF_CATALOG_AUTO_REFRESH_ENABLED,
    CONF_DEBUG_ENABLED,
    CONF_FUZZY_ENABLED,
    CONF_FUZZY_THRESHOLD,
    CONF_HIGH_RISK_THRESHOLD,
    CONF_LLM_AGENT_ID,
    CONF_LLM_FALLBACK_ENABLED,
    CONF_LLM_TRANSLATE_ENABLED,
    CONF_LOCAL_AGENT_ID,
    CONF_MANUAL_TARGETS,
    CONF_MAX_LLM_CANDIDATES,
    DEFAULT_AMBIGUITY_GAP,
    DEFAULT_CATALOG_AUTO_REFRESH_ENABLED,
    DEFAULT_DEBUG_ENABLED,
    DEFAULT_FUZZY_ENABLED,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_LLM_FALLBACK_ENABLED,
    DEFAULT_LLM_TRANSLATE_ENABLED,
    DEFAULT_MAX_LLM_CANDIDATES,
    DOMAIN,
)

BUILTIN_LOCAL_AGENT_ID = "homeassistant"
NO_AGENT_PLACEHOLDER = "__no_agent_available__"

CONVERSATION_AGENT_DOMAINS = {
    "assist",
    "conversation",
    "openai_conversation",
    "google_generative_ai_conversation",
    "chatgpt",
    "ollama",
    "llama_conversation",
    "assist_satellite",
}

LLM_AGENT_DOMAINS = {
    "openai_conversation",
    "google_generative_ai_conversation",
    "ollama",
    "anthropic",
    "llama_conversation",
    "chatgpt",
}


def build_router_config(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize user data with defaults and type coercion."""
    fuzzy_threshold = float(data.get(CONF_FUZZY_THRESHOLD, DEFAULT_FUZZY_THRESHOLD))
    ambiguity_gap = float(data.get(CONF_AMBIGUITY_GAP, DEFAULT_AMBIGUITY_GAP))
    high_risk_threshold = float(
        data.get(CONF_HIGH_RISK_THRESHOLD, DEFAULT_HIGH_RISK_THRESHOLD)
    )
    max_llm_candidates = int(data.get(CONF_MAX_LLM_CANDIDATES, DEFAULT_MAX_LLM_CANDIDATES))

    return {
        CONF_LOCAL_AGENT_ID: str(data[CONF_LOCAL_AGENT_ID]).strip(),
        CONF_LLM_AGENT_ID: str(data[CONF_LLM_AGENT_ID]).strip(),
        CONF_LANGUAGE: str(data.get(CONF_LANGUAGE, "en")).strip() or "en",
        CONF_FUZZY_ENABLED: bool(data.get(CONF_FUZZY_ENABLED, DEFAULT_FUZZY_ENABLED)),
        CONF_FUZZY_THRESHOLD: min(max(fuzzy_threshold, 0.0), 1.0),
        CONF_AMBIGUITY_GAP: min(max(ambiguity_gap, 0.0), 1.0),
        CONF_LLM_TRANSLATE_ENABLED: bool(
            data.get(CONF_LLM_TRANSLATE_ENABLED, DEFAULT_LLM_TRANSLATE_ENABLED)
        ),
        CONF_LLM_FALLBACK_ENABLED: bool(
            data.get(CONF_LLM_FALLBACK_ENABLED, DEFAULT_LLM_FALLBACK_ENABLED)
        ),
        CONF_DEBUG_ENABLED: bool(data.get(CONF_DEBUG_ENABLED, DEFAULT_DEBUG_ENABLED)),
        CONF_CATALOG_AUTO_REFRESH_ENABLED: bool(
            data.get(
                CONF_CATALOG_AUTO_REFRESH_ENABLED,
                DEFAULT_CATALOG_AUTO_REFRESH_ENABLED,
            )
        ),
        CONF_HIGH_RISK_THRESHOLD: min(max(high_risk_threshold, 0.0), 1.0),
        CONF_MAX_LLM_CANDIDATES: max(1, max_llm_candidates),
        CONF_MANUAL_TARGETS: data.get(CONF_MANUAL_TARGETS, []),
    }


def parse_manual_targets(raw_json: str) -> list[dict[str, Any]]:
    """Parse manual targets JSON from options flow text field."""
    if not raw_json.strip():
        return []
    payload = json.loads(raw_json)
    if not isinstance(payload, list):
        raise ValueError("manual targets must be a JSON list")
    return [item for item in payload if isinstance(item, dict)]


async def async_get_local_agent_options(hass) -> list[selector.SelectOptionDict]:
    """Get local-capable conversation agent options."""
    options: list[selector.SelectOptionDict] = [
        selector.SelectOptionDict(label="Home Assistant", value=BUILTIN_LOCAL_AGENT_ID)
    ]
    seen: set[str] = {BUILTIN_LOCAL_AGENT_ID}

    for entry in hass.config_entries.async_entries():
        state_name = getattr(getattr(entry, "state", None), "name", "")
        if state_name and state_name != "LOADED":
            continue
        domain = entry.domain
        if domain not in CONVERSATION_AGENT_DOMAINS and "conversation" not in domain:
            continue

        agent_id = entry.entry_id
        if agent_id in seen:
            continue
        seen.add(agent_id)

        title = entry.title or domain.replace("_", " ").title()
        options.append(selector.SelectOptionDict(label=title, value=agent_id))

    options.sort(key=lambda item: item["label"].lower())
    return options


async def async_get_llm_agent_options(hass) -> list[selector.SelectOptionDict]:
    """Get likely LLM conversation agent options."""
    options: list[selector.SelectOptionDict] = []
    seen: set[str] = set()

    for entry in hass.config_entries.async_entries():
        state_name = getattr(getattr(entry, "state", None), "name", "")
        if state_name and state_name != "LOADED":
            continue

        domain = entry.domain
        if domain not in LLM_AGENT_DOMAINS and "conversation" not in domain:
            continue

        agent_id = entry.entry_id
        if agent_id in seen:
            continue
        seen.add(agent_id)

        title = entry.title or domain.replace("_", " ").title()
        options.append(selector.SelectOptionDict(label=title, value=agent_id))

    options.sort(key=lambda item: item["label"].lower())
    if not options:
        options.append(
            selector.SelectOptionDict(
                label="No LLM conversation agents found",
                value=NO_AGENT_PLACEHOLDER,
            )
        )
    return options


class CatalogConversationRouterConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle config flow."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}

        local_options = await async_get_local_agent_options(self.hass)
        llm_options = await async_get_llm_agent_options(self.hass)
        local_selector = _build_select_selector(local_options)
        llm_selector = _build_select_selector(llm_options)

        if llm_options and llm_options[0]["value"] == NO_AGENT_PLACEHOLDER:
            errors["base"] = "no_llm_agents_found"

        if user_input is not None:
            try:
                config = build_router_config(user_input)
                if not config[CONF_LOCAL_AGENT_ID] or not config[CONF_LLM_AGENT_ID]:
                    errors["base"] = "agent_required"
                elif config[CONF_LLM_AGENT_ID] == NO_AGENT_PLACEHOLDER:
                    errors["base"] = "no_llm_agents_found"
                elif config[CONF_LOCAL_AGENT_ID] == config[CONF_LLM_AGENT_ID]:
                    errors["base"] = "agents_must_differ"
                else:
                    return self.async_create_entry(
                        title="Catalog Conversation Router",
                        data=config,
                    )
            except Exception:
                errors["base"] = "invalid_config"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_LOCAL_AGENT_ID,
                        default=BUILTIN_LOCAL_AGENT_ID,
                    ): local_selector,
                    vol.Required(
                        CONF_LLM_AGENT_ID,
                        default=llm_options[0]["value"],
                    ): llm_selector,
                    vol.Optional(CONF_LANGUAGE, default="en"): str,
                    vol.Optional(CONF_FUZZY_ENABLED, default=DEFAULT_FUZZY_ENABLED): bool,
                    vol.Optional(
                        CONF_FUZZY_THRESHOLD,
                        default=DEFAULT_FUZZY_THRESHOLD,
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                    vol.Optional(
                        CONF_AMBIGUITY_GAP,
                        default=DEFAULT_AMBIGUITY_GAP,
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                    vol.Optional(
                        CONF_LLM_TRANSLATE_ENABLED,
                        default=DEFAULT_LLM_TRANSLATE_ENABLED,
                    ): bool,
                    vol.Optional(
                        CONF_LLM_FALLBACK_ENABLED,
                        default=DEFAULT_LLM_FALLBACK_ENABLED,
                    ): bool,
                    vol.Optional(CONF_DEBUG_ENABLED, default=DEFAULT_DEBUG_ENABLED): bool,
                    vol.Optional(
                        CONF_CATALOG_AUTO_REFRESH_ENABLED,
                        default=DEFAULT_CATALOG_AUTO_REFRESH_ENABLED,
                    ): bool,
                    vol.Optional(
                        CONF_HIGH_RISK_THRESHOLD,
                        default=DEFAULT_HIGH_RISK_THRESHOLD,
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                    vol.Optional(
                        CONF_MAX_LLM_CANDIDATES,
                        default=DEFAULT_MAX_LLM_CANDIDATES,
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=200)),
                }
            ),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        return CatalogConversationRouterOptionsFlow(config_entry)


class CatalogConversationRouterOptionsFlow(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry) -> None:
        self._entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        current_manual = self._entry.options.get(
            CONF_MANUAL_TARGETS,
            self._entry.data.get(CONF_MANUAL_TARGETS, []),
        )

        local_options = await async_get_local_agent_options(self.hass)
        llm_options = await async_get_llm_agent_options(self.hass)
        local_selector = _build_select_selector(local_options)
        llm_selector = _build_select_selector(llm_options)

        if llm_options and llm_options[0]["value"] == NO_AGENT_PLACEHOLDER:
            errors["base"] = "no_llm_agents_found"

        if user_input is not None:
            try:
                manual_targets = parse_manual_targets(user_input.get("manual_targets_json", ""))
                data = {
                    **self._entry.data,
                    **self._entry.options,
                    **user_input,
                    CONF_MANUAL_TARGETS: manual_targets,
                }
                data.pop("manual_targets_json", None)
                normalized = build_router_config(data)

                if (
                    not normalized[CONF_LOCAL_AGENT_ID]
                    or not normalized[CONF_LLM_AGENT_ID]
                ):
                    errors["base"] = "agent_required"
                elif normalized[CONF_LLM_AGENT_ID] == NO_AGENT_PLACEHOLDER:
                    errors["base"] = "no_llm_agents_found"
                elif normalized[CONF_LOCAL_AGENT_ID] == normalized[CONF_LLM_AGENT_ID]:
                    errors["base"] = "agents_must_differ"
                else:
                    return self.async_create_entry(title="", data=normalized)
            except Exception:
                errors["base"] = "invalid_manual_targets"

        return self.async_show_form(
            step_id="init",
            data_schema=self._build_options_schema(
                current_manual=current_manual,
                local_selector=local_selector,
                llm_selector=llm_selector,
                llm_options=llm_options,
            ),
            errors=errors,
        )

    def _build_options_schema(
        self,
        *,
        current_manual: list[dict[str, Any]],
        local_selector,
        llm_selector,
        llm_options: list[selector.SelectOptionDict],
    ) -> vol.Schema:
        """Build options form schema."""
        return vol.Schema(
            {
                vol.Required(
                    CONF_LOCAL_AGENT_ID,
                    default=self._entry.options.get(
                        CONF_LOCAL_AGENT_ID,
                        self._entry.data.get(CONF_LOCAL_AGENT_ID, BUILTIN_LOCAL_AGENT_ID),
                    ),
                ): local_selector,
                vol.Required(
                    CONF_LLM_AGENT_ID,
                    default=self._entry.options.get(
                        CONF_LLM_AGENT_ID,
                        self._entry.data.get(CONF_LLM_AGENT_ID, llm_options[0]["value"]),
                    ),
                ): llm_selector,
                vol.Optional(
                    CONF_LANGUAGE,
                    default=self._entry.options.get(
                        CONF_LANGUAGE,
                        self._entry.data.get(CONF_LANGUAGE, "en"),
                    ),
                ): str,
                vol.Optional(
                    CONF_FUZZY_ENABLED,
                    default=self._entry.options.get(
                        CONF_FUZZY_ENABLED,
                        self._entry.data.get(CONF_FUZZY_ENABLED, DEFAULT_FUZZY_ENABLED),
                    ),
                ): bool,
                vol.Optional(
                    CONF_FUZZY_THRESHOLD,
                    default=self._entry.options.get(
                        CONF_FUZZY_THRESHOLD,
                        self._entry.data.get(CONF_FUZZY_THRESHOLD, DEFAULT_FUZZY_THRESHOLD),
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                vol.Optional(
                    CONF_AMBIGUITY_GAP,
                    default=self._entry.options.get(
                        CONF_AMBIGUITY_GAP,
                        self._entry.data.get(CONF_AMBIGUITY_GAP, DEFAULT_AMBIGUITY_GAP),
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                vol.Optional(
                    CONF_LLM_TRANSLATE_ENABLED,
                    default=self._entry.options.get(
                        CONF_LLM_TRANSLATE_ENABLED,
                        self._entry.data.get(
                            CONF_LLM_TRANSLATE_ENABLED,
                            DEFAULT_LLM_TRANSLATE_ENABLED,
                        ),
                    ),
                ): bool,
                vol.Optional(
                    CONF_LLM_FALLBACK_ENABLED,
                    default=self._entry.options.get(
                        CONF_LLM_FALLBACK_ENABLED,
                        self._entry.data.get(
                            CONF_LLM_FALLBACK_ENABLED,
                            DEFAULT_LLM_FALLBACK_ENABLED,
                        ),
                    ),
                ): bool,
                vol.Optional(
                    CONF_DEBUG_ENABLED,
                    default=self._entry.options.get(
                        CONF_DEBUG_ENABLED,
                        self._entry.data.get(CONF_DEBUG_ENABLED, DEFAULT_DEBUG_ENABLED),
                    ),
                ): bool,
                vol.Optional(
                    CONF_CATALOG_AUTO_REFRESH_ENABLED,
                    default=self._entry.options.get(
                        CONF_CATALOG_AUTO_REFRESH_ENABLED,
                        self._entry.data.get(
                            CONF_CATALOG_AUTO_REFRESH_ENABLED,
                            DEFAULT_CATALOG_AUTO_REFRESH_ENABLED,
                        ),
                    ),
                ): bool,
                vol.Optional(
                    CONF_HIGH_RISK_THRESHOLD,
                    default=self._entry.options.get(
                        CONF_HIGH_RISK_THRESHOLD,
                        self._entry.data.get(
                            CONF_HIGH_RISK_THRESHOLD,
                            DEFAULT_HIGH_RISK_THRESHOLD,
                        ),
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                vol.Optional(
                    CONF_MAX_LLM_CANDIDATES,
                    default=self._entry.options.get(
                        CONF_MAX_LLM_CANDIDATES,
                        self._entry.data.get(
                            CONF_MAX_LLM_CANDIDATES,
                            DEFAULT_MAX_LLM_CANDIDATES,
                        ),
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=200)),
                vol.Optional(
                    "manual_targets_json",
                    default=json.dumps(current_manual, indent=2),
                ): str,
            }
        )


def _build_select_selector(
    options: list[selector.SelectOptionDict],
) -> selector.SelectSelector:
    """Build a dropdown selector with fixed options."""
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=options,
            mode=selector.SelectSelectorMode.DROPDOWN,
            custom_value=False,
            sort=False,
        )
    )
