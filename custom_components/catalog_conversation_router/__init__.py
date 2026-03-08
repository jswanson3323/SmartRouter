"""Catalog Conversation Router integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

try:  # pragma: no cover - import-safe for unit tests without HA
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.event import async_track_time_interval
except Exception:  # pragma: no cover
    ConfigEntry = object  # type: ignore[assignment]
    HomeAssistant = object  # type: ignore[assignment]

    def async_track_time_interval(hass, action, period):  # type: ignore[no-redef]
        return lambda: None

from .agent_router import AgentRouter
from .catalog import CatalogManager
from .const import (
    CONF_AMBIGUITY_GAP,
    CONF_CATALOG_AUTO_REFRESH_ENABLED,
    CONF_DEBUG_ENABLED,
    CONF_FUZZY_ENABLED,
    CONF_FUZZY_THRESHOLD,
    CONF_HIGH_RISK_THRESHOLD,
    CONF_LANGUAGE,
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
from .conversation import CatalogRouterConversationAgent, async_register_agent, async_unregister_agent
from .llm_adapter import LLMAdapter
from .local_agent_adapter import AgentAdapter
from .matcher import FuzzyMatcher
from .models import RouterConfig
from .services import async_register_services, async_unregister_services

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IntegrationRuntime:
    """Live runtime objects for a config entry."""

    config: RouterConfig
    catalog_manager: CatalogManager
    router: AgentRouter
    conversation_agent: CatalogRouterConversationAgent
    unsub_refresh: callable | None = None


def _entry_to_config(entry: ConfigEntry) -> RouterConfig:
    merged = {**entry.data, **entry.options}
    return RouterConfig(
        local_agent_id=merged[CONF_LOCAL_AGENT_ID],
        llm_agent_id=merged[CONF_LLM_AGENT_ID],
        language=merged.get(CONF_LANGUAGE, "en"),
        fuzzy_enabled=merged.get(CONF_FUZZY_ENABLED, DEFAULT_FUZZY_ENABLED),
        fuzzy_threshold=merged.get(CONF_FUZZY_THRESHOLD, DEFAULT_FUZZY_THRESHOLD),
        ambiguity_gap=merged.get(CONF_AMBIGUITY_GAP, DEFAULT_AMBIGUITY_GAP),
        llm_translate_enabled=merged.get(
            CONF_LLM_TRANSLATE_ENABLED,
            DEFAULT_LLM_TRANSLATE_ENABLED,
        ),
        llm_fallback_enabled=merged.get(
            CONF_LLM_FALLBACK_ENABLED,
            DEFAULT_LLM_FALLBACK_ENABLED,
        ),
        debug_enabled=merged.get(CONF_DEBUG_ENABLED, DEFAULT_DEBUG_ENABLED),
        catalog_auto_refresh_enabled=merged.get(
            CONF_CATALOG_AUTO_REFRESH_ENABLED,
            DEFAULT_CATALOG_AUTO_REFRESH_ENABLED,
        ),
        high_risk_threshold=merged.get(
            CONF_HIGH_RISK_THRESHOLD,
            DEFAULT_HIGH_RISK_THRESHOLD,
        ),
        max_llm_candidates=merged.get(
            CONF_MAX_LLM_CANDIDATES,
            DEFAULT_MAX_LLM_CANDIDATES,
        ),
        manual_targets=merged.get(CONF_MANUAL_TARGETS, []),
    )


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up integration domain."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up config entry runtime."""
    cfg = _entry_to_config(entry)
    catalog_manager = CatalogManager(hass, cfg)
    await catalog_manager.async_rebuild()

    agent_adapter = AgentAdapter(hass)
    matcher = FuzzyMatcher(cfg.fuzzy_threshold, cfg.ambiguity_gap)
    llm_adapter = LLMAdapter(agent_adapter)
    router = AgentRouter(
        config=cfg,
        catalog_manager=catalog_manager,
        matcher=matcher,
        agent_adapter=agent_adapter,
        llm_adapter=llm_adapter,
    )
    conv_agent = CatalogRouterConversationAgent(router=router, language=cfg.language)

    runtime = IntegrationRuntime(
        config=cfg,
        catalog_manager=catalog_manager,
        router=router,
        conversation_agent=conv_agent,
    )

    if cfg.catalog_auto_refresh_enabled:
        async def _periodic_refresh(now):
            await catalog_manager.async_rebuild()

        runtime.unsub_refresh = async_track_time_interval(
            hass,
            _periodic_refresh,
            period=timedelta(minutes=15),
        )

    await async_register_agent(hass, entry, conv_agent)

    hass.data[DOMAIN][entry.entry_id] = runtime
    if len(hass.data[DOMAIN]) == 1:
        await async_register_services(hass)

    entry.async_on_unload(entry.add_update_listener(async_update_options))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload config entry runtime."""
    runtime: IntegrationRuntime = hass.data[DOMAIN].pop(entry.entry_id)

    await async_unregister_agent(hass, entry)
    if runtime.unsub_refresh:
        runtime.unsub_refresh()

    if not hass.data[DOMAIN]:
        await async_unregister_services(hass)

    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
