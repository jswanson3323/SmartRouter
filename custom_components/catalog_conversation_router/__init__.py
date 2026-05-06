"""Catalog Conversation Router integration."""

from __future__ import annotations

import asyncio
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
    CONF_TRANSLATE_LLM_AGENT_ID,
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
    PLATFORMS,
)
from .conversation import (
    CatalogRouterConversationAgent,
    CatalogRouterLegacyAgentAlias,
    async_register_agent_alias,
    async_unregister_agent_alias,
)
from .ha_conversation_agents import (
    get_registered_conversation_agents,
    get_registered_llm_agents,
)
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
    legacy_agent_alias: object | None = None
    migration_task: asyncio.Task | None = None
    unsub_refresh: callable | None = None


def _available_agent_ids(hass: HomeAssistant) -> set[str]:
    """Return currently callable conversation agent ids."""
    return {descriptor.agent_id for descriptor in get_registered_conversation_agents(hass)}


def _available_llm_agent_ids(hass: HomeAssistant) -> set[str]:
    """Return currently callable LLM-capable conversation agent ids."""
    return {descriptor.agent_id for descriptor in get_registered_llm_agents(hass)}


def _entry_to_config(entry: ConfigEntry) -> RouterConfig:
    merged = {**entry.data, **entry.options}
    return RouterConfig(
        local_agent_id=merged[CONF_LOCAL_AGENT_ID],
        llm_agent_id=merged[CONF_LLM_AGENT_ID],
        translate_llm_agent_id=merged.get(
            CONF_TRANSLATE_LLM_AGENT_ID,
            merged[CONF_LLM_AGENT_ID],
        ),
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
    existing_runtime: IntegrationRuntime | None = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if existing_runtime is not None:
        _LOGGER.warning(
            "Config entry %s is being set up while an existing runtime is still present; cleaning up stale runtime first",
            entry.entry_id,
        )
        await _async_teardown_runtime(hass, entry, existing_runtime, remove_from_store=False)

    cfg = _entry_to_config(entry)
    available_agents = _available_agent_ids(hass)
    available_llm_agents = _available_llm_agent_ids(hass)
    if cfg.local_agent_id not in {"homeassistant", "__default__", "default"} and cfg.local_agent_id not in available_agents:
        _LOGGER.warning(
            "Configured local agent_id %s is not currently available; falling back to Home Assistant local agent",
            cfg.local_agent_id,
        )
        cfg.local_agent_id = "homeassistant"
    if cfg.llm_agent_id not in available_llm_agents:
        if len(available_llm_agents) == 1:
            fallback_llm_agent_id = next(iter(available_llm_agents))
            _LOGGER.warning(
                "Configured llm_agent_id %s is not currently callable; using the only available LLM agent %s for this runtime",
                cfg.llm_agent_id,
                fallback_llm_agent_id,
            )
            cfg.llm_agent_id = fallback_llm_agent_id
        else:
            _LOGGER.warning(
                "Configured llm_agent_id %s is not currently callable during setup; leaving LLM features enabled and retrying resolution at request time. Available LLM agents right now: %s",
                cfg.llm_agent_id,
                sorted(available_llm_agents),
            )
    if cfg.llm_agent_id == entry.entry_id:
        _LOGGER.error(
            "Configured llm_agent_id for entry %s points back to the router itself; disabling direct self-recursive selection at setup",
            entry.entry_id,
        )
        non_router_llm_agents = {
            agent_id for agent_id in available_llm_agents if agent_id != entry.entry_id
        }
        if len(non_router_llm_agents) == 1:
            cfg.llm_agent_id = next(iter(non_router_llm_agents))
            _LOGGER.warning(
                "Using non-router LLM agent %s instead of recursive self-selection",
                cfg.llm_agent_id,
            )

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
        hass=hass,
        router_agent_id=entry.entry_id,
    )
    conv_agent = CatalogRouterConversationAgent(
        router=router,
        language=cfg.language,
        entry_id=entry.entry_id,
    )
    legacy_agent_alias = CatalogRouterLegacyAgentAlias(
        legacy_agent_id=entry.entry_id,
        entity_agent=conv_agent,
        language=cfg.language,
    )

    runtime = IntegrationRuntime(
        config=cfg,
        catalog_manager=catalog_manager,
        router=router,
        conversation_agent=conv_agent,
        legacy_agent_alias=legacy_agent_alias,
    )

    if cfg.catalog_auto_refresh_enabled:
        async def _periodic_refresh(now):
            await catalog_manager.async_rebuild()

        runtime.unsub_refresh = async_track_time_interval(
            hass,
            _periodic_refresh,
            timedelta(minutes=15),
        )

    hass.data[DOMAIN][entry.entry_id] = runtime
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    await async_register_agent_alias(hass, entry, legacy_agent_alias)
    runtime.migration_task = hass.async_create_task(
        _async_retry_assist_pipeline_engine_migration(
            hass,
            old_engine_id=entry.entry_id,
            new_engine_id=conv_agent.entity_id,
        ),
        name=f"catalog_router_pipeline_migration_{entry.entry_id}",
    )
    _LOGGER.info(
        "Catalog router runtime ready for entry %s: local_agent_id=%s llm_agent_id=%s available_llm_agents=%s runtime_count=%s",
        entry.entry_id,
        cfg.local_agent_id,
        cfg.llm_agent_id,
        sorted(available_llm_agents),
        len(hass.data[DOMAIN]),
    )
    if len(hass.data[DOMAIN]) == 1:
        await async_register_services(hass)

    entry.async_on_unload(entry.add_update_listener(async_update_options))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload config entry runtime."""
    runtime: IntegrationRuntime = hass.data[DOMAIN].pop(entry.entry_id)
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    await _async_teardown_runtime(hass, entry, runtime, remove_from_store=False)

    if not hass.data[DOMAIN]:
        await async_unregister_services(hass)

    return unload_ok


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload when options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def _async_teardown_runtime(
    hass: HomeAssistant,
    entry: ConfigEntry,
    runtime: IntegrationRuntime,
    *,
    remove_from_store: bool,
) -> None:
    """Tear down a runtime defensively."""
    if remove_from_store:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    if runtime.migration_task is not None and not runtime.migration_task.done():
        runtime.migration_task.cancel()
    try:
        await async_unregister_agent_alias(hass, entry)
    except Exception:
        _LOGGER.exception("Failed to unregister conversation agent alias for entry %s", entry.entry_id)
    if runtime.unsub_refresh:
        try:
            runtime.unsub_refresh()
        except Exception:
            _LOGGER.exception("Failed to unsubscribe refresh listener for entry %s", entry.entry_id)


async def _async_migrate_assist_pipeline_engine_ids(
    hass: HomeAssistant,
    *,
    old_engine_id: str,
    new_engine_id: str,
) -> bool:
    """Migrate stored Assist pipelines from legacy manager ids to entity ids."""
    if old_engine_id == new_engine_id:
        return True
    try:
        from homeassistant.components import assist_pipeline
    except Exception:
        return False

    try:
        if getattr(assist_pipeline, "KEY_ASSIST_PIPELINE", None) not in hass.data:
            return False
        pipelines = assist_pipeline.async_get_pipelines(hass)
    except Exception:
        _LOGGER.exception("Failed to load Assist pipelines for conversation engine migration")
        return False

    migrated = 0
    for pipeline in pipelines:
        pipeline_engine = getattr(pipeline, "conversation_engine", None)
        if pipeline_engine != old_engine_id:
            continue
        try:
            await assist_pipeline.async_update_pipeline(
                hass,
                pipeline,
                conversation_engine=new_engine_id,
            )
            migrated += 1
        except Exception:
            _LOGGER.exception(
                "Failed to migrate Assist pipeline %s from %s to %s",
                getattr(pipeline, "id", "<unknown>"),
                old_engine_id,
                new_engine_id,
            )
    migrated += await _async_migrate_stale_raw_pipeline_engine_ids(
        hass,
        pipelines=pipelines,
        new_engine_id=new_engine_id,
    )
    if migrated:
        _LOGGER.warning(
            "Migrated %s Assist pipeline(s) to conversation engine %s",
            migrated,
            new_engine_id,
        )
    return True


async def _async_migrate_stale_raw_pipeline_engine_ids(
    hass: HomeAssistant,
    *,
    pipelines,
    new_engine_id: str,
) -> int:
    """Migrate stale raw pipeline engine ids that no longer map to any config entry."""
    try:
        from homeassistant.components import assist_pipeline
    except Exception:
        return 0

    migrated = 0
    for pipeline in pipelines:
        pipeline_engine = getattr(pipeline, "conversation_engine", None)
        if not isinstance(pipeline_engine, str) or "." in pipeline_engine:
            continue
        if hass.config_entries.async_get_entry(pipeline_engine) is not None:
            continue
        try:
            await assist_pipeline.async_update_pipeline(
                hass,
                pipeline,
                conversation_engine=new_engine_id,
            )
            migrated += 1
            _LOGGER.warning(
                "Migrated stale raw Assist pipeline engine %s on pipeline %s to %s",
                pipeline_engine,
                getattr(pipeline, "id", "<unknown>"),
                new_engine_id,
            )
        except Exception:
            _LOGGER.exception(
                "Failed migrating stale raw Assist pipeline engine %s on pipeline %s",
                pipeline_engine,
                getattr(pipeline, "id", "<unknown>"),
            )
    return migrated


async def _async_retry_assist_pipeline_engine_migration(
    hass: HomeAssistant,
    *,
    old_engine_id: str,
    new_engine_id: str,
) -> None:
    """Retry Assist pipeline migration until the assist pipeline store is ready."""
    for delay_s in (0, 5, 15, 30, 60):
        if delay_s:
            await asyncio.sleep(delay_s)
        ready = await _async_migrate_assist_pipeline_engine_ids(
            hass,
            old_engine_id=old_engine_id,
            new_engine_id=new_engine_id,
        )
        if ready:
            return
    _LOGGER.warning(
        "Assist pipeline storage was not ready for conversation engine migration after retries; legacy engine id %s may remain configured until the next reload",
        old_engine_id,
    )
