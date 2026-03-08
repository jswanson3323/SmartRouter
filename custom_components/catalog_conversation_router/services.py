"""Service registration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from .const import (
    ATTR_AREA,
    ATTR_TEXT,
    DOMAIN,
    SERVICE_DUMP_CATALOG,
    SERVICE_GET_CATALOG_STATS,
    SERVICE_REBUILD_CATALOG,
    SERVICE_TEST_UTTERANCE,
)

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - HA runtime specific
    from homeassistant.core import SupportsResponse
except Exception:  # pragma: no cover
    SupportsResponse = None  # type: ignore[assignment]


async def async_register_services(hass) -> None:
    """Register integration services."""

    async def _rebuild(call) -> None:
        for runtime in hass.data.get(DOMAIN, {}).values():
            await runtime.catalog_manager.async_rebuild()

    async def _stats(call) -> dict[str, Any]:
        payload = {
            entry_id: runtime.catalog_manager.stats()
            for entry_id, runtime in hass.data.get(DOMAIN, {}).items()
        }
        return {"entries": payload}

    async def _test_utterance(call) -> dict[str, Any]:
        text = call.data[ATTR_TEXT]
        origin_area = call.data.get(ATTR_AREA)
        output: dict[str, Any] = {}
        for entry_id, runtime in hass.data.get(DOMAIN, {}).items():
            result = await runtime.router.async_route(
                text=text,
                language=runtime.config.language,
                conversation_id=None,
                context=call.context,
                dry_run=True,
                origin_area=origin_area,
            )
            output[entry_id] = {
                "path": result.path.value,
                "trace": result.trace.as_dict(),
            }
        return {"entries": output}

    async def _dump_catalog(call) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for entry_id, runtime in hass.data.get(DOMAIN, {}).items():
            catalog = runtime.catalog_manager.get_catalog()
            output[entry_id] = {
                "stats": runtime.catalog_manager.stats(),
                "entity_targets": [
                    {
                        "entity_id": target.entity_id,
                        "name": target.name,
                        "domain": target.domain,
                        "area": target.area,
                        "floor": target.floor,
                        "device_name": target.device_name,
                        "aliases": list(target.aliases),
                        "capabilities": list(target.capabilities),
                        "tokens": list(target.tokens),
                        "phonetic_tokens": list(target.phonetic_tokens),
                    }
                    for target in catalog.entity_targets
                ],
                "conversation_targets": [
                    {
                        "target_id": target.target_id,
                        "type": target.type,
                        "display_name": target.display_name,
                        "canonical_phrase": target.canonical_phrase,
                        "sample_phrases": list(target.sample_phrases),
                        "source": target.source,
                        "slots": list(target.slots),
                        "aliases": list(target.aliases),
                        "enabled": target.enabled,
                        "tokens": list(target.tokens),
                        "phonetic_tokens": list(target.phonetic_tokens),
                    }
                    for target in catalog.conversation_targets
                ],
            }
        return {"entries": output}

    if not hass.services.has_service(DOMAIN, SERVICE_REBUILD_CATALOG):
        hass.services.async_register(
            DOMAIN,
            SERVICE_REBUILD_CATALOG,
            _rebuild,
            schema=vol.Schema({}),
        )

    stats_schema = vol.Schema({})
    dump_schema = vol.Schema({})
    test_schema = vol.Schema({vol.Required(ATTR_TEXT): str, vol.Optional(ATTR_AREA): str})

    if SupportsResponse is not None:
        if not hass.services.has_service(DOMAIN, SERVICE_GET_CATALOG_STATS):
            hass.services.async_register(
                DOMAIN,
                SERVICE_GET_CATALOG_STATS,
                _stats,
                schema=stats_schema,
                supports_response=SupportsResponse.ONLY,
            )
        if not hass.services.has_service(DOMAIN, SERVICE_TEST_UTTERANCE):
            hass.services.async_register(
                DOMAIN,
                SERVICE_TEST_UTTERANCE,
                _test_utterance,
                schema=test_schema,
                supports_response=SupportsResponse.ONLY,
            )
        if not hass.services.has_service(DOMAIN, SERVICE_DUMP_CATALOG):
            hass.services.async_register(
                DOMAIN,
                SERVICE_DUMP_CATALOG,
                _dump_catalog,
                schema=dump_schema,
                supports_response=SupportsResponse.ONLY,
            )
    else:  # pragma: no cover
        _LOGGER.warning("SupportsResponse not available; response services downgraded")
        if not hass.services.has_service(DOMAIN, SERVICE_GET_CATALOG_STATS):
            hass.services.async_register(
                DOMAIN,
                SERVICE_GET_CATALOG_STATS,
                _stats,
                schema=stats_schema,
            )
        if not hass.services.has_service(DOMAIN, SERVICE_TEST_UTTERANCE):
            hass.services.async_register(
                DOMAIN,
                SERVICE_TEST_UTTERANCE,
                _test_utterance,
                schema=test_schema,
            )
        if not hass.services.has_service(DOMAIN, SERVICE_DUMP_CATALOG):
            hass.services.async_register(
                DOMAIN,
                SERVICE_DUMP_CATALOG,
                _dump_catalog,
                schema=dump_schema,
            )


async def async_unregister_services(hass) -> None:
    """Remove integration services when last entry unloads."""
    for service in (
        SERVICE_REBUILD_CATALOG,
        SERVICE_GET_CATALOG_STATS,
        SERVICE_TEST_UTTERANCE,
        SERVICE_DUMP_CATALOG,
    ):
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)
