"""Service registration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from .const import (
    ATTR_TEXT,
    DOMAIN,
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
        output: dict[str, Any] = {}
        for entry_id, runtime in hass.data.get(DOMAIN, {}).items():
            result = await runtime.router.async_route(
                text=text,
                language=runtime.config.language,
                conversation_id=None,
                context=call.context,
                dry_run=True,
            )
            output[entry_id] = {
                "path": result.path.value,
                "trace": result.trace.as_dict(),
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
    test_schema = vol.Schema({vol.Required(ATTR_TEXT): str})

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


async def async_unregister_services(hass) -> None:
    """Remove integration services when last entry unloads."""
    for service in (
        SERVICE_REBUILD_CATALOG,
        SERVICE_GET_CATALOG_STATS,
        SERVICE_TEST_UTTERANCE,
    ):
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)
