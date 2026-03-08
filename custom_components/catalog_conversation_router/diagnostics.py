"""Diagnostics support."""

from __future__ import annotations

from typing import Any

from .const import CONF_MANUAL_TARGETS, DOMAIN

REDACT_KEYS = {CONF_MANUAL_TARGETS}


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key in REDACT_KEYS:
            result[key] = "***"
        else:
            result[key] = value
    return result


async def async_get_config_entry_diagnostics(hass, entry) -> dict[str, Any]:
    """Return diagnostics data."""
    runtime = hass.data[DOMAIN][entry.entry_id]
    return {
        "entry": _redact(dict(entry.data)),
        "options": _redact(dict(entry.options)),
        "catalog_stats": runtime.catalog_manager.stats(),
    }
