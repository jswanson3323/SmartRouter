"""Catalog manager and manual target tests."""

import asyncio

from custom_components.catalog_conversation_router.catalog import CatalogManager
from custom_components.catalog_conversation_router.models import RouterConfig


class _FakeState:
    def __init__(self, entity_id: str, name: str) -> None:
        self.entity_id = entity_id
        self.name = name
        self.attributes = {}


class _FakeStates:
    def __init__(self, states):
        self._states = states

    def async_all(self, domain=None):
        if domain is None:
            return list(self._states)
        return [s for s in self._states if s.entity_id.startswith(f"{domain}.")]


class _FakeHass:
    def __init__(self, states):
        self.states = _FakeStates(states)
        self.data = {}


def _config() -> RouterConfig:
    return RouterConfig(
        local_agent_id="local",
        llm_agent_id="llm",
        language="en",
        fuzzy_enabled=True,
        fuzzy_threshold=0.84,
        ambiguity_gap=0.08,
        llm_translate_enabled=True,
        llm_fallback_enabled=True,
        debug_enabled=False,
        catalog_auto_refresh_enabled=True,
        high_risk_threshold=0.96,
        max_llm_candidates=20,
        manual_targets=[
            {
                "display_name": "Movie Mode",
                "target_type": "manual",
                "sample_phrases": ["movie time"],
                "canonical_phrase": "activate movie mode",
                "aliases": ["movie moat"],
                "enabled": True,
            }
        ],
    )


def test_build_entity_catalog_and_merge_manual_targets() -> None:
    hass = _FakeHass([_FakeState("light.kitchen", "Kitchen Light")])
    manager = CatalogManager(hass, _config())
    catalog = asyncio.run(manager.async_rebuild())
    assert catalog.metadata.entity_count == 1
    assert catalog.metadata.conversation_target_count >= 1


def test_catalog_stats_shape() -> None:
    hass = _FakeHass([_FakeState("sensor.temp", "Temperature")])
    manager = CatalogManager(hass, _config())
    asyncio.run(manager.async_rebuild())
    stats = manager.stats()
    assert set(stats) >= {
        "revision",
        "last_refreshed",
        "entity_count",
        "conversation_target_count",
        "refresh_failures",
    }
