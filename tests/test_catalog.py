"""Catalog manager and manual target tests."""

import asyncio
import sys
import types

import pytest

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


class _FakeEntityEntry:
    def __init__(
        self,
        *,
        entity_id: str,
        disabled_by=None,
        hidden_by=None,
        exposed_by="assist",
        aliases=None,
        area_id=None,
        device_id=None,
        labels=None,
    ) -> None:
        self.entity_id = entity_id
        self.disabled_by = disabled_by
        self.hidden_by = hidden_by
        self.exposed_by = exposed_by
        self.aliases = aliases or []
        self.area_id = area_id
        self.device_id = device_id
        self.labels = labels or []


class _FakeEntityRegistry:
    def __init__(self, entries) -> None:
        self.entities = {entry.entity_id: entry for entry in entries}


class _FakeArea:
    def __init__(self, name: str, floor_id=None, labels=None) -> None:
        self.name = name
        self.floor_id = floor_id
        self.labels = labels or []


class _FakeAreaRegistry:
    def __init__(self, areas) -> None:
        self._areas = areas

    def async_get_area(self, area_id):
        return self._areas.get(area_id)


class _FakeDevice:
    def __init__(self, name: str, area_id=None, labels=None) -> None:
        self.name = name
        self.name_by_user = None
        self.area_id = area_id
        self.labels = labels or []


class _FakeDeviceRegistry:
    def __init__(self, devices) -> None:
        self._devices = devices

    def async_get(self, device_id):
        return self._devices.get(device_id)


class _FakeFloor:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeFloorRegistry:
    def __init__(self, floors) -> None:
        self._floors = floors

    def async_get_floor(self, floor_id):
        return self._floors.get(floor_id)


class _FakeHass:
    def __init__(self, states, entity_entries, exposure_map=None):
        self.states = _FakeStates(states)
        self.data = {}
        self._entity_reg = _FakeEntityRegistry(entity_entries)
        self._area_reg = _FakeAreaRegistry({"area_kitchen": _FakeArea("Kitchen", "f1")})
        self._device_reg = _FakeDeviceRegistry({"dev1": _FakeDevice("Kitchen Device", "area_kitchen")})
        self._floor_reg = _FakeFloorRegistry({"f1": _FakeFloor("First Floor")})
        self._label_reg = types.SimpleNamespace(labels={})
        self._exposure_map = exposure_map or {}


def _install_fake_registry_modules(monkeypatch, hass: _FakeHass) -> None:
    homeassistant_mod = types.ModuleType("homeassistant")
    helpers_mod = types.ModuleType("homeassistant.helpers")
    components_mod = types.ModuleType("homeassistant.components")
    ha_comp_mod = types.ModuleType("homeassistant.components.homeassistant")

    ar_mod = types.ModuleType("homeassistant.helpers.area_registry")
    ar_mod.async_get = lambda _hass: hass._area_reg

    dr_mod = types.ModuleType("homeassistant.helpers.device_registry")
    dr_mod.async_get = lambda _hass: hass._device_reg

    er_mod = types.ModuleType("homeassistant.helpers.entity_registry")
    er_mod.async_get = lambda _hass: hass._entity_reg

    fr_mod = types.ModuleType("homeassistant.helpers.floor_registry")
    fr_mod.async_get = lambda _hass: hass._floor_reg

    lr_mod = types.ModuleType("homeassistant.helpers.label_registry")
    lr_mod.async_get = lambda _hass: hass._label_reg

    ee_mod = types.ModuleType("homeassistant.components.homeassistant.exposed_entities")

    async def _async_should_expose(_hass, _assistant, entity_id):
        return hass._exposure_map.get(entity_id)

    ee_mod.async_should_expose = _async_should_expose

    monkeypatch.setitem(sys.modules, "homeassistant", homeassistant_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers", helpers_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.components", components_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.components.homeassistant", ha_comp_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.area_registry", ar_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.device_registry", dr_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.entity_registry", er_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.floor_registry", fr_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.label_registry", lr_mod)
    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.homeassistant.exposed_entities",
        ee_mod,
    )


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


def test_build_entity_catalog_and_merge_manual_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = _FakeHass(
        [_FakeState("light.kitchen", "Kitchen Light")],
        [
            _FakeEntityEntry(
                entity_id="light.kitchen",
                exposed_by="assist",
                aliases=["Kitchen Lamp"],
                area_id="area_kitchen",
                device_id="dev1",
            )
        ],
    )
    _install_fake_registry_modules(monkeypatch, hass)

    manager = CatalogManager(hass, _config())
    catalog = asyncio.run(manager.async_rebuild())
    assert catalog.metadata.entity_count == 1
    assert catalog.entity_targets[0].exposed is True
    assert catalog.metadata.conversation_target_count >= 1


def test_non_exposed_entity_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = _FakeHass(
        [
            _FakeState("light.kitchen", "Kitchen Light"),
            _FakeState("light.secret", "Secret Light"),
        ],
        [
            _FakeEntityEntry(entity_id="light.kitchen", exposed_by="assist"),
            _FakeEntityEntry(entity_id="light.secret", exposed_by=None),
        ],
    )
    _install_fake_registry_modules(monkeypatch, hass)

    manager = CatalogManager(hass, _config())
    catalog = asyncio.run(manager.async_rebuild())
    assert [target.entity_id for target in catalog.entity_targets] == ["light.kitchen"]


def test_exposed_by_empty_but_assist_api_exposed_is_included(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = _FakeHass(
        [_FakeState("light.kitchen", "Kitchen Light")],
        [_FakeEntityEntry(entity_id="light.kitchen", exposed_by=None)],
        exposure_map={"light.kitchen": True},
    )
    _install_fake_registry_modules(monkeypatch, hass)

    manager = CatalogManager(hass, _config())
    catalog = asyncio.run(manager.async_rebuild())
    assert [target.entity_id for target in catalog.entity_targets] == ["light.kitchen"]


def test_catalog_stats_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = _FakeHass(
        [_FakeState("sensor.temp", "Temperature")],
        [_FakeEntityEntry(entity_id="sensor.temp", exposed_by="assist")],
    )
    _install_fake_registry_modules(monkeypatch, hass)

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


def test_area_super_area_label_is_added_to_entity_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = _FakeHass(
        [_FakeState("light.kitchen", "Kitchen Light")],
        [
            _FakeEntityEntry(
                entity_id="light.kitchen",
                exposed_by="assist",
                area_id="area_kitchen",
                device_id="dev1",
            )
        ],
    )
    hass._area_reg = _FakeAreaRegistry(
        {"area_kitchen": _FakeArea("Kitchen", "f1", labels=["label_upstairs"])}
    )
    hass._label_reg = types.SimpleNamespace(
        labels={"label_upstairs": types.SimpleNamespace(name="SuperArea: Upstairs")}
    )
    _install_fake_registry_modules(monkeypatch, hass)

    manager = CatalogManager(hass, _config())
    catalog = asyncio.run(manager.async_rebuild())

    assert catalog.entity_targets[0].area == "Kitchen"
    assert catalog.entity_targets[0].super_area == "Upstairs"


def test_area_super_area_label_ids_are_added_to_entity_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = _FakeHass(
        [_FakeState("light.kitchen", "Kitchen Light")],
        [
            _FakeEntityEntry(
                entity_id="light.kitchen",
                exposed_by="assist",
                area_id="area_kitchen",
                device_id="dev1",
            )
        ],
    )
    hass._area_reg = _FakeAreaRegistry(
        {"area_kitchen": types.SimpleNamespace(name="Kitchen", floor_id="f1", label_ids=["label_upstairs"])}
    )
    hass._label_reg = types.SimpleNamespace(
        labels={"label_upstairs": types.SimpleNamespace(name="SuperArea: Upstairs")}
    )
    _install_fake_registry_modules(monkeypatch, hass)

    manager = CatalogManager(hass, _config())
    catalog = asyncio.run(manager.async_rebuild())

    assert catalog.entity_targets[0].area == "Kitchen"
    assert catalog.entity_targets[0].super_area == "Upstairs"


def test_area_super_area_slug_label_ids_are_added_to_entity_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = _FakeHass(
        [_FakeState("light.kitchen", "Kitchen Light")],
        [
            _FakeEntityEntry(
                entity_id="light.kitchen",
                exposed_by="assist",
                area_id="area_kitchen",
                device_id="dev1",
            )
        ],
    )
    hass._area_reg = _FakeAreaRegistry(
        {"area_kitchen": types.SimpleNamespace(name="Kitchen", floor_id="f1", labels={"superarea_great_room"})}
    )
    hass._label_reg = types.SimpleNamespace(
        labels=[],
        async_get_label=lambda label_id: types.SimpleNamespace(name="SuperArea: Great Room")
        if label_id == "superarea_great_room"
        else None,
    )
    _install_fake_registry_modules(monkeypatch, hass)

    manager = CatalogManager(hass, _config())
    catalog = asyncio.run(manager.async_rebuild())

    assert catalog.entity_targets[0].area == "Kitchen"
    assert catalog.entity_targets[0].super_area == "Great Room"
