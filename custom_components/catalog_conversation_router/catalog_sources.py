"""Catalog source adapters."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from collections.abc import Iterable
from typing import Any

from .models import ConversationTarget, EntityTarget
from .phonetics import normalize_text, phonetic_tokens, tokenize

_LOGGER = logging.getLogger(__name__)


def _entity_capabilities_from_domain(domain: str) -> list[str]:
    mapping = {
        "light": ["turn_on", "turn_off", "set"],
        "switch": ["turn_on", "turn_off"],
        "climate": ["set", "query", "turn_on", "turn_off"],
        "cover": ["open", "close", "set"],
        "lock": ["lock", "unlock"],
        "alarm_control_panel": ["arm", "disarm"],
        "sensor": ["query"],
        "binary_sensor": ["query"],
        "scene": ["activate"],
        "script": ["activate"],
    }
    return mapping.get(domain, ["activate", "query"])


class EntityCatalogSource:
    """Build entity target catalog from Home Assistant state + registries."""

    async def async_collect(self, hass: Any) -> list[EntityTarget]:
        """Collect entity targets best-effort."""
        area_lookup: dict[str, str] = {}
        device_lookup: dict[str, str] = {}
        floor_lookup: dict[str, str] = {}
        aliases_lookup: dict[str, list[str]] = {}
        included_entity_ids: set[str] = set()

        try:
            from homeassistant.helpers import entity_registry as er
            entity_reg = er.async_get(hass)
        except Exception as err:  # pragma: no cover - depends on HA internals
            _LOGGER.warning("Entity registry unavailable; skipping entity catalog build: %s", err)
            return []

        area_reg = None
        device_reg = None
        floor_reg = None

        try:
            from homeassistant.helpers import area_registry as ar
            from homeassistant.helpers import device_registry as dr

            area_reg = ar.async_get(hass)
            device_reg = dr.async_get(hass)
        except Exception as err:  # pragma: no cover - optional enrichment
            _LOGGER.debug("Area/device registry enrichment unavailable: %s", err)

        try:
            from homeassistant.helpers import floor_registry as fr

            floor_reg = fr.async_get(hass)
        except Exception as err:  # pragma: no cover - optional enrichment
            _LOGGER.debug("Floor registry enrichment unavailable: %s", err)

        exposure_checker = await _async_build_exposure_checker(hass)
        _LOGGER.debug("Assist exposure checker built=%s", exposure_checker is not None)

        for entry in entity_reg.entities.values():
            if entry.disabled_by is not None or entry.hidden_by is not None:
                _LOGGER.debug(
                    "Entity exposure check: entity_id=%s exposed_by=%s included=%s reason=%s",
                    entry.entity_id,
                    getattr(entry, "exposed_by", None),
                    False,
                    "hidden_or_disabled",
                )
                continue

            is_exposed = await _async_is_exposed_to_assist(
                entry=entry,
                checker=exposure_checker,
            )
            _LOGGER.debug(
                "Entity exposure check: entity_id=%s exposed_by=%s included=%s",
                entry.entity_id,
                getattr(entry, "exposed_by", None),
                is_exposed,
            )
            if not is_exposed:
                continue

            included_entity_ids.add(entry.entity_id)
            aliases_lookup[entry.entity_id] = list(entry.aliases or [])

            try:
                if area_reg is not None and entry.area_id:
                    if area := area_reg.async_get_area(entry.area_id):
                        area_lookup[entry.entity_id] = area.name
            except Exception as err:  # pragma: no cover - optional enrichment
                _LOGGER.debug("Area lookup failed for %s: %s", entry.entity_id, err)

            try:
                if device_reg is not None and entry.device_id:
                    device = device_reg.async_get(entry.device_id)
                    if device:
                        device_lookup[entry.entity_id] = device.name_by_user or device.name
                        if area_reg is not None and device.area_id:
                            if area := area_reg.async_get_area(device.area_id):
                                area_lookup.setdefault(entry.entity_id, area.name)
                                if floor_reg is not None and area.floor_id:
                                    if floor := floor_reg.async_get_floor(area.floor_id):
                                        floor_lookup[entry.entity_id] = floor.name
            except Exception as err:  # pragma: no cover - optional enrichment
                _LOGGER.debug("Device/floor enrichment failed for %s: %s", entry.entity_id, err)

        targets: list[EntityTarget] = []
        for state in hass.states.async_all():
            entity_id = state.entity_id
            if entity_id not in included_entity_ids:
                continue
            domain, _, _ = entity_id.partition(".")
            name = state.name or entity_id
            aliases = aliases_lookup.get(entity_id, [])
            area_name = area_lookup.get(entity_id)
            floor = floor_lookup.get(entity_id)
            device_name = device_lookup.get(entity_id)

            tokens = tokenize(name)
            for alias in aliases:
                tokens.extend(tokenize(alias))
            if area_name:
                tokens.extend(tokenize(area_name))
            tokens = sorted(set(tokens))

            targets.append(
                EntityTarget(
                    entity_id=entity_id,
                    name=name,
                    normalized_name=normalize_text(name),
                    aliases=aliases,
                    domain=domain,
                    area=area_name,
                    floor=floor,
                    device_name=device_name,
                    exposed=True,
                    capabilities=_entity_capabilities_from_domain(domain),
                    tokens=tokens,
                    phonetic_tokens=phonetic_tokens(tokens),
                )
            )

        _LOGGER.debug("Entity catalog included entity_count=%s", len(targets))
        return targets


async def _async_is_exposed_to_assist(
    *,
    entry: Any,
    checker: Callable[[str], Awaitable[bool | None]] | None,
) -> bool:
    """Determine exposure with fallback to Assist exposure API when needed."""
    if getattr(entry, "exposed_by", None) is not None:
        return True

    if checker is None:
        return False

    checked = await checker(entry.entity_id)
    return checked is True


async def _async_build_exposure_checker(
    hass: Any,
) -> Callable[[str], Awaitable[bool | None]] | None:
    """Build a checker for Assist exposure using Home Assistant exposed-entities helpers."""
    try:
        from homeassistant.components.homeassistant import exposed_entities as ee
    except Exception as err:  # pragma: no cover - optional API
        _LOGGER.debug("Assist exposed-entities API unavailable: %s", err)
        return None

    async def _check(entity_id: str) -> bool | None:
        if hasattr(ee, "async_should_expose"):
            return bool(await ee.async_should_expose(hass, "conversation", entity_id))
        if hasattr(ee, "async_is_exposed"):
            return bool(await ee.async_is_exposed(hass, "conversation", entity_id))
        if hasattr(ee, "should_expose"):
            return bool(ee.should_expose(hass, "conversation", entity_id))
        if hasattr(ee, "is_exposed"):
            return bool(ee.is_exposed(hass, "conversation", entity_id))
        return None

    return _check


class ConversationTargetSource:
    """Build conversation targets from discoverable HA sources."""

    async def async_collect(self, hass: Any) -> list[ConversationTarget]:
        """Collect conversation target candidates from optional sources."""
        targets: list[ConversationTarget] = []
        targets.extend(await self._from_intent_script(hass))
        targets.extend(await self._from_sentence_automations(hass))
        return targets

    async def _from_intent_script(self, hass: Any) -> list[ConversationTarget]:
        data: list[ConversationTarget] = []
        intent_domain = hass.data.get("intent_script")
        if not intent_domain:
            return data

        scripts = intent_domain.get("intent_scripts") or {}
        for intent_name, script in scripts.items():
            sample = intent_name.replace("_", " ").lower()
            canonical = f"{sample}"
            data.append(
                ConversationTarget(
                    target_id=f"intent_script:{intent_name}",
                    type="intent_script",
                    display_name=intent_name,
                    normalized_name=normalize_text(intent_name),
                    sample_phrases=[sample],
                    canonical_phrase=canonical,
                    source="intent_script",
                    slots=list((script or {}).get("slot_schema", {}).keys()),
                    tokens=tokenize(intent_name),
                    phonetic_tokens=phonetic_tokens(tokenize(intent_name)),
                )
            )
        return data

    async def _from_sentence_automations(self, hass: Any) -> list[ConversationTarget]:
        data: list[ConversationTarget] = []
        for state in hass.states.async_all("automation"):
            triggers = state.attributes.get("triggers") or []
            for idx, trigger in enumerate(triggers):
                if trigger.get("trigger") not in {"conversation", "sentence"}:
                    continue
                sentence = trigger.get("command") or trigger.get("sentence")
                if not sentence:
                    continue
                display_name = state.name or state.entity_id
                tokens = tokenize(display_name + " " + sentence)
                data.append(
                    ConversationTarget(
                        target_id=f"automation:{state.entity_id}:{idx}",
                        type="automation_sentence",
                        display_name=display_name,
                        normalized_name=normalize_text(display_name),
                        sample_phrases=[sentence],
                        canonical_phrase=sentence,
                        source="automation",
                        slots=[],
                        tokens=tokens,
                        phonetic_tokens=phonetic_tokens(tokens),
                    )
                )
        return data


class ManualConversationTargetSource:
    """Merge user-provided manual conversation targets."""

    async def async_collect(self, manual_targets: Iterable[dict[str, Any]]) -> list[ConversationTarget]:
        """Build targets from manual settings."""
        output: list[ConversationTarget] = []
        for idx, target in enumerate(manual_targets):
            display_name = str(target.get("display_name", "")).strip()
            canonical_phrase = str(target.get("canonical_phrase", "")).strip()
            if not display_name or not canonical_phrase:
                continue

            sample_phrases = [str(v).strip() for v in target.get("sample_phrases", []) if str(v).strip()]
            aliases = [str(v).strip() for v in target.get("aliases", []) if str(v).strip()]
            enabled = bool(target.get("enabled", True))
            target_type = str(target.get("target_type", "manual"))

            token_basis = [display_name, canonical_phrase, *sample_phrases, *aliases]
            token_values: list[str] = []
            for value in token_basis:
                token_values.extend(tokenize(value))
            token_values = sorted(set(token_values))

            output.append(
                ConversationTarget(
                    target_id=f"manual:{idx}:{normalize_text(display_name)}",
                    type=target_type,
                    display_name=display_name,
                    normalized_name=normalize_text(display_name),
                    sample_phrases=sample_phrases,
                    canonical_phrase=canonical_phrase,
                    source="manual",
                    slots=[str(v) for v in target.get("slots", []) if str(v).strip()],
                    tokens=token_values,
                    phonetic_tokens=phonetic_tokens(token_values),
                    aliases=aliases,
                    enabled=enabled,
                )
            )
        return output
