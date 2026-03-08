"""Catalog source adapters."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from typing import Any

from .models import ConversationTarget, EntityTarget
from .phonetics import normalize_text, phonetic_tokens, tokenize

_LOGGER = logging.getLogger(__name__)


def _entity_capabilities_from_domain(domain: str) -> list[str]:
    mapping = {
        "light": ["turn_on", "turn_off", "set"],
        "switch": ["turn_on", "turn_off"],
        "fan": ["turn_on", "turn_off", "set"],
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


def _coerce_bool(value: Any) -> bool | None:
    """Best-effort coercion for mixed HA API return types."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "on", "yes", "1"}:
            return True
        if lowered in {"false", "off", "no", "0"}:
            return False
    return bool(value)


class EntityCatalogSource:
    """Build entity target catalog from Home Assistant state + registries."""

    async def async_collect(self, hass: Any) -> list[EntityTarget]:
        """Collect entity targets best-effort."""
        states_by_entity_id = {state.entity_id: state for state in hass.states.async_all()}
        area_lookup: dict[str, str] = {}
        device_lookup: dict[str, str] = {}
        floor_lookup: dict[str, str] = {}
        aliases_lookup: dict[str, list[str]] = {}
        included_entity_ids: set[str] = set()

        try:
            from homeassistant.helpers import entity_registry as er

            entity_reg = er.async_get(hass)
        except Exception as err:  # pragma: no cover - depends on HA internals
            _LOGGER.warning(
                "Entity registry unavailable; skipping entity catalog build: %s",
                err,
            )
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

        for entry in entity_reg.entities.values():
            if entry.disabled_by is not None or entry.hidden_by is not None:
                _LOGGER.debug(
                    "Entity skipped: entity_id=%s reason=hidden_or_disabled",
                    entry.entity_id,
                )
                continue

            state = states_by_entity_id.get(entry.entity_id)
            is_exposed = await _async_is_exposed_to_assist(
                hass=hass,
                entry=entry,
                state=state,
            )
            _LOGGER.debug(
                "Entity exposure check: entity_id=%s exposed_by=%s state_assist_exposed=%s included=%s",
                entry.entity_id,
                getattr(entry, "exposed_by", None),
                None if state is None else state.attributes.get("assist_exposed"),
                is_exposed,
            )
            if not is_exposed:
                continue

            included_entity_ids.add(entry.entity_id)
            aliases_lookup[entry.entity_id] = list(getattr(entry, "aliases", []) or [])

            try:
                if area_reg is not None and entry.area_id:
                    area = area_reg.async_get_area(entry.area_id)
                    if area:
                        area_lookup[entry.entity_id] = area.name
                        if floor_reg is not None and getattr(area, "floor_id", None):
                            floor = floor_reg.async_get_floor(area.floor_id)
                            if floor:
                                floor_lookup[entry.entity_id] = floor.name
            except Exception as err:  # pragma: no cover - optional enrichment
                _LOGGER.debug("Area lookup failed for %s: %s", entry.entity_id, err)

            try:
                if device_reg is not None and entry.device_id:
                    device = device_reg.async_get(entry.device_id)
                    if device:
                        device_lookup[entry.entity_id] = device.name_by_user or device.name
                        if area_reg is not None and device.area_id:
                            area = area_reg.async_get_area(device.area_id)
                            if area:
                                area_lookup.setdefault(entry.entity_id, area.name)
                                if floor_reg is not None and getattr(area, "floor_id", None):
                                    floor = floor_reg.async_get_floor(area.floor_id)
                                    if floor:
                                        floor_lookup[entry.entity_id] = floor.name
            except Exception as err:  # pragma: no cover - optional enrichment
                _LOGGER.debug("Device/floor enrichment failed for %s: %s", entry.entity_id, err)

        targets: list[EntityTarget] = []
        for entity_id in included_entity_ids:
            state = states_by_entity_id.get(entity_id)
            if state is None:
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
            if device_name:
                tokens.extend(tokenize(device_name))
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

        _LOGGER.debug(
            "Entity catalog included entity_count=%s included_entity_ids=%s",
            len(targets),
            sorted(included_entity_ids),
        )
        return targets


async def _async_is_exposed_to_assist(
    *,
    hass: Any,
    entry: Any,
    state: Any | None,
) -> bool:
    """Determine whether an entity is exposed to Assist.

    Preference order:
    1. Home Assistant exposed_entities helper API
    2. entity_registry.exposed_by
    3. state attribute fallback for older/custom setups
    """
    helper_result = await _async_check_assist_exposure_via_helper(
        hass=hass,
        entity_id=entry.entity_id,
    )
    if helper_result is not None:
        return helper_result

    if getattr(entry, "exposed_by", None) is not None:
        return True

    if state is not None:
        state_attr_result = _coerce_bool(state.attributes.get("assist_exposed"))
        if state_attr_result is not None:
            return state_attr_result

        for key in (
            "conversation_exposed",
            "exposed_to_assist",
            "voice_assistant_exposed",
        ):
            fallback = _coerce_bool(state.attributes.get(key))
            if fallback is not None:
                return fallback

    return False


async def _async_check_assist_exposure_via_helper(
    *,
    hass: Any,
    entity_id: str,
) -> bool | None:
    """Best-effort check using Home Assistant exposed_entities helpers."""
    try:
        from homeassistant.components.homeassistant import exposed_entities as ee
    except Exception as err:  # pragma: no cover - optional API
        _LOGGER.debug("Assist exposed-entities module unavailable: %s", err)
        return None

    candidate_calls: list[tuple[str, tuple[Any, ...]]] = [
        ("async_should_expose", (hass, "conversation", entity_id)),
        ("async_is_exposed", (hass, "conversation", entity_id)),
        ("should_expose", (hass, "conversation", entity_id)),
        ("is_exposed", (hass, "conversation", entity_id)),
        # Older/internal manager style fallbacks
        ("async_should_expose", ("conversation", entity_id)),
        ("async_is_exposed", ("conversation", entity_id)),
        ("should_expose", ("conversation", entity_id)),
        ("is_exposed", ("conversation", entity_id)),
    ]

    for attr_name, args in candidate_calls:
        func = getattr(ee, attr_name, None)
        if func is None:
            continue

        try:
            result = func(*args)
            if inspect.isawaitable(result):
                result = await result
            coerced = _coerce_bool(result)
            if coerced is not None:
                return coerced
        except TypeError:
            continue
        except Exception as err:  # pragma: no cover - version specific
            _LOGGER.debug(
                "Assist exposure helper failed for %s via %s: %s",
                entity_id,
                attr_name,
                err,
            )

    return None


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
        _LOGGER.debug("Intent script discovery: intent_script_domain_exists=%s", bool(intent_domain))
        if not intent_domain:
            return data

        scripts = intent_domain.get("intent_scripts") or {}
        _LOGGER.debug("Intent script discovery: intent_script_count=%s", len(scripts))
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
        _LOGGER.debug("Intent script discovery: conversation_targets_built=%s", len(data))
        return data

    async def _from_sentence_automations(self, hass: Any) -> list[ConversationTarget]:
        data: list[ConversationTarget] = []
        automation_states = list(hass.states.async_all("automation"))
        _LOGGER.debug("Automation sentence discovery: automation_state_count=%s", len(automation_states))

        trigger_attr_present = 0
        sentence_trigger_count = 0

        for state in automation_states:
            has_triggers_attr = "triggers" in state.attributes
            if has_triggers_attr:
                trigger_attr_present += 1

            triggers = state.attributes.get("triggers")
            if not isinstance(triggers, list):
                # In some HA versions trigger metadata is not exposed on runtime state attributes.
                # Keep this path conservative and skip without inventing targets.
                _LOGGER.debug(
                    "Automation sentence discovery: entity_id=%s has_triggers_attr=%s usable=%s",
                    state.entity_id,
                    has_triggers_attr,
                    False,
                )
                continue

            for idx, trigger in enumerate(triggers):
                if trigger.get("trigger") not in {"conversation", "sentence"}:
                    continue
                sentence = trigger.get("command") or trigger.get("sentence")
                if not sentence:
                    continue
                sentence_trigger_count += 1
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
        _LOGGER.debug(
            "Automation sentence discovery: trigger_attr_present_count=%s sentence_trigger_count=%s "
            "conversation_targets_built=%s",
            trigger_attr_present,
            sentence_trigger_count,
            len(data),
        )
        return data


class ManualConversationTargetSource:
    """Merge user-provided manual conversation targets."""

    async def async_collect(
        self,
        manual_targets: Iterable[dict[str, Any]],
    ) -> list[ConversationTarget]:
        """Build targets from manual settings."""
        output: list[ConversationTarget] = []
        for idx, target in enumerate(manual_targets):
            display_name = str(target.get("display_name", "")).strip()
            canonical_phrase = str(target.get("canonical_phrase", "")).strip()
            if not display_name or not canonical_phrase:
                continue

            sample_phrases = [
                str(v).strip()
                for v in target.get("sample_phrases", [])
                if str(v).strip()
            ]
            aliases = [
                str(v).strip() for v in target.get("aliases", []) if str(v).strip()
            ]
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
