"""Catalog source adapters."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

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


def _safe_list(value: Any) -> list[Any]:
    """Normalize a possibly-scalar/list YAML value into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_commands(command_value: Any) -> list[str]:
    """Extract command strings from scalar/list command fields."""
    commands: list[str] = []
    for item in _safe_list(command_value):
        if isinstance(item, str):
            text = item.strip()
            if text:
                commands.append(text)
    return commands


def _tokenize_all(values: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(tokenize(value))
    return sorted(set(tokens))


class _LenientLoader(yaml.SafeLoader):
    """YAML loader that tolerates Home Assistant custom tags like !input."""


def _construct_unknown_tag(
    loader: _LenientLoader,
    node: yaml.Node,
) -> Any:
    """Handle unknown YAML tags by returning the underlying Python value."""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return None


_LenientLoader.add_constructor(None, _construct_unknown_tag)


def _read_yaml_file(path: Path) -> Any:
    """Read YAML file best-effort."""
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return yaml.load(text, Loader=_LenientLoader)
    except Exception as err:  # pragma: no cover - filesystem/runtime specific
        _LOGGER.warning("Failed reading YAML file %s: %s", path, err)
        return None


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

        _LOGGER.warning(
            "Entity catalog build complete: entity_count=%s",
            len(targets),
        )
        _LOGGER.debug(
            "Entity catalog included entity_ids=%s",
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
        targets.extend(await self._from_intent_script_runtime(hass))
        targets.extend(await self._from_intent_script_yaml(hass))
        targets.extend(await self._from_automation_runtime(hass))
        targets.extend(await self._from_automation_yaml(hass))

        deduped: dict[tuple[str, str, str], ConversationTarget] = {}
        for target in targets:
            key = (
                target.type,
                normalize_text(target.display_name),
                normalize_text(target.canonical_phrase),
            )
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = target
                continue

            merged_phrases = sorted(
                {
                    *existing.sample_phrases,
                    *target.sample_phrases,
                }
            )
            merged_tokens = sorted({*existing.tokens, *target.tokens})
            merged_phonetics = sorted({*existing.phonetic_tokens, *target.phonetic_tokens})
            existing.sample_phrases = merged_phrases
            existing.tokens = merged_tokens
            existing.phonetic_tokens = merged_phonetics

        final_targets = list(deduped.values())
        _LOGGER.warning(
            "Conversation target catalog build complete: target_count=%s raw_count=%s",
            len(final_targets),
            len(targets),
        )
        return final_targets

    async def _from_intent_script_runtime(self, hass: Any) -> list[ConversationTarget]:
        """Best-effort runtime discovery for intent_script."""
        targets: list[ConversationTarget] = []
        intent_domain = hass.data.get("intent_script")

        _LOGGER.warning(
            "Intent script runtime discovery: domain_present=%s domain_type=%s",
            intent_domain is not None,
            type(intent_domain).__name__ if intent_domain is not None else "None",
        )
        if intent_domain is None:
            return targets

        scripts: dict[str, Any] = {}

        if isinstance(intent_domain, dict):
            for key in ("intent_scripts", "intents", "scripts"):
                candidate = intent_domain.get(key)
                if isinstance(candidate, dict):
                    scripts = candidate
                    _LOGGER.warning(
                        "Intent script runtime discovery: using dict key=%s count=%s",
                        key,
                        len(candidate),
                    )
                    break
            if not scripts:
                # Some versions may already be keyed by intent name.
                dict_like_values = any(isinstance(v, (dict, str)) for v in intent_domain.values())
                if dict_like_values:
                    scripts = intent_domain
                    _LOGGER.warning(
                        "Intent script runtime discovery: using top-level mapping count=%s",
                        len(scripts),
                    )
        else:
            _LOGGER.warning(
                "Intent script runtime discovery: unsupported runtime type=%s",
                type(intent_domain).__name__,
            )

        for intent_name, script in scripts.items():
            built = self._build_intent_script_target(intent_name, script, source="intent_script_runtime")
            if built is not None:
                targets.append(built)

        _LOGGER.warning(
            "Intent script runtime discovery complete: built_count=%s",
            len(targets),
        )
        return targets

    async def _from_intent_script_yaml(self, hass: Any) -> list[ConversationTarget]:
        """File-based discovery for intent_script.yaml."""
        targets: list[ConversationTarget] = []
        config_path = Path(hass.config.path("intent_script.yaml"))
        data = _read_yaml_file(config_path)

        _LOGGER.warning(
            "Intent script YAML discovery: path=%s exists=%s loaded=%s",
            config_path,
            config_path.exists(),
            data is not None,
        )
        if not isinstance(data, dict):
            return targets

        for intent_name, script in data.items():
            built = self._build_intent_script_target(intent_name, script, source="intent_script_yaml")
            if built is not None:
                targets.append(built)

        _LOGGER.warning(
            "Intent script YAML discovery complete: built_count=%s",
            len(targets),
        )
        return targets

    def _build_intent_script_target(
        self,
        intent_name: str,
        script: Any,
        *,
        source: str,
    ) -> ConversationTarget | None:
        """Create a conversation target from an intent_script definition."""
        if not isinstance(intent_name, str) or not intent_name.strip():
            return None

        script_dict = script if isinstance(script, dict) else {}
        friendly_name = str(script_dict.get("name") or intent_name).strip()
        sample_phrases = [
            text
            for text in {
                intent_name.replace("_", " ").strip(),
                friendly_name.strip(),
            }
            if text
        ]
        slots = list((script_dict.get("slot_schema") or {}).keys()) if isinstance(script_dict, dict) else []
        token_values = _tokenize_all(sample_phrases + [friendly_name])

        return ConversationTarget(
            target_id=f"{source}:{intent_name}",
            type="intent_script",
            display_name=friendly_name,
            normalized_name=normalize_text(friendly_name),
            sample_phrases=sample_phrases,
            canonical_phrase=intent_name.replace("_", " ").strip().lower(),
            source=source,
            slots=slots,
            tokens=token_values,
            phonetic_tokens=phonetic_tokens(token_values),
        )

    async def _from_automation_runtime(self, hass: Any) -> list[ConversationTarget]:
        """Very conservative runtime discovery.

        Runtime automation entity states usually do not expose trigger definitions,
        but we still log what is available so failures are visible.
        """
        targets: list[ConversationTarget] = []
        automation_states = list(hass.states.async_all("automation"))
        attr_trigger_count = 0

        for state in automation_states:
            triggers = state.attributes.get("triggers")
            if isinstance(triggers, list):
                attr_trigger_count += len(triggers)
                for idx, trigger in enumerate(triggers):
                    built = self._build_automation_target_from_trigger(
                        trigger=trigger,
                        target_id=f"automation_runtime:{state.entity_id}:{idx}",
                        display_name=state.name or state.entity_id,
                        source="automation_runtime",
                    )
                    if built is not None:
                        targets.append(built)

        _LOGGER.warning(
            "Automation runtime discovery complete: automation_state_count=%s trigger_attr_count=%s built_count=%s",
            len(automation_states),
            attr_trigger_count,
            len(targets),
        )
        return targets

    async def _from_automation_yaml(self, hass: Any) -> list[ConversationTarget]:
        """YAML-based discovery for automations, including blueprint-backed instances."""
        targets: list[ConversationTarget] = []
        automations_path = Path(hass.config.path("automations.yaml"))
        data = _read_yaml_file(automations_path)

        _LOGGER.warning(
            "Automation YAML discovery: path=%s exists=%s loaded=%s",
            automations_path,
            automations_path.exists(),
            data is not None,
        )
        if not isinstance(data, list):
            return targets

        direct_count = 0
        blueprint_count = 0

        for idx, automation in enumerate(data):
            if not isinstance(automation, dict):
                continue

            display_name = str(
                automation.get("alias")
                or automation.get("id")
                or f"automation_{idx}"
            ).strip()

            use_blueprint = automation.get("use_blueprint")
            if isinstance(use_blueprint, dict):
                blueprint_targets = self._build_targets_from_blueprint_reference(
                    hass=hass,
                    use_blueprint=use_blueprint,
                    display_name=display_name,
                    automation_index=idx,
                )
                blueprint_count += len(blueprint_targets)
                targets.extend(blueprint_targets)
                continue

            triggers = automation.get("trigger") or automation.get("triggers") or []
            for trigger_index, trigger in enumerate(_safe_list(triggers)):
                built = self._build_automation_target_from_trigger(
                    trigger=trigger,
                    target_id=f"automation_yaml:{idx}:{trigger_index}",
                    display_name=display_name,
                    source="automation_yaml",
                )
                if built is not None:
                    direct_count += 1
                    targets.append(built)

        _LOGGER.warning(
            "Automation YAML discovery complete: automation_count=%s direct_built=%s blueprint_built=%s total_built=%s",
            len(data),
            direct_count,
            blueprint_count,
            len(targets),
        )
        return targets

    def _build_targets_from_blueprint_reference(
        self,
        *,
        hass: Any,
        use_blueprint: dict[str, Any],
        display_name: str,
        automation_index: int,
    ) -> list[ConversationTarget]:
        """Build targets from a blueprint file referenced by an automation."""
        blueprint_path = use_blueprint.get("path")
        if not isinstance(blueprint_path, str) or not blueprint_path.strip():
            _LOGGER.warning(
                "Blueprint automation discovery skipped: automation=%s reason=missing_path",
                display_name,
            )
            return []

        resolved_path = Path(hass.config.path("blueprints", "automation", blueprint_path))
        blueprint_data = _read_yaml_file(resolved_path)

        _LOGGER.warning(
            "Blueprint automation discovery: automation=%s path=%s exists=%s loaded=%s",
            display_name,
            resolved_path,
            resolved_path.exists(),
            blueprint_data is not None,
        )
        if not isinstance(blueprint_data, dict):
            return []

        triggers = blueprint_data.get("trigger") or blueprint_data.get("triggers") or []
        built_targets: list[ConversationTarget] = []
        for trigger_index, trigger in enumerate(_safe_list(triggers)):
            built = self._build_automation_target_from_trigger(
                trigger=trigger,
                target_id=f"automation_blueprint:{automation_index}:{trigger_index}",
                display_name=display_name,
                source=f"automation_blueprint:{blueprint_path}",
            )
            if built is not None:
                built_targets.append(built)

        _LOGGER.warning(
            "Blueprint automation discovery complete: automation=%s built_count=%s",
            display_name,
            len(built_targets),
        )
        return built_targets

    def _build_automation_target_from_trigger(
        self,
        *,
        trigger: Any,
        target_id: str,
        display_name: str,
        source: str,
    ) -> ConversationTarget | None:
        """Create a conversation target from a single trigger definition."""
        if not isinstance(trigger, dict):
            return None

        platform_value = trigger.get("platform") or trigger.get("trigger")
        if platform_value not in {"conversation", "sentence"}:
            return None

        commands = _extract_commands(trigger.get("command") or trigger.get("sentence"))
        if not commands:
            return None

        trigger_id = str(trigger.get("id") or target_id).strip()
        slots = []
        if isinstance(trigger.get("slots"), dict):
            slots = list(trigger["slots"].keys())

        token_values = _tokenize_all([display_name, *commands, trigger_id])

        return ConversationTarget(
            target_id=f"{source}:{trigger_id}",
            type="automation_sentence",
            display_name=display_name,
            normalized_name=normalize_text(display_name),
            sample_phrases=commands,
            canonical_phrase=commands[0],
            source=source,
            slots=slots,
            tokens=token_values,
            phonetic_tokens=phonetic_tokens(token_values),
        )


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

        _LOGGER.warning(
            "Manual conversation target merge complete: built_count=%s",
            len(output),
        )
        return output