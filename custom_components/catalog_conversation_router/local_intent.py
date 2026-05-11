"""Local intent resolver backed by Hassil sentence matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .matcher import CANONICAL_ACTION_TEXT
from .models import Catalog, ConversationTarget, EntityTarget, LLMTranslationResult
from .phonetics import normalize_text, tokenize
from .phrase_renderer import render_conversation_pattern

SLOT_RE = re.compile(r"\{([^}]+)\}")

DAY_TOKENS = {
    "today",
    "tomorrow",
    "tonight",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}
TIME_TOKENS = {
    "minute",
    "minutes",
    "hour",
    "hours",
    "second",
    "seconds",
    "am",
    "pm",
}
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
}
ENTITY_SKIP_WORDS = ["please", "can you", "would you"]
PHRASE_SKIP_WORDS = ["please"]

ENTITY_ACTION_RULES: dict[str, dict[str, Any]] = {
    "turn_on": {
        "rule": "(turn on | switch on | activate | start | enable)",
        "tool_groups": {"lighting", "fan", "climate", "media"},
    },
    "turn_off": {
        "rule": "(turn off | switch off | deactivate | stop | disable | kill)",
        "tool_groups": {"lighting", "fan", "climate", "media"},
    },
    "open": {
        "rule": "(open)",
        "tool_groups": {"covers"},
    },
    "close": {
        "rule": "(close | shut)",
        "tool_groups": {"covers"},
    },
    "lock": {
        "rule": "(lock)",
        "tool_groups": {"locks"},
    },
    "unlock": {
        "rule": "(unlock)",
        "tool_groups": {"locks"},
    },
    "query": {
        "rule": "(what is | whats | what are | status of | status for | tell me | is)",
        "tool_groups": {"lighting", "fan", "climate", "media", "covers", "locks", "timers", "lists"},
    },
}
DOMAIN_SLOT_DEFS: dict[str, dict[str, Any]] = {
    "lighting": {
        "slot": "lighting_domain",
        "values": [
            {"in": "light", "out": "lighting"},
            {"in": "lights", "out": "lighting"},
            {"in": "lamp", "out": "lighting"},
            {"in": "lamps", "out": "lighting"},
        ],
    },
    "fan": {
        "slot": "fan_domain",
        "values": [
            {"in": "fan", "out": "fan"},
            {"in": "fans", "out": "fan"},
        ],
    },
    "climate": {
        "slot": "climate_domain",
        "values": [
            {"in": "thermostat", "out": "climate"},
            {"in": "temperature", "out": "climate"},
            {"in": "heater", "out": "climate"},
            {"in": "climate", "out": "climate"},
            {"in": "ac", "out": "climate"},
        ],
    },
    "locks": {
        "slot": "locks_domain",
        "values": [
            {"in": "lock", "out": "locks"},
            {"in": "locks", "out": "locks"},
            {"in": "door lock", "out": "locks"},
        ],
    },
    "covers": {
        "slot": "covers_domain",
        "values": [
            {"in": "garage door", "out": "covers"},
            {"in": "door", "out": "covers"},
            {"in": "cover", "out": "covers"},
            {"in": "covers", "out": "covers"},
            {"in": "shade", "out": "covers"},
            {"in": "shades", "out": "covers"},
            {"in": "blind", "out": "covers"},
            {"in": "blinds", "out": "covers"},
        ],
    },
    "media": {
        "slot": "media_domain",
        "values": [
            {"in": "speaker", "out": "media"},
            {"in": "speakers", "out": "media"},
            {"in": "tv", "out": "media"},
            {"in": "television", "out": "media"},
            {"in": "receiver", "out": "media"},
            {"in": "media", "out": "media"},
        ],
    },
    "timers": {
        "slot": "timers_domain",
        "values": [
            {"in": "timer", "out": "timers"},
            {"in": "timers", "out": "timers"},
            {"in": "alarm", "out": "timers"},
            {"in": "alarms", "out": "timers"},
            {"in": "reminder", "out": "timers"},
            {"in": "reminders", "out": "timers"},
        ],
    },
    "lists": {
        "slot": "lists_domain",
        "values": [
            {"in": "shopping list", "out": "lists"},
            {"in": "list", "out": "lists"},
            {"in": "lists", "out": "lists"},
        ],
    },
}
LIGHTING_LOCATION_RULES: dict[str, str] = {
    "turn_on": "(turn on | switch on | illuminate)",
    "turn_off": "(turn off | switch off | darken)",
}


@dataclass(slots=True)
class BuilderTarget:
    """Entity-builder target record."""

    name: str
    normalized_name: str
    tool_group: str
    actions: set[str]
    area: str | None
    super_area: str | None
    domain: str | None
    synthetic: bool


class LocalIntentResolver:
    """Resolve local translations without using the LLM translation layer."""

    def resolve(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        origin_area: str | None = None,
        origin_super_area: str | None = None,
    ) -> LLMTranslationResult:
        if self._import_hassil() is None:
            return LLMTranslationResult(
                mode="general",
                canonical_text=None,
                tool_group=None,
                confidence=0.0,
                notes="hassil_unavailable",
                valid=False,
                source=None,
                intent_family=None,
                confidence_reason="missing_hassil_dependency",
                debug={"match_engine": "hassil"},
                raw_text=None,
            )

        phrase_result = self._match_phrase_intent(utterance=utterance, catalog=catalog)
        if phrase_result is not None:
            return phrase_result

        entity_result = self._build_entity_command(
            utterance=utterance,
            catalog=catalog,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )
        if entity_result is not None:
            return entity_result

        return LLMTranslationResult(
            mode="general",
            canonical_text=None,
            tool_group=None,
            confidence=0.0,
            notes="no_local_intent_match",
            valid=False,
            source=None,
            intent_family=None,
            confidence_reason="no_fully_bound_result",
            debug={"match_engine": "hassil"},
            raw_text=None,
        )

    def _import_hassil(self) -> Any | None:
        try:
            import hassil
        except ImportError:
            return None
        return hassil

    def _match_phrase_intent(
        self,
        *,
        utterance: str,
        catalog: Catalog,
    ) -> LLMTranslationResult | None:
        hassil = self._import_hassil()
        if hassil is None:
            return None

        intents = self._build_phrase_intents(catalog)
        result = hassil.recognize_best(
            utterance,
            intents,
            allow_unmatched_entities=False,
            language=catalog.metadata.language or "en",
        )
        if result is None:
            return None

        metadata = result.intent_metadata or {}
        pattern = str(metadata.get("pattern", ""))
        canonical_pattern = str(metadata.get("canonical_phrase", pattern))
        target_id = str(metadata.get("target_id", result.intent.name))
        tool_group = str(metadata.get("tool_group", "general"))
        if not pattern or not canonical_pattern:
            return None

        slots = {
            name: entity.text_clean
            for name, entity in result.entities.items()
            if entity.text_clean
        }
        if not self._phrase_slots_are_valid(slots):
            return None

        rendered = render_conversation_pattern(utterance, canonical_pattern)
        if not rendered.text:
            return None

        return LLMTranslationResult(
            mode="translate",
            canonical_text=rendered.text,
            tool_group=tool_group,
            confidence=0.9,
            notes="phrase_match",
            valid=True,
            source="phrase_matcher",
            intent_family=target_id,
            confidence_reason="fully_bound_phrase_template",
            debug={
                "match_engine": "hassil",
                "pattern": pattern,
                "canonical_pattern": canonical_pattern,
                "target_id": target_id,
                "slots": slots,
            },
            raw_text=None,
        )

    def _build_phrase_intents(self, catalog: Catalog):
        hassil = self._import_hassil()
        assert hassil is not None

        slot_lists: dict[str, dict[str, Any]] = {}
        intents_data: dict[str, Any] = {"language": catalog.metadata.language or "en", "skip_words": PHRASE_SKIP_WORDS, "lists": slot_lists, "intents": {}}

        for target in catalog.conversation_targets:
            patterns = self._conversation_target_patterns(target)
            if not patterns:
                continue

            intent_name = self._safe_intent_name(f"phrase_{target.target_id}")
            data_entries: list[dict[str, Any]] = []
            for pattern in patterns:
                for slot_name in SLOT_RE.findall(pattern):
                    self._ensure_phrase_slot_list(slot_lists, normalize_text(slot_name))
                data_entries.append(
                    {
                        "sentences": [pattern],
                        "metadata": {
                            "target_id": target.target_id,
                            "canonical_phrase": target.canonical_phrase,
                            "pattern": pattern,
                            "tool_group": self._infer_tool_group_from_phrase(pattern),
                        },
                    }
                )
            if data_entries:
                intents_data["intents"][intent_name] = {"data": data_entries}

        return hassil.Intents.from_dict(intents_data)

    def _ensure_phrase_slot_list(self, slot_lists: dict[str, dict[str, Any]], slot_name: str) -> None:
        if slot_name in slot_lists:
            return
        if slot_name == "day":
            slot_lists[slot_name] = {
                "values": [{"in": day, "out": day} for day in sorted(DAY_TOKENS)]
            }
            return
        slot_lists[slot_name] = {"wildcard": True}

    def _conversation_target_patterns(self, target: ConversationTarget) -> list[str]:
        patterns: list[str] = []
        if self._is_user_sayable_phrase(target.canonical_phrase):
            patterns.append(target.canonical_phrase)
        for phrase in target.sample_phrases:
            if self._is_user_sayable_phrase(phrase):
                patterns.append(phrase)
        return patterns

    def _is_user_sayable_phrase(self, phrase: str | None) -> bool:
        if not phrase:
            return False
        lowered = normalize_text(phrase)
        if not lowered:
            return False
        return not any(
            token in lowered
            for token in ("command prompt", "trigger phrases", "computednametype")
        )

    def _phrase_slots_are_valid(self, slots: dict[str, str]) -> bool:
        for slot_name, slot_value in slots.items():
            normalized_value = normalize_text(slot_value)
            tokens = set(tokenize(normalized_value, drop_stop_words=False))
            if not tokens:
                return False
            if slot_name == "day" and not (tokens & DAY_TOKENS):
                return False
            if slot_name == "when":
                if not (
                    tokens & (DAY_TOKENS | TIME_TOKENS | NUMBER_WORDS)
                    or any(token.isdigit() for token in tokens)
                ):
                    return False
            if slot_name == "amount":
                if not (tokens & NUMBER_WORDS or any(token.isdigit() for token in tokens)):
                    return False
        return True

    def _build_entity_command(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> LLMTranslationResult | None:
        hassil = self._import_hassil()
        if hassil is None:
            return None

        targets = self._build_targets(catalog)
        intents = self._build_entity_intents(catalog, targets)
        result = hassil.recognize_best(
            utterance,
            intents,
            allow_unmatched_entities=False,
            language=catalog.metadata.language or "en",
        )
        if result is None:
            return None

        action = self._result_slot_value(result, "action")
        if not action:
            return None
        tool_group = self._result_slot_value(result, "tool_group")
        name_entity = result.entities.get("name")
        location_entity = result.entities.get("location")
        domain_entity = self._domain_entity_for_result(result)

        if name_entity is not None:
            target_name = self._normalize_display_name(str(name_entity.value))
            matched = next(
                (target for target in targets if target.normalized_name == target_name),
                None,
            )
            if matched is None or action not in matched.actions:
                return None
        else:
            matched = self._resolve_domain_target(
                targets=targets,
                action=action,
                tool_group=tool_group,
                domain_entity=domain_entity,
                location_entity=location_entity,
                origin_area=origin_area,
                origin_super_area=origin_super_area,
            )
            if matched is None:
                return None

        canonical_text = self._canonical_text_from_action_target(action, matched.name)
        return LLMTranslationResult(
            mode="translate",
            canonical_text=canonical_text,
            tool_group=matched.tool_group,
            confidence=0.88 if not matched.synthetic else 0.84,
            notes="entity_builder_match",
            valid=True,
            source="entity_builder",
            intent_family="entity_control" if action != "query" else "entity_query",
            confidence_reason="fully_resolved_entity_command",
            debug={
                "match_engine": "hassil",
                "target": matched.name,
                "action": action,
                "tool_group": matched.tool_group,
                "area": matched.area,
                "super_area": matched.super_area,
                "synthetic": matched.synthetic,
            },
            raw_text=None,
        )

    def _build_entity_intents(self, catalog: Catalog, targets: list[BuilderTarget]):
        hassil = self._import_hassil()
        assert hassil is not None

        lists: dict[str, Any] = {
            "name": {"values": self._name_slot_values(targets)},
            "location": {"values": self._location_slot_values(targets)},
        }
        for tool_group, spec in DOMAIN_SLOT_DEFS.items():
            if any(target.tool_group == tool_group for target in targets):
                lists[spec["slot"]] = {"values": spec["values"]}

        intents_data: dict[str, Any] = {
            "language": catalog.metadata.language or "en",
            "skip_words": ENTITY_SKIP_WORDS,
            "expansion_rules": self._entity_expansion_rules(),
            "lists": lists,
            "intents": {"LocalEntityIntent": {"data": self._entity_sentence_data(targets)}},
        }
        return hassil.Intents.from_dict(intents_data)

    def _entity_expansion_rules(self) -> dict[str, str]:
        rules = {
            f"{action}_verb": spec["rule"]
            for action, spec in ENTITY_ACTION_RULES.items()
        }
        rules["lighting_location_on"] = LIGHTING_LOCATION_RULES["turn_on"]
        rules["lighting_location_off"] = LIGHTING_LOCATION_RULES["turn_off"]
        return rules

    def _entity_sentence_data(self, targets: list[BuilderTarget]) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []

        for action, spec in ENTITY_ACTION_RULES.items():
            tool_groups = spec["tool_groups"]
            data.append(
                {
                    "sentences": [f"<{action}_verb> [all] [the] {{name}}"],
                    "slots": {"action": action, "strategy": "name"},
                }
            )

            for tool_group in sorted(tool_groups):
                domain_spec = DOMAIN_SLOT_DEFS.get(tool_group)
                if domain_spec is None:
                    continue
                if not any(target.tool_group == tool_group for target in targets):
                    continue
                if not any(action in target.actions for target in targets if target.tool_group == tool_group):
                    continue

                domain_slot = domain_spec["slot"]
                data.extend(
                    [
                        {
                            "sentences": [
                                f"<{action}_verb> [all] [the] {{{domain_slot}}} in [the] {{location}}",
                                f"<{action}_verb> [all] [the] {{location}} {{{domain_slot}}}",
                            ],
                            "slots": {
                                "action": action,
                                "tool_group": tool_group,
                                "strategy": "domain_location",
                            },
                        },
                        {
                            "sentences": [f"<{action}_verb> [all] [the] {{{domain_slot}}}"],
                            "slots": {
                                "action": action,
                                "tool_group": tool_group,
                                "strategy": "domain_origin",
                            },
                        },
                    ]
                )

        if any(target.tool_group == "lighting" for target in targets):
            data.extend(
                [
                    {
                        "sentences": ["<lighting_location_on> [the] {location}"],
                        "slots": {
                            "action": "turn_on",
                            "tool_group": "lighting",
                            "strategy": "domain_location",
                        },
                    },
                    {
                        "sentences": ["<lighting_location_off> [the] {location}"],
                        "slots": {
                            "action": "turn_off",
                            "tool_group": "lighting",
                            "strategy": "domain_location",
                        },
                    },
                ]
            )

        return data

    def _name_slot_values(self, targets: list[BuilderTarget]) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for target in targets:
            values.append(
                {
                    "in": target.name,
                    "out": target.name,
                    "metadata": {
                        "tool_group": target.tool_group,
                        "area": target.area,
                        "super_area": target.super_area,
                        "synthetic": target.synthetic,
                    },
                }
            )
        return values

    def _location_slot_values(self, targets: list[BuilderTarget]) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for target in targets:
            if target.area:
                key = ("area", target.area)
                if key not in seen:
                    seen.add(key)
                    values.append(
                        {
                            "in": target.area,
                            "out": target.area,
                            "metadata": {"location_type": "area"},
                        }
                    )
            if target.super_area:
                key = ("super_area", target.super_area)
                if key not in seen:
                    seen.add(key)
                    values.append(
                        {
                            "in": target.super_area,
                            "out": target.super_area,
                            "metadata": {"location_type": "super_area"},
                        }
                    )
        return values

    def _build_targets(self, catalog: Catalog) -> list[BuilderTarget]:
        targets: dict[str, BuilderTarget] = {}

        def add_target(
            *,
            name: str,
            tool_group: str,
            actions: set[str],
            area: str | None = None,
            super_area: str | None = None,
            domain: str | None = None,
            synthetic: bool = False,
        ) -> None:
            normalized_name = self._normalize_display_name(name)
            if not normalized_name:
                return
            existing = targets.get(normalized_name)
            if existing is not None:
                return
            targets[normalized_name] = BuilderTarget(
                name=normalized_name,
                normalized_name=normalized_name,
                tool_group=tool_group,
                actions=actions,
                area=self._normalize_display_name(area) or None,
                super_area=self._normalize_display_name(super_area) or None,
                domain=domain,
                synthetic=synthetic,
            )

        add_target(name="timers", tool_group="timers", actions={"list", "query", "cancel", "start"})
        add_target(name="alarms", tool_group="timers", actions={"list", "query", "cancel", "start"})
        add_target(name="reminders", tool_group="timers", actions={"list", "query", "cancel", "start"})

        seen_area_groups: set[tuple[str, str]] = set()
        for entity in catalog.entity_targets:
            tool_group = self._infer_tool_group_from_domain(entity.domain)
            actions = self._actions_for_entity(entity)
            if not actions:
                continue
            area = self._normalize_display_name(entity.area)
            super_area = self._normalize_display_name(entity.super_area)
            add_target(
                name=entity.name,
                tool_group=tool_group,
                actions=actions,
                area=area,
                super_area=super_area,
                domain=entity.domain,
            )
            if area and tool_group in {"lighting", "fan", "climate"}:
                area_key = (tool_group, area)
                if area_key in seen_area_groups:
                    continue
                seen_area_groups.add(area_key)
                if tool_group == "lighting":
                    add_target(
                        name=f"{area} lights",
                        tool_group="lighting",
                        actions={"turn_on", "turn_off", "set", "query"},
                        area=area,
                        super_area=super_area,
                        domain="light",
                        synthetic=True,
                    )
                elif tool_group == "fan":
                    add_target(
                        name=f"{area} fans",
                        tool_group="fan",
                        actions={"turn_on", "turn_off", "set", "query"},
                        area=area,
                        super_area=super_area,
                        domain="fan",
                        synthetic=True,
                    )
                elif tool_group == "climate":
                    add_target(
                        name=f"{area} temperature",
                        tool_group="climate",
                        actions={"set", "query"},
                        area=area,
                        super_area=super_area,
                        domain="climate",
                        synthetic=True,
                    )

        return list(targets.values())

    def _resolve_domain_target(
        self,
        *,
        targets: list[BuilderTarget],
        action: str,
        tool_group: str | None,
        domain_entity: Any | None,
        location_entity: Any | None,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> BuilderTarget | None:
        if not tool_group:
            return None

        area_hint: str | None = None
        super_area_hint: str | None = None
        if location_entity is not None:
            location_value = self._normalize_display_name(str(location_entity.value))
            location_type = (location_entity.metadata or {}).get("location_type")
            if location_type == "area":
                area_hint = location_value
            elif location_type == "super_area":
                super_area_hint = location_value
        else:
            area_hint = self._normalize_display_name(origin_area) or None
            super_area_hint = self._normalize_display_name(origin_super_area) or None

        candidates = [
            target
            for target in targets
            if action in target.actions and target.tool_group == tool_group
        ]
        candidates = self._filter_by_location(
            candidates,
            area_hint=area_hint,
            super_area_hint=super_area_hint,
        )
        singular_preferred = self._is_singular_domain_request(domain_entity)
        if singular_preferred:
            single_entity = self._prefer_single_entity_candidate(
                candidates=candidates,
                tool_group=tool_group,
                area_hint=area_hint,
            )
            if single_entity is not None:
                return single_entity
            return None
        exact_area_group = [
            target for target in candidates if target.synthetic and target.area == area_hint
        ]
        if len(exact_area_group) == 1:
            return exact_area_group[0]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _filter_by_location(
        self,
        targets: list[BuilderTarget],
        *,
        area_hint: str | None,
        super_area_hint: str | None,
    ) -> list[BuilderTarget]:
        if area_hint:
            exact = [target for target in targets if target.area == area_hint]
            if exact:
                return exact
        if super_area_hint:
            exact = [target for target in targets if target.super_area == super_area_hint]
            if exact:
                return exact
        return targets

    def _result_slot_value(self, result: Any, slot_name: str) -> str | None:
        entity = result.entities.get(slot_name)
        if entity is None:
            return None
        value = str(entity.value).strip().lower()
        return value or None

    def _domain_entity_for_result(self, result: Any) -> Any | None:
        for spec in DOMAIN_SLOT_DEFS.values():
            entity = result.entities.get(spec["slot"])
            if entity is not None:
                return entity
        return None

    def _is_singular_domain_request(self, domain_entity: Any | None) -> bool:
        if domain_entity is None:
            return False
        raw = normalize_text(getattr(domain_entity, "text_clean", "") or getattr(domain_entity, "text", ""))
        if not raw:
            return False
        return raw in {
            "light",
            "lamp",
            "fan",
            "thermostat",
            "temperature",
            "heater",
            "speaker",
            "tv",
            "receiver",
            "lock",
            "garage door",
            "door",
            "cover",
            "shade",
            "blind",
            "timer",
            "alarm",
            "reminder",
            "list",
        }

    def _prefer_single_entity_candidate(
        self,
        *,
        candidates: list[BuilderTarget],
        tool_group: str,
        area_hint: str | None,
    ) -> BuilderTarget | None:
        real_candidates = [target for target in candidates if not target.synthetic]
        if not real_candidates:
            return None

        if area_hint:
            expected_name = {
                "lighting": f"{area_hint} light",
                "fan": f"{area_hint} fan",
                "climate": f"{area_hint} thermostat",
                "media": f"{area_hint} speaker",
                "locks": f"{area_hint} lock",
                "covers": f"{area_hint} door",
            }.get(tool_group)
            if expected_name:
                exact = [
                    target for target in real_candidates if target.normalized_name == expected_name
                ]
                if len(exact) == 1:
                    return exact[0]

        if len(real_candidates) == 1:
            return real_candidates[0]
        return None

    def _safe_intent_name(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
        return cleaned.strip("_") or "intent"

    def _infer_tool_group_from_phrase(self, phrase: str) -> str:
        normalized = normalize_text(phrase)
        tokens = set(tokenize(normalized, drop_stop_words=False))
        if tokens & {"timer", "timers", "alarm", "alarms", "reminder", "reminders"}:
            return "timers"
        if tokens & {"light", "lights", "lamp", "lamps"}:
            return "lighting"
        if tokens & {"temperature", "thermostat", "climate", "heat", "cool", "ac"}:
            return "climate"
        if tokens & {"weather", "forecast", "outside"}:
            return "general"
        if tokens & {"speaker", "tv", "receiver", "media"}:
            return "media"
        return "general"

    def _infer_tool_group_from_domain(self, domain: str | None) -> str:
        domain_key = (domain or "").strip().lower()
        return {
            "light": "lighting",
            "switch": "lighting",
            "fan": "fan",
            "climate": "climate",
            "media_player": "media",
            "lock": "locks",
            "cover": "covers",
            "timer": "timers",
            "todo": "lists",
        }.get(domain_key, "mixed")

    def _actions_for_entity(self, entity: EntityTarget) -> set[str]:
        actions: set[str] = set()
        for capability in entity.capabilities:
            action = {
                "turn_on": "turn_on",
                "turn_off": "turn_off",
                "activate": "turn_on",
                "open": "open",
                "close": "close",
                "lock": "lock",
                "unlock": "unlock",
                "query": "query",
                "set": "set",
            }.get(capability)
            if action:
                actions.add(action)
        return actions

    def _canonical_text_from_action_target(self, action: str, target: str) -> str:
        target = self._normalize_display_name(target)
        action_text = CANONICAL_ACTION_TEXT.get(action)
        if action == "query":
            if target in {"timers", "alarms", "reminders"}:
                return f"what {target} do i have"
            return f"what is {target}"
        if action_text:
            return f"{action_text} {target}"
        return target

    def _normalize_display_name(self, value: str | None) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", " ", normalize_text(value)).strip()
