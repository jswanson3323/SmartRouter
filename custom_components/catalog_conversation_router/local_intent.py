"""Local intent resolver backed by Hassil sentence matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .matcher import CANONICAL_ACTION_TEXT
from .models import Catalog, ConversationTarget, EntityTarget, LLMTranslationResult, ResolvedLocalCommand
from .phonetics import normalize_text, tokenize
from .phrase_renderer import render_conversation_pattern
from .semantic_intent import (
    ENTITY_AMBIGUITY_GAP,
    ENTITY_DIRECT_THRESHOLD,
    ENTITY_FAMILY_AMBIGUITY_GAP,
    PHRASE_DIRECT_THRESHOLD,
    PHRASE_FILTER_THRESHOLD,
    SemanticEntityCandidate,
    SemanticIntentRanker,
    SemanticPhraseCandidate,
    SemanticRequestClassification,
    infer_command_intent_family,
)

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
DEVICE_CONTROL_TOOL_GROUPS = {"lighting", "fan", "climate", "media", "locks", "covers"}
ENTITY_ACTION_RULES: dict[str, dict[str, Any]] = {
    "turn_on": {
        "rule": "(turn on | switch on)",
        "tool_groups": {"lighting", "fan", "climate", "media"},
    },
    "turn_off": {
        "rule": "(turn off | switch off)",
        "tool_groups": {"lighting", "fan", "climate", "media"},
    },
    "pause": {
        "rule": "(pause)",
        "tool_groups": {"media"},
    },
    "play": {
        "rule": "(play | resume)",
        "tool_groups": {"media"},
    },
    "stop": {
        "rule": "(stop)",
        "tool_groups": {"media"},
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
        "rule": "(what is | whats | what are | status of | status for | tell me)",
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
    "turn_on": "(turn on | switch on)",
    "turn_off": "(turn off | switch off)",
}
COMPOUND_ACTION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("switch off", "turn_off"),
    ("turn off", "turn_off"),
    ("switch on", "turn_on"),
    ("turn on", "turn_on"),
    ("unlock", "unlock"),
    ("close", "close"),
    ("open", "open"),
    ("lock", "lock"),
)


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

    def __init__(self, semantic_ranker: SemanticIntentRanker | None = None) -> None:
        self._semantic_ranker = semantic_ranker or SemanticIntentRanker()

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
                debug={
                    "match_engine": "hassil",
                    "semantic_available": self._semantic_ranker.available(),
                    "semantic_error": self._semantic_ranker.unavailable_reason(),
                },
                raw_text=None,
            )

        phrase_result = self._match_phrase_intent(utterance=utterance, catalog=catalog)
        if phrase_result is not None:
            return phrase_result

        compound_result = self._match_compound_entity_intent(
            utterance=utterance,
            catalog=catalog,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )
        if compound_result is not None:
            return compound_result

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
            debug={
                "match_engine": "hassil",
                "semantic_available": self._semantic_ranker.available(),
                "semantic_error": self._semantic_ranker.unavailable_reason(),
                **self._semantic_debug_snapshot(
                    utterance=utterance,
                    catalog=catalog,
                    origin_area=origin_area,
                    origin_super_area=origin_super_area,
                ),
            },
            raw_text=None,
        )

    def classify_request(
        self,
        *,
        utterance: str,
        catalog: Catalog,
    ) -> SemanticRequestClassification | None:
        """Classify whether a request is tool-oriented or open-domain."""
        if not self._semantic_ranker.available():
            return None
        return self._semantic_ranker.classify_request(
            utterance=utterance,
            catalog=catalog,
            command_docs=self._semantic_entity_command_docs(self._build_targets(catalog)),
            infer_tool_group=self._infer_tool_group_from_phrase,
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

        semantic_candidates = self._semantic_ranker.rank_phrase_candidates(
            utterance=utterance,
            catalog=catalog,
            infer_tool_group=self._infer_tool_group_from_phrase,
        )
        allowed_target_ids = [
            candidate.target_id
            for candidate in semantic_candidates
            if candidate.score >= PHRASE_FILTER_THRESHOLD
        ]

        intents = self._build_phrase_intents(
            catalog,
            allowed_target_ids=set(allowed_target_ids) if allowed_target_ids else None,
        )
        result = hassil.recognize_best(
            utterance,
            intents,
            allow_unmatched_entities=False,
            language=catalog.metadata.language or "en",
        )
        if result is None:
            return self._semantic_direct_phrase_match(semantic_candidates)

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
                "semantic_candidates": [
                    {
                        "target_id": candidate.target_id,
                        "score": round(candidate.score, 4),
                        "example_text": candidate.example_text,
                    }
                    for candidate in semantic_candidates[:5]
                ],
            },
            raw_text=None,
        )

    def _build_phrase_intents(self, catalog: Catalog, allowed_target_ids: set[str] | None = None):
        hassil = self._import_hassil()
        assert hassil is not None

        slot_lists: dict[str, dict[str, Any]] = {}
        intents_data: dict[str, Any] = {"language": catalog.metadata.language or "en", "skip_words": PHRASE_SKIP_WORDS, "lists": slot_lists, "intents": {}}

        for target in catalog.conversation_targets:
            if allowed_target_ids is not None and target.target_id not in allowed_target_ids:
                continue
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

    def _semantic_direct_phrase_match(
        self,
        semantic_candidates: list[SemanticPhraseCandidate],
    ) -> LLMTranslationResult | None:
        if not semantic_candidates:
            return None
        best = semantic_candidates[0]
        if (
            best.score < PHRASE_DIRECT_THRESHOLD
            or not best.concrete
            or best.has_slots
            or best.tool_group in DEVICE_CONTROL_TOOL_GROUPS
        ):
            return None
        return LLMTranslationResult(
            mode="translate",
            canonical_text=normalize_text(best.raw_text),
            tool_group=best.tool_group,
            confidence=min(0.97, best.score),
            notes="semantic_phrase_match",
            valid=True,
            source="semantic_phrase_matcher",
            intent_family=best.target_id,
            confidence_reason="semantic_phrase_family_match",
            debug={
                "match_engine": "fastembed+hassil",
                "semantic_candidates": [
                    {
                        "target_id": candidate.target_id,
                        "score": round(candidate.score, 4),
                        "example_text": candidate.example_text,
                        "raw_text": candidate.raw_text,
                        "concrete": candidate.concrete,
                    }
                    for candidate in semantic_candidates[:5]
                ],
            },
            raw_text=None,
        )

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

    def _match_compound_entity_intent(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> LLMTranslationResult | None:
        segments = self._parse_compound_segments(utterance)
        if segments is None or len(segments) < 2:
            return None

        targets = self._build_targets(catalog)
        resolved_commands: list[ResolvedLocalCommand] = []
        effective_area = origin_area
        effective_super_area = origin_super_area
        debug_segments: list[dict[str, Any]] = []

        for action, target_text, explicit_action in segments:
            resolved = self._resolve_compound_segment_target(
                targets=targets,
                action=action,
                target_text=target_text,
                origin_area=effective_area,
                origin_super_area=effective_super_area,
            )
            if resolved is None:
                segment_utterance = self._compound_segment_utterance(
                    action=action,
                    target_text=target_text,
                )
                segment_result = self._build_entity_command(
                    utterance=segment_utterance,
                    catalog=catalog,
                    origin_area=effective_area,
                    origin_super_area=effective_super_area,
                )
                if segment_result is None or not segment_result.valid or not segment_result.canonical_text:
                    return None

                segment_debug = dict(segment_result.debug or {})
                resolved = ResolvedLocalCommand(
                    canonical_text=segment_result.canonical_text,
                    action=str(segment_debug.get("action") or action),
                    target_name=str(segment_debug.get("target") or ""),
                    tool_group=str(segment_debug.get("tool_group") or segment_result.tool_group or ""),
                    area=str(segment_debug.get("area") or "") or None,
                    super_area=str(segment_debug.get("super_area") or "") or None,
                    source=segment_result.source,
                )

            segment_utterance = self._compound_segment_utterance(
                action=action,
                target_text=target_text,
            )
            resolved_commands.append(resolved)
            debug_segments.append(
                {
                    "input": segment_utterance,
                    "action": resolved.action,
                    "explicit_action": explicit_action,
                    "canonical_text": resolved.canonical_text,
                    "target": resolved.target_name,
                    "area": resolved.area,
                    "super_area": resolved.super_area,
                    "tool_group": resolved.tool_group,
                    "source": resolved.source,
                }
            )
            if resolved.area:
                effective_area = resolved.area
            if resolved.super_area:
                effective_super_area = resolved.super_area

        canonical_commands = [command.canonical_text for command in resolved_commands]
        return LLMTranslationResult(
            mode="translate",
            canonical_text=" and ".join(canonical_commands),
            tool_group="mixed" if len({command.tool_group for command in resolved_commands}) > 1 else resolved_commands[0].tool_group,
            confidence=0.9,
            notes="compound_entity_builder_match",
            valid=True,
            source="compound_entity_matcher",
            intent_family="compound_entity_control",
            confidence_reason="fully_resolved_compound_entity_command",
            debug={
                "match_engine": "compound+hassil",
                "segments": debug_segments,
            },
            raw_text=None,
            resolved_commands=resolved_commands,
        )

    def _parse_compound_segments(
        self,
        utterance: str,
    ) -> list[tuple[str, str, bool]] | None:
        normalized = normalize_text(utterance)
        if " and " not in normalized:
            return None

        chunks = [chunk.strip() for chunk in normalized.split(" and ") if chunk.strip()]
        if len(chunks) < 2:
            return None

        segments: list[tuple[str, str, bool]] = []
        current_action: str | None = None
        for index, chunk in enumerate(chunks):
            parsed = self._parse_compound_segment_chunk(chunk)
            if parsed is None:
                if current_action is None:
                    return None
                action = current_action
                target_text = self._strip_leading_articles(chunk)
                explicit_action = False
            else:
                action, target_text = parsed
                current_action = action
                explicit_action = True

            if action not in {"turn_on", "turn_off", "open", "close", "lock", "unlock"}:
                return None
            if not target_text:
                return None
            segments.append((action, target_text, explicit_action))

            if index == 0 and not explicit_action:
                return None

        if not any(not explicit_action for _, _, explicit_action in segments[1:]) and not any(
            explicit_action and index > 0 for index, (_, _, explicit_action) in enumerate(segments)
        ):
            return None
        return segments

    def _parse_compound_segment_chunk(self, chunk: str) -> tuple[str, str] | None:
        for prefix, action in COMPOUND_ACTION_PREFIXES:
            if chunk == prefix or chunk.startswith(f"{prefix} "):
                remainder = chunk.removeprefix(prefix).strip()
                return action, self._strip_leading_articles(remainder)
        return None

    def _compound_segment_utterance(
        self,
        *,
        action: str,
        target_text: str,
    ) -> str:
        action_text = CANONICAL_ACTION_TEXT.get(action, action)
        return f"{action_text} {self._strip_leading_articles(target_text)}".strip()

    def _strip_leading_articles(self, text: str) -> str:
        cleaned = normalize_text(text)
        tokens = cleaned.split()
        while tokens and tokens[0] in {"the", "a", "an"}:
            tokens.pop(0)
        return " ".join(tokens)

    def _resolve_compound_segment_target(
        self,
        *,
        targets: list[BuilderTarget],
        action: str,
        target_text: str,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> ResolvedLocalCommand | None:
        normalized_target = self._normalize_display_name(target_text)
        if not normalized_target:
            return None

        generic = self._resolve_compound_generic_domain_target(
            targets=targets,
            action=action,
            target_text=normalized_target,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )
        if generic is not None:
            return generic

        area_hint = self._normalize_display_name(origin_area) or None
        super_area_hint = self._normalize_display_name(origin_super_area) or None
        candidates = [
            target
            for target in targets
            if action in target.actions
            and (
                target.normalized_name == normalized_target
                or target.normalized_name.endswith(f" {normalized_target}")
            )
        ]
        candidates = self._filter_by_location(
            candidates,
            area_hint=area_hint,
            super_area_hint=super_area_hint,
        )
        if len(candidates) != 1:
            return None
        matched = candidates[0]
        return ResolvedLocalCommand(
            canonical_text=self._canonical_text_from_action_target(action, matched.name),
            action=action,
            target_name=matched.name,
            tool_group=matched.tool_group,
            area=matched.area,
            super_area=matched.super_area,
            source="compound_fast_matcher",
        )

    def _resolve_compound_generic_domain_target(
        self,
        *,
        targets: list[BuilderTarget],
        action: str,
        target_text: str,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> ResolvedLocalCommand | None:
        tool_group, singular = self._infer_compound_domain_target(target_text)
        if tool_group is None:
            return None

        class _CompoundDomainEntity:
            def __init__(self, text: str) -> None:
                self.value = text
                self.text_clean = text

        matched = self._resolve_domain_target(
            targets=targets,
            action=action,
            tool_group=tool_group,
            domain_entity=_CompoundDomainEntity(target_text),
            location_entity=None,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )
        if matched is None:
            return None
        if singular and matched.synthetic:
            return None
        return ResolvedLocalCommand(
            canonical_text=self._canonical_text_from_action_target(action, matched.name),
            action=action,
            target_name=matched.name,
            tool_group=matched.tool_group,
            area=matched.area,
            super_area=matched.super_area,
            source="compound_fast_matcher",
        )

    def _infer_compound_domain_target(self, target_text: str) -> tuple[str | None, bool]:
        normalized = self._normalize_display_name(target_text)
        mapping = {
            "light": ("lighting", True),
            "lights": ("lighting", False),
            "lamp": ("lighting", True),
            "lamps": ("lighting", False),
            "fan": ("fan", True),
            "fans": ("fan", False),
            "thermostat": ("climate", True),
            "temperature": ("climate", True),
            "heater": ("climate", True),
            "speaker": ("media", True),
            "speakers": ("media", False),
            "tv": ("media", True),
            "television": ("media", True),
            "receiver": ("media", True),
            "lock": ("locks", True),
            "locks": ("locks", False),
            "garage door": ("covers", True),
            "door": ("covers", True),
            "cover": ("covers", True),
            "covers": ("covers", False),
            "shade": ("covers", True),
            "shades": ("covers", False),
            "blind": ("covers", True),
            "blinds": ("covers", False),
        }
        return mapping.get(normalized, (None, False))

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
        semantic_command_docs = self._semantic_entity_command_docs(targets)
        semantic_candidates = self._semantic_ranker.rank_entity_commands(
            utterance=utterance,
            command_docs=semantic_command_docs,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )
        intents = self._build_entity_intents(catalog, targets)
        result = hassil.recognize_best(
            utterance,
            intents,
            allow_unmatched_entities=False,
            language=catalog.metadata.language or "en",
        )
        if result is None:
            return self._semantic_direct_entity_match(
                semantic_candidates=semantic_candidates,
                targets=targets,
                utterance=utterance,
                origin_area=origin_area,
                origin_super_area=origin_super_area,
            )

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
                "semantic_candidates": [
                    {
                        "action": candidate.action,
                        "target_name": candidate.target_name,
                        "score": round(candidate.score, 4),
                    }
                    for candidate in semantic_candidates[:5]
                ],
            },
            raw_text=None,
        )

    def _semantic_direct_entity_match(
        self,
        *,
        semantic_candidates: list[SemanticEntityCandidate],
        targets: list[BuilderTarget],
        utterance: str,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> LLMTranslationResult | None:
        if not semantic_candidates:
            return None
        resolved_candidates: list[tuple[SemanticEntityCandidate, BuilderTarget]] = []
        for candidate in semantic_candidates:
            matched = next(
                (
                    target
                    for target in targets
                    if target.normalized_name == candidate.target_name
                    and target.tool_group == candidate.tool_group
                ),
                None,
            )
            if matched is None or candidate.action not in matched.actions:
                continue
            resolved_candidates.append((candidate, matched))

        if not resolved_candidates:
            return None

        resolved_candidates = self._prefer_semantic_query_candidate(
            resolved_candidates=resolved_candidates
        )
        if not resolved_candidates:
            return None

        best, matched = resolved_candidates[0]
        second = resolved_candidates[1][0] if len(resolved_candidates) > 1 else None
        if best.score < ENTITY_DIRECT_THRESHOLD:
            return None

        utterance_norm = normalize_text(utterance)
        if second is not None and (best.score - second.score) < ENTITY_AMBIGUITY_GAP:
            preferred = self._prefer_origin_semantic_candidate(
                resolved_candidates=resolved_candidates,
                utterance=utterance_norm,
                origin_area=origin_area,
                origin_super_area=origin_super_area,
            )
            if preferred is None:
                return None
            best, matched = preferred

        return LLMTranslationResult(
            mode="translate",
            canonical_text=self._canonical_text_from_action_target(best.action, matched.name),
            tool_group=matched.tool_group,
            confidence=min(0.97, best.score),
            notes="semantic_entity_match",
            valid=True,
            source="semantic_entity_matcher",
            intent_family="entity_control" if best.action != "query" else "entity_query",
            confidence_reason="semantic_entity_command_match",
            debug={
                "match_engine": "fastembed+hassil",
                "target": matched.name,
                "action": best.action,
                "semantic_candidates": [
                    {
                        "action": candidate.action,
                        "target_name": candidate.target_name,
                        "score": round(candidate.score, 4),
                        "synthetic": candidate.synthetic,
                    }
                    for candidate in semantic_candidates[:5]
                ],
            },
            raw_text=None,
        )

    def _prefer_semantic_query_candidate(
        self,
        *,
        resolved_candidates: list[tuple[SemanticEntityCandidate, BuilderTarget]],
    ) -> list[tuple[SemanticEntityCandidate, BuilderTarget]]:
        """Prefer query when the semantic model marks the same target as family-ambiguous."""
        if not resolved_candidates:
            return resolved_candidates

        best_candidate, best_target = resolved_candidates[0]
        if best_candidate.action == "query":
            return resolved_candidates

        query_candidates = [
            pair
            for pair in resolved_candidates
            if pair[0].action == "query"
            and pair[1].normalized_name == best_target.normalized_name
            and pair[1].tool_group == best_target.tool_group
        ]
        if not query_candidates:
            return resolved_candidates

        best_query_candidate, _best_query_target = query_candidates[0]
        if best_query_candidate.score < ENTITY_DIRECT_THRESHOLD:
            return resolved_candidates
        if (best_candidate.score - best_query_candidate.score) > ENTITY_FAMILY_AMBIGUITY_GAP:
            return resolved_candidates

        return [
            pair for pair in resolved_candidates if pair[0].action == "query"
        ]

    def _prefer_origin_semantic_candidate(
        self,
        *,
        resolved_candidates: list[tuple[SemanticEntityCandidate, BuilderTarget]],
        utterance: str,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> tuple[SemanticEntityCandidate, BuilderTarget] | None:
        if self._utterance_mentions_known_location(
            utterance,
            [target for _, target in resolved_candidates],
        ):
            return None

        normalized_origin_area = self._normalize_display_name(origin_area) or None
        normalized_origin_super_area = self._normalize_display_name(origin_super_area) or None

        if normalized_origin_area:
            area_matches = [
                pair for pair in resolved_candidates if pair[1].area == normalized_origin_area
            ]
            if area_matches:
                return area_matches[0]

        if normalized_origin_super_area:
            super_area_matches = [
                pair for pair in resolved_candidates if pair[1].super_area == normalized_origin_super_area
            ]
            if super_area_matches:
                return super_area_matches[0]

        return None

    def _utterance_mentions_known_location(
        self,
        utterance: str,
        targets: list[BuilderTarget],
    ) -> bool:
        for target in targets:
            if target.area and target.area in utterance:
                return True
            if target.super_area and target.super_area in utterance:
                return True
        return False

    def _semantic_debug_snapshot(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> dict[str, Any]:
        if not self._semantic_ranker.available():
            return {}

        phrase_candidates = self._semantic_ranker.rank_phrase_candidates(
            utterance=utterance,
            catalog=catalog,
            infer_tool_group=self._infer_tool_group_from_phrase,
            limit=3,
        )
        targets = self._build_targets(catalog)
        entity_candidates = self._semantic_ranker.rank_entity_commands(
            utterance=utterance,
            command_docs=self._semantic_entity_command_docs(targets),
            origin_area=origin_area,
            origin_super_area=origin_super_area,
            limit=3,
        )
        return {
            "semantic_phrase_candidates": [
                {
                    "target_id": candidate.target_id,
                    "score": round(candidate.score, 4),
                    "example_text": candidate.example_text,
                    "raw_text": candidate.raw_text,
                    "concrete": candidate.concrete,
                    "has_slots": candidate.has_slots,
                }
                for candidate in phrase_candidates
            ],
            "semantic_entity_candidates": [
                {
                    "action": candidate.action,
                    "target_name": candidate.target_name,
                    "tool_group": candidate.tool_group,
                    "score": round(candidate.score, 4),
                    "synthetic": candidate.synthetic,
                    "singular": candidate.singular,
                }
                for candidate in entity_candidates
            ],
        }

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

    def _semantic_entity_command_docs(self, targets: list[BuilderTarget]) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for target in targets:
            for action in sorted(target.actions):
                if action == "set":
                    continue
                semantic_texts = self._semantic_texts_for_action_target(
                    action=action,
                    target=target,
                )
                if target.synthetic and target.tool_group == "lighting" and target.area:
                    semantic_texts.append(f"{CANONICAL_ACTION_TEXT.get(action, action)} {target.area}")
                for semantic_text in semantic_texts:
                    docs.append(
                        {
                            "action": action,
                            "target_name": target.normalized_name,
                            "tool_group": target.tool_group,
                            "intent_family": infer_command_intent_family(action),
                            "synthetic": target.synthetic,
                            "singular": not target.synthetic and not target.normalized_name.endswith("s"),
                            "area": target.area,
                            "super_area": target.super_area,
                            "semantic_text": semantic_text,
                        }
                    )
        return docs

    def _semantic_texts_for_action_target(
        self,
        *,
        action: str,
        target: BuilderTarget,
    ) -> list[str]:
        """Return semantically useful phrasings for a target/action pair."""
        target_name = target.name
        semantic_texts = [self._canonical_text_from_action_target(action, target_name)]
        if target.tool_group == "media" and action != "query":
            semantic_texts.extend(
                self._media_semantic_texts_for_action_target(
                    action=action,
                    target=target,
                )
            )
        if action != "query":
            return self._dedupe_semantic_texts(semantic_texts)

        semantic_texts.extend(
            [
                f"status of {target_name}",
                f"status for {target_name}",
                f"tell me the status of {target_name}",
                f"what is the status of {target_name}",
            ]
        )

        if target.synthetic and target.area:
            semantic_texts.extend(
                [
                    f"what is in {target.area}",
                    f"what is the status of {target.area}",
                    f"tell me the status of {target.area}",
                ]
            )

        return self._dedupe_semantic_texts(semantic_texts)

    def _media_semantic_texts_for_action_target(
        self,
        *,
        action: str,
        target: BuilderTarget,
    ) -> list[str]:
        area = target.area
        if action == "pause":
            phrases = [
                "pause the music",
                "pause music",
                "pause the media",
                "pause media",
            ]
            if area:
                phrases.extend(
                    [
                        f"pause the music in {area}",
                        f"pause the {area} music",
                        f"pause media in {area}",
                    ]
                )
            return phrases

        if action == "play":
            phrases = [
                "play the music",
                "play music",
                "play the media",
                "play media",
                "resume the music",
                "resume music",
                "resume the media",
                "resume media",
            ]
            if area:
                phrases.extend(
                    [
                        f"play the music in {area}",
                        f"play the {area} music",
                        f"resume the music in {area}",
                        f"resume the {area} music",
                    ]
                )
            return phrases

        if action == "stop":
            phrases = [
                "stop the music",
                "stop music",
                "stop the media",
                "stop media",
                "kill the music",
                "kill music",
                "kill the media",
                "kill media",
            ]
            if area:
                phrases.extend(
                    [
                        f"stop the music in {area}",
                        f"stop the {area} music",
                        f"kill the music in {area}",
                        f"kill the {area} music",
                    ]
                )
            return phrases

        return []

    def _dedupe_semantic_texts(self, semantic_texts: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for semantic_text in semantic_texts:
            normalized = self._normalize_display_name(semantic_text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

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
                "pause": "pause",
                "play": "play",
                "stop": "stop",
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
