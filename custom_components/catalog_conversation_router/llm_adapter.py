"""LLM translation adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from time import perf_counter
from functools import partial
from typing import Any

from .local_intent import LocalIntentResolver
from .models import Catalog, LLMTranslationResult
from .phonetics import normalize_text, tokenize
from .semantic_intent import SemanticIntentRanker, SemanticRequestClassification
from .semantic_service import RemoteSemanticIntentRanker

_LOGGER = logging.getLogger(__name__)

SLOW_SEMANTIC_CLASSIFICATION_MS = 1500.0

JSON_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")

VALID_TRANSLATION_ACTIONS = {
    "turn_on",
    "turn_off",
    "pause",
    "play",
    "stop",
    "open",
    "close",
    "lock",
    "unlock",
    "set",
    "query",
    "list",
    "cancel",
    "start",
    "unknown",
}

VALID_TRANSLATION_MODES = {"translate", "state", "control", "general"}
VALID_TOOL_GROUPS = {
    "general",
    "timers",
    "lighting",
    "climate",
    "media",
    "locks",
    "covers",
    "fan",
    "lists",
    "mixed",
}

TRANSLATION_PROMPT_TEMPLATE = """You are a smart-home intent parser. Return ONLY one compact minified JSON object. No markdown. No code fences. No <think>. No explanations.

Valid modes:
- translate: use when the user clearly names an exact Device catalog target or has the same meaning as a Known phrase catalog row
- control: use when the user wants smart-home control but did not clearly name an exact target
- state: use when the user asks about smart-home state/status
- general: use when unrelated to the smart home

Valid tool_group values:
- general
- timers
- lighting
- climate
- media
- locks
- covers
- fan
- lists
- mixed

Valid action values:
- turn_on
- turn_off
- pause
- play
- stop
- open
- close
- lock
- unlock
- set
- query
- list
- cancel
- start
- unknown

Action inference:
- Infer the closest valid action from the user's intent, even when wording is informal.
- Use only the valid action values.
- If no valid action is clear, use action unknown.

Origin context:
- origin_area: {origin_area}
- origin_super_area: {origin_super_area}

Instructions:
1. Check the Known phrase catalog first. If the user asks the same question or gives the same command as a Known phrase row, even with different wording, return mode translate using that exact phrase text as both target and translated_text.
2. For Known phrase translations, use action unknown. Do not invent phrase actions.
3. If the user clearly names an exact Device catalog target, return mode translate with that exact target.
4. If the user wants smart-home control but does not clearly name an exact Device catalog target and does not match a Known phrase, return mode control with the inferred tool_group and action. Use target null and translated_text null.
5. For area-less smart-home control commands, do not choose a room or device. Return mode control. Python will resolve origin_area.
6. For state/status questions, return mode state with action query or list when appropriate. Use target null and translated_text null unless a Known phrase match is clear.
7. For unrelated questions, return mode general with action unknown, target null, translated_text null.
8. Only use Device catalog targets or Known phrase texts that appear in the catalogs below.

Return format:
{{"mode":"translate|control|state|general","tool_group":"general|timers|lighting|climate|media|locks|covers|fan|lists|mixed","action":"turn_on|turn_off|pause|play|stop|open|close|lock|unlock|set|query|list|cancel|start|unknown","target":null,"translated_text":null,"confidence":0.0}}

Device catalog:
{device_catalog_lines}

Known phrase catalog:
{intent_catalog_lines}

If a Known phrase row has the same meaning as the user request, copy that phrase text exactly into target and translated_text.

User: {utterance} /no_think
JSON:"""


# Store extra payloads for LLMTranslationResult by id.
TRANSLATION_PAYLOADS: dict[int, dict[str, Any]] = {}

class LLMAdapter:
    """Build prompts and parse structured translation output."""

    def __init__(self, agent_adapter, *, semantic_service_url: str = "") -> None:
        self._agent_adapter = agent_adapter
        semantic_ranker = (
            RemoteSemanticIntentRanker(semantic_service_url)
            if semantic_service_url.strip()
            else SemanticIntentRanker()
        )
        self._semantic_ranker = semantic_ranker
        self._local_intent_resolver = LocalIntentResolver(semantic_ranker=semantic_ranker)

    async def async_classify_request(
        self,
        *,
        utterance: str,
        catalog: Catalog,
    ) -> SemanticRequestClassification | None:
        """Classify whether an utterance is tool-oriented or general/open-domain."""
        if not getattr(self._semantic_ranker, "available", lambda: False)():
            return None
        started = perf_counter()
        classify = partial(
            self._local_intent_resolver.classify_request,
            utterance=utterance,
            catalog=catalog,
        )
        hass = getattr(self._agent_adapter, "hass", None)
        try:
            if hass is not None and hasattr(hass, "async_add_executor_job"):
                result = await hass.async_add_executor_job(classify)
            else:
                result = classify()
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            if elapsed_ms >= SLOW_SEMANTIC_CLASSIFICATION_MS:
                unavailable_reason = getattr(
                    self._semantic_ranker,
                    "unavailable_reason",
                    lambda: None,
                )()
                _LOGGER.warning(
                    "Slow semantic classification: %.1f ms utterance=%r available=%s unavailable_reason=%r",
                    elapsed_ms,
                    utterance[:120],
                    getattr(self._semantic_ranker, "available", lambda: False)(),
                    unavailable_reason,
                )
            else:
                _LOGGER.debug(
                    "Semantic classification finished in %.1f ms utterance=%r",
                    elapsed_ms,
                    utterance[:120],
                )
        return result

    async def async_translate_for_local(
        self,
        *,
        llm_agent_id: str,
        utterance: str,
        language: str,
        catalog: Catalog,
        max_candidates: int,
        conversation_id: str | None,
        context: Any,
        origin_area: str | None = None,
        origin_super_area: str | None = None,
        preserve_raw_text: bool = False,
    ) -> LLMTranslationResult:
        """Resolve a local translation without using the LLM."""
        del llm_agent_id, language, max_candidates, conversation_id, context, preserve_raw_text
        hass = getattr(self._agent_adapter, "hass", None)
        if hass is not None and hasattr(hass, "async_add_executor_job"):
            return await hass.async_add_executor_job(
                partial(
                    self._local_intent_resolver.resolve,
                    utterance=utterance,
                    catalog=catalog,
                    origin_area=origin_area,
                    origin_super_area=origin_super_area,
                )
            )
        return self._local_intent_resolver.resolve(
            utterance=utterance,
            catalog=catalog,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )

    async def async_final_fallback(
        self,
        *,
        llm_agent_id: str,
        utterance: str,
        language: str,
        conversation_id: str | None,
        context: Any,
        device_id: str | None = None,
        satellite_id: str | None = None,
        extra_system_prompt: str | None = None,
    ):
        """Final direct LLM handling."""
        outcome = await self._agent_adapter.async_process(
            agent_id=llm_agent_id,
            text=utterance,
            language=language,
            conversation_id=conversation_id,
            context=context,
            device_id=device_id,
            satellite_id=satellite_id,
            extra_system_prompt=extra_system_prompt,
        )

        if outcome.response_text:
            outcome.response_text = self._strip_leading_router_json(
                outcome.response_text
            )

        return outcome

    def _strip_leading_router_json(self, text: str | None) -> str | None:
        """Remove accidental leading router JSON from fallback LLM speech."""
        if not text:
            return text

        stripped = text.lstrip()
        match = JSON_BLOCK_RE.match(stripped)
        if not match:
            return text

        try:
            payload = json.loads(match.group(0))
        except Exception:
            return text

        if isinstance(payload, dict) and "mode" in payload and (
            "canonical_text" in payload or "translated_text" in payload
        ):
            remainder = stripped[match.end():].lstrip(" \n,:-")
            return remainder or text

        return text

    def _build_translation_prompt(
        self,
        *,
        utterance: str,
        language: str,
        catalog: Catalog,
        max_candidates: int,
        origin_area: str | None,
        origin_super_area: str | None = None,
    ) -> str:
        del language, max_candidates
        origin_area_text = self._normalize_display_name(origin_area) if isinstance(origin_area, str) and origin_area.strip() else "null"
        origin_super_area_text = self._normalize_display_name(origin_super_area) if isinstance(origin_super_area, str) and origin_super_area.strip() else "null"
        device_catalog_lines = self._render_device_catalog(catalog)
        intent_catalog_lines = self._render_intent_phrase_catalog(catalog)
        for line in device_catalog_lines:
            if "timer" in line.casefold():
                _LOGGER.warning("LLM catalog timer line: %s", line)
        device_catalog_text = (
            "\n".join(device_catalog_lines) if device_catalog_lines else "[general] unavailable"
        )
        intent_catalog_text = (
            "\n".join(intent_catalog_lines) if intent_catalog_lines else "[general] unavailable"
        )
        prompt = TRANSLATION_PROMPT_TEMPLATE.format(
            device_catalog_lines=device_catalog_text,
            intent_catalog_lines=intent_catalog_text,
            origin_area=origin_area_text,
            origin_super_area=origin_super_area_text,
            utterance=utterance,
        )
        device_catalog_hash = hashlib.sha256(device_catalog_text.encode("utf-8")).hexdigest()[:12]
        intent_catalog_hash = hashlib.sha256(intent_catalog_text.encode("utf-8")).hexdigest()[:12]
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        _LOGGER.warning(
            "LLM translation prompt diagnostics: catalog_lines=%s catalog_chars=%s catalog_hash=%s intent_lines=%s intent_chars=%s intent_hash=%s prompt_chars=%s prompt_hash=%s",
            len(device_catalog_lines),
            len(device_catalog_text),
            device_catalog_hash,
            len(intent_catalog_lines),
            len(intent_catalog_text),
            intent_catalog_hash,
            len(prompt),
            prompt_hash,
        )
        return prompt

    def _validate_translation_result(
        self,
        result: LLMTranslationResult,
        catalog: Catalog,
        *,
        utterance: str,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> LLMTranslationResult:
        """Reject translation text that is not a known normalized catalog command."""
        payload = TRANSLATION_PAYLOADS.pop(id(result), {}) or {}
        action = self._normalize_action(payload.get("action"))
        target_map = self._build_target_map(catalog)

        repaired = self._repair_extracted_control_result(
            result=result,
            action=action,
            utterance=utterance,
            target_map=target_map,
            origin_area=origin_area,
        )
        if repaired is not None:
            return repaired

        if not result.valid:
            return result

        if result.mode != "translate":
            result.canonical_text = None
            return result

        target = payload.get("target") if isinstance(payload.get("target"), str) else None
        normalized_target = self._normalize_catalog_phrase(target)
        matched = target_map.get(normalized_target)
        if matched is None:
            intent_map = self._build_intent_phrase_map(catalog)
            matched_intent = intent_map.get(normalized_target)
            if matched_intent is not None:
                result.tool_group = matched_intent["tool_group"]
                result.canonical_text = matched_intent["phrase"]
                result.valid = True
                result.notes = None
                return result

        if action is None or action == "unknown":
            result.valid = False
            result.notes = "invalid_action"
            result.canonical_text = None
            return result

        if matched is None:
            generic_origin_match = self._find_origin_area_match(
                action=action,
                utterance=utterance,
                tool_group=str(result.tool_group or ""),
                target_map=target_map,
                origin_area=str(origin_area or "").strip(),
            )
            if generic_origin_match is not None:
                matched = generic_origin_match

        if matched is None:
            result.valid = False
            result.notes = "invalid_target"
            result.canonical_text = None
            return result

        if action not in matched["actions"]:
            result.valid = False
            result.notes = "unsupported_action_for_target"
            result.canonical_text = None
            return result

        scoped_match = self._scope_translation_to_origin_area(
            matched=matched,
            action=action,
            utterance=utterance,
            target_map=target_map,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )
        if scoped_match is None:
            result.valid = False
            result.notes = "origin_area_conflict"
            result.canonical_text = None
            return result
        matched = scoped_match

        result.tool_group = matched["tool_group"]
        result.canonical_text = self._canonical_text_from_action_target(action, matched["target"])
        return result

    def _repair_extracted_control_result(
        self,
        *,
        result: LLMTranslationResult,
        action: str | None,
        utterance: str,
        target_map: dict[str, dict[str, Any]],
        origin_area: str | None,
    ) -> LLMTranslationResult | None:
        """Resolve mode=control intent extraction into a deterministic catalog command."""
        if result.mode != "control":
            return None
        if action is None or action == "unknown":
            return None
        if result.tool_group in {None, "general"}:
            return None

        if result.tool_group == "timers":
            timer_target = self._timer_target_from_utterance(utterance)
            if timer_target is None:
                return None
            if action == "query":
                action = "list"
            matched = target_map.get(timer_target)
            if matched is None or action not in matched.get("actions", set()):
                return None
            result.mode = "translate"
            result.valid = True
            result.notes = None
            result.tool_group = matched["tool_group"]
            result.canonical_text = self._canonical_text_from_action_target(action, matched["target"])
            return result

        origin_match = self._find_origin_area_match(
            action=action,
            utterance=utterance,
            tool_group=str(result.tool_group or ""),
            target_map=target_map,
            origin_area=self._normalize_display_name(origin_area),
        )
        if origin_match is None:
            return None

        result.mode = "translate"
        result.valid = True
        result.notes = None
        result.tool_group = origin_match["tool_group"]
        result.canonical_text = self._canonical_text_from_action_target(action, origin_match["target"])
        return result

    def _timer_target_from_utterance(self, utterance: str) -> str | None:
        tokens = set(tokenize(normalize_text(utterance)))
        if tokens & {"alarm", "alarms"}:
            return "alarms"
        if tokens & {"reminder", "reminders"}:
            return "reminders"
        if tokens & {"timer", "timers"}:
            return "timers"
        return None

    def _scope_translation_to_origin_area(
        self,
        *,
        matched: dict[str, Any],
        action: str,
        utterance: str,
        target_map: dict[str, dict[str, Any]],
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> dict[str, Any] | None:
        """Prevent area-less utterances from executing against the wrong room."""
        clean_origin_area = str(origin_area or "").strip()
        clean_origin_super_area = str(origin_super_area or "").strip()
        if not clean_origin_area and not clean_origin_super_area:
            return matched

        tool_group = str(matched.get("tool_group") or "").strip().lower()
        if tool_group in {"timers", "lists", "general"}:
            return matched

        known_locations = self._known_locations_from_target_map(target_map)
        utterance_location = self._known_location_mentioned(utterance, known_locations)
        matched_target = str(matched.get("target") or "")
        translated_location = self._known_location_mentioned(matched_target, known_locations)

        matched_area = str(matched.get("area") or "").strip()
        matched_super_area = str(matched.get("super_area") or "").strip()

        # If the user explicitly named a known area/super-area, trust the catalog
        # match only when it belongs to that known location. Do not use origin_area.
        if utterance_location:
            if self._same_location(matched_area, utterance_location) or self._same_location(matched_super_area, utterance_location):
                return matched
            return None

        # If the translated target does not contain a known area/super-area, do not
        # rewrite it. It may be a real device name such as "Drum Light".
        if not translated_location:
            return matched

        # The LLM selected/inserted a known location that the user did not say.
        # If it conflicts with origin context, try a deterministic origin-area repair.
        if clean_origin_area and not self._same_location(translated_location, clean_origin_area):
            origin_match = self._find_origin_area_match(
                action=action,
                utterance=utterance,
                tool_group=str(matched.get("tool_group") or ""),
                target_map=target_map,
                origin_area=clean_origin_area,
            )
            return origin_match

        if clean_origin_super_area and not self._same_location(translated_location, clean_origin_super_area):
            origin_match = self._find_origin_area_match(
                action=action,
                utterance=utterance,
                tool_group=str(matched.get("tool_group") or ""),
                target_map=target_map,
                origin_area=clean_origin_area,
            )
            return origin_match

        return matched
    
    def _known_locations_from_target_map(self, target_map: dict[str, dict[str, Any]]) -> set[str]:
        locations: set[str] = set()
        for item in target_map.values():
            area = str(item.get("area") or "").strip()
            super_area = str(item.get("super_area") or "").strip()
            if area:
                locations.add(area)
            if super_area:
                locations.add(super_area)
        return locations

    def _known_location_mentioned(self, text: str, known_locations: set[str]) -> str | None:
        normalized_text = self._normalize_catalog_phrase(text)
        if not normalized_text:
            return None
        for location in sorted(known_locations, key=len, reverse=True):
            normalized_location = self._normalize_catalog_phrase(location)
            if normalized_location and self._phrase_contains_token_sequence(normalized_text, normalized_location):
                return location
        return None

    def _find_origin_area_match(
        self,
        *,
        action: str,
        utterance: str,
        tool_group: str,
        target_map: dict[str, dict[str, Any]],
        origin_area: str,
    ) -> dict[str, Any] | None:
        if not origin_area:
            return None

        candidates = [
            item
            for item in target_map.values()
            if action in item.get("actions", set())
            and self._same_location(str(item.get("area") or ""), origin_area)
            and (
                item.get("tool_group") == tool_group
                or self._target_name_matches_requested_kind(
                    utterance,
                    str(item.get("target") or ""),
                )
            )
            and self._target_kind_matches_utterance(
                utterance,
                str(item.get("tool_group") or ""),
                str(item.get("target") or ""),
            )
        ]
        if not candidates:
            return None

        requested_scope = self._requested_device_scope(utterance, tool_group)
        group_candidates = [item for item in candidates if bool(item.get("synthetic"))]
        specific_candidates = [item for item in candidates if not bool(item.get("synthetic"))]

        if requested_scope == "plural":
            if len(group_candidates) == 1:
                return group_candidates[0]
            if len(specific_candidates) == 1:
                return specific_candidates[0]
            return None

        if requested_scope == "singular":
            preferred_singular_targets = self._preferred_singular_targets_for_origin(
                tool_group=tool_group,
                origin_area=origin_area,
            )
            exact_singular_candidates = [
                item
                for item in specific_candidates
                if self._normalize_catalog_phrase(str(item.get("target") or ""))
                in preferred_singular_targets
            ]
            if len(exact_singular_candidates) == 1:
                return exact_singular_candidates[0]
            if len(specific_candidates) == 1:
                return specific_candidates[0]
            if len(candidates) == 1 and not group_candidates:
                return candidates[0]
            return None

        if len(candidates) == 1:
            return candidates[0]
        if len(group_candidates) == 1:
            return group_candidates[0]
        if len(specific_candidates) == 1:
            return specific_candidates[0]

        return None

    def _target_name_matches_requested_kind(self, utterance: str, target: str) -> bool:
        utterance_tokens = set(tokenize(normalize_text(utterance)))
        target_tokens = set(tokenize(normalize_text(target)))
        device_tokens = {
            "light",
            "lights",
            "lamp",
            "lamps",
            "fan",
            "fans",
            "thermostat",
            "thermostats",
            "temperature",
            "blind",
            "blinds",
            "shade",
            "shades",
            "curtain",
            "curtains",
            "lock",
            "locks",
            "door",
        }
        return bool(utterance_tokens & target_tokens & device_tokens)


    def _preferred_singular_targets_for_origin(
        self,
        *,
        tool_group: str,
        origin_area: str,
    ) -> set[str]:
        normalized_origin = self._normalize_catalog_phrase(origin_area)
        if not normalized_origin:
            return set()
        if tool_group == "lighting":
            return {
                f"{normalized_origin} light",
                f"{normalized_origin} lamp",
            }
        if tool_group == "fan":
            return {f"{normalized_origin} fan"}
        if tool_group == "climate":
            return {
                f"{normalized_origin} thermostat",
                f"{normalized_origin} temperature",
            }
        if tool_group == "covers":
            return {
                f"{normalized_origin} blind",
                f"{normalized_origin} shade",
                f"{normalized_origin} curtain",
                f"{normalized_origin} cover",
            }
        if tool_group == "locks":
            return {
                f"{normalized_origin} lock",
                f"{normalized_origin} door lock",
            }
        return set()


    def _requested_device_scope(self, utterance: str, tool_group: str) -> str | None:
        tokens = set(tokenize(normalize_text(utterance)))
        if tool_group == "lighting":
            if tokens & {"lights", "lamps"}:
                return "plural"
            if tokens & {"light", "lamp"}:
                return "singular"
        if tool_group == "fan":
            if "fans" in tokens:
                return "plural"
            if "fan" in tokens:
                return "singular"
        if tool_group == "climate":
            if tokens & {"thermostats"}:
                return "plural"
            if tokens & {"thermostat", "temperature", "climate", "heat", "cool", "ac"}:
                return "singular"
        if tool_group == "covers":
            if tokens & {"covers", "blinds", "shades", "curtains", "shutters"}:
                return "plural"
            if tokens & {"cover", "blind", "shade", "curtain", "shutter", "garage"}:
                return "singular"
        if tool_group == "locks":
            if "locks" in tokens:
                return "plural"
            if tokens & {"lock", "door"}:
                return "singular"
        return None

    def _utterance_mentions_location(self, utterance: str, location: str) -> bool:
        normalized_location = self._normalize_catalog_phrase(location)
        if not normalized_location:
            return False
        normalized_utterance = self._normalize_catalog_phrase(utterance)
        return self._phrase_contains_token_sequence(normalized_utterance, normalized_location)

    def _phrase_contains_token_sequence(self, text: str, phrase: str) -> bool:
        if not text or not phrase:
            return False
        return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))

    def _same_location(self, left: str, right: str) -> bool:
        return bool(
            left
            and right
            and self._normalize_catalog_phrase(left) == self._normalize_catalog_phrase(right)
        )

    def _target_kind_matches_utterance(self, utterance: str, tool_group: str, target: str) -> bool:
        normalized = normalize_text(utterance)
        tokens = set(tokenize(normalized))
        target_tokens = set(tokenize(normalize_text(target)))
        if tool_group == "fan":
            return bool(tokens & {"fan", "fans"})
        if tool_group == "lighting":
            return bool(tokens & {"light", "lights", "lamp", "lamps"}) or bool(
                target_tokens & tokens & {"light", "lights", "lamp", "lamps"}
            )
        if tool_group == "climate":
            return bool(tokens & {"temperature", "thermostat", "climate", "heat", "cool", "ac"})
        if tool_group == "covers":
            return bool(tokens & {"cover", "covers", "blind", "blinds", "shade", "shades", "curtain", "curtains", "garage"})
        if tool_group == "locks":
            return bool(tokens & {"lock", "locks", "door"})
        return bool(target_tokens & tokens)

    def _parse_translation_json(self, text: str) -> LLMTranslationResult:
        # First try strict JSON parsing (best case: model returned pure JSON)
        try:
            payload = json.loads(text.strip())
            text = json.dumps(payload)
        except Exception:
            pass

        match = JSON_BLOCK_RE.search(text)
        if not match:
            _LOGGER.debug("LLM translation returned no JSON block. Raw text suppressed.")
            return LLMTranslationResult(
                mode="general",
                canonical_text=None,
                tool_group="general",
                confidence=0.0,
                notes="no_json_found",
                valid=False,
            )

        try:
            payload = json.loads(match.group(0).strip())
        except json.JSONDecodeError:
            _LOGGER.debug("LLM translation produced invalid JSON. Raw text suppressed.")
            _LOGGER.debug("Invalid LLM JSON payload: %s", text)
            return LLMTranslationResult(
                mode="general",
                canonical_text=None,
                tool_group="general",
                confidence=0.0,
                notes="invalid_json",
                valid=False,
            )

        mode = str(payload.get("mode", "general"))
        mode = mode.strip().lower()
        translated_text = payload.get("translated_text")
        action = payload.get("action")
        target = payload.get("target")
        canonical_text = translated_text.strip() if isinstance(translated_text, str) else None
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        tool_group = payload.get("tool_group")
        if isinstance(tool_group, str):
            tool_group = tool_group.strip().lower()
        else:
            tool_group = None
        notes = payload.get("notes")
        valid_mode = mode in VALID_TRANSLATION_MODES
        valid_confidence = 0.0 <= confidence <= 1.0
        valid = valid_mode and valid_confidence
        if mode == "translate":
            valid = valid and self._normalize_action(action) is not None and isinstance(target, str) and target.strip() != ""
        elif mode in {"control", "state"}:
            valid = valid and self._normalize_action(action) is not None and canonical_text is None
        else:
            valid = valid and canonical_text is None
        if tool_group is not None and tool_group not in VALID_TOOL_GROUPS:
            tool_group = None

        result = LLMTranslationResult(
            mode=mode,
            canonical_text=canonical_text.strip() if isinstance(canonical_text, str) else None,
            tool_group=tool_group,
            confidence=confidence,
            notes=str(notes) if notes is not None else None,
            valid=valid,
        )
        TRANSLATION_PAYLOADS[id(result)] = payload
        return result

    def _render_device_catalog(self, catalog: Catalog) -> list[str]:
        lines: list[str] = []
        for item in self._build_target_map(catalog).values():
            parts = [
                f"[{item['tool_group']}] {item['target']}",
                f"actions: {','.join(sorted(item['actions']))}",
            ]
            area = item.get("area")
            super_area = item.get("super_area")
            if area:
                parts.append(f"area: {area}")
            if super_area:
                parts.append(f"super_area: {super_area}")
            if item.get("synthetic"):
                parts.append("scope: area_group")
            lines.append(" | ".join(parts))
        return lines

    def _render_intent_phrase_catalog(self, catalog: Catalog) -> list[str]:
        return [
            f"[phrase] {item['phrase']}"
            for item in self._build_intent_phrase_map(catalog).values()
        ]

    def _normalize_display_name(self, value: str | None) -> str:
        if not value:
            return ""
        return self._normalize_catalog_phrase(value)

    def _build_target_map(self, catalog: Catalog) -> dict[str, dict[str, Any]]:
        targets: dict[str, dict[str, Any]] = {}

        def add_target(
            name: str,
            tool_group: str,
            actions: set[str],
            *,
            area: str | None = None,
            super_area: str | None = None,
            synthetic: bool = False,
        ) -> None:
            name = self._normalize_display_name(name)
            if not name:
                return
            normalized = self._normalize_catalog_phrase(name)
            if not normalized or normalized in targets:
                return

            clean_area = self._normalize_display_name(area)
            clean_super_area = self._normalize_display_name(super_area)

            targets[normalized] = {
                "target": name,
                "tool_group": tool_group,
                "actions": {
                    action
                    for action in actions
                    if action in VALID_TRANSLATION_ACTIONS and action != "unknown"
                },
                "area": clean_area or None,
                "super_area": clean_super_area or None,
                "synthetic": synthetic,
            }

        # Virtual/common targets.
        add_target("timers", "timers", {"list", "query", "cancel", "start"})
        add_target("alarms", "timers", {"list", "query", "cancel", "start"})
        add_target("reminders", "timers", {"list", "query", "cancel", "start"})

        seen_area_groups: set[tuple[str, str]] = set()
        sorted_entities = sorted(
            catalog.entity_targets,
            key=lambda entity: (
                self._normalize_catalog_phrase(str(getattr(entity, "name", "") or "")),
                str(getattr(entity, "domain", "") or "").casefold(),
                self._normalize_catalog_phrase(str(getattr(entity, "area", "") or "")),
                self._normalize_catalog_phrase(str(getattr(entity, "super_area", "") or "")),
            ),
        )
        for entity in sorted_entities:
            tool_group = self._infer_tool_group_from_domain(entity.domain)
            actions = self._actions_for_entity(entity)
            area = self._normalize_display_name(getattr(entity, "area", "") or "")
            super_area = self._normalize_display_name(getattr(entity, "super_area", "") or "")
            add_target(entity.name, tool_group, actions, area=area, super_area=super_area)
            if area and tool_group in {"lighting", "fan", "climate"}:
                area_key = (tool_group, area)
                if area_key not in seen_area_groups:
                    seen_area_groups.add(area_key)
                    if tool_group == "lighting":
                        add_target(
                            f"{area} lights",
                            "lighting",
                            {"turn_on", "turn_off", "set", "query"},
                            area=area,
                            super_area=super_area,
                            synthetic=True,
                        )
                    elif tool_group == "fan":
                        add_target(
                            f"{area} fans",
                            "fan",
                            {"turn_on", "turn_off", "set", "query"},
                            area=area,
                            super_area=super_area,
                            synthetic=True,
                        )
                    elif tool_group == "climate":
                        add_target(
                            f"{area} temperature",
                            "climate",
                            {"set", "query"},
                            area=area,
                            super_area=super_area,
                            synthetic=True,
                        )

        return dict(sorted(targets.items(), key=lambda item: item[0]))

    def _build_intent_phrase_map(self, catalog: Catalog) -> dict[str, dict[str, Any]]:
        phrases: dict[str, dict[str, Any]] = {}

        for target_obj in catalog.conversation_targets:
            for phrase in self._conversation_target_phrases(target_obj):
                tool_group = self._infer_tool_group_from_phrase(phrase)
                normalized_phrase = self._normalize_catalog_phrase(phrase)
                if not normalized_phrase or normalized_phrase in phrases:
                    continue
                clean_phrase = self._normalize_display_name(phrase)
                phrases[normalized_phrase] = {
                    "phrase": clean_phrase,
                    "tool_group": tool_group,
                    "target": clean_phrase,
                    "action": "unknown",
                }

        return dict(sorted(phrases.items(), key=lambda item: item[0]))

    def _conversation_target_phrases(self, target) -> list[str]:
        """Return clean, user-sayable phrases for a conversation target."""
        phrases: list[str] = []
        canonical_phrase = getattr(target, "canonical_phrase", None)
        if isinstance(canonical_phrase, str) and self._is_user_sayable_phrase(canonical_phrase):
            phrases.append(self._normalize_display_name(canonical_phrase))
        for phrase in getattr(target, "sample_phrases", []) or []:
            if isinstance(phrase, str) and self._is_user_sayable_phrase(phrase):
                phrases.append(self._normalize_display_name(phrase))
        return phrases

    def _is_user_sayable_phrase(self, phrase: str | None) -> bool:
        """Filter out internal/template phrases that hurt the translation prompt."""
        if not phrase:
            return False
        text = phrase.strip()
        lowered = text.casefold()
        if not text:
            return False
        if "command_prompt" in lowered or "trigger_phrases" in lowered:
            return False
        if "computednametype" in lowered:
            return False
        return True

    def _actions_for_entity(self, entity) -> set[str]:
        actions: set[str] = set()
        for capability in getattr(entity, "capabilities", []) or []:
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

    def _normalize_action(self, action: Any) -> str | None:
        if not isinstance(action, str):
            return None
        normalized = action.strip().lower()
        return normalized if normalized in VALID_TRANSLATION_ACTIONS else None

    def _canonical_text_from_action_target(self, action: str, target: str) -> str:
        target = self._normalize_display_name(target)

        if action == "turn_on":
            return f"turn on {target}"
        if action == "turn_off":
            return f"turn off {target}"
        if action == "open":
            return f"open {target}"
        if action == "close":
            return f"close {target}"
        if action == "lock":
            return f"lock {target}"
        if action == "unlock":
            return f"unlock {target}"
        if action == "set":
            return f"set {target}"
        if action == "query":
            if target in {"timers", "alarms", "reminders"}:
                return f"what {target} do I have"
            return f"what is {target}"
        if action == "list":
            if target in {"timers", "alarms", "reminders"}:
                return f"what {target} do I have"
            return f"list {target}"
        if action == "cancel":
            return f"cancel {target}"
        if action == "start":
            return f"start {target}"
        return str(target)



    def _normalize_catalog_phrase(self, phrase: str | None) -> str:
        if not phrase:
            return ""
        return re.sub(r"\s+", " ", phrase).strip().casefold()

    def _normalize_tool_group(self, tool_group: str | None) -> str | None:
        if not tool_group:
            return None
        normalized = tool_group.strip().lower()
        return normalized if normalized in VALID_TOOL_GROUPS else None

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

    def _infer_tool_group_from_phrase(self, phrase: str) -> str:
        normalized = normalize_text(phrase)
        tokens = set(tokenize(normalized))
        if tokens & {"timer", "timers", "alarm", "alarms", "reminder", "reminders"}:
            return "timers"
        if tokens & {"light", "lights", "lamp", "lamps"}:
            return "lighting"
        if tokens & {"climate", "temperature", "thermostat", "heat", "cool", "ac"}:
            return "climate"
        if tokens & {
            "media",
            "music",
            "movie",
            "movies",
            "song",
            "songs",
            "speaker",
            "speakers",
            "tv",
            "receiver",
            "volume",
            "pause",
            "play",
        }:
            return "media"
        if tokens & {"lock", "locks", "unlock", "door"}:
            return "locks"
        if tokens & {"cover", "covers", "blind", "blinds", "shade", "shades", "curtain", "curtains", "shutter", "shutters", "garage"}:
            return "covers"
        if tokens & {"fan", "fans"}:
            return "fan"
        if tokens & {"list", "lists", "todo", "shopping", "grocery", "groceries"}:
            return "lists"
        return "mixed"
