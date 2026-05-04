"""LLM translation adapter."""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any

from .matcher import CANONICAL_ACTION_TEXT
from .models import CandidateType, Catalog, LLMTranslationResult
from .phonetics import normalize_text, tokenize

_LOGGER = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")
MAX_ENTITY_PROMPT_CANDIDATES = 8
MAX_CONVERSATION_PROMPT_CANDIDATES = 6
MAX_LOCALITY_PROMPT_CANDIDATES = 4

class LLMAdapter:
    """Build prompts and parse structured translation output."""

    def __init__(self, agent_adapter) -> None:
        self._agent_adapter = agent_adapter

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
        """Ask LLM agent to output strict translation JSON."""
        prompt = self._build_translation_prompt(
            utterance=utterance,
            language=language,
            catalog=catalog,
            max_candidates=max_candidates,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )

        outcome = await self._agent_adapter.async_process(
            agent_id=llm_agent_id,
            text=prompt,
            language=language,
            conversation_id=None,
            context=context,
        )

        if not outcome.response_text:
            return LLMTranslationResult(
                mode="fallback_answer",
                canonical_text=None,
                confidence=0.0,
                target_type=CandidateType.UNKNOWN,
                notes="empty_response",
                valid=False,
                raw_text=None,
            )

        parsed = self._parse_translation_json(outcome.response_text)
        parsed.raw_text = outcome.response_text
        parsed = self._validate_translation_result(parsed, catalog)

        # Prevent accidental speech/debug leakage from LLM output unless explicitly
        # requested for a debug trace.
        if not preserve_raw_text:
            parsed.raw_text = None

        return parsed

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

        if isinstance(payload, dict) and "mode" in payload and "canonical_text" in payload:
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
        entity_candidates = self._select_entity_candidates(
            utterance=utterance,
            catalog=catalog,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
            limit=min(max_candidates, MAX_ENTITY_PROMPT_CANDIDATES),
        )
        conversation_candidates = self._select_conversation_candidates(
            utterance=utterance,
            catalog=catalog,
            limit=min(max_candidates, MAX_CONVERSATION_PROMPT_CANDIDATES),
        )
        area_entity_candidates = self._select_locality_entity_candidates(
            utterance=utterance,
            catalog=catalog,
            area=origin_area,
            limit=MAX_LOCALITY_PROMPT_CANDIDATES,
        )
        origin_super_area = origin_super_area or self._infer_super_area_from_origin_area(
            origin_area=origin_area,
            catalog=catalog,
        )
        super_area_entity_candidates = self._select_locality_entity_candidates(
            utterance=utterance,
            catalog=catalog,
            area=origin_super_area,
            area_attr="super_area",
            limit=MAX_LOCALITY_PROMPT_CANDIDATES,
            exclude_area=origin_area,
        )

        schema = {
            "mode": "translate_for_local | fallback_answer",
            "canonical_text": "string or null",
            "confidence": "float 0..1",
            "target_type": "entity_command | conversation_target | unknown",
            "notes": "string",
        }

        return (
            "You are a translation layer for Home Assistant conversation routing. "
            "Do NOT execute commands, only translate when confident. "
            "Correct likely ASR mistakes but only to listed valid targets. "
            "Never invent entities, areas, custom targets, or canonical commands. canonical_text must use only the exact listed entity target names or exact listed conversation target phrases; otherwise return mode=fallback_answer with canonical_text=null. "
            "canonical_text must be a full natural-language command or exact listed conversation phrase. "
            "Never return tool names, function names, API names, service names, domain names, or internal identifiers such as HassTurnOn, turn_on, light, fan, script, scene, automation, or homeassistant. "
            "If origin area is provided, strongly prefer matching entities in that area first. "
            "If no entity in the origin area fits, use the origin SuperArea as a second pass. "
            "For generic room-local requests like 'turn on the light', choose an in-area entity when available; otherwise prefer an entity from the same SuperArea. "
            f"Language: {language}. "
            f"Original utterance: {utterance!r}. "
            f"Origin area: {origin_area!r}. "
            f"Origin SuperArea: {origin_super_area!r}. "
            f"Origin-area entity targets: {area_entity_candidates}. "
            f"Origin-SuperArea entity targets: {super_area_entity_candidates}. "
            f"Relevant entity targets: {entity_candidates}. "
            f"Relevant conversation targets: {conversation_candidates}. "
            "Return ONLY a single JSON object and nothing else. Do not include explanations, prefixes, suffixes, or text before or after the JSON. If unsure, return mode=fallback_answer with canonical_text=null. JSON schema: "
            f"{json.dumps(schema)}"
        )

    def _select_entity_candidates(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        origin_area: str | None,
        origin_super_area: str | None,
        limit: int,
    ) -> list[str]:
        normalized = normalize_text(utterance)
        utterance_tokens = set(tokenize(utterance))
        scored: list[tuple[float, str]] = []
        for entity in catalog.entity_targets:
            entity_token_set = set(entity.tokens)
            overlap = len(utterance_tokens & entity_token_set)
            similarity = SequenceMatcher(a=normalized, b=entity.normalized_name).ratio()
            locality_bonus = 0.0
            if origin_area and entity.area and origin_area.strip().lower() == entity.area.strip().lower():
                locality_bonus = 0.15
            elif (
                origin_super_area
                and entity.super_area
                and origin_super_area.strip().lower() == entity.super_area.strip().lower()
            ):
                locality_bonus = 0.08
            score = overlap * 0.4 + similarity * 0.6 + locality_bonus
            if overlap == 0 and similarity < 0.24 and locality_bonus == 0.0:
                continue
            scored.append((score, entity.name))
        scored.sort(key=lambda item: (-item[0], item[1].lower()))
        names = [name for _, name in scored[:limit]]
        if names:
            return names
        return [entity.name for entity in catalog.entity_targets[:limit]]

    def _select_conversation_candidates(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        limit: int,
    ) -> list[str]:
        normalized = normalize_text(utterance)
        utterance_tokens = set(tokenize(utterance))
        scored: list[tuple[float, str]] = []
        for target in catalog.conversation_targets:
            token_set = set(target.tokens)
            overlap = len(utterance_tokens & token_set)
            similarity = SequenceMatcher(a=normalized, b=target.normalized_name).ratio()
            score = overlap * 0.45 + similarity * 0.55
            if overlap == 0 and similarity < 0.24:
                continue
            scored.append((score, target.display_name))
        scored.sort(key=lambda item: (-item[0], item[1].lower()))
        names = [name for _, name in scored[:limit]]
        if names:
            return names
        return [target.display_name for target in catalog.conversation_targets[:limit]]

    def _select_locality_entity_candidates(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        area: str | None,
        limit: int,
        area_attr: str = "area",
        exclude_area: str | None = None,
    ) -> list[str]:
        if not area:
            return []
        normalized_area = area.strip().lower()
        normalized_exclude_area = exclude_area.strip().lower() if exclude_area else None
        normalized = normalize_text(utterance)
        utterance_tokens = set(tokenize(utterance))
        scored: list[tuple[float, str]] = []
        for entity in catalog.entity_targets:
            entity_area = getattr(entity, area_attr, None)
            if not entity_area or entity_area.strip().lower() != normalized_area:
                continue
            if (
                area_attr == "super_area"
                and normalized_exclude_area
                and entity.area
                and entity.area.strip().lower() == normalized_exclude_area
            ):
                continue
            overlap = len(utterance_tokens & set(entity.tokens))
            similarity = SequenceMatcher(a=normalized, b=entity.normalized_name).ratio()
            score = overlap * 0.45 + similarity * 0.55
            scored.append((score, entity.name))
        scored.sort(key=lambda item: (-item[0], item[1].lower()))
        return [name for _, name in scored[:limit]]

    def _infer_super_area_from_origin_area(
        self,
        *,
        origin_area: str | None,
        catalog: Catalog,
    ) -> str | None:
        if not origin_area:
            return None

        area_name = origin_area.strip().lower()
        matches = {
            e.super_area.strip()
            for e in catalog.entity_targets
            if e.area and e.super_area and e.area.strip().lower() == area_name
        }
        if len(matches) == 1:
            return next(iter(matches))
        return None

    def _validate_translation_result(
        self,
        result: LLMTranslationResult,
        catalog: Catalog,
    ) -> LLMTranslationResult:
        """Reject LLM canonical text that is not a known catalog command."""
        if not result.valid or not result.canonical_text:
            return result

        valid_phrases: set[str] = {
            target.canonical_phrase.strip().lower()
            for target in catalog.conversation_targets
            if target.canonical_phrase
        }
        for entity in catalog.entity_targets:
            for capability in entity.capabilities:
                prefix = CANONICAL_ACTION_TEXT.get(capability)
                if not prefix:
                    continue
                if prefix == "what is":
                    valid_phrases.add(f"what is {entity.name}".strip().lower())
                else:
                    valid_phrases.add(f"{prefix} {entity.name}".strip().lower())

        if result.canonical_text.strip().lower() in valid_phrases:
            return result

        result.valid = False
        result.mode = "fallback_answer"
        result.notes = "invalid_canonical_text"
        result.canonical_text = None
        return result

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
                mode="fallback_answer",
                canonical_text=None,
                confidence=0.0,
                target_type=CandidateType.UNKNOWN,
                notes="no_json_found",
                valid=False,
            )

        try:
            payload = json.loads(match.group(0).strip())
        except json.JSONDecodeError:
            _LOGGER.debug("LLM translation produced invalid JSON. Raw text suppressed.")
            _LOGGER.debug("Invalid LLM JSON payload: %s", text)
            return LLMTranslationResult(
                mode="fallback_answer",
                canonical_text=None,
                confidence=0.0,
                target_type=CandidateType.UNKNOWN,
                notes="invalid_json",
                valid=False,
            )

        mode = str(payload.get("mode", "fallback_answer"))
        mode = mode.strip().lower()
        canonical_text = payload.get("canonical_text")
        if isinstance(canonical_text, str):
            canonical_text = canonical_text.strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        raw_target_type = str(payload.get("target_type", "unknown"))
        notes = payload.get("notes")

        try:
            target_type = CandidateType(raw_target_type)
        except ValueError:
            target_type = CandidateType.UNKNOWN

        valid = (
            mode == "translate_for_local"
            and isinstance(canonical_text, str)
            and canonical_text.strip() != ""
            and 0.0 <= confidence <= 1.0
        )

        return LLMTranslationResult(
            mode=mode,
            canonical_text=canonical_text.strip() if isinstance(canonical_text, str) else None,
            confidence=confidence,
            target_type=target_type,
            notes=str(notes) if notes is not None else None,
            valid=valid,
        )
