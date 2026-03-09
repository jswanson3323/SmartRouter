"""LLM translation adapter."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .models import CandidateType, Catalog, LLMTranslationResult

_LOGGER = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


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
    ) -> LLMTranslationResult:
        """Ask LLM agent to output strict translation JSON."""
        prompt = self._build_translation_prompt(
            utterance=utterance,
            language=language,
            catalog=catalog,
            max_candidates=max_candidates,
            origin_area=origin_area,
        )

        outcome = await self._agent_adapter.async_process(
            agent_id=llm_agent_id,
            text=prompt,
            language=language,
            conversation_id=conversation_id,
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

        # Prevent accidental speech/debug leakage from LLM output
        # Only structured fields should propagate through the router
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
    ):
        """Final direct LLM handling."""
        return await self._agent_adapter.async_process(
            agent_id=llm_agent_id,
            text=utterance,
            language=language,
            conversation_id=conversation_id,
            context=context,
        )

    def _build_translation_prompt(
        self,
        *,
        utterance: str,
        language: str,
        catalog: Catalog,
        max_candidates: int,
        origin_area: str | None,
    ) -> str:
        entity_candidates = [e.name for e in catalog.entity_targets[:max_candidates]]
        conversation_candidates = [
            t.display_name for t in catalog.conversation_targets[:max_candidates]
        ]
        area_entity_candidates = [
            e.name
            for e in catalog.entity_targets
            if origin_area
            and e.area
            and origin_area.strip().lower() == e.area.strip().lower()
        ][:max_candidates]

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
            "Never invent entities, areas, or custom targets. "
            "If origin area is provided, strongly prefer matching entities in that area. "
            "For generic room-local requests like 'turn on the light', choose an in-area entity when available. "
            f"Language: {language}. "
            f"Original utterance: {utterance!r}. "
            f"Origin area: {origin_area!r}. "
            f"Origin-area entity targets: {area_entity_candidates}. "
            f"Entity targets: {entity_candidates}. "
            f"Conversation targets: {conversation_candidates}. "
            "Return ONLY a single JSON object and nothing else. Do not include explanations, prefixes, suffixes, or text before or after the JSON. If unsure, return mode=fallback_answer with canonical_text=null. JSON schema: "
            f"{json.dumps(schema)}"
        )

    def _parse_translation_json(self, text: str) -> LLMTranslationResult:
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
            payload = json.loads(match.group(0))
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
        canonical_text = payload.get("canonical_text")
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
