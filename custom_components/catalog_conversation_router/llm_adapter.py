"""LLM translation adapter."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .matcher import CANONICAL_ACTION_TEXT
from .models import Catalog, LLMTranslationResult
from .phonetics import normalize_text, tokenize

_LOGGER = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")
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
TRANSLATION_PROMPT_TEMPLATE = """You are a translation/classification layer for a smart home AI system. Return ONLY one compact minified JSON object. No markdown. No code fences. No <think>. No explanations.

Valid modes:
- translate
- state
- control
- general

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

Rules:
1. Try translation first.
2. If the request is semantically equivalent to a catalog phrase but uses different wording, mode MUST be translate.
3. Do not classify as state/control when a catalog phrase can answer the same request.
4. Use translate only when the user request can be confidently mapped to one of the provided catalog phrases.
5. When using translate, translated_text MUST exactly match one catalog phrase.
6. If no confident catalog phrase matches, classify instead.
7. Use state when the user asks for information about the smart home, devices, rooms, timers, sensors, or current status.
8. Use control when the user wants to control, change, activate, disable, adjust, or operate the smart home.
9. Use general for everything unrelated to smart home state or control.
10. Only translate may include a non-null translated_text.
11. For state, control, and general, translated_text MUST be null.
12. tool_group can never be translate.

Return format:
{{"mode":"translate|state|control|general","tool_group":"general|timers|lighting|climate|media|locks|covers|fan|lists|mixed","translated_text":null,"confidence":0.0}}

Catalog phrases:
{catalog_lines}

Examples:
User: did I set a timer
JSON: {{"mode":"translate","tool_group":"timers","translated_text":"what timers do I have","confidence":0.95}}

User: kill the kitchen lights
JSON: {{"mode":"translate","tool_group":"lighting","translated_text":"turn off kitchen lights","confidence":0.95}}

User: how do I boil eggs
JSON: {{"mode":"general","tool_group":"general","translated_text":null,"confidence":0.95}}

User: {utterance}"""

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
                mode="general",
                canonical_text=None,
                tool_group="general",
                confidence=0.0,
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
        del language, max_candidates, origin_area, origin_super_area
        catalog_lines = self._render_catalog_phrases(catalog)
        return TRANSLATION_PROMPT_TEMPLATE.format(
            catalog_lines="\n".join(catalog_lines) if catalog_lines else "[general] unavailable",
            utterance=utterance,
        )

    def _validate_translation_result(
        self,
        result: LLMTranslationResult,
        catalog: Catalog,
    ) -> LLMTranslationResult:
        """Reject translation text that is not a known normalized catalog command."""
        if not result.valid:
            return result

        if result.mode != "translate":
            result.canonical_text = None
            return result

        if not result.canonical_text:
            result.valid = False
            result.notes = "missing_translated_text"
            return result

        catalog_map = self._build_catalog_phrase_map(catalog)
        normalized_text = self._normalize_catalog_phrase(result.canonical_text)
        matched = catalog_map.get(normalized_text)
        if matched is None:
            result.valid = False
            result.notes = "invalid_canonical_text"
            result.canonical_text = None
            return result

        result.canonical_text = matched["phrase"]
        normalized_group = self._normalize_tool_group(result.tool_group)
        if normalized_group is None:
            result.tool_group = matched["tool_group"]
            result.notes = "tool_group_inferred"
            return result

        result.tool_group = normalized_group
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
            valid = valid and isinstance(canonical_text, str) and canonical_text.strip() != ""
        else:
            valid = valid and canonical_text is None
        if tool_group is not None and tool_group not in VALID_TOOL_GROUPS:
            tool_group = None

        return LLMTranslationResult(
            mode=mode,
            canonical_text=canonical_text.strip() if isinstance(canonical_text, str) else None,
            tool_group=tool_group,
            confidence=confidence,
            notes=str(notes) if notes is not None else None,
            valid=valid,
        )

    def _render_catalog_phrases(self, catalog: Catalog) -> list[str]:
        return [
            f"[{item['tool_group']}] {item['phrase']}"
            for item in self._build_catalog_phrase_map(catalog).values()
        ]

    def _build_catalog_phrase_map(self, catalog: Catalog) -> dict[str, dict[str, str]]:
        entries: dict[str, dict[str, str]] = {}
        for target in catalog.conversation_targets:
            if not target.canonical_phrase:
                continue
            phrase = target.canonical_phrase.strip()
            normalized = self._normalize_catalog_phrase(phrase)
            if not normalized or normalized in entries:
                continue
            entries[normalized] = {
                "phrase": phrase,
                "tool_group": self._infer_tool_group_from_phrase(phrase),
            }

        for entity in catalog.entity_targets:
            tool_group = self._infer_tool_group_from_domain(entity.domain)
            for capability in entity.capabilities:
                prefix = CANONICAL_ACTION_TEXT.get(capability)
                if not prefix:
                    continue
                if prefix == "what is":
                    phrase = f"what is {entity.name}"
                else:
                    phrase = f"{prefix} {entity.name}"
                phrase = phrase.strip()
                normalized = self._normalize_catalog_phrase(phrase)
                if not normalized or normalized in entries:
                    continue
                entries[normalized] = {
                    "phrase": phrase,
                    "tool_group": tool_group,
                }
        return entries

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
