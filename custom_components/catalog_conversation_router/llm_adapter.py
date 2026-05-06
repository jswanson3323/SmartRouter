"""LLM translation adapter."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from .models import Catalog, LLMTranslationResult
from .phonetics import normalize_text, tokenize

_LOGGER = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")

VALID_TRANSLATION_ACTIONS = {
    "turn_on",
    "turn_off",
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

Valid action values:
- turn_on
- turn_off
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

Capabilities:
- lighting: turn_on, turn_off, set, query
- fan: turn_on, turn_off, set, query
- climate: set, query, turn_on, turn_off
- media: turn_on, turn_off, query
- covers: open, close, set, query
- locks: lock, unlock, query
- timers: start, cancel, list, query
- lists: turn_on, query

Rules:
1. Try translation first.
2. Use translate when the user request can be mapped to one known target and one valid action.
3. Use the exact target name from the Device catalog when a device, area, or supported virtual target matches.
4. translated_text must be a simple Home Assistant-style phrase built from the action and target.
5. If no confident target/action pair matches, classify instead.
6. Use state when the user asks for information about the smart home, devices, rooms, timers, sensors, or current status.
7. Use control when the user wants to control, change, activate, disable, adjust, or operate the smart home.
8. Use general for everything unrelated to smart home state or control.
9. For state, control, and general: action must be unknown, target must be null, and translated_text must be null.
10. tool_group can never be translate.

Return format:
{{"mode":"translate|state|control|general","tool_group":"general|timers|lighting|climate|media|locks|covers|fan|lists|mixed","action":"turn_on|turn_off|open|close|lock|unlock|set|query|list|cancel|start|unknown","target":null,"translated_text":null,"confidence":0.0}}

Device catalog:
{catalog_lines}

Examples:
User: did I set a timer
JSON: {{"mode":"translate","tool_group":"timers","action":"list","target":"timers","translated_text":"what timers do I have","confidence":0.95}}

User: kill the kitchen lights
JSON: {{"mode":"translate","tool_group":"lighting","action":"turn_off","target":"Kitchen lights","translated_text":"turn off Kitchen lights","confidence":0.95}}

User: kill the teen room light
JSON: {{"mode":"translate","tool_group":"lighting","action":"turn_off","target":"Teen Room Light","translated_text":"turn off Teen Room Light","confidence":0.95}}

User: how do I boil eggs
JSON: {{"mode":"general","tool_group":"general","action":"unknown","target":null,"translated_text":null,"confidence":0.95}}

User: {utterance} /no_think
JSON:"""

MAX_TRANSLATION_CATALOG_LINES = 220

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
        catalog_lines = self._render_device_catalog(catalog)
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

        payload = getattr(result, "payload", {}) or {}
        action = self._normalize_action(payload.get("action"))
        target = payload.get("target") if isinstance(payload.get("target"), str) else None
        normalized_target = self._normalize_catalog_phrase(target)
        target_map = self._build_target_map(catalog)
        matched = target_map.get(normalized_target)
        if matched is None:
            result.valid = False
            result.notes = "invalid_target"
            result.canonical_text = None
            return result

        if action is None or action == "unknown":
            result.valid = False
            result.notes = "invalid_action"
            result.canonical_text = None
            return result

        if action not in matched["actions"]:
            result.valid = False
            result.notes = "unsupported_action_for_target"
            result.canonical_text = None
            return result

        result.tool_group = matched["tool_group"]
        result.canonical_text = self._canonical_text_from_action_target(action, matched["target"])
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
            payload=payload,
            valid=valid,
        )

    def _render_device_catalog(self, catalog: Catalog) -> list[str]:
        lines = [
            f"[{item['tool_group']}] {item['target']} | actions: {','.join(sorted(item['actions']))}"
            for item in self._build_target_map(catalog).values()
        ]
        return lines[:MAX_TRANSLATION_CATALOG_LINES]

    def _build_target_map(self, catalog: Catalog) -> dict[str, dict[str, Any]]:
        targets: dict[str, dict[str, Any]] = {}

        def add_target(name: str, tool_group: str, actions: set[str]) -> None:
            name = str(name or "").strip()
            if not name:
                return
            normalized = self._normalize_catalog_phrase(name)
            if not normalized or normalized in targets:
                return
            targets[normalized] = {
                "target": name,
                "tool_group": tool_group,
                "actions": {action for action in actions if action in VALID_TRANSLATION_ACTIONS and action != "unknown"},
            }

        # Virtual/common targets.
        add_target("timers", "timers", {"list", "query", "cancel", "start"})
        add_target("alarms", "timers", {"list", "query", "cancel", "start"})
        add_target("reminders", "timers", {"list", "query", "cancel", "start"})

        seen_area_groups: set[tuple[str, str]] = set()
        for entity in catalog.entity_targets:
            tool_group = self._infer_tool_group_from_domain(entity.domain)
            actions = self._actions_for_entity(entity)
            add_target(entity.name, tool_group, actions)

            area = str(getattr(entity, "area", "") or "").strip()
            if area and tool_group in {"lighting", "fan", "climate"}:
                area_key = (tool_group, area.casefold())
                if area_key not in seen_area_groups:
                    seen_area_groups.add(area_key)
                    if tool_group == "lighting":
                        add_target(f"{area} lights", "lighting", {"turn_on", "turn_off", "set", "query"})
                    elif tool_group == "fan":
                        add_target(f"{area} fan", "fan", {"turn_on", "turn_off", "set", "query"})
                    elif tool_group == "climate":
                        add_target(f"{area} temperature", "climate", {"set", "query"})

        for target in catalog.conversation_targets:
            for phrase in self._conversation_target_phrases(target):
                tool_group = self._infer_tool_group_from_phrase(phrase)
                action = self._infer_action_from_phrase(phrase)
                if action != "unknown":
                    add_target(phrase, tool_group, {action})

        return targets

    def _conversation_target_phrases(self, target) -> list[str]:
        """Return clean, user-sayable phrases for a conversation target."""
        phrases: list[str] = []
        canonical_phrase = getattr(target, "canonical_phrase", None)
        if isinstance(canonical_phrase, str) and self._is_user_sayable_phrase(canonical_phrase):
            phrases.append(canonical_phrase.strip())
        for phrase in getattr(target, "sample_phrases", []) or []:
            if isinstance(phrase, str) and self._is_user_sayable_phrase(phrase):
                phrases.append(phrase.strip())
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

    def _infer_action_from_phrase(self, phrase: str) -> str:
        normalized = normalize_text(phrase)
        tokens = set(tokenize(normalized))
        if tokens & {"cancel", "stop", "clear", "delete"}:
            return "cancel"
        if tokens & {"start", "set", "create", "begin"} and tokens & {"timer", "timers", "alarm", "alarms", "reminder", "reminders"}:
            return "start"
        if tokens & {"open", "show", "display"}:
            return "open"
        if tokens & {"close", "shut"}:
            return "close"
        if tokens & {"lock"}:
            return "lock"
        if tokens & {"unlock"}:
            return "unlock"
        if tokens & {"turn", "activate", "enable", "on"} and "off" not in tokens:
            return "turn_on"
        if tokens & {"off", "deactivate", "disable"}:
            return "turn_off"
        if tokens & {"what", "is", "are", "list", "show", "tell", "how", "when"}:
            if "list" in tokens:
                return "list"
            return "query"
        return "unknown"


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
