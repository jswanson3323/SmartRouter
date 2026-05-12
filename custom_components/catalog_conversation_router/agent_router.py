"""Main routing pipeline implementation."""


from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from time import perf_counter
from typing import Any

from .catalog import CatalogManager
from .ha_conversation_agents import get_registered_llm_agents
from .llm_adapter import LLMAdapter
from .matcher import FuzzyMatcher
from .models import (
    ActiveConversationState,
    EntityTarget,
    LocalAgentOutcome,
    ParsedStateQuery,
    ResolutionPath,
    ResolutionTrace,
    RouterConfig,
    RouterResult,
    StreamingFallbackRequest,
)
from .phrase_renderer import render_conversation_pattern
from .phonetics import normalize_text, tokenize
from .semantic_intent import (
    REQUEST_GENERAL_BYPASS_THRESHOLD,
    REQUEST_TOOL_HINT_THRESHOLD,
    SemanticRequestClassification,
)
from .safety import validate_fuzzy_execution

_LOGGER = logging.getLogger(__name__)
SUPER_AREA_LABEL_RE = re.compile(r"^superarea\s*:\s*(.+)$", re.IGNORECASE)
STATUS_QUERY_HINTS = (
    "status",
    "state",
    "what about",
    "tell me about",
    "can you tell me about",
    "what is",
    "whats",
    "is",
    "are",
    "temperature",
    "temp",
)
STATE_ENRICHMENT_INTRO = (
    "Router-resolved live state for this turn. "
    "Use this live state as the source of truth for the current answer. "
    "Answer only about the resolved target or targets below. "
    "Do not summarize unrelated rooms, devices, or the whole home. "
    "Do not say you cannot see the live state if router-resolved live state is provided."
)
TEMPERATURE_WORD_RE = re.compile(r"\btemp(?:e?r?a?t?u?r?e|urature|rature)?\b", re.IGNORECASE)
BINARY_STATE_ENDING_RE = re.compile(
    r"\b(on|off|open|opened|closed|close|locked|unlocked)\b\s*\??$",
    re.IGNORECASE,
)
DOMAIN_STATE_QUERY_RE = re.compile(
    r"^(what|which)\s+([a-z_ ]+?)\s+are\s+(on|off|open|opened|closed|close|locked|unlocked)\s+(?:in|at|of)\s+(.+?)\??$",
    re.IGNORECASE,
)
HA_QUERY_OPENERS = (
    "what is",
    "whats",
    "tell me",
    "how is",
    "how s",
)
LOCAL_ACTION_PREFIXES = (
    "turn on",
    "turn off",
    "switch on",
    "switch off",
    "open",
    "close",
    "lock",
    "unlock",
    "set",
    "activate",
    "deactivate",
)
CONTROL_LIKE_VERB_TOKENS = {
    "activate",
    "adjust",
    "arm",
    "blast",
    "cancel",
    "change",
    "clear",
    "close",
    "deactivate",
    "delete",
    "disable",
    "disarm",
    "enable",
    "kill",
    "lock",
    "open",
    "raise",
    "set",
    "shut",
    "start",
    "stop",
    "switch",
    "toggle",
    "turn",
    "unlock",
}
ACTIVE_CONVERSATION_TTL = timedelta(hours=6)


def _starts_with_supported_local_action(text: str) -> bool:
    """Return true only when text starts with a deterministic local action phrase."""
    normalized = normalize_text(text)
    return any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in LOCAL_ACTION_PREFIXES
    )


def _looks_like_unknown_local_control_request(text: str) -> bool:
    """Detect control-like utterances whose leading verb is not locally supported."""
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    if not tokens:
        return False
    if _starts_with_supported_local_action(normalized):
        return False
    return tokens[0] in CONTROL_LIKE_VERB_TOKENS


MAX_ACTIVE_CONVERSATIONS = 256
OPEN_DOMAIN_PREFIXES = (
    "how do i",
    "how do you",
    "how to",
    "why is",
    "why are",
    "what is",
    "what are",
    "who is",
    "who are",
    "when is",
    "where is",
    "tell me about",
    "give me",
    "explain",
    "describe",
    "write",
    "summarize",
    "help me",
)
SMART_HOME_HINT_TOKENS = {
    "light",
    "lights",
    "lamp",
    "fan",
    "switch",
    "thermostat",
    "temperature",
    "temp",
    "climate",
    "door",
    "garage",
    "lock",
    "cover",
    "speaker",
    "tv",
    "scene",
    "script",
    "alarm",
    "vacuum",
    "kitchen",
    "office",
    "bedroom",
    "guest",
    "great",
    "garage",
    "laundry",
    "gym",
    "pool",
    "spa",
}


class AgentRouter:
    """Coordinates local-first routing and fallback chain."""

    def __init__(
        self,
        *,
        config: RouterConfig,
        catalog_manager: CatalogManager,
        matcher: FuzzyMatcher,
        agent_adapter,
        llm_adapter: LLMAdapter,
        hass: Any,
        router_agent_id: str | None = None,
    ) -> None:
        self._config = config
        self._catalog = catalog_manager
        self._matcher = matcher
        self._agent_adapter = agent_adapter
        self._llm_adapter = llm_adapter
        self._hass = hass
        self._router_agent_id = router_agent_id
        self._active_conversations: dict[str, ActiveConversationState] = {}

    async def async_route(
        self,
        *,
        text: str,
        language: str,
        conversation_id: str | None,
        context: Any,
        dry_run: bool = False,
        debug_collect_all: bool = False,
        execute_debug_branches: bool = False,
        allow_streaming_llm_fallback: bool = False,
        origin_area: str | None = None,
        device_id: str | None = None,
        satellite_id: str | None = None,
        extra_system_prompt: str | None = None,
    ) -> RouterResult:
        """Run deterministic routing pipeline with bounded attempts."""
        route_started = perf_counter()
        self._prune_active_conversations()
        _LOGGER.debug("ROUTER START: text=%r, device_id=%r, satellite_id=%r", text, device_id, satellite_id)
        catalog = self._catalog.get_catalog()
        resolved_origin_area = origin_area or self._resolve_origin_area(device_id=device_id, satellite_id=satellite_id, context=context)
        resolved_origin_super_area = self._resolve_origin_super_area(
            origin_area=resolved_origin_area,
            device_id=device_id,
            satellite_id=satellite_id,
            context=context,
        )
        trace = ResolutionTrace(
            original_utterance=text,
            normalized_utterance=normalize_text(text),
            catalog_revision=catalog.metadata.revision,
            origin_area=resolved_origin_area,
            origin_super_area=resolved_origin_super_area,
        )
        selected_path: ResolutionPath | None = None
        selected_outcome = None

        def _should_execute_branch() -> bool:
            return (not dry_run) or execute_debug_branches

        def _result(
            *,
            path: ResolutionPath,
            outcome,
            streaming_request: StreamingFallbackRequest | None = None,
        ) -> RouterResult:
            trace.route_duration_ms = round((perf_counter() - route_started) * 1000, 3)
            return RouterResult(
                path=path,
                outcome=outcome,
                trace=trace,
                streaming_request=streaming_request,
            )

        def _record_outcome(prefix: str, outcome) -> None:
            executed = outcome is not None
            setattr(trace, f"{prefix}_executed", executed)
            if outcome is None:
                return
            setattr(trace, f"{prefix}_outcome", "success" if outcome.success else "failed")
            setattr(trace, f"{prefix}_response_text", outcome.response_text)
            setattr(trace, f"{prefix}_response_type", outcome.response_type)
            setattr(trace, f"{prefix}_error_code", outcome.error_code)
            setattr(trace, f"{prefix}_processed_locally", outcome.processed_locally)
            setattr(trace, f"{prefix}_conversation_id", outcome.conversation_id)
            setattr(trace, f"{prefix}_continue_conversation", outcome.continue_conversation)

        active_state = self._active_conversations.get(conversation_id) if conversation_id else None
        if active_state is not None:
            trace.active_conversation_executor = active_state.executor_type
            trace.active_conversation_agent_id = active_state.agent_id
            trace.downstream_conversation_id = active_state.downstream_conversation_id
            trace.continuation_routed_directly = True

        if active_state is not None and active_state.executor_type == "llm":
            state_result = await self._try_local_state_query(
                text=text,
                language=language,
                conversation_id=conversation_id,
                context=context,
                trace=trace,
                catalog=catalog,
                origin_area=resolved_origin_area,
                origin_super_area=resolved_origin_super_area,
                device_id=device_id,
                satellite_id=satellite_id,
                extra_system_prompt=extra_system_prompt,
                preserve_llm_owner=True,
            )
            if state_result is not None:
                trace.state_query_intercepted_during_llm = True
                trace.selected_path = ResolutionPath.LOCAL_STATE_QUERY
                trace.final_executor = "local"
                trace.route_duration_ms = round((perf_counter() - route_started) * 1000, 3)
                return state_result
            action_result = await self._try_local_action_during_llm(
                text=text,
                language=language,
                context=context,
                trace=trace,
                catalog=catalog,
                origin_area=resolved_origin_area,
                origin_super_area=resolved_origin_super_area,
                device_id=device_id,
                satellite_id=satellite_id,
                extra_system_prompt=extra_system_prompt,
            )
            if action_result is not None:
                trace.local_action_intercepted_during_llm = True
                self._update_conversation_tracking(
                    conversation_id,
                    action_result.outcome,
                    executor="local",
                    agent_id=self._config.local_agent_id,
                )
                trace.selected_path = action_result.path
                trace.final_executor = "local"
                trace.route_duration_ms = round((perf_counter() - route_started) * 1000, 3)
                return action_result
            local_intent_result = await self._try_local_intent_during_llm(
                text=text,
                language=language,
                conversation_id=conversation_id,
                context=context,
                trace=trace,
                catalog=catalog,
                origin_area=resolved_origin_area,
                origin_super_area=resolved_origin_super_area,
                device_id=device_id,
                satellite_id=satellite_id,
                extra_system_prompt=extra_system_prompt,
            )
            if local_intent_result is not None:
                trace.selected_path = local_intent_result.path
                trace.final_executor = "local"
                trace.route_duration_ms = round((perf_counter() - route_started) * 1000, 3)
                return local_intent_result
            trace.assist_pipeline_input = text
            trace.rendered_from_pattern = False
            trace.rendered_slots = {}
            trace.llm_translation_summary = {
                "mode": "skipped",
                "confidence": 0.0,
                "tool_group": None,
                "valid": False,
                "notes": "active_llm_conversation",
            }
            trace.llm_translated_local_executed = False
            llm_system_prompt = self._build_llm_fallback_system_prompt(
                trace=trace,
                text=text,
                catalog=catalog,
                origin_area=resolved_origin_area,
                origin_super_area=resolved_origin_super_area,
                extra_system_prompt=extra_system_prompt,
            )
            if self._config.llm_fallback_enabled:
                if _should_execute_branch():
                    runtime_llm_agent_id = self._resolve_runtime_llm_agent_id(
                        active_state.agent_id
                    )
                    if allow_streaming_llm_fallback:
                        trace.llm_fallback_stream_attempted = True
                        trace.llm_fallback_stream_supported = None
                        trace.llm_fallback_stream_used = None
                        trace.llm_fallback_stream_fallback_reason = None
                        return _result(
                            path=ResolutionPath.LLM_FALLBACK,
                            outcome=self._empty_outcome(),
                            streaming_request=StreamingFallbackRequest(
                                llm_agent_id=runtime_llm_agent_id,
                                utterance=text,
                                language=language,
                                conversation_id=active_state.downstream_conversation_id,
                                context=context,
                                device_id=device_id,
                                satellite_id=satellite_id,
                                extra_system_prompt=llm_system_prompt,
                            ),
                        )
                    llm_fallback_started = perf_counter()
                    llm_outcome = await self._llm_adapter.async_final_fallback(
                        llm_agent_id=runtime_llm_agent_id,
                        utterance=text,
                        language=language,
                        conversation_id=active_state.downstream_conversation_id,
                        context=context,
                        device_id=device_id,
                        satellite_id=satellite_id,
                        extra_system_prompt=llm_system_prompt,
                    )
                    trace.llm_fallback_duration_ms = round(
                        (perf_counter() - llm_fallback_started) * 1000,
                        3,
                    )
                else:
                    llm_outcome = None
                _record_outcome("llm_fallback", llm_outcome)
                self._update_conversation_tracking(
                    conversation_id,
                    llm_outcome,
                    executor="llm",
                    agent_id=runtime_llm_agent_id if llm_outcome is not None else active_state.agent_id,
                )
                trace.selected_path = ResolutionPath.LLM_FALLBACK
                trace.final_executor = "llm"
                return _result(
                    path=ResolutionPath.LLM_FALLBACK,
                    outcome=llm_outcome,
                )
        if active_state is not None and active_state.executor_type == "local":
            trace.assist_pipeline_input = text
            trace.rendered_from_pattern = False
            trace.rendered_slots = {}
            trace.llm_translation_summary = {
                "mode": "skipped",
                "confidence": 0.0,
                "tool_group": None,
                "valid": False,
                "notes": "active_local_conversation",
            }
            trace.llm_translated_local_executed = False
            if _should_execute_branch():
                local_outcome = await self._agent_adapter.async_process(
                    agent_id=active_state.agent_id,
                    text=text,
                    language=language,
                    conversation_id=active_state.downstream_conversation_id,
                    context=context,
                    device_id=device_id,
                    satellite_id=satellite_id,
                    extra_system_prompt=extra_system_prompt,
                )
            else:
                local_outcome = None
            _record_outcome("exact_local", local_outcome)
            self._update_conversation_tracking(
                conversation_id,
                local_outcome,
                executor="local",
                agent_id=active_state.agent_id,
            )
            trace.selected_path = ResolutionPath.EXACT_LOCAL
            trace.final_executor = "local"
            trace.route_duration_ms = round((perf_counter() - route_started) * 1000, 3)
            return _result(
                path=ResolutionPath.EXACT_LOCAL,
                outcome=local_outcome,
            )

        state_result = await self._try_local_state_query(
            text=text,
            language=language,
            conversation_id=conversation_id,
            context=context,
            trace=trace,
            catalog=catalog,
            origin_area=resolved_origin_area,
            origin_super_area=resolved_origin_super_area,
            device_id=device_id,
            satellite_id=satellite_id,
            extra_system_prompt=extra_system_prompt,
            preserve_llm_owner=False,
        )
        if state_result is not None:
            trace.selected_path = ResolutionPath.LOCAL_STATE_QUERY
            trace.final_executor = "local"
            trace.route_duration_ms = round((perf_counter() - route_started) * 1000, 3)
            return state_result

        fuzzy_started = perf_counter()
        match_result = self._matcher.match(
            text,
            catalog,
            origin_area=resolved_origin_area,
            origin_super_area=resolved_origin_super_area,
        )
        trace.fuzzy_match_duration_ms = round((perf_counter() - fuzzy_started) * 1000, 3)
        _LOGGER.debug(
            "FUZZY MATCH: best=%s score=%s",
            getattr(match_result.best, "canonical_phrase", None),
            getattr(match_result.best, "score", None),
        )
        trace.normalized_utterance = match_result.normalized_utterance
        trace.effective_area_hint = match_result.effective_area_hint
        trace.effective_super_area_hint = match_result.effective_super_area_hint
        unknown_local_control_request = _looks_like_unknown_local_control_request(text)

        # 1) fuzzy attempt
        if self._config.fuzzy_enabled and match_result.best is not None and not unknown_local_control_request:
            trace.top_fuzzy_candidates = [
                {
                    "candidate_id": c.candidate_id,
                    "candidate_type": c.candidate_type.value,
                    "target_name": c.target_name,
                    "canonical_phrase": c.canonical_phrase,
                    "score": round(c.score, 4),
                    "detail": c.detail,
                }
                for c in match_result.top_candidates
            ]
            second = match_result.top_candidates[1].score if len(match_result.top_candidates) > 1 else 0.0
            decision = validate_fuzzy_execution(
                inferred_action=match_result.inferred_action,
                candidate_action=match_result.best.action,
                canonical_phrase=match_result.best.canonical_phrase,
                best_score=match_result.best.score,
                second_score=second,
                fuzzy_threshold=self._config.fuzzy_threshold,
                ambiguity_gap=self._config.ambiguity_gap,
                high_risk_threshold=self._config.high_risk_threshold,
            )
            trace.fuzzy_decision = {
                "allowed": decision.allowed,
                "reason": decision.reason,
                "risk_tier": decision.risk_tier.value,
                "best_score": match_result.best.score,
                "second_score": second,
                "threshold": self._config.fuzzy_threshold,
                "ambiguity_gap": self._config.ambiguity_gap,
            }

            if decision.allowed:
                trace.chosen_canonical_phrase = match_result.best.canonical_phrase
                candidate_detail = match_result.best.detail or {}
                trace.matched_sample_phrase_raw = candidate_detail.get("matched_sample_phrase_raw")
                trace.matched_sample_phrase_normalized_for_scoring = candidate_detail.get(
                    "matched_sample_phrase_normalized_for_scoring"
                )

                assist_input = match_result.best.canonical_phrase
                trace.rendered_from_pattern = False
                trace.rendered_slots = {}
                skip_fuzzy_execution = False

                if match_result.best.candidate_type.value == "conversation_target":
                    matched_pattern = (
                        trace.matched_sample_phrase_raw
                        or match_result.best.canonical_phrase
                    )
                    rendered = render_conversation_pattern(
                        original_utterance=text,
                        pattern=matched_pattern,
                    )
                    assist_input = rendered.text or normalize_text(text)
                    trace.rendered_from_pattern = rendered.rendered_from_pattern
                    trace.rendered_slots = rendered.slots
                    if self._rendered_pattern_has_empty_slots(rendered.slots):
                        skip_fuzzy_execution = True
                        trace.fuzzy_decision["allowed"] = False
                        trace.fuzzy_decision["reason"] = "missing_required_slots"
                        trace.fuzzy_decision["missing_slots"] = [
                            slot_name
                            for slot_name, value in rendered.slots.items()
                            if not value.strip()
                        ]
                        _LOGGER.debug(
                            "Skipping fuzzy conversation-target execution due to empty rendered slots: %s",
                            trace.fuzzy_decision["missing_slots"],
                        )

                trace.assist_pipeline_input = assist_input
                if skip_fuzzy_execution:
                    _LOGGER.debug(
                        "Fuzzy candidate rejected after render validation: %s",
                        trace.fuzzy_decision["reason"],
                    )
                else:
                    _LOGGER.debug("ROUTER PATH: FUZZY_LOCAL → calling adapter with input=%r", assist_input)
                    if _should_execute_branch():
                        fuzzy_outcome = await self._agent_adapter.async_process(
                            agent_id=self._config.local_agent_id,
                            text=assist_input,
                            language=language,
                            conversation_id=conversation_id,
                            context=context,
                            device_id=device_id,
                            satellite_id=satellite_id,
                        )
                    else:
                        fuzzy_outcome = None
                    _record_outcome("fuzzy_local", fuzzy_outcome)
                    self._update_conversation_tracking(
                        conversation_id,
                        fuzzy_outcome,
                        executor="local",
                        agent_id=self._config.local_agent_id,
                    )

                    if dry_run or (fuzzy_outcome is not None and fuzzy_outcome.success):
                        _LOGGER.debug("ROUTER DECISION: FUZZY_LOCAL")
                        if selected_path is None:
                            selected_path = ResolutionPath.FUZZY_LOCAL
                            selected_outcome = fuzzy_outcome
                            trace.selected_path = ResolutionPath.FUZZY_LOCAL
                            trace.final_executor = "local"
                        if not debug_collect_all:
                            return _result(
                                path=ResolutionPath.FUZZY_LOCAL,
                                outcome=fuzzy_outcome,
                            )
            else:
                candidate_detail = match_result.best.detail or {}
                forced_area_resolution = (
                    decision.reason == "ambiguity_gap_too_small"
                    and (
                        candidate_detail.get("area_scoped_domain_resolution") == 1.0
                        or candidate_detail.get("super_area_scoped_domain_resolution") == 1.0
                    )
                )
                trace.fuzzy_decision["forced_area_resolution"] = forced_area_resolution
                if forced_area_resolution:
                    trace.chosen_canonical_phrase = match_result.best.canonical_phrase
                    trace.assist_pipeline_input = match_result.best.canonical_phrase
                    trace.rendered_from_pattern = False
                    trace.rendered_slots = {}
                    _LOGGER.debug("ROUTER PATH: FUZZY_LOCAL (forced area) → calling adapter with input=%r", match_result.best.canonical_phrase)
                    if _should_execute_branch():
                        fuzzy_outcome = await self._agent_adapter.async_process(
                            agent_id=self._config.local_agent_id,
                            text=match_result.best.canonical_phrase,
                            language=language,
                            conversation_id=conversation_id,
                            context=context,
                            device_id=device_id,
                            satellite_id=satellite_id,
                        )
                    else:
                        fuzzy_outcome = None
                    _record_outcome("fuzzy_local", fuzzy_outcome)
                    self._update_conversation_tracking(
                        conversation_id,
                        fuzzy_outcome,
                        executor="local",
                        agent_id=self._config.local_agent_id,
                    )

                    if dry_run or (fuzzy_outcome is not None and fuzzy_outcome.success):
                        _LOGGER.debug("ROUTER DECISION: FUZZY_LOCAL")
                        if selected_path is None:
                            selected_path = ResolutionPath.FUZZY_LOCAL
                            selected_outcome = fuzzy_outcome
                            trace.selected_path = ResolutionPath.FUZZY_LOCAL
                            trace.final_executor = "local"
                        if not debug_collect_all:
                            return _result(
                                path=ResolutionPath.FUZZY_LOCAL,
                                outcome=fuzzy_outcome,
                            )
                _LOGGER.debug("Fuzzy candidate rejected: %s", decision.reason)

        if unknown_local_control_request:
            trace.fuzzy_decision = {
                "allowed": False,
                "reason": "unknown_local_action",
                "risk_tier": "high",
                "best_score": getattr(match_result.best, "score", 0.0),
                "threshold": self._config.fuzzy_threshold,
                "ambiguity_gap": self._config.ambiguity_gap,
            }
            _LOGGER.debug("Fuzzy candidate rejected before execution: unknown local action")

        # 2) LLM translation to local
        semantic_request_classification = await self._llm_adapter.async_classify_request(
            utterance=text,
            catalog=catalog,
        )
        trace.semantic_request_classification_available = (
            semantic_request_classification is not None
        )
        if semantic_request_classification is not None:
            trace.semantic_request_classification_kind = (
                semantic_request_classification.kind
            )
            trace.semantic_request_classification_confidence = round(
                semantic_request_classification.confidence,
                4,
            )
            trace.semantic_request_classification_intent_family = (
                semantic_request_classification.intent_family
            )
            trace.semantic_request_classification_reason = (
                semantic_request_classification.reason
            )
            trace.semantic_request_classification_debug = (
                semantic_request_classification.debug
            )
            direct_llm_only = bool(
                semantic_request_classification.kind == "general_request"
                and semantic_request_classification.confidence
                >= REQUEST_GENERAL_BYPASS_THRESHOLD
            )
            trace.semantic_request_routing_source = (
                "semantic_classifier" if direct_llm_only else "semantic_classifier_advisory"
            )
        else:
            direct_llm_only = self._should_bypass_local_pipeline_for_llm(
                text=text,
                catalog=catalog,
            )
            trace.semantic_request_routing_source = (
                "heuristic_bypass" if direct_llm_only else "heuristic_fallback"
            )
        should_run_llm_translation = (
            self._config.llm_translate_enabled
            and selected_path is None
            and not direct_llm_only
        )
        fuzzy_found_good_match = selected_path == ResolutionPath.FUZZY_LOCAL
        translation_found_good_match = False
        if not should_run_llm_translation:
            trace.llm_translation_summary = {
                "mode": "skipped",
                "confidence": 0.0,
                "tool_group": None,
                "valid": False,
                "notes": (
                    "skipped_due_to_fuzzy_match"
                    if selected_path == ResolutionPath.FUZZY_LOCAL
                    else "semantic_general_request_bypass"
                    if direct_llm_only
                    and trace.semantic_request_routing_source == "semantic_classifier"
                    else "direct_llm_only"
                    if direct_llm_only
                    else "skipped"
                ),
                "debug": {
                    "semantic_request_routing_source": trace.semantic_request_routing_source,
                    "semantic_request_classification_kind": trace.semantic_request_classification_kind,
                    "semantic_request_classification_confidence": trace.semantic_request_classification_confidence,
                    "semantic_request_classification_intent_family": trace.semantic_request_classification_intent_family,
                },
            }
            trace.llm_translated_local_executed = False
        if should_run_llm_translation:
            runtime_llm_agent_id = self._resolve_runtime_llm_agent_id(
                self._config.translate_llm_agent_id
            )
            translation_started = perf_counter()
            translation = await self._llm_adapter.async_translate_for_local(
                llm_agent_id=runtime_llm_agent_id,
                utterance=text,
                language=language,
                catalog=catalog,
                max_candidates=self._config.max_llm_candidates,
                conversation_id=conversation_id,
                context=context,
                origin_area=resolved_origin_area,
                origin_super_area=resolved_origin_super_area,
                preserve_raw_text=debug_collect_all,
            )
            trace.llm_translation_duration_ms = round(
                (perf_counter() - translation_started) * 1000,
                3,
            )
            _LOGGER.debug(
                "LLM TRANSLATION: valid=%s canonical=%s mode=%s",
                translation.valid,
                translation.canonical_text,
                translation.mode,
            )
            inferred_tool_group = translation.notes == "tool_group_inferred"
            trace.llm_translation_summary = {
                "mode": translation.mode,
                "tool_group": translation.tool_group,
                "confidence": translation.confidence,
                "valid": translation.valid,
                "notes": translation.notes,
                "source": translation.source,
                "intent_family": translation.intent_family,
                "confidence_reason": translation.confidence_reason,
                "debug": translation.debug,
            }
            trace.llm_translation_raw_text = translation.raw_text
            trace.llm_translation_tool_group = translation.tool_group
            trace.llm_translation_catalog_match = (
                translation.mode == "translate"
                and translation.canonical_text is not None
                and translation.valid
            )
            trace.llm_translation_tool_group_inferred = inferred_tool_group
            translation_found_good_match = bool(
                translation.valid and translation.canonical_text
            )

            if translation.valid and translation.canonical_text:
                if selected_path is None:
                    trace.chosen_canonical_phrase = translation.canonical_text
                    trace.assist_pipeline_input = translation.canonical_text
                    trace.rendered_from_pattern = False
                    trace.rendered_slots = {}
                _LOGGER.debug("ROUTER PATH: LLM_TRANSLATED_LOCAL → calling adapter with input=%r", translation.canonical_text)
                if _should_execute_branch():
                    translated_outcome = await self._agent_adapter.async_process(
                        agent_id=self._config.local_agent_id,
                        text=translation.canonical_text,
                        language=language,
                        conversation_id=conversation_id,
                        context=context,
                        device_id=device_id,
                        satellite_id=satellite_id,
                    )
                else:
                    translated_outcome = None
                _record_outcome("llm_translated_local", translated_outcome)
                self._update_conversation_tracking(
                    conversation_id,
                    translated_outcome,
                    executor="local",
                    agent_id=self._config.local_agent_id,
                )

                if dry_run or (translated_outcome is not None and translated_outcome.success):
                    _LOGGER.debug("ROUTER DECISION: LLM_TRANSLATED_LOCAL")
                    if selected_path is None:
                        selected_path = ResolutionPath.LLM_TRANSLATED_LOCAL
                        selected_outcome = translated_outcome
                        trace.selected_path = ResolutionPath.LLM_TRANSLATED_LOCAL
                        trace.final_executor = "local"
                    if not debug_collect_all:
                        return _result(
                            path=ResolutionPath.LLM_TRANSLATED_LOCAL,
                            outcome=translated_outcome,
                        )

        # 3) raw local attempt
        exact = None
        if direct_llm_only:
            trace.exact_local_executed = False
            trace.exact_local_outcome = "skipped"
        elif unknown_local_control_request:
            trace.exact_local_executed = False
            trace.exact_local_outcome = "skipped"
            trace.exact_local_error_code = "unknown_local_action"
            _LOGGER.debug("EXACT_LOCAL skipped: unknown local action should route through LLM translation/fallback")
        else:
            _LOGGER.debug("ROUTER PATH: EXACT_LOCAL → calling adapter with input=%r", text)
            if _should_execute_branch():
                exact = await self._agent_adapter.async_process(
                    agent_id=self._config.local_agent_id,
                    text=text,
                    language=language,
                    conversation_id=conversation_id,
                    context=context,
                    device_id=device_id,
                    satellite_id=satellite_id,
                )
            if exact is not None:
                _LOGGER.debug(
                    "EXACT LOCAL: success=%s response=%r processed_locally=%s",
                    exact.success,
                    exact.response_text,
                    exact.processed_locally,
                )
                trace.exact_local_executed = True
                trace.exact_local_outcome = "success" if exact.success else "failed"
                trace.exact_local_response_text = exact.response_text
                trace.exact_local_response_type = exact.response_type
                trace.exact_local_error_code = exact.error_code
                trace.exact_local_processed_locally = exact.processed_locally
                trace.failure_category = (
                    exact.failure_category.value if exact.failure_category is not None else None
                )
                self._update_conversation_tracking(
                    conversation_id,
                    exact,
                    executor="local",
                    agent_id=self._config.local_agent_id,
                )
            else:
                trace.exact_local_executed = False
                trace.exact_local_outcome = "skipped"

            if exact is not None and exact.success and exact.processed_locally is True:
                _LOGGER.debug("ROUTER DECISION: EXACT_LOCAL")
                if selected_path is None:
                    trace.assist_pipeline_input = text
                    trace.rendered_from_pattern = False
                    trace.rendered_slots = {}
                    trace.selected_path = ResolutionPath.EXACT_LOCAL
                    trace.final_executor = "local"
                    selected_path = ResolutionPath.EXACT_LOCAL
                    selected_outcome = exact
                if not debug_collect_all:
                    return _result(path=ResolutionPath.EXACT_LOCAL, outcome=exact)

            if exact is not None and exact.success and exact.processed_locally is not True:
                _LOGGER.debug(
                    "EXACT LOCAL rejected: success=%s processed_locally=%s response=%r",
                    exact.success,
                    exact.processed_locally,
                    exact.response_text,
                )

        allow_final_llm_fallback = bool(
            direct_llm_only
            or (
                selected_path is None
                and not fuzzy_found_good_match
                and not translation_found_good_match
            )
        )
        if not allow_final_llm_fallback:
            trace.llm_fallback_executed = False
            trace.llm_fallback_outcome = "skipped"
            trace.llm_fallback_error_code = "not_allowed_after_local_match"
        # 4) final direct llm fallback
        if self._config.llm_fallback_enabled and allow_final_llm_fallback:
            if _should_execute_branch():
                _LOGGER.debug("ROUTER DECISION: LLM_FALLBACK (calling LLM)")
                runtime_llm_agent_id = self._resolve_runtime_llm_agent_id(
                    self._config.llm_agent_id
                )
                llm_system_prompt = self._build_llm_fallback_system_prompt(
                    trace=trace,
                    text=text,
                    catalog=catalog,
                    origin_area=resolved_origin_area,
                    origin_super_area=resolved_origin_super_area,
                    extra_system_prompt=extra_system_prompt,
                    semantic_request_classification=semantic_request_classification,
                )
                fallback_conversation_id = self._resolve_initial_llm_fallback_conversation_id()
                if allow_streaming_llm_fallback:
                    trace.llm_fallback_stream_attempted = True
                    trace.llm_fallback_stream_supported = None
                    trace.llm_fallback_stream_used = None
                    trace.llm_fallback_stream_fallback_reason = None
                    return _result(
                        path=ResolutionPath.LLM_FALLBACK,
                        outcome=self._empty_outcome(),
                        streaming_request=StreamingFallbackRequest(
                            llm_agent_id=runtime_llm_agent_id,
                            utterance=text,
                            language=language,
                            conversation_id=fallback_conversation_id,
                            context=context,
                            device_id=device_id,
                            satellite_id=satellite_id,
                            extra_system_prompt=llm_system_prompt,
                        ),
                    )
                llm_fallback_started = perf_counter()
                llm_outcome = await self._llm_adapter.async_final_fallback(
                    llm_agent_id=runtime_llm_agent_id,
                    utterance=text,
                    language=language,
                    conversation_id=fallback_conversation_id,
                    context=context,
                    device_id=device_id,
                    satellite_id=satellite_id,
                    extra_system_prompt=llm_system_prompt,
                )
                trace.llm_fallback_duration_ms = round(
                    (perf_counter() - llm_fallback_started) * 1000,
                    3,
                )
            else:
                llm_outcome = None
            _record_outcome("llm_fallback", llm_outcome)
            self._update_conversation_tracking(
                conversation_id,
                llm_outcome,
                executor="llm",
                agent_id=runtime_llm_agent_id if llm_outcome is not None else self._config.llm_agent_id,
            )
            _LOGGER.debug(
                "LLM FALLBACK RESPONSE: %r",
                getattr(llm_outcome, "response_text", None) if llm_outcome is not None else None,
            )
            if selected_path is None:
                trace.selected_path = ResolutionPath.LLM_FALLBACK
                trace.final_executor = "llm"
                selected_path = ResolutionPath.LLM_FALLBACK
                selected_outcome = llm_outcome
            if not debug_collect_all:
                return _result(path=ResolutionPath.LLM_FALLBACK, outcome=llm_outcome)

        if selected_path is None:
            _LOGGER.warning(
                "Router failed to resolve utterance locally or through fallback: text=%r origin_area=%r origin_super_area=%r",
                text,
                resolved_origin_area,
                resolved_origin_super_area,
            )
            trace.selected_path = ResolutionPath.FAILED
            trace.final_executor = "none"
            return _result(path=ResolutionPath.FAILED, outcome=exact)

        return _result(path=selected_path, outcome=selected_outcome)

    def _empty_outcome(self):
        """Return a placeholder outcome when execution is deferred."""
        return LocalAgentOutcome(
            success=False,
            response=None,
            response_text=None,
            failure_category=None,
            raw=None,
        )

    def finalize_streaming_fallback(
        self,
        *,
        conversation_id: str | None,
        outcome: LocalAgentOutcome,
        agent_id: str,
    ) -> None:
        """Update continuation tracking after a streamed fallback completes."""
        self._update_conversation_tracking(
            conversation_id,
            outcome,
            executor="llm",
            agent_id=agent_id,
        )

    def _update_conversation_tracking(
        self,
        conversation_id: str | None,
        outcome: Any,
        *,
        executor: str,
        agent_id: str,
    ) -> None:
        """Track whether a conversation should continue and where it should return."""
        if not conversation_id:
            return

        should_continue = getattr(outcome, "continue_conversation", None) is True
        if should_continue:
            downstream_conversation_id = getattr(outcome, "conversation_id", None) or conversation_id
            state = self._active_conversations.get(conversation_id)
            started_at = state.started_at if state is not None else ActiveConversationState(
                outer_conversation_id=conversation_id,
                executor_type=executor,
                agent_id=agent_id,
                downstream_conversation_id=downstream_conversation_id,
            ).started_at
            self._active_conversations[conversation_id] = ActiveConversationState(
                outer_conversation_id=conversation_id,
                executor_type=executor,
                agent_id=agent_id,
                downstream_conversation_id=downstream_conversation_id,
                started_at=started_at,
            )
            self._enforce_active_conversation_limit()
        else:
            self._active_conversations.pop(conversation_id, None)

    def _resolve_runtime_llm_agent_id(self, configured_agent_id: str) -> str:
        """Resolve the best callable LLM agent id at request time."""
        descriptors = [
            descriptor
            for descriptor in get_registered_llm_agents(self._hass)
            if descriptor.agent_id != self._router_agent_id
        ]
        available_ids = {descriptor.agent_id for descriptor in descriptors}
        if configured_agent_id == self._router_agent_id:
            _LOGGER.error(
                "Configured llm_agent_id %s resolves to the router itself; refusing recursive fallback",
                configured_agent_id,
            )
            if len(descriptors) == 1:
                fallback = descriptors[0].agent_id
                _LOGGER.warning(
                    "Using the only non-router LLM agent %s instead of recursive router agent %s",
                    fallback,
                    configured_agent_id,
                )
                return fallback
            return configured_agent_id
        if configured_agent_id in available_ids:
            return configured_agent_id
        if len(descriptors) == 1:
            fallback = descriptors[0].agent_id
            _LOGGER.info(
                "Configured llm_agent_id %s is not currently callable; using runtime fallback agent %s",
                configured_agent_id,
                fallback,
            )
            return fallback
        return configured_agent_id

    def _rendered_pattern_has_empty_slots(self, slots: dict[str, str]) -> bool:
        """Return True when a rendered conversation pattern left any slot blank."""
        return any(not value.strip() for value in slots.values())

    def _prune_active_conversations(self) -> None:
        """Drop stale active continuation state."""
        now = datetime.now(tz=UTC)
        stale_ids = [
            conversation_id
            for conversation_id, state in self._active_conversations.items()
            if self._is_state_stale(state, now)
        ]
        for conversation_id in stale_ids:
            self._active_conversations.pop(conversation_id, None)

    def _is_state_stale(self, state: ActiveConversationState, now: datetime) -> bool:
        """Return True when continuation state is older than the TTL."""
        try:
            last_updated = datetime.fromisoformat(state.last_updated_at)
        except Exception:
            return True
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=UTC)
        return now - last_updated > ACTIVE_CONVERSATION_TTL

    def _enforce_active_conversation_limit(self) -> None:
        """Bound the active continuation map size."""
        overflow = len(self._active_conversations) - MAX_ACTIVE_CONVERSATIONS
        if overflow <= 0:
            return
        sorted_items = sorted(
            self._active_conversations.items(),
            key=lambda item: item[1].last_updated_at,
        )
        for conversation_id, _ in sorted_items[:overflow]:
            self._active_conversations.pop(conversation_id, None)

    def _should_bypass_local_pipeline_for_llm(self, *, text: str, catalog) -> bool:
        """Return True when an utterance looks open-domain and should skip translation/local probes."""
        normalized = normalize_text(text)
        if not normalized:
            return False
        if self._parse_state_query(text) is not None:
            return False
        if any(normalized.startswith(prefix) for prefix in LOCAL_ACTION_PREFIXES):
            return False

        utterance_tokens = set(tokenize(normalized))
        has_smart_home_hint = bool(SMART_HOME_HINT_TOKENS & utterance_tokens)
        if has_smart_home_hint:
            return False

        # Obvious open-domain voice requests should not pay for LLM translation
        # or raw local Assist probing. Let them hit the streaming LLM fallback
        # immediately. Home/status/action requests are excluded above.
        if any(normalized.startswith(prefix) for prefix in OPEN_DOMAIN_PREFIXES):
            return True

        for entity in catalog.entity_targets:
            if utterance_tokens & set(entity.tokens):
                return False
            if entity.area and utterance_tokens & set(tokenize(entity.area)):
                return False
        for target in catalog.conversation_targets:
            if utterance_tokens & set(target.tokens):
                return False

        return len(utterance_tokens) >= 4

    async def _try_local_state_query(
        self,
        *,
        text: str,
        language: str,
        conversation_id: str | None,
        context: Any,
        trace: ResolutionTrace,
        catalog,
        origin_area: str | None,
        origin_super_area: str | None,
        device_id: str | None,
        satellite_id: str | None,
        extra_system_prompt: str | None,
        preserve_llm_owner: bool,
    ) -> RouterResult | None:
        parsed = self._parse_state_query(text)
        trace.state_query_detected = parsed is not None
        if parsed is None:
            return None

        trace.state_query_kind = parsed.query_kind
        trace.state_query_target_phrase = parsed.target_phrase
        trace.state_query_requested_state = parsed.requested_state

        if parsed.query_kind == "domain_state":
            canonical_text = self._render_local_state_query(parsed=parsed, resolved=None)
            if not canonical_text:
                return None
            trace.state_query_canonical_text = canonical_text
            local_outcome = await self._agent_adapter.async_process(
                agent_id=self._config.local_agent_id,
                text=canonical_text,
                language=language,
                conversation_id=None if preserve_llm_owner else conversation_id,
                context=context,
                device_id=device_id,
                satellite_id=satellite_id,
                extra_system_prompt=extra_system_prompt,
            )
            trace.state_query_local_executed = True
            trace.state_query_local_response_text = local_outcome.response_text
            if not preserve_llm_owner:
                self._update_conversation_tracking(
                    conversation_id,
                    local_outcome,
                    executor="local",
                    agent_id=self._config.local_agent_id,
                )
            if not local_outcome.success:
                return None
            return RouterResult(
                path=ResolutionPath.LOCAL_STATE_QUERY,
                outcome=local_outcome,
                trace=trace,
            )

        resolved = self._resolve_state_query_target(
            parsed=parsed,
            entities=catalog.entity_targets,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )
        if resolved is None:
            return None

        canonical_text = self._render_local_state_query(parsed=parsed, resolved=resolved)
        if not canonical_text:
            return None

        trace.state_query_canonical_text = canonical_text
        trace.state_query_fuzzy_match_target = resolved["label"]
        trace.state_query_fuzzy_match_score = round(resolved["score"], 4)

        local_outcome = await self._agent_adapter.async_process(
            agent_id=self._config.local_agent_id,
            text=canonical_text,
            language=language,
            conversation_id=None if preserve_llm_owner else conversation_id,
            context=context,
            device_id=device_id,
            satellite_id=satellite_id,
            extra_system_prompt=extra_system_prompt,
        )
        trace.state_query_local_executed = True
        trace.state_query_local_response_text = local_outcome.response_text
        if not preserve_llm_owner:
            self._update_conversation_tracking(
                conversation_id,
                local_outcome,
                executor="local",
                agent_id=self._config.local_agent_id,
            )
        if not local_outcome.success:
            return None

        return RouterResult(
            path=ResolutionPath.LOCAL_STATE_QUERY,
            outcome=local_outcome,
            trace=trace,
        )

    async def _try_local_action_during_llm(
        self,
        *,
        text: str,
        language: str,
        context: Any,
        trace: ResolutionTrace,
        catalog,
        origin_area: str | None,
        origin_super_area: str | None,
        device_id: str | None,
        satellite_id: str | None,
        extra_system_prompt: str | None,
    ) -> RouterResult | None:
        normalized = normalize_text(text)
        unknown_local_control_request = _looks_like_unknown_local_control_request(normalized)
        if unknown_local_control_request:
            trace.local_action_detected = False
            return None

        trace.local_action_detected = _starts_with_supported_local_action(normalized)
        if not trace.local_action_detected:
            return None

        fuzzy_started = perf_counter()
        match_result = self._matcher.match(
            text,
            catalog,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )
        trace.fuzzy_match_duration_ms = round((perf_counter() - fuzzy_started) * 1000, 3)
        if match_result.best is not None:
            second = match_result.top_candidates[1].score if len(match_result.top_candidates) > 1 else 0.0
            decision = validate_fuzzy_execution(
                inferred_action=match_result.inferred_action,
                candidate_action=match_result.best.action,
                canonical_phrase=match_result.best.canonical_phrase,
                best_score=match_result.best.score,
                second_score=second,
                fuzzy_threshold=self._config.fuzzy_threshold,
                ambiguity_gap=self._config.ambiguity_gap,
                high_risk_threshold=self._config.high_risk_threshold,
            )
            if decision.allowed:
                assist_input = match_result.best.canonical_phrase
                if match_result.best.candidate_type.value == "conversation_target":
                    matched_pattern = (
                        (match_result.best.detail or {}).get("matched_sample_phrase_raw")
                        or match_result.best.canonical_phrase
                    )
                    rendered = render_conversation_pattern(
                        original_utterance=text,
                        pattern=matched_pattern,
                    )
                    assist_input = rendered.text or normalize_text(text)
                trace.local_action_canonical_text = assist_input
                fuzzy_outcome = await self._agent_adapter.async_process(
                    agent_id=self._config.local_agent_id,
                    text=assist_input,
                    language=language,
                    conversation_id=None,
                    context=context,
                    device_id=device_id,
                    satellite_id=satellite_id,
                    extra_system_prompt=extra_system_prompt,
                )
                trace.local_action_response_text = fuzzy_outcome.response_text
                if fuzzy_outcome.success:
                    return RouterResult(
                        path=ResolutionPath.FUZZY_LOCAL,
                        outcome=fuzzy_outcome,
                        trace=trace,
                    )

        trace.local_action_canonical_text = text
        exact_outcome = await self._agent_adapter.async_process(
            agent_id=self._config.local_agent_id,
            text=text,
            language=language,
            conversation_id=None,
            context=context,
            device_id=device_id,
            satellite_id=satellite_id,
            extra_system_prompt=extra_system_prompt,
        )
        trace.local_action_response_text = exact_outcome.response_text
        if not exact_outcome.success:
            return None
        return RouterResult(
            path=ResolutionPath.EXACT_LOCAL,
            outcome=exact_outcome,
            trace=trace,
        )

    async def _try_local_intent_during_llm(
        self,
        *,
        text: str,
        language: str,
        conversation_id: str | None,
        context: Any,
        trace: ResolutionTrace,
        catalog,
        origin_area: str | None,
        origin_super_area: str | None,
        device_id: str | None,
        satellite_id: str | None,
        extra_system_prompt: str | None,
    ) -> RouterResult | None:
        """Allow strong local phrase/entity intent matches to interrupt an active LLM turn."""
        translation_started = perf_counter()
        translation = await self._llm_adapter.async_translate_for_local(
            llm_agent_id=self._config.translate_llm_agent_id,
            utterance=text,
            language=language,
            catalog=catalog,
            max_candidates=self._config.max_llm_candidates,
            conversation_id=conversation_id,
            context=context,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
            preserve_raw_text=False,
        )
        trace.llm_translation_duration_ms = round(
            (perf_counter() - translation_started) * 1000,
            3,
        )
        trace.llm_translation_summary = {
            "mode": translation.mode,
            "tool_group": translation.tool_group,
            "confidence": translation.confidence,
            "valid": translation.valid,
            "notes": (
                "active_llm_local_intent_interrupt"
                if translation.valid and translation.canonical_text
                else "active_llm_conversation"
            ),
            "source": translation.source,
            "intent_family": translation.intent_family,
            "confidence_reason": translation.confidence_reason,
            "debug": translation.debug,
        }
        if not (translation.valid and translation.canonical_text):
            trace.llm_translated_local_executed = False
            return None

        trace.chosen_canonical_phrase = translation.canonical_text
        trace.assist_pipeline_input = translation.canonical_text
        trace.rendered_from_pattern = False
        trace.rendered_slots = {}
        translated_outcome = await self._agent_adapter.async_process(
            agent_id=self._config.local_agent_id,
            text=translation.canonical_text,
            language=language,
            conversation_id=None,
            context=context,
            device_id=device_id,
            satellite_id=satellite_id,
            extra_system_prompt=extra_system_prompt,
        )
        trace.llm_translated_local_executed = True
        trace.llm_translated_local_outcome = (
            "success" if translated_outcome.success else "failed"
        )
        trace.llm_translated_local_response_text = translated_outcome.response_text
        trace.llm_translated_local_response_type = translated_outcome.response_type
        trace.llm_translated_local_error_code = translated_outcome.error_code
        trace.llm_translated_local_processed_locally = translated_outcome.processed_locally
        trace.llm_translated_local_conversation_id = translated_outcome.conversation_id
        trace.llm_translated_local_continue_conversation = (
            translated_outcome.continue_conversation
        )
        if not translated_outcome.success:
            return None

        self._update_conversation_tracking(
            conversation_id,
            translated_outcome,
            executor="local",
            agent_id=self._config.local_agent_id,
        )
        return RouterResult(
            path=ResolutionPath.LLM_TRANSLATED_LOCAL,
            outcome=translated_outcome,
            trace=trace,
        )

    def _build_llm_fallback_system_prompt(
        self,
        *,
        trace: ResolutionTrace,
        text: str,
        catalog,
        origin_area: str | None,
        origin_super_area: str | None,
        extra_system_prompt: str | None,
        semantic_request_classification: SemanticRequestClassification | None = None,
    ) -> str | None:
        """Build the compact system prompt sent to the fallback LLM."""
        upstream_prompt = extra_system_prompt.strip() if extra_system_prompt else None
        semantic_hint = self._build_semantic_fallback_hint(
            classification=semantic_request_classification,
            trace=trace,
        )
        state_enrichment = self._build_llm_state_enrichment(
            text=text,
            catalog=catalog,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        )

        prompt_parts: list[str] = []
        if semantic_hint:
            prompt_parts.append(semantic_hint)
            trace.llm_fallback_prompt_hint_applied = True
            trace.llm_fallback_prompt_hint = semantic_hint
        else:
            trace.llm_fallback_prompt_hint_applied = False
            trace.llm_fallback_prompt_hint = None

        if state_enrichment:
            prompt_parts.append(state_enrichment["prompt"])
            trace.llm_state_enrichment_applied = True
            trace.llm_state_enrichment_targets = state_enrichment["targets"]
            trace.llm_state_enrichment_prompt = state_enrichment["prompt"]
        else:
            trace.llm_state_enrichment_applied = False
            trace.llm_state_enrichment_targets = []
            trace.llm_state_enrichment_prompt = None

        fallback_prompt = "\n\n".join(prompt_parts) if prompt_parts else None

        trace.llm_fallback_upstream_prompt_suppressed = bool(upstream_prompt)
        trace.llm_fallback_upstream_prompt_chars = (
            len(upstream_prompt) if upstream_prompt is not None else 0
        )
        trace.llm_fallback_prompt_chars = len(fallback_prompt) if fallback_prompt else 0
        return fallback_prompt

    def _build_semantic_fallback_hint(
        self,
        *,
        classification: SemanticRequestClassification | None,
        trace: ResolutionTrace,
    ) -> str | None:
        """Build a short hint for the fallback LLM from semantic request classification."""
        if classification is None:
            return None
        if (
            classification.kind == "general_request"
            and classification.confidence >= REQUEST_GENERAL_BYPASS_THRESHOLD
        ):
            return (
                "This appears to be a general/open-domain request. "
                "Prefer a general assistant answer unless the user explicitly asks for a Home Assistant action."
            )
        if (
            classification.kind == "tool_request"
            and classification.confidence >= REQUEST_TOOL_HINT_THRESHOLD
            and trace.selected_path == ResolutionPath.FAILED
        ):
            return (
                "This appears to be a smart-home/tool request that local routing could not fully resolve. "
                "Prefer a smart-home interpretation if it fits the user's wording."
            )
        return None

    def _resolve_initial_llm_fallback_conversation_id(self) -> str | None:
        """Start first-turn LLM fallback fresh to avoid replaying outer chat history."""
        return None

    def _build_llm_state_enrichment(
        self,
        *,
        text: str,
        catalog,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> dict[str, Any] | None:
        """Build compact state context for active LLM turns when matches are clear."""
        if self._hass is None or not self._looks_like_status_followup(text):
            return None

        query_kind = self._classify_enrichment_query(text)
        resolved = self._resolve_state_enrichment_entities(
            text=text,
            catalog=catalog,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
            query_kind=query_kind,
        )
        if not resolved:
            return None

        lines: list[str] = [STATE_ENRICHMENT_INTRO]
        if query_kind == "binary":
            lines.append(
                "For yes/no state questions, answer whether the requested state is true or false using the resolved live state below."
            )
        elif query_kind == "temperature":
            lines.append(
                "For temperature questions, answer with the current temperature only. Prefer current temperature over target or set temperature."
            )
        target_names: list[str] = []
        for entity in resolved:
            rendered = self._render_entity_live_state(entity, query_kind=query_kind)
            if not rendered:
                continue
            lines.append(f"- {rendered}")
            target_names.append(entity.name)

        if not target_names:
            return None

        return {
            "targets": target_names,
            "prompt": "\n".join(lines),
        }

    def _looks_like_status_followup(self, text: str) -> bool:
        normalized = normalize_text(text)
        if not normalized:
            return False
        normalized = self._normalize_temperature_text(normalized)
        if any(normalized.startswith(prefix) for prefix in STATUS_QUERY_HINTS):
            return True
        if any(token in normalized.split() for token in ("light", "lights", "fan", "temperature", "temp", "thermostat", "lock", "door", "cover")):
            return True
        return len(normalized.split()) <= 6

    def _parse_state_query(self, text: str) -> ParsedStateQuery | None:
        normalized = self._normalize_temperature_text(normalize_text(text))
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return None

        binary_match = re.match(r"^(is|are)\s+(.+?)\s+(on|off|open|opened|closed|close|locked|unlocked)\??$", normalized)
        if binary_match:
            target = re.sub(r"^(the|a|an)\s+", "", binary_match.group(2)).strip()
            return ParsedStateQuery(
                query_kind="binary",
                target_phrase=target,
                requested_state=binary_match.group(3),
                normalized_text=normalized,
            )

        domain_state_match = DOMAIN_STATE_QUERY_RE.match(normalized)
        if domain_state_match:
            return ParsedStateQuery(
                query_kind="domain_state",
                target_phrase=re.sub(r"^(the|a|an)\s+", "", domain_state_match.group(4)).strip(),
                requested_state=domain_state_match.group(3),
                normalized_text=normalized,
                domain_hint=domain_state_match.group(2).strip(),
            )

        if "temperature" in normalized.split() or "temp" in normalized.split() or re.search(r"\bhow (hot|cold|warm|cool)\b", normalized):
            target = normalized
            for opener in HA_QUERY_OPENERS:
                if target.startswith(opener):
                    target = target.removeprefix(opener).strip()
                    break
            target = re.sub(r"\bthe current temperature\b", "", target).strip()
            target = re.sub(r"\bcurrent temperature\b", "", target).strip()
            target = re.sub(r"\btemperature\b", "", target).strip()
            target = re.sub(r"\btemp\b", "", target).strip()
            target = re.sub(r"\bhow (hot|cold|warm|cool) is (it|)\b", "", target).strip()
            target = re.sub(r"^(in|of|for|at)\s+", "", target).strip()
            target = re.sub(r"^(the|a|an)\s+", "", target).strip()
            if target:
                return ParsedStateQuery(
                    query_kind="temperature",
                    target_phrase=target,
                    requested_state=None,
                    normalized_text=normalized,
                )

        if any(normalized.startswith(opener) for opener in HA_QUERY_OPENERS) or " state" in normalized or " status" in normalized:
            target = normalized
            for opener in HA_QUERY_OPENERS:
                if target.startswith(opener):
                    target = target.removeprefix(opener).strip()
                    break
            target = re.sub(r"\b(state|status)\b", "", target).strip()
            target = re.sub(r"^(the|a|an)\s+", "", target).strip()
            if target:
                return ParsedStateQuery(
                    query_kind="status",
                    target_phrase=target,
                    requested_state=None,
                    normalized_text=normalized,
                )

        return None

    def _classify_enrichment_query(self, text: str) -> str:
        normalized = self._normalize_temperature_text(normalize_text(text))
        if "temperature" in normalized.split() or "temp" in normalized.split():
            return "temperature"
        if normalized.startswith("is ") or normalized.startswith("are "):
            return "binary"
        return "status"

    def _resolve_state_enrichment_entities(
        self,
        *,
        text: str,
        catalog,
        origin_area: str | None,
        origin_super_area: str | None,
        query_kind: str,
    ) -> list[EntityTarget]:
        segments = self._extract_enrichment_segments(text, query_kind=query_kind)
        if not segments:
            return []

        resolved: list[EntityTarget] = []
        used_ids: set[str] = set()
        for segment in segments:
            entity = self._resolve_best_enrichment_entity(
                segment=segment,
                entities=catalog.entity_targets,
                origin_area=origin_area,
                origin_super_area=origin_super_area,
                query_kind=query_kind,
            )
            if entity is None or entity.entity_id in used_ids:
                continue
            resolved.append(entity)
            used_ids.add(entity.entity_id)
            if len(resolved) >= 3:
                break
        return resolved

    def _resolve_state_query_target(
        self,
        *,
        parsed: ParsedStateQuery,
        entities: list[EntityTarget],
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> dict[str, Any] | None:
        target = parsed.target_phrase
        target_tokens = tokenize(target)
        if not target_tokens:
            return None

        candidates: list[tuple[float, int, dict[str, Any]]] = []
        normalized_target = normalize_text(target)
        normalized_target_tokens = set(target_tokens)
        for entity in entities:
            if parsed.query_kind == "temperature" and not self._entity_supports_temperature(entity):
                continue

            entity_tokens = set(entity.tokens)
            alias_tokens = [set(tokenize(alias)) for alias in entity.aliases]
            overlap = len(normalized_target_tokens & entity_tokens)
            alias_overlap = max((len(normalized_target_tokens & alias) for alias in alias_tokens), default=0)
            entity_area_tokens = set(tokenize(entity.area or ""))
            area_overlap = len(normalized_target_tokens & entity_area_tokens)
            label = entity.name
            similarity = SequenceMatcher(a=normalized_target, b=entity.normalized_name).ratio()
            if not overlap and not alias_overlap and not area_overlap and similarity < 0.58:
                continue

            score = 0.0
            score += 0.45 * (overlap / max(1, len(normalized_target_tokens)))
            score += 0.20 * (alias_overlap / max(1, len(normalized_target_tokens)))
            score += 0.20 * similarity
            score += 0.15 * (area_overlap / max(1, len(normalized_target_tokens)))
            locality = self._entity_locality_rank(
                entity=entity,
                origin_area=origin_area,
                origin_super_area=origin_super_area,
            )
            score += 0.08 * locality
            if parsed.query_kind == "temperature":
                score += self._temperature_entity_bonus(entity)
            candidates.append(
                (
                    min(score, 1.0),
                    locality,
                    {
                        "kind": "entity",
                        "entity": entity,
                        "label": label,
                        "score": min(score, 1.0),
                    },
                )
            )

        if parsed.query_kind == "temperature":
            area_match = self._resolve_temperature_area_target(parsed.target_phrase, entities)
            if area_match is not None:
                candidates.append((area_match["score"], 3, area_match))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best = candidates[0][2]
        best_score = candidates[0][0]
        if best_score < 0.68:
            return None
        if len(candidates) > 1:
            second_score = candidates[1][0]
            second_locality = candidates[1][1]
            if second_score >= (best_score - 0.06) and second_locality == candidates[0][1]:
                return None
        return best

    def _resolve_temperature_area_target(
        self,
        target_phrase: str,
        entities: list[EntityTarget],
    ) -> dict[str, Any] | None:
        normalized_target = normalize_text(target_phrase)
        target_tokens = set(tokenize(normalized_target))
        if not target_tokens:
            return None

        best: dict[str, Any] | None = None
        best_score = 0.0
        seen_areas: set[str] = set()
        for entity in entities:
            if not self._entity_supports_temperature(entity) or not entity.area:
                continue
            area_key = normalize_text(entity.area)
            if area_key in seen_areas:
                continue
            seen_areas.add(area_key)
            area_tokens = set(tokenize(entity.area))
            overlap = len(target_tokens & area_tokens)
            if not overlap:
                continue
            score = overlap / max(1, len(target_tokens))
            if score > best_score:
                best_score = score
                best = {
                    "kind": "area",
                    "area": entity.area,
                    "label": entity.area,
                    "score": min(0.72 + (0.2 * score), 0.92),
                }
        return best

    def _render_local_state_query(
        self,
        *,
        parsed: ParsedStateQuery,
        resolved: dict[str, Any] | None,
    ) -> str | None:
        if parsed.query_kind == "domain_state":
            domain_text = parsed.domain_hint or "devices"
            state_text = parsed.requested_state or "on"
            return f"what {domain_text} are {state_text} in {parsed.target_phrase}"

        if resolved is None:
            return None

        if resolved["kind"] == "area" and parsed.query_kind == "temperature":
            return f"what is the current temperature in {resolved['area']}"

        entity: EntityTarget = resolved["entity"]
        if parsed.query_kind == "binary" and parsed.requested_state:
            return f"what is the state of {entity.name}"
        if parsed.query_kind == "temperature":
            if entity.area and parsed.target_phrase == normalize_text(entity.area):
                return f"what is the current temperature in {entity.area}"
            return f"what is the current temperature of {entity.name}"
        return f"what is the state of {entity.name}"

    def _extract_enrichment_segments(self, text: str, *, query_kind: str) -> list[str]:
        normalized = self._normalize_temperature_text(normalize_text(text))
        if query_kind == "binary":
            normalized = re.sub(r"^(is|are)\s+", "", normalized).strip()
            normalized = BINARY_STATE_ENDING_RE.sub("", normalized).strip()
        for prefix in (
            "can you tell me about ",
            "tell me about ",
            "what about ",
            "what is ",
            "whats ",
            "status of ",
            "the status of ",
        ):
            if normalized.startswith(prefix):
                normalized = normalized.removeprefix(prefix).strip()
                break
        if query_kind == "temperature":
            normalized = re.sub(r"^(what is|whats)\s+", "", normalized).strip()
            normalized = re.sub(r"\btemperature\b", "", normalized).strip()
            normalized = re.sub(r"\btemp\b", "", normalized).strip()
            normalized = re.sub(r"^(in|at|for)\s+", "", normalized).strip()

        normalized = re.sub(r"\b(can you|could you|please)\b", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return []

        raw_segments = re.split(r"\b(?:and|,)\b", normalized)
        segments: list[str] = []
        for raw in raw_segments:
            segment = raw.strip()
            if not segment:
                continue
            segment = re.sub(r"^(about|the|a|an)\s+", "", segment).strip()
            segment = re.sub(r"\?$", "", segment).strip()
            if segment:
                segments.append(segment)
        if not segments:
            return []
        return segments[:3]

    def _resolve_best_enrichment_entity(
        self,
        *,
        segment: str,
        entities: list[EntityTarget],
        origin_area: str | None,
        origin_super_area: str | None,
        query_kind: str,
    ) -> EntityTarget | None:
        segment_normalized = self._normalize_temperature_text(normalize_text(segment))
        segment_tokens = tokenize(segment_normalized)
        if not segment_tokens:
            return None

        ranked: list[tuple[float, int, int, EntityTarget]] = []
        for entity in entities:
            score = self._enrichment_entity_score(
                segment_normalized=segment_normalized,
                segment_tokens=segment_tokens,
                entity=entity,
                origin_area=origin_area,
                origin_super_area=origin_super_area,
                query_kind=query_kind,
            )
            if score is None:
                continue
            locality_rank = self._entity_locality_rank(
                entity=entity,
                origin_area=origin_area,
                origin_super_area=origin_super_area,
            )
            ranked.append((score, locality_rank, len(entity.tokens), entity))

        if not ranked:
            return None

        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best_score, best_locality, _, best_entity = ranked[0]
        if best_score < 0.72:
            return None
        if len(ranked) > 1:
            second_score, second_locality, _, _ = ranked[1]
            if second_score >= (best_score - 0.08) and second_locality == best_locality:
                return None
        return best_entity

    def _enrichment_entity_score(
        self,
        *,
        segment_normalized: str,
        segment_tokens: list[str],
        entity: EntityTarget,
        origin_area: str | None,
        origin_super_area: str | None,
        query_kind: str,
    ) -> float | None:
        entity_tokens = entity.tokens
        if not entity_tokens:
            return None

        if query_kind == "temperature" and not self._entity_supports_temperature(entity):
            return None

        overlap = len(set(segment_tokens) & set(entity_tokens))
        if overlap == 0 and query_kind != "temperature":
            return None

        token_coverage = overlap / max(1, len(set(segment_tokens)))
        entity_coverage = overlap / max(1, len(set(entity_tokens)))
        phrase_score = 1.0 if segment_normalized == entity.normalized_name else 0.0
        contains_score = 1.0 if segment_normalized in entity.normalized_name else 0.0
        score = max(
            phrase_score,
            (0.65 * token_coverage) + (0.20 * entity_coverage) + (0.15 * contains_score),
        )

        if self._entity_locality_rank(
            entity=entity,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        ) == 2:
            score += 0.06
        elif self._entity_locality_rank(
            entity=entity,
            origin_area=origin_area,
            origin_super_area=origin_super_area,
        ) == 1:
            score += 0.03

        if query_kind == "temperature":
            score += self._temperature_entity_bonus(entity)

        return min(score, 1.0)

    def _entity_locality_rank(
        self,
        *,
        entity: EntityTarget,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> int:
        if origin_area and entity.area and normalize_text(entity.area) == normalize_text(origin_area):
            return 2
        if (
            origin_super_area
            and entity.super_area
            and normalize_text(entity.super_area) == normalize_text(origin_super_area)
        ):
            return 1
        return 0

    def _render_entity_live_state(self, entity: EntityTarget, *, query_kind: str) -> str | None:
        states = getattr(self._hass, "states", None)
        if states is None or not hasattr(states, "get"):
            return None

        state = states.get(entity.entity_id)
        if state is None:
            return None

        attrs = getattr(state, "attributes", {}) or {}
        value = getattr(state, "state", None)
        if value is None:
            return None

        summary = f"{entity.name}: {value}"
        if query_kind == "temperature":
            return self._render_temperature_live_state(entity, state)
        if entity.domain == "climate":
            current_temp = attrs.get("current_temperature")
            unit = attrs.get("temperature_unit")
            if current_temp is not None:
                summary = f"{entity.name}: current temperature {current_temp}{unit or ''}, mode {value}"
        elif entity.domain == "sensor":
            unit = attrs.get("unit_of_measurement")
            device_class = attrs.get("device_class")
            if unit:
                summary = f"{entity.name}: {value}{unit}"
            elif device_class:
                summary = f"{entity.name}: {value} ({device_class})"
        elif entity.domain in {"light", "switch", "fan", "lock", "cover"}:
            summary = f"{entity.name}: {value}"
        return summary

    def _render_temperature_live_state(self, entity: EntityTarget, state: Any) -> str | None:
        attrs = getattr(state, "attributes", {}) or {}
        value = getattr(state, "state", None)
        if entity.domain == "climate":
            current_temp = attrs.get("current_temperature")
            unit = attrs.get("temperature_unit")
            if current_temp is None:
                return None
            return f"{entity.name}: current temperature {current_temp}{unit or ''}"
        unit = attrs.get("unit_of_measurement")
        if value is None:
            return None
        if unit:
            return f"{entity.name}: {value}{unit}"
        return f"{entity.name}: {value}"

    def _entity_supports_temperature(self, entity: EntityTarget) -> bool:
        if entity.domain == "climate":
            return True
        tokens = set(entity.tokens)
        return "temperature" in tokens or "temp" in tokens

    def _temperature_entity_bonus(self, entity: EntityTarget) -> float:
        tokens = set(entity.tokens)
        if entity.domain == "climate":
            return 0.25
        if "temperature" in tokens:
            return 0.20
        if "temp" in tokens:
            return 0.18
        return 0.0

    def _normalize_temperature_text(self, text: str) -> str:
        return TEMPERATURE_WORD_RE.sub("temperature", text)

    def _resolve_origin_area(
        self,
        *,
        device_id: str | None,
        satellite_id: str | None,
        context: Any,
    ) -> str | None:
        """Best-effort resolve the area associated with the utterance origin."""
        try:
            from homeassistant.helpers import area_registry as ar
            from homeassistant.helpers import device_registry as dr
            from homeassistant.helpers import entity_registry as er
        except Exception:
            return None

        area_reg = ar.async_get(self._hass)
        device_reg = dr.async_get(self._hass)
        entity_reg = er.async_get(self._hass)

        def _area_name_for_device_id(candidate_device_id: str | None) -> str | None:
            if not candidate_device_id:
                return None
            device = device_reg.async_get(candidate_device_id)
            if device is None or not getattr(device, 'area_id', None):
                return None
            area = area_reg.async_get_area(device.area_id)
            return area.name if area else None

        def _device_id_for_entity_id(entity_id: str | None) -> str | None:
            if not entity_id:
                return None
            entry = entity_reg.async_get(entity_id)
            return getattr(entry, 'device_id', None) if entry else None

        area_name = _area_name_for_device_id(device_id)
        if area_name:
            return area_name

        area_name = _area_name_for_device_id(_device_id_for_entity_id(satellite_id))
        if area_name:
            return area_name

        context_device_id = getattr(context, 'device_id', None)
        area_name = _area_name_for_device_id(context_device_id)
        if area_name:
            return area_name

        return None

    def _resolve_origin_super_area(
        self,
        *,
        origin_area: str | None,
        device_id: str | None,
        satellite_id: str | None,
        context: Any,
    ) -> str | None:
        """Best-effort resolve SuperArea label for the utterance origin area."""
        try:
            from homeassistant.helpers import area_registry as ar
            from homeassistant.helpers import device_registry as dr
            from homeassistant.helpers import entity_registry as er
            from homeassistant.helpers import label_registry as lr
        except Exception:
            return None

        area_reg = ar.async_get(self._hass)
        device_reg = dr.async_get(self._hass)
        entity_reg = er.async_get(self._hass)
        label_reg = lr.async_get(self._hass)

        label_lookup = {}
        labels = getattr(label_reg, "labels", None)
        if isinstance(labels, dict):
            for label_id, label in labels.items():
                name = getattr(label, "name", None)
                if isinstance(name, str) and name.strip():
                    label_lookup[str(label_id)] = name.strip()

        def _extract_super_area_from_labels(obj: Any) -> str | None:
            raw_labels = (
                getattr(obj, "labels", None)
                or getattr(obj, "label_ids", None)
                or []
            )
            for raw_label in raw_labels:
                if hasattr(raw_label, "name"):
                    label_name = getattr(raw_label, "name", None)
                elif isinstance(raw_label, str):
                    label_name = label_lookup.get(raw_label, raw_label)
                else:
                    label_name = None
                if not isinstance(label_name, str):
                    continue
                match = SUPER_AREA_LABEL_RE.match(label_name.strip())
                if match and match.group(1).strip():
                    return match.group(1).strip()
            return None

        def _area_for_device_id(candidate_device_id: str | None):
            if not candidate_device_id:
                return None
            device = device_reg.async_get(candidate_device_id)
            if device is None or not getattr(device, "area_id", None):
                return None
            return area_reg.async_get_area(device.area_id)

        def _device_id_for_entity_id(entity_id: str | None) -> str | None:
            if not entity_id:
                return None
            entry = entity_reg.async_get(entity_id)
            return getattr(entry, "device_id", None) if entry else None

        candidate_areas: list[Any] = []
        registry_area_maps = []
        for attr_name in ("areas", "_areas"):
            raw = getattr(area_reg, attr_name, None)
            if isinstance(raw, dict):
                registry_area_maps.append(raw)
        if origin_area:
            for area_map in registry_area_maps:
                for area in area_map.values():
                    if getattr(area, "name", None) and str(area.name).strip().lower() == origin_area.strip().lower():
                        candidate_areas.append(area)

        for candidate_device_id in (
            device_id,
            _device_id_for_entity_id(satellite_id),
            getattr(context, "device_id", None),
        ):
            area = _area_for_device_id(candidate_device_id)
            if area is not None:
                candidate_areas.append(area)

        seen_ids: set[str] = set()
        for area in candidate_areas:
            area_id = getattr(area, "id", None) or getattr(area, "area_id", None) or str(id(area))
            if area_id in seen_ids:
                continue
            seen_ids.add(str(area_id))
            super_area = _extract_super_area_from_labels(area)
            if super_area:
                return super_area

        return None
