"""Main routing pipeline implementation."""


from __future__ import annotations

import logging
import re
from typing import Any

from .catalog import CatalogManager
from .llm_adapter import LLMAdapter
from .matcher import FuzzyMatcher
from .models import ResolutionPath, ResolutionTrace, RouterConfig, RouterResult
from .phrase_renderer import render_conversation_pattern
from .phonetics import normalize_text
from .safety import validate_fuzzy_execution

_LOGGER = logging.getLogger(__name__)
SUPER_AREA_LABEL_RE = re.compile(r"^superarea\s*:\s*(.+)$", re.IGNORECASE)


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
    ) -> None:
        self._config = config
        self._catalog = catalog_manager
        self._matcher = matcher
        self._agent_adapter = agent_adapter
        self._llm_adapter = llm_adapter
        self._hass = hass

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
        origin_area: str | None = None,
        device_id: str | None = None,
        satellite_id: str | None = None,
    ) -> RouterResult:
        """Run deterministic routing pipeline with bounded attempts."""
        _LOGGER.debug("ROUTER START: text=%r, device_id=%r, satellite_id=%r", text, device_id, satellite_id)
        _LOGGER.warning("ROUTER INPUT: %r", text)
        catalog = self._catalog.get_catalog()
        resolved_origin_area = origin_area or self._resolve_origin_area(device_id=device_id, satellite_id=satellite_id, context=context)
        resolved_origin_super_area = self._resolve_origin_super_area(
            origin_area=resolved_origin_area,
            device_id=device_id,
            satellite_id=satellite_id,
            context=context,
        )
        match_result = self._matcher.match(
            text,
            catalog,
            origin_area=resolved_origin_area,
            origin_super_area=resolved_origin_super_area,
        )
        _LOGGER.warning("FUZZY MATCH: best=%s score=%s", getattr(match_result.best, "canonical_phrase", None), getattr(match_result.best, "score", None))
        trace = ResolutionTrace(
            original_utterance=text,
            normalized_utterance=match_result.normalized_utterance,
            catalog_revision=catalog.metadata.revision,
            origin_area=resolved_origin_area,
            origin_super_area=resolved_origin_super_area,
            effective_area_hint=match_result.effective_area_hint,
            effective_super_area_hint=match_result.effective_super_area_hint,
        )
        selected_path: ResolutionPath | None = None
        selected_outcome = None

        def _should_execute_branch() -> bool:
            return (not dry_run) or execute_debug_branches

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

        # 1) fuzzy attempt
        if self._config.fuzzy_enabled and match_result.best is not None:
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

                trace.assist_pipeline_input = assist_input
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

                if dry_run or (fuzzy_outcome is not None and fuzzy_outcome.success):
                    _LOGGER.warning("ROUTER DECISION: FUZZY_LOCAL")
                    if selected_path is None:
                        selected_path = ResolutionPath.FUZZY_LOCAL
                        selected_outcome = fuzzy_outcome
                        trace.selected_path = ResolutionPath.FUZZY_LOCAL
                        trace.final_executor = "local"
                    if not debug_collect_all:
                        return RouterResult(
                            path=ResolutionPath.FUZZY_LOCAL,
                            outcome=fuzzy_outcome,
                            trace=trace,
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

                    if dry_run or (fuzzy_outcome is not None and fuzzy_outcome.success):
                        _LOGGER.warning("ROUTER DECISION: FUZZY_LOCAL")
                        if selected_path is None:
                            selected_path = ResolutionPath.FUZZY_LOCAL
                            selected_outcome = fuzzy_outcome
                            trace.selected_path = ResolutionPath.FUZZY_LOCAL
                            trace.final_executor = "local"
                        if not debug_collect_all:
                            return RouterResult(
                                path=ResolutionPath.FUZZY_LOCAL,
                                outcome=fuzzy_outcome,
                                trace=trace,
                            )
                _LOGGER.debug("Fuzzy candidate rejected: %s", decision.reason)

        # 2) LLM translation to local
        if self._config.llm_translate_enabled:
            translation = await self._llm_adapter.async_translate_for_local(
                llm_agent_id=self._config.llm_agent_id,
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
            _LOGGER.warning("LLM TRANSLATION: valid=%s canonical=%s mode=%s", translation.valid, translation.canonical_text, translation.mode)
            trace.llm_translation_summary = {
                "mode": translation.mode,
                "confidence": translation.confidence,
                "target_type": translation.target_type.value,
                "valid": translation.valid,
                "notes": translation.notes,
            }
            trace.llm_translation_raw_text = translation.raw_text

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

                if dry_run or (translated_outcome is not None and translated_outcome.success):
                    _LOGGER.warning("ROUTER DECISION: LLM_TRANSLATED_LOCAL")
                    if selected_path is None:
                        selected_path = ResolutionPath.LLM_TRANSLATED_LOCAL
                        selected_outcome = translated_outcome
                        trace.selected_path = ResolutionPath.LLM_TRANSLATED_LOCAL
                        trace.final_executor = "local"
                    if not debug_collect_all:
                        return RouterResult(
                            path=ResolutionPath.LLM_TRANSLATED_LOCAL,
                            outcome=translated_outcome,
                            trace=trace,
                        )

        # 3) raw local attempt
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
        else:
            exact = None
        if exact is not None:
            _LOGGER.warning("EXACT LOCAL: success=%s response=%r processed_locally=%s", exact.success, exact.response_text, exact.processed_locally)
            trace.exact_local_executed = True
            trace.exact_local_outcome = "success" if exact.success else "failed"
            trace.exact_local_response_text = exact.response_text
            trace.exact_local_response_type = exact.response_type
            trace.exact_local_error_code = exact.error_code
            trace.exact_local_processed_locally = exact.processed_locally
            trace.failure_category = (
                exact.failure_category.value if exact.failure_category is not None else None
            )
        else:
            trace.exact_local_executed = False
            trace.exact_local_outcome = "skipped"

        if exact is not None and exact.success and exact.processed_locally is True:
            _LOGGER.warning("ROUTER DECISION: EXACT_LOCAL")
            if selected_path is None:
                trace.assist_pipeline_input = text
                trace.rendered_from_pattern = False
                trace.rendered_slots = {}
                trace.selected_path = ResolutionPath.EXACT_LOCAL
                trace.final_executor = "local"
                selected_path = ResolutionPath.EXACT_LOCAL
                selected_outcome = exact
            if not debug_collect_all:
                return RouterResult(path=ResolutionPath.EXACT_LOCAL, outcome=exact, trace=trace)

        if exact is not None and exact.success and exact.processed_locally is not True:
            _LOGGER.warning(
                "EXACT LOCAL rejected: success=%s processed_locally=%s response=%r",
                exact.success,
                exact.processed_locally,
                exact.response_text,
            )

        # 4) final direct llm fallback
        if self._config.llm_fallback_enabled:
            if _should_execute_branch():
                _LOGGER.warning("ROUTER DECISION: LLM_FALLBACK (calling LLM)")
                llm_outcome = await self._llm_adapter.async_final_fallback(
                    llm_agent_id=self._config.llm_agent_id,
                    utterance=text,
                    language=language,
                    conversation_id=conversation_id,
                    context=context,
                )
            else:
                llm_outcome = None
            _record_outcome("llm_fallback", llm_outcome)
            _LOGGER.warning("LLM FALLBACK RESPONSE: %r", getattr(llm_outcome, "response_text", None) if llm_outcome is not None else None)
            if selected_path is None:
                trace.selected_path = ResolutionPath.LLM_FALLBACK
                trace.final_executor = "llm"
                selected_path = ResolutionPath.LLM_FALLBACK
                selected_outcome = llm_outcome
            if not debug_collect_all:
                return RouterResult(path=ResolutionPath.LLM_FALLBACK, outcome=llm_outcome, trace=trace)

        if selected_path is None:
            trace.selected_path = ResolutionPath.FAILED
            trace.final_executor = "none"
            return RouterResult(path=ResolutionPath.FAILED, outcome=exact, trace=trace)

        return RouterResult(path=selected_path, outcome=selected_outcome, trace=trace)

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
