"""Main routing pipeline implementation."""

from __future__ import annotations

import logging
from typing import Any

from .catalog import CatalogManager
from .llm_adapter import LLMAdapter
from .matcher import FuzzyMatcher
from .models import ResolutionPath, ResolutionTrace, RouterConfig, RouterResult
from .safety import validate_fuzzy_execution

_LOGGER = logging.getLogger(__name__)


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
    ) -> None:
        self._config = config
        self._catalog = catalog_manager
        self._matcher = matcher
        self._agent_adapter = agent_adapter
        self._llm_adapter = llm_adapter

    async def async_route(
        self,
        *,
        text: str,
        language: str,
        conversation_id: str | None,
        context: Any,
        dry_run: bool = False,
    ) -> RouterResult:
        """Run deterministic routing pipeline with bounded attempts."""
        catalog = self._catalog.get_catalog()
        match_result = self._matcher.match(text, catalog)
        trace = ResolutionTrace(
            original_utterance=text,
            normalized_utterance=match_result.normalized_utterance,
            catalog_revision=catalog.metadata.revision,
        )

        # 1) exact local attempt
        exact = await self._agent_adapter.async_process(
            agent_id=self._config.local_agent_id,
            text=text,
            language=language,
            conversation_id=conversation_id,
            context=context,
        )
        trace.exact_local_outcome = "success" if exact.success else "failed"
        trace.exact_local_response_text = exact.response_text
        trace.failure_category = (
            exact.failure_category.value if exact.failure_category is not None else None
        )

        if exact.success:
            trace.selected_path = ResolutionPath.EXACT_LOCAL
            trace.final_executor = "local"
            return RouterResult(path=ResolutionPath.EXACT_LOCAL, outcome=exact, trace=trace)

        # 2) fuzzy attempt
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

            if decision.allowed:
                trace.chosen_canonical_phrase = match_result.best.canonical_phrase
                if not dry_run:
                    fuzzy_outcome = await self._agent_adapter.async_process(
                        agent_id=self._config.local_agent_id,
                        text=match_result.best.canonical_phrase,
                        language=language,
                        conversation_id=conversation_id,
                        context=context,
                    )
                else:
                    fuzzy_outcome = exact

                if fuzzy_outcome.success or dry_run:
                    trace.selected_path = ResolutionPath.FUZZY_LOCAL
                    trace.final_executor = "local"
                    return RouterResult(
                        path=ResolutionPath.FUZZY_LOCAL,
                        outcome=fuzzy_outcome,
                        trace=trace,
                    )
            else:
                _LOGGER.debug("Fuzzy candidate rejected: %s", decision.reason)

        # 3) LLM translation to local
        if self._config.llm_translate_enabled:
            translation = await self._llm_adapter.async_translate_for_local(
                llm_agent_id=self._config.llm_agent_id,
                utterance=text,
                language=language,
                catalog=catalog,
                max_candidates=self._config.max_llm_candidates,
                conversation_id=conversation_id,
                context=context,
            )
            trace.llm_translation_summary = {
                "mode": translation.mode,
                "confidence": translation.confidence,
                "target_type": translation.target_type.value,
                "valid": translation.valid,
                "notes": translation.notes,
            }

            if translation.valid and translation.canonical_text:
                trace.chosen_canonical_phrase = translation.canonical_text
                if not dry_run:
                    translated_outcome = await self._agent_adapter.async_process(
                        agent_id=self._config.local_agent_id,
                        text=translation.canonical_text,
                        language=language,
                        conversation_id=conversation_id,
                        context=context,
                    )
                else:
                    translated_outcome = exact

                if translated_outcome.success or dry_run:
                    trace.selected_path = ResolutionPath.LLM_TRANSLATED_LOCAL
                    trace.final_executor = "local"
                    return RouterResult(
                        path=ResolutionPath.LLM_TRANSLATED_LOCAL,
                        outcome=translated_outcome,
                        trace=trace,
                    )

        # 4) final direct llm fallback
        if self._config.llm_fallback_enabled:
            if not dry_run:
                llm_outcome = await self._llm_adapter.async_final_fallback(
                    llm_agent_id=self._config.llm_agent_id,
                    utterance=text,
                    language=language,
                    conversation_id=conversation_id,
                    context=context,
                )
            else:
                llm_outcome = exact
            trace.selected_path = ResolutionPath.LLM_FALLBACK
            trace.final_executor = "llm"
            return RouterResult(path=ResolutionPath.LLM_FALLBACK, outcome=llm_outcome, trace=trace)

        trace.selected_path = ResolutionPath.FAILED
        trace.final_executor = "none"
        return RouterResult(path=ResolutionPath.FAILED, outcome=exact, trace=trace)
