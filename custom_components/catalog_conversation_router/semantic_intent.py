"""Semantic intent retrieval for local phrase/entity matching."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .models import Catalog, ConversationTarget
from .phonetics import normalize_text

GRAMMAR_RE = re.compile(r"[\[\]\(\)\|]")
SLOT_RE = re.compile(r"\{([^}]+)\}")
CONCRETE_PATTERN_RE = re.compile(r"[\[\]\(\)\{\}\|]")
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
PHRASE_FILTER_THRESHOLD = 0.74
PHRASE_DIRECT_THRESHOLD = 0.84
ENTITY_DIRECT_THRESHOLD = 0.82
ENTITY_AMBIGUITY_GAP = 0.025


@dataclass(slots=True)
class SemanticPhraseCandidate:
    """A semantically ranked phrase candidate."""

    target_id: str
    example_text: str
    raw_text: str
    score: float
    concrete: bool
    has_slots: bool
    tool_group: str


@dataclass(slots=True)
class SemanticEntityCandidate:
    """A semantically ranked entity command candidate."""

    action: str
    target_name: str
    tool_group: str
    score: float
    synthetic: bool
    singular: bool


class SemanticIntentRanker:
    """Embedding-backed semantic ranking for phrase and entity intents."""

    def __init__(self) -> None:
        self._embedder: Any | None = None
        self._model_error: str | None = None
        self._cache_dir = os.getenv("CCR_SEMANTIC_CACHE_DIR")
        self._model_name = os.getenv("CCR_SEMANTIC_MODEL", DEFAULT_MODEL_NAME)

    def available(self) -> bool:
        return self._get_embedder() is not None

    def unavailable_reason(self) -> str | None:
        self._get_embedder()
        return self._model_error

    def rank_phrase_candidates(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        infer_tool_group: Any,
        limit: int = 8,
    ) -> list[SemanticPhraseCandidate]:
        docs: list[tuple[str, str, str, bool, bool, str]] = []
        for target in catalog.conversation_targets:
            patterns = self._conversation_target_patterns(target)
            tool_group = infer_tool_group(target.canonical_phrase)
            for pattern in patterns:
                semantic_text = self._semantic_phrase_text(pattern)
                if not semantic_text:
                    continue
                docs.append(
                    (
                        target.target_id,
                        pattern,
                        semantic_text,
                        self._is_concrete_phrase(pattern),
                        bool(SLOT_RE.search(pattern)),
                        tool_group,
                    )
                )

        ranked = self._rank(
            utterance=self._semantic_phrase_text(utterance),
            docs=[doc_text for _, _, doc_text, _, _, _ in docs],
            limit=limit,
        )
        results: list[SemanticPhraseCandidate] = []
        for idx, score in ranked:
            target_id, raw_text, doc_text, concrete, has_slots, tool_group = docs[idx]
            results.append(
                SemanticPhraseCandidate(
                    target_id=target_id,
                    example_text=doc_text,
                    raw_text=raw_text,
                    score=score,
                    concrete=concrete,
                    has_slots=has_slots,
                    tool_group=tool_group,
                )
            )
        return self._dedupe_phrase_candidates(results, limit=limit)

    def rank_entity_commands(
        self,
        *,
        utterance: str,
        command_docs: list[dict[str, Any]],
        origin_area: str | None,
        origin_super_area: str | None,
        limit: int = 8,
    ) -> list[SemanticEntityCandidate]:
        query_text = normalize_text(utterance)
        if origin_area and origin_area.casefold() not in query_text:
            query_text = f"{query_text} {normalize_text(origin_area)}"
        elif origin_super_area and origin_super_area.casefold() not in query_text:
            query_text = f"{query_text} {normalize_text(origin_super_area)}"

        ranked = self._rank(
            utterance=query_text,
            docs=[doc["semantic_text"] for doc in command_docs],
            limit=limit,
        )
        results: list[SemanticEntityCandidate] = []
        utterance_norm = normalize_text(utterance)
        for idx, score in ranked:
            doc = command_docs[idx]
            adjusted = score + self._number_agreement_bonus(
                utterance=utterance_norm,
                target_name=doc["target_name"],
                synthetic=doc["synthetic"],
                singular=doc["singular"],
            )
            results.append(
                SemanticEntityCandidate(
                    action=doc["action"],
                    target_name=doc["target_name"],
                    tool_group=doc["tool_group"],
                    score=adjusted,
                    synthetic=doc["synthetic"],
                    singular=doc["singular"],
                )
            )
        deduped: dict[tuple[str, str], SemanticEntityCandidate] = {}
        for candidate in results:
            key = (candidate.action, candidate.target_name)
            current = deduped.get(key)
            if current is None or candidate.score > current.score:
                deduped[key] = candidate
        return sorted(deduped.values(), key=lambda item: item.score, reverse=True)[:limit]

    def _rank(self, *, utterance: str, docs: list[str], limit: int) -> list[tuple[int, float]]:
        embedder = self._get_embedder()
        if embedder is None or not utterance or not docs:
            return []
        texts = [utterance, *docs]
        vectors = list(embedder.embed(texts))
        if len(vectors) != len(texts):
            return []
        utter_vector = vectors[0]
        results: list[tuple[int, float]] = []
        for idx, doc_vector in enumerate(vectors[1:]):
            score = self._cosine(utter_vector, doc_vector)
            results.append((idx, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:limit]

    def _get_embedder(self) -> Any | None:
        if self._embedder is not None:
            return self._embedder
        if self._model_error is not None:
            return None
        try:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
                lazy_load=False,
            )
            return self._embedder
        except Exception as err:  # pragma: no cover - dependency/runtime specific
            self._model_error = str(err)
            return None

    def _cosine(self, left: Any, right: Any) -> float:
        left_arr = np.asarray(left, dtype=np.float32)
        right_arr = np.asarray(right, dtype=np.float32)
        left_norm = float(np.linalg.norm(left_arr))
        right_norm = float(np.linalg.norm(right_arr))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left_arr, right_arr) / (left_norm * right_norm))

    def _semantic_phrase_text(self, text: str) -> str:
        if not text:
            return ""
        stripped = SLOT_RE.sub(lambda match: f" {normalize_text(match.group(1))} ", text)
        stripped = GRAMMAR_RE.sub(" ", stripped)
        return re.sub(r"\s+", " ", normalize_text(stripped)).strip()

    def _is_concrete_phrase(self, pattern: str) -> bool:
        return not CONCRETE_PATTERN_RE.search(pattern)

    def _conversation_target_patterns(self, target: ConversationTarget) -> Iterable[str]:
        if target.canonical_phrase:
            yield target.canonical_phrase
        for phrase in target.sample_phrases:
            if phrase:
                yield phrase

    def _dedupe_phrase_candidates(
        self,
        candidates: list[SemanticPhraseCandidate],
        *,
        limit: int,
    ) -> list[SemanticPhraseCandidate]:
        best_by_target: dict[str, SemanticPhraseCandidate] = {}
        for candidate in candidates:
            current = best_by_target.get(candidate.target_id)
            if current is None or candidate.score > current.score:
                best_by_target[candidate.target_id] = candidate
        return sorted(best_by_target.values(), key=lambda item: item.score, reverse=True)[:limit]

    def _number_agreement_bonus(
        self,
        *,
        utterance: str,
        target_name: str,
        synthetic: bool,
        singular: bool,
    ) -> float:
        tokens = set(utterance.split())
        if synthetic and any(token in tokens for token in {"fans", "lights", "lamps", "timers", "alarms", "reminders"}):
            return 0.02
        if singular and any(token in tokens for token in {"fan", "light", "lamp", "timer", "alarm", "reminder"}):
            return 0.02
        if synthetic and target_name.endswith("lights") and not (
            tokens & {"light", "lights", "lamp", "lamps"}
        ):
            return 0.03
        if synthetic and any(token in tokens for token in {"fan", "light", "lamp", "timer", "alarm", "reminder"}):
            return -0.03
        if singular and any(token in tokens for token in {"fans", "lights", "lamps", "timers", "alarms", "reminders"}):
            return -0.02
        return 0.0
