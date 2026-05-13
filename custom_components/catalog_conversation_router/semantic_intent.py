"""Semantic intent retrieval for local phrase/entity matching."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

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
ENTITY_FAMILY_AMBIGUITY_GAP = 0.04
REQUEST_GENERAL_BYPASS_THRESHOLD = 0.84
REQUEST_TOOL_HINT_THRESHOLD = 0.72
GENERAL_EXEMPLARS: tuple[tuple[str, str], ...] = (
    ("general", "how do i boil eggs"),
    ("general", "explain this concept to me"),
    ("general", "tell me a joke"),
    ("general", "summarize this article"),
    ("general", "write an email for me"),
    ("general", "what does this mean"),
    ("general", "help me brainstorm ideas"),
    ("general", "who is the president"),
)


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


@dataclass(slots=True)
class SemanticRequestClassification:
    """Top-level semantic request classification."""

    kind: str
    confidence: float
    intent_family: str | None
    reason: str | None = None
    debug: dict[str, Any] | None = None


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

    def classify_request(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        command_docs: list[dict[str, Any]],
        infer_tool_group: Any,
        limit: int = 5,
    ) -> SemanticRequestClassification | None:
        docs: list[tuple[str, str | None, str]] = []
        for target in catalog.conversation_targets:
            tool_group = infer_tool_group(target.canonical_phrase)
            intent_family = infer_phrase_intent_family(
                target.canonical_phrase,
                tool_group=tool_group,
            )
            for pattern in self._conversation_target_patterns(target):
                semantic_text = self._semantic_phrase_text(pattern)
                if semantic_text:
                    docs.append(("tool_request", intent_family, semantic_text))

        for doc in command_docs:
            semantic_text = str(doc.get("semantic_text") or "").strip()
            if semantic_text:
                docs.append(
                    (
                        "tool_request",
                        str(doc.get("intent_family") or infer_command_intent_family(doc.get("action"))),
                        semantic_text,
                    )
                )

        for family, text in GENERAL_EXEMPLARS:
            docs.append(("general_request", family, text))

        ranked = self._rank(
            utterance=self._semantic_phrase_text(utterance),
            docs=[doc_text for _, _, doc_text in docs],
        )
        if not ranked:
            return None

        best_tool_score = 0.0
        best_tool_family: str | None = None
        best_tool_text: str | None = None
        best_general_score = 0.0
        best_general_family: str | None = None
        best_general_text: str | None = None
        debug_candidates: list[dict[str, Any]] = []
        for idx, score in ranked:
            kind, family, text = docs[idx]
            if len(debug_candidates) < limit:
                debug_candidates.append(
                    {
                        "kind": kind,
                        "intent_family": family,
                        "text": text,
                        "score": round(score, 4),
                    }
                )
            if kind == "tool_request" and score > best_tool_score:
                best_tool_score = score
                best_tool_family = family
                best_tool_text = text
            elif kind == "general_request" and score > best_general_score:
                best_general_score = score
                best_general_family = family
                best_general_text = text

        if best_general_score >= best_tool_score:
            kind = "general_request"
            winning_score = best_general_score
            losing_score = best_tool_score
            intent_family = best_general_family or "general"
            winner_text = best_general_text
        else:
            kind = "tool_request"
            winning_score = best_tool_score
            losing_score = best_general_score
            intent_family = best_tool_family
            winner_text = best_tool_text

        confidence = max(
            0.0,
            min(1.0, winning_score + (0.5 * max(0.0, winning_score - losing_score))),
        )
        return SemanticRequestClassification(
            kind=kind,
            confidence=confidence,
            intent_family=intent_family,
            reason=winner_text,
            debug={
                "top_candidates": debug_candidates,
                "tool_score": round(best_tool_score, 4),
                "general_score": round(best_general_score, 4),
            },
        )

    def _rank(self, *, utterance: str, docs: list[str]) -> list[tuple[int, float]]:
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
        return results

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
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
        if len(left_values) != len(right_values):
            return 0.0
        left_norm = sum(value * value for value in left_values) ** 0.5
        right_norm = sum(value * value for value in right_values) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot = sum(left_value * right_value for left_value, right_value in zip(left_values, right_values, strict=False))
        return float(dot / (left_norm * right_norm))

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


def infer_command_intent_family(action: Any) -> str:
    """Infer a coarse family for an entity command."""
    if str(action) == "query":
        return "entity_query"
    return "entity_control"


def infer_phrase_intent_family(text: str, *, tool_group: str | None = None) -> str:
    """Infer a coarse phrase family for classification/debugging."""
    normalized = normalize_text(text)
    if tool_group == "timers" or any(token in normalized for token in ("timer", "alarm", "reminder")):
        return "timer"
    if any(token in normalized for token in ("forecast", "weather")):
        return "weather"
    if any(token in normalized for token in ("camera", "cam", "feed", "view")):
        return "camera"
    if any(token in normalized for token in ("spa", "hot tub", "pool lights", "lights on", "lights off")):
        return "entity_control"
    return tool_group or "conversation_phrase"
