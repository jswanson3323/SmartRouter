"""Semantic ranking service for the Home Assistant add-on."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import re
from pathlib import Path
import threading
from typing import Any

from fastapi import FastAPI
from fastembed import TextEmbedding
from pydantic import BaseModel, Field

GRAMMAR_RE = re.compile(r"[\[\]\(\)\|]")
SLOT_RE = re.compile(r"\{([^}]+)\}")
CONCRETE_PATTERN_RE = re.compile(r"[\[\]\(\)\{\}\|]")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8099
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_DOC_VECTOR_CACHE_BYTES = 100 * 1024 * 1024
QUERY_OPENERS = (
    "what is",
    "whats",
    "what are",
    "status of",
    "status for",
    "what s",
    "how is",
    "how are",
)
QUERY_HINT_TOKENS = {
    "status",
    "state",
    "temperature",
    "temp",
    "on",
    "off",
    "open",
    "closed",
    "locked",
    "unlocked",
}

app = FastAPI(title="Catalog Router Semantic Service")


class PhraseTarget(BaseModel):
    target_id: str
    tool_group: str
    intent_family: str | None = None
    patterns: list[str] = Field(default_factory=list)


class PhraseRequest(BaseModel):
    utterance: str
    limit: int = 8
    phrases: list[PhraseTarget] = Field(default_factory=list)


class CommandDoc(BaseModel):
    action: str
    target_name: str
    tool_group: str
    intent_family: str | None = None
    synthetic: bool
    singular: bool
    area: str | None = None
    super_area: str | None = None
    semantic_text: str


class EntityRequest(BaseModel):
    utterance: str
    origin_area: str | None = None
    origin_super_area: str | None = None
    limit: int = 8
    commands: list[CommandDoc] = Field(default_factory=list)


class ClassificationRequest(BaseModel):
    utterance: str
    limit: int = 5
    phrases: list[PhraseTarget] = Field(default_factory=list)
    commands: list[CommandDoc] = Field(default_factory=list)


class SemanticService:
    def __init__(self) -> None:
        options = self._load_options()
        self.model_name = str(options.get("model_name") or DEFAULT_MODEL_NAME)
        self.embedder = TextEmbedding(model_name=self.model_name, lazy_load=False)
        self._doc_vector_caches: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._doc_vector_cache_bytes = 0

    def rank_phrase(self, request: PhraseRequest) -> dict[str, Any]:
        docs: list[tuple[str, str, str, bool, bool, str]] = []
        for target in request.phrases:
            for pattern in target.patterns:
                semantic_text = self._semantic_phrase_text(pattern)
                if not semantic_text:
                    continue
                docs.append(
                    (
                        target.target_id,
                        pattern,
                        semantic_text,
                        not CONCRETE_PATTERN_RE.search(pattern),
                        bool(SLOT_RE.search(pattern)),
                        target.tool_group,
                    )
                )

        doc_texts = [doc[2] for doc in docs]
        ranked = self._rank_cached(
            self._semantic_phrase_text(request.utterance),
            doc_texts,
            cache_namespace="phrase",
        )
        candidates = []
        seen: set[str] = set()
        for idx, score in ranked:
            target_id, raw_text, example_text, concrete, has_slots, tool_group = docs[idx]
            if target_id in seen:
                continue
            seen.add(target_id)
            candidates.append(
                {
                    "target_id": target_id,
                    "raw_text": raw_text,
                    "example_text": example_text,
                    "concrete": concrete,
                    "has_slots": has_slots,
                    "tool_group": tool_group,
                    "score": score,
                }
            )
        return {"candidates": candidates}

    def rank_entity(self, request: EntityRequest) -> dict[str, Any]:
        query_text = self._normalize_text(request.utterance)
        utterance_norm = self._normalize_text(request.utterance)

        ranked = self._rank_cached(
            query_text,
            [doc.semantic_text for doc in request.commands],
            cache_namespace="entity",
        )
        candidates = []
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for idx, score in ranked:
            doc = request.commands[idx]
            adjusted = score + self._number_agreement_bonus(
                utterance=utterance_norm,
                target_name=doc.target_name,
                synthetic=doc.synthetic,
                singular=doc.singular,
            )
            adjusted += self._target_overlap_bonus(
                utterance=utterance_norm,
                target_name=doc.target_name,
            )
            adjusted += self._origin_area_bonus(
                utterance=utterance_norm,
                area=doc.area,
                super_area=doc.super_area,
                origin_area=request.origin_area,
                origin_super_area=request.origin_super_area,
            )
            adjusted += self._query_action_bonus(
                utterance=utterance_norm,
                action=doc.action,
            )
            item = {
                "action": doc.action,
                "target_name": doc.target_name,
                "tool_group": doc.tool_group,
                "synthetic": doc.synthetic,
                "singular": doc.singular,
                "score": adjusted,
            }
            key = (doc.action, doc.target_name)
            current = deduped.get(key)
            if current is None or item["score"] > current["score"]:
                deduped[key] = item
        candidates = sorted(deduped.values(), key=lambda item: item["score"], reverse=True)[
            : request.limit
        ]
        return {"candidates": candidates}

    def classify_request(self, request: ClassificationRequest) -> dict[str, Any]:
        docs: list[tuple[str, str | None, str]] = []
        for target in request.phrases:
            for pattern in target.patterns:
                semantic_text = self._semantic_phrase_text(pattern)
                if semantic_text:
                    docs.append(("tool_request", target.intent_family, semantic_text))

        for command in request.commands:
            if command.semantic_text:
                docs.append(
                    (
                        "tool_request",
                        command.intent_family,
                        self._normalize_text(command.semantic_text),
                    )
                )

        for family, text in (
            ("general", "how do i boil eggs"),
            ("general", "explain this concept to me"),
            ("general", "tell me a joke"),
            ("general", "summarize this article"),
            ("general", "write an email for me"),
            ("general", "what does this mean"),
            ("general", "help me brainstorm ideas"),
            ("general", "who is the president"),
        ):
            docs.append(("general_request", family, text))

        ranked = self._rank_cached(
            self._semantic_phrase_text(request.utterance),
            [doc[2] for doc in docs],
            cache_namespace="classification",
        )
        if not ranked:
            return {
                "kind": "general_request",
                "confidence": 0.0,
                "intent_family": "general",
                "reason": None,
                "debug": {"top_candidates": []},
            }

        best_tool_score = 0.0
        best_tool_family: str | None = None
        best_tool_text: str | None = None
        best_general_score = 0.0
        best_general_family: str | None = None
        best_general_text: str | None = None
        debug_candidates: list[dict[str, Any]] = []
        for idx, score in ranked:
            kind, family, text = docs[idx]
            adjusted_score = score + self._classification_intent_bonus(
                utterance=self._semantic_phrase_text(request.utterance),
                intent_family=family,
            )
            if len(debug_candidates) < request.limit:
                debug_candidates.append(
                    {
                        "kind": kind,
                        "intent_family": family,
                        "text": text,
                        "score": round(adjusted_score, 4),
                    }
                )
            if kind == "tool_request" and adjusted_score > best_tool_score:
                best_tool_score = adjusted_score
                best_tool_family = family
                best_tool_text = text
            elif kind == "general_request" and adjusted_score > best_general_score:
                best_general_score = adjusted_score
                best_general_family = family
                best_general_text = text

        if best_general_score >= best_tool_score:
            kind = "general_request"
            winning_score = best_general_score
            losing_score = best_tool_score
            intent_family = best_general_family or "general"
            reason = best_general_text
        else:
            kind = "tool_request"
            winning_score = best_tool_score
            losing_score = best_general_score
            intent_family = best_tool_family
            reason = best_tool_text

        confidence = max(
            0.0,
            min(1.0, winning_score + (0.5 * max(0.0, winning_score - losing_score))),
        )
        return {
            "kind": kind,
            "confidence": confidence,
            "intent_family": intent_family,
            "reason": reason,
            "debug": {
                "tool_score": round(best_tool_score, 4),
                "general_score": round(best_general_score, 4),
                "top_candidates": debug_candidates,
            },
        }

    def _rank(self, utterance: str, docs: list[str]) -> list[tuple[int, float]]:
        if not utterance or not docs:
            return []
        vectors = list(self.embedder.embed([utterance, *docs]))
        utter_vector = vectors[0]
        results: list[tuple[int, float]] = []
        for idx, doc_vector in enumerate(vectors[1:]):
            results.append((idx, self._cosine(utter_vector, doc_vector)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def _rank_cached(
        self,
        utterance: str,
        docs: list[str],
        *,
        cache_namespace: str,
    ) -> list[tuple[int, float]]:
        if not utterance or not docs:
            return []
        doc_vectors = self._get_cached_doc_vectors(
            cache_namespace=cache_namespace,
            docs=docs,
        )
        utter_vector = list(self.embedder.embed([utterance]))[0]
        results: list[tuple[int, float]] = []
        for idx, doc_vector in enumerate(doc_vectors):
            results.append((idx, self._cosine(utter_vector, doc_vector)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def _get_cached_doc_vectors(
        self,
        *,
        cache_namespace: str,
        docs: list[str],
    ) -> list[Any]:
        cache_key = self._doc_cache_key(cache_namespace=cache_namespace, docs=docs)
        with self._cache_lock:
            cache_entry = self._doc_vector_caches.get(cache_key)
            if cache_entry is not None:
                self._doc_vector_caches.move_to_end(cache_key)
                return cache_entry["vectors"]

        vectors = list(self.embedder.embed(docs))
        vector_bytes = self._estimate_vectors_bytes(vectors)

        with self._cache_lock:
            self._doc_vector_caches[cache_key] = {
                "namespace": cache_namespace,
                "doc_count": len(docs),
                "bytes": vector_bytes,
                "vectors": vectors,
            }
            self._doc_vector_cache_bytes += vector_bytes
            self._doc_vector_caches.move_to_end(cache_key)
            while self._doc_vector_cache_bytes > MAX_DOC_VECTOR_CACHE_BYTES:
                _evicted_key, evicted_entry = self._doc_vector_caches.popitem(last=False)
                self._doc_vector_cache_bytes -= int(evicted_entry.get("bytes", 0))
            self._doc_vector_cache_bytes = max(0, self._doc_vector_cache_bytes)

        return vectors

    def _doc_cache_key(self, *, cache_namespace: str, docs: list[str]) -> str:
        payload = {
            "namespace": cache_namespace,
            "model_name": self.model_name,
            "docs": docs,
        }
        key_json = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def _estimate_vectors_bytes(self, vectors: list[Any]) -> int:
        total = 0
        for vector in vectors:
            nbytes = getattr(vector, "nbytes", None)
            if isinstance(nbytes, int):
                total += nbytes
                continue
            try:
                total += len(vector) * 8
            except TypeError:
                total += 8
        return total

    def _looks_like_query_utterance(self, utterance: str) -> bool:
        normalized = self._normalize_text(utterance)
        if not normalized:
            return False
        if any(normalized.startswith(prefix) for prefix in QUERY_OPENERS):
            return True
        tokens = set(normalized.split())
        return "status" in tokens or "state" in tokens or (
            ("what" in tokens or "how" in tokens)
            and bool(tokens & QUERY_HINT_TOKENS)
        )

    def _query_action_bonus(self, *, utterance: str, action: str) -> float:
        if not self._looks_like_query_utterance(utterance):
            return 0.0
        if action == "query":
            return 0.12
        if action in {"turn_on", "turn_off", "open", "close", "lock", "unlock", "start", "cancel"}:
            return -0.08
        return 0.0

    def _classification_intent_bonus(
        self,
        *,
        utterance: str,
        intent_family: str | None,
    ) -> float:
        if not self._looks_like_query_utterance(utterance):
            return 0.0
        if intent_family == "entity_query":
            return 0.1
        if intent_family in {"entity_control", "compound_entity_control"}:
            return -0.06
        return 0.0

    def _semantic_phrase_text(self, text: str) -> str:
        stripped = SLOT_RE.sub(lambda match: f" {self._normalize_text(match.group(1))} ", text)
        stripped = GRAMMAR_RE.sub(" ", stripped)
        return re.sub(r"\s+", " ", self._normalize_text(stripped)).strip()

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\s]+", " ", text.lower()).strip()

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

    def _target_overlap_bonus(self, *, utterance: str, target_name: str) -> float:
        utterance_tokens = set(utterance.split())
        target_tokens = [token for token in self._normalize_text(target_name).split() if token not in {"the"}]
        if not utterance_tokens or not target_tokens:
            return 0.0
        overlap = sum(1 for token in target_tokens if token in utterance_tokens)
        if overlap == 0:
            return 0.0
        if " ".join(target_tokens) in utterance:
            return 0.1
        return min(0.08, 0.08 * (overlap / len(target_tokens)))

    def _origin_area_bonus(
        self,
        *,
        utterance: str,
        area: str | None,
        super_area: str | None,
        origin_area: str | None,
        origin_super_area: str | None,
    ) -> float:
        utterance_norm = self._normalize_text(utterance)
        if area and self._normalize_text(area) and self._normalize_text(area) in utterance_norm:
            return 0.0
        if super_area and self._normalize_text(super_area) and self._normalize_text(super_area) in utterance_norm:
            return 0.0
        if origin_area:
            normalized_origin_area = self._normalize_text(origin_area)
            normalized_area = self._normalize_text(area) if area else ""
            if normalized_area:
                if normalized_origin_area == normalized_area:
                    return 0.18
                return -0.06
        if origin_super_area:
            normalized_origin_super_area = self._normalize_text(origin_super_area)
            normalized_super_area = self._normalize_text(super_area) if super_area else ""
            if normalized_super_area:
                if normalized_origin_super_area == normalized_super_area:
                    return 0.1
                return -0.03
        return 0.0

    def _load_options(self) -> dict[str, Any]:
        options_path = Path("/data/options.json")
        if not options_path.exists():
            return {}
        try:
            return json.loads(options_path.read_text(encoding="utf-8"))
        except Exception:
            return {}


SERVICE = SemanticService()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_name": SERVICE.model_name,
        "doc_vector_cache_count": len(SERVICE._doc_vector_caches),
        "doc_vector_cache_bytes": SERVICE._doc_vector_cache_bytes,
    }


@app.post("/rank/phrase")
def rank_phrase(request: PhraseRequest) -> dict[str, Any]:
    return SERVICE.rank_phrase(request)


@app.post("/rank/entity")
def rank_entity(request: EntityRequest) -> dict[str, Any]:
    return SERVICE.rank_entity(request)


@app.post("/classify/request")
def classify_request(request: ClassificationRequest) -> dict[str, Any]:
    return SERVICE.classify_request(request)


if __name__ == "__main__":
    import uvicorn

    options_path = Path("/data/options.json")
    options: dict[str, Any] = {}
    if options_path.exists():
        try:
            options = json.loads(options_path.read_text(encoding="utf-8"))
        except Exception:
            options = {}
    host = str(options.get("host") or os.getenv("CCR_HOST") or DEFAULT_HOST)
    port = int(options.get("port") or os.getenv("CCR_PORT") or DEFAULT_PORT)
    uvicorn.run(app, host=host, port=port)
