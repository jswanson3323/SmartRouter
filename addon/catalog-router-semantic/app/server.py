"""Semantic ranking service for the Home Assistant add-on."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
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

app = FastAPI(title="Catalog Router Semantic Service")


class PhraseTarget(BaseModel):
    target_id: str
    tool_group: str
    patterns: list[str] = Field(default_factory=list)


class PhraseRequest(BaseModel):
    utterance: str
    limit: int = 8
    phrases: list[PhraseTarget] = Field(default_factory=list)


class CommandDoc(BaseModel):
    action: str
    target_name: str
    tool_group: str
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


class SemanticService:
    def __init__(self) -> None:
        options = self._load_options()
        self.model_name = str(options.get("model_name") or DEFAULT_MODEL_NAME)
        self.embedder = TextEmbedding(model_name=self.model_name, lazy_load=False)

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

        ranked = self._rank(self._semantic_phrase_text(request.utterance), [doc[2] for doc in docs], request.limit)
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

        ranked = self._rank(query_text, [doc.semantic_text for doc in request.commands], request.limit)
        candidates = []
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for idx, score in ranked:
            doc = request.commands[idx]
            adjusted = score + self._number_agreement_bonus(
                utterance=self._normalize_text(request.utterance),
                target_name=doc.target_name,
                synthetic=doc.synthetic,
                singular=doc.singular,
            )
            adjusted += self._target_overlap_bonus(
                utterance=self._normalize_text(request.utterance),
                target_name=doc.target_name,
            )
            adjusted += self._origin_area_bonus(
                utterance=self._normalize_text(request.utterance),
                area=doc.area,
                super_area=doc.super_area,
                origin_area=request.origin_area,
                origin_super_area=request.origin_super_area,
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

    def _rank(self, utterance: str, docs: list[str], limit: int) -> list[tuple[int, float]]:
        if not utterance or not docs:
            return []
        vectors = list(self.embedder.embed([utterance, *docs]))
        utter_vector = vectors[0]
        results: list[tuple[int, float]] = []
        for idx, doc_vector in enumerate(vectors[1:]):
            results.append((idx, self._cosine(utter_vector, doc_vector)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:limit]

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
    return {"ok": True, "model_name": SERVICE.model_name}


@app.post("/rank/phrase")
def rank_phrase(request: PhraseRequest) -> dict[str, Any]:
    return SERVICE.rank_phrase(request)


@app.post("/rank/entity")
def rank_entity(request: EntityRequest) -> dict[str, Any]:
    return SERVICE.rank_entity(request)


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
