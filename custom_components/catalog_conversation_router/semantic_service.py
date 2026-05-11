"""Optional HTTP client for semantic intent ranking."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .models import Catalog
from .phonetics import normalize_text
from .semantic_intent import SemanticEntityCandidate, SemanticPhraseCandidate


class RemoteSemanticIntentRanker:
    """Use an external semantic service for intent ranking."""

    def __init__(self, service_url: str) -> None:
        self._service_url = service_url.rstrip("/")
        self._last_error: str | None = None

    def available(self) -> bool:
        return bool(self._service_url)

    def unavailable_reason(self) -> str | None:
        return self._last_error

    def rank_phrase_candidates(
        self,
        *,
        utterance: str,
        catalog: Catalog,
        infer_tool_group: Any,
        limit: int = 8,
    ) -> list[SemanticPhraseCandidate]:
        payload = {
            "utterance": utterance,
            "limit": limit,
            "phrases": [
                {
                    "target_id": target.target_id,
                    "tool_group": infer_tool_group(target.canonical_phrase),
                    "patterns": [
                        phrase
                        for phrase in [target.canonical_phrase, *target.sample_phrases]
                        if phrase
                    ],
                }
                for target in catalog.conversation_targets
            ],
        }
        data = self._post_json("/rank/phrase", payload)
        if data is None:
            return []
        return [
            SemanticPhraseCandidate(
                target_id=str(item["target_id"]),
                example_text=str(item["example_text"]),
                raw_text=str(item["raw_text"]),
                score=float(item["score"]),
                concrete=bool(item["concrete"]),
                has_slots=bool(item["has_slots"]),
                tool_group=str(item["tool_group"]),
            )
            for item in data.get("candidates", [])
        ]

    def rank_entity_commands(
        self,
        *,
        utterance: str,
        command_docs: list[dict[str, Any]],
        origin_area: str | None,
        origin_super_area: str | None,
        limit: int = 8,
    ) -> list[SemanticEntityCandidate]:
        payload = {
            "utterance": utterance,
            "origin_area": normalize_text(origin_area) if origin_area else None,
            "origin_super_area": normalize_text(origin_super_area) if origin_super_area else None,
            "limit": limit,
            "commands": command_docs,
        }
        data = self._post_json("/rank/entity", payload)
        if data is None:
            return []
        return [
            SemanticEntityCandidate(
                action=str(item["action"]),
                target_name=str(item["target_name"]),
                tool_group=str(item["tool_group"]),
                score=float(item["score"]),
                synthetic=bool(item["synthetic"]),
                singular=bool(item["singular"]),
            )
            for item in data.get("candidates", [])
        ]

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        url = f"{self._service_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=0.75) as response:
                response_body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, ValueError) as err:
            self._last_error = str(err)
            return None
        try:
            return json.loads(response_body)
        except json.JSONDecodeError as err:
            self._last_error = str(err)
            return None
