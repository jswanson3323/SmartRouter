"""Catalog-grounded fuzzy matcher."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import CandidateScore, CandidateType, Catalog, MatchResult
from .phonetics import normalize_text, phonetic_key, phonetic_tokens, tokenize

ACTION_MAP = {
    "turn on": "turn_on",
    "switch on": "turn_on",
    "turn off": "turn_off",
    "switch off": "turn_off",
    "activate": "activate",
    "deactivate": "deactivate",
    "set": "set",
    "what is": "query",
    "whats": "query",
    "status": "query",
    "lock": "lock",
    "unlock": "unlock",
    "arm": "arm",
    "disarm": "disarm",
    "open": "open",
    "close": "close",
    "enable": "enable",
    "disable": "disable",
}

CANONICAL_ACTION_TEXT = {
    "turn_on": "turn on",
    "turn_off": "turn off",
    "activate": "activate",
    "deactivate": "deactivate",
    "query": "what is",
    "set": "set",
    "lock": "lock",
    "unlock": "unlock",
    "arm": "arm",
    "disarm": "disarm",
    "open": "open",
    "close": "close",
    "enable": "enable",
    "disable": "disable",
}


@dataclass(slots=True)
class ParsedUtterance:
    """Action/target parse from utterance."""

    action: str | None
    action_phrase: str | None
    target_phrase: str
    area_hint: str | None


class FuzzyMatcher:
    """Deterministic fuzzy matcher with multi-signal scoring."""

    def __init__(self, fuzzy_threshold: float, ambiguity_gap: float) -> None:
        self._fuzzy_threshold = fuzzy_threshold
        self._ambiguity_gap = ambiguity_gap

    def parse_utterance(self, utterance: str) -> ParsedUtterance:
        """Infer action and target slices from utterance."""
        normalized = normalize_text(utterance)

        chosen_action_phrase: str | None = None
        chosen_action: str | None = None
        for phrase in sorted(ACTION_MAP, key=len, reverse=True):
            if normalized.startswith(f"{phrase} ") or normalized == phrase:
                chosen_action_phrase = phrase
                chosen_action = ACTION_MAP[phrase]
                break

        target = normalized
        if chosen_action_phrase:
            target = normalized.removeprefix(chosen_action_phrase).strip()

        # Remove leading determiners from the extracted target phrase.
        target_tokens = target.split()
        while target_tokens and target_tokens[0] in {"the", "a", "an"}:
            target_tokens.pop(0)
        target = " ".join(target_tokens)

        area_hint = None
        # Parse location qualifiers using word boundaries to avoid splitting inside words
        # (e.g., "great" must not match "at").
        qualifier_match = re.search(r"\b(in|at|on)\b\s+(.+)$", target)
        if qualifier_match:
            before = target[: qualifier_match.start()].strip()
            after = qualifier_match.group(2).strip()
            if before and after:
                area_hint = after.split(" ")[0]
                target = before

        return ParsedUtterance(
            action=chosen_action,
            action_phrase=chosen_action_phrase,
            target_phrase=target,
            area_hint=area_hint,
        )

    def match(self, utterance: str, catalog: Catalog) -> MatchResult:
        """Score entity and conversation target candidates."""
        normalized = normalize_text(utterance)
        parsed = self.parse_utterance(utterance)
        parsed_target_before = normalize_text(parsed.target_phrase or normalized)
        utter_tokens = self._normalize_asr_target_tokens(parsed_target_before)
        parsed_target_after = " ".join(utter_tokens)
        utter_phonetic = set(phonetic_tokens(utter_tokens))

        scores: list[CandidateScore] = []

        for entity in catalog.entity_targets:
            target_tokens = entity.tokens
            target_phonetic = set(entity.phonetic_tokens)
            alias_scores = [
                self._token_similarity(utter_tokens, tokenize(alias)) for alias in entity.aliases
            ]
            alias_similarity = max(alias_scores) if alias_scores else 0.0

            score_detail = self._score_signals(
                utter_tokens=utter_tokens,
                utter_phonetic=utter_phonetic,
                target_tokens=target_tokens,
                target_phonetic=target_phonetic,
                utter_target_text=parsed_target_after,
                candidate_name=entity.normalized_name,
                alias_similarity=alias_similarity,
                action=parsed.action,
                capabilities=entity.capabilities,
                area_hint=parsed.area_hint,
                candidate_area=entity.area,
                candidate_phrase=entity.normalized_name,
            )
            final_score = self._weighted_score(score_detail)
            scores.append(
                CandidateScore(
                    candidate_id=entity.entity_id,
                    candidate_type=CandidateType.ENTITY_COMMAND,
                    canonical_phrase=self._build_entity_canonical(parsed.action, entity.name),
                    score=final_score,
                    action=parsed.action,
                    target_name=entity.name,
                    detail={
                        **score_detail,
                        "parsed_target_before_normalization": parsed_target_before,
                        "parsed_target_after_normalization": parsed_target_after,
                    },
                )
            )

        for target in catalog.conversation_targets:
            if not target.enabled:
                continue
            target_tokens = target.tokens
            target_phonetic = set(target.phonetic_tokens)
            alias_scores = [
                self._token_similarity(utter_tokens, tokenize(alias)) for alias in target.aliases
            ]
            alias_similarity = max(alias_scores) if alias_scores else 0.0

            score_detail = self._score_signals(
                utter_tokens=utter_tokens,
                utter_phonetic=utter_phonetic,
                target_tokens=target_tokens,
                target_phonetic=target_phonetic,
                utter_target_text=parsed_target_after,
                candidate_name=target.normalized_name,
                alias_similarity=alias_similarity,
                action=parsed.action,
                capabilities=["activate", "query", "set"],
                area_hint=parsed.area_hint,
                candidate_area=None,
                candidate_phrase=target.normalized_name,
            )
            final_score = self._weighted_score(score_detail)
            scores.append(
                CandidateScore(
                    candidate_id=target.target_id,
                    candidate_type=CandidateType.CONVERSATION_TARGET,
                    canonical_phrase=target.canonical_phrase,
                    score=final_score,
                    action=parsed.action,
                    target_name=target.display_name,
                    detail={
                        **score_detail,
                        "parsed_target_before_normalization": parsed_target_before,
                        "parsed_target_after_normalization": parsed_target_after,
                    },
                )
            )

        ranked = sorted(scores, key=lambda item: item.score, reverse=True)
        ranked = self._dedupe_by_canonical_phrase(ranked)
        top = ranked[:3]
        best = top[0] if top else None
        second = top[1] if len(top) > 1 else None

        matched = bool(
            best
            and best.score >= self._fuzzy_threshold
            and (best.score - (second.score if second else 0.0)) >= self._ambiguity_gap
        )

        return MatchResult(
            matched=matched,
            best=best,
            top_candidates=top,
            inferred_action=parsed.action,
            normalized_utterance=normalized,
            parsed_target_before_normalization=parsed_target_before,
            parsed_target_after_normalization=parsed_target_after,
        )

    def _weighted_score(self, detail: dict[str, float]) -> float:
        return (
            0.30 * detail["token_similarity"]
            + 0.20 * detail["whole_target_similarity"]
            + 0.15 * detail["phonetic_similarity"]
            + 0.10 * detail["edit_similarity"]
            + 0.10 * detail["action_compatibility"]
            + 0.05 * detail["alias_similarity"]
            + 0.10 * detail["structure_similarity"]
        )

    def _score_signals(
        self,
        *,
        utter_tokens: list[str],
        utter_phonetic: set[str],
        target_tokens: list[str],
        target_phonetic: set[str],
        utter_target_text: str,
        candidate_name: str,
        alias_similarity: float,
        action: str | None,
        capabilities: list[str],
        area_hint: str | None,
        candidate_area: str | None,
        candidate_phrase: str,
    ) -> dict[str, float]:
        token_similarity = self._token_similarity(utter_tokens, target_tokens)
        phonetic_similarity = self._set_similarity(utter_phonetic, target_phonetic)
        edit_similarity = SequenceMatcher(a=utter_target_text, b=candidate_name).ratio()
        whole_target_similarity = self._whole_target_similarity(utter_target_text, candidate_phrase)
        action_compatibility = self._action_compatibility(action, capabilities)
        structure_similarity = self._structure_similarity(utter_tokens, target_tokens)

        if area_hint and candidate_area and area_hint in candidate_area.lower():
            structure_similarity = min(1.0, structure_similarity + 0.10)

        return {
            "token_similarity": token_similarity,
            "whole_target_similarity": whole_target_similarity,
            "phonetic_similarity": phonetic_similarity,
            "edit_similarity": edit_similarity,
            "action_compatibility": action_compatibility,
            "alias_similarity": alias_similarity,
            "structure_similarity": structure_similarity,
        }

    def _token_similarity(self, left: list[str], right: list[str]) -> float:
        return self._set_similarity(set(left), set(right))

    def _set_similarity(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _action_compatibility(self, action: str | None, capabilities: list[str]) -> float:
        if action is None:
            return 0.6
        if action in {"turn_on", "turn_off"}:
            return 1.0
        if action in capabilities:
            return 1.0
        if action == "query" and "query" in capabilities:
            return 1.0
        return 0.2

    def _structure_similarity(self, utter_tokens: list[str], target_tokens: list[str]) -> float:
        if not utter_tokens or not target_tokens:
            return 0.0
        ratio = min(len(utter_tokens), len(target_tokens)) / max(
            len(utter_tokens), len(target_tokens)
        )
        first_token_bonus = 0.2 if utter_tokens[0] == target_tokens[0] else 0.0
        return min(1.0, ratio + first_token_bonus)

    def _build_entity_canonical(self, action: str | None, name: str) -> str:
        prefix = CANONICAL_ACTION_TEXT.get(action or "", "activate")
        if prefix == "what is":
            return f"what is {name}"
        return f"{prefix} {name}"

    def _normalize_asr_target_tokens(self, target_phrase: str) -> list[str]:
        """Normalize likely ASR noun substitutions before scoring."""
        return tokenize(target_phrase)

    def _whole_target_similarity(self, parsed_target: str, candidate_target: str) -> float:
        """Compare full target phrases using edit and phonetic phrase similarity."""
        candidate_normalized = normalize_text(candidate_target)
        edit_ratio = SequenceMatcher(a=parsed_target, b=candidate_normalized).ratio()

        parsed_phonetic = " ".join(phonetic_key(token) for token in parsed_target.split())
        candidate_phonetic = " ".join(phonetic_key(token) for token in candidate_normalized.split())
        phonetic_ratio = SequenceMatcher(a=parsed_phonetic, b=candidate_phonetic).ratio()
        return (edit_ratio + phonetic_ratio) / 2

    def _dedupe_by_canonical_phrase(self, ranked: list[CandidateScore]) -> list[CandidateScore]:
        """Deduplicate by canonical phrase, keeping highest-score candidate for each phrase."""
        deduped: list[CandidateScore] = []
        seen: set[str] = set()
        for candidate in ranked:
            key = normalize_text(candidate.canonical_phrase)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped
