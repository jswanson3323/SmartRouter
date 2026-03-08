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

    def match(self, utterance: str, catalog: Catalog, origin_area: str | None = None) -> MatchResult:
        """Score entity and conversation target candidates."""
        normalized = normalize_text(utterance)
        parsed = self.parse_utterance(utterance)
        parsed_target_before = normalize_text(parsed.target_phrase or normalized)
        utter_tokens = self._normalize_asr_target_tokens(parsed_target_before)
        parsed_target_after = " ".join(utter_tokens)
        utter_phonetic = set(phonetic_tokens(utter_tokens))
        utter_full_tokens = tokenize(normalized)
        utter_full_phonetic = set(phonetic_tokens(utter_full_tokens))
        effective_area_hint = parsed.area_hint or (normalize_text(origin_area) if origin_area else None)

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
                area_hint=effective_area_hint,
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

            alias_scores = [
                self._token_similarity(utter_tokens, tokenize(alias)) for alias in target.aliases
            ]
            alias_similarity = max(alias_scores) if alias_scores else 0.0

            raw_phrase_candidates = [
                phrase
                for phrase in [*target.sample_phrases, target.canonical_phrase, target.display_name]
                if phrase
            ]
            phrase_candidates: list[str] = []
            for phrase in raw_phrase_candidates:
                phrase_candidates.extend(self._expand_conversation_phrase_variants(phrase))
            if not phrase_candidates:
                phrase_candidates = [target.display_name]

            best_phrase = normalize_text(target.display_name)
            best_target_tokens = target.tokens
            best_target_phonetic = set(target.phonetic_tokens)
            best_score_detail: dict[str, float] | None = None
            best_phrase_score = -1.0
            best_phrase_raw = target.display_name
            best_phrase_for_scoring = normalize_text(target.display_name)

            for phrase in phrase_candidates:
                normalized_phrase = normalize_text(phrase)
                scoring_phrase = self._normalize_conversation_phrase_for_scoring(phrase)
                phrase_tokens = tokenize(scoring_phrase)
                phrase_phonetic = set(phonetic_tokens(phrase_tokens))

                score_detail = self._score_signals(
                    utter_tokens=utter_full_tokens,
                    utter_phonetic=utter_full_phonetic,
                    target_tokens=phrase_tokens,
                    target_phonetic=phrase_phonetic,
                    utter_target_text=normalized,
                    candidate_name=scoring_phrase,
                    alias_similarity=alias_similarity,
                    action=parsed.action,
                    capabilities=self._infer_phrase_capabilities(normalized_phrase),
                    area_hint=effective_area_hint,
                    candidate_area=None,
                    candidate_phrase=scoring_phrase,
                )
                score_detail["token_similarity"] = max(
                    score_detail["token_similarity"],
                    self._target_token_coverage_similarity(utter_full_tokens, phrase_tokens),
                )
                score_detail["phonetic_similarity"] = max(
                    score_detail["phonetic_similarity"],
                    self._target_token_coverage_similarity(
                        list(utter_full_phonetic),
                        list(phrase_phonetic),
                    ),
                )
                score_detail["structure_similarity"] = max(
                    score_detail["structure_similarity"],
                    self._ordered_token_subsequence_similarity(utter_full_tokens, phrase_tokens),
                )
                phrase_score = self._weighted_score(score_detail)
                if phrase_score > best_phrase_score:
                    best_phrase_score = phrase_score
                    best_phrase = normalized_phrase
                    best_phrase_raw = phrase
                    best_phrase_for_scoring = scoring_phrase
                    best_target_tokens = phrase_tokens
                    best_target_phonetic = phrase_phonetic
                    best_score_detail = score_detail

            score_detail = best_score_detail or self._score_signals(
                utter_tokens=utter_full_tokens,
                utter_phonetic=utter_full_phonetic,
                target_tokens=best_target_tokens,
                target_phonetic=best_target_phonetic,
                utter_target_text=normalized,
                candidate_name=best_phrase,
                alias_similarity=alias_similarity,
                action=parsed.action,
                capabilities=self._infer_phrase_capabilities(best_phrase),
                area_hint=effective_area_hint,
                candidate_area=None,
                candidate_phrase=best_phrase,
            )
            score_detail["token_similarity"] = max(
                score_detail["token_similarity"],
                self._target_token_coverage_similarity(utter_full_tokens, best_target_tokens),
            )
            score_detail["phonetic_similarity"] = max(
                score_detail["phonetic_similarity"],
                self._target_token_coverage_similarity(
                    list(utter_full_phonetic),
                    list(best_target_phonetic),
                ),
            )
            score_detail["structure_similarity"] = max(
                score_detail["structure_similarity"],
                self._ordered_token_subsequence_similarity(utter_full_tokens, best_target_tokens),
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
                        "matched_sample_phrase": best_phrase,
                        "matched_sample_phrase_raw": best_phrase_raw,
                        "matched_sample_phrase_normalized_for_scoring": best_phrase_for_scoring,
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
            origin_area=origin_area,
            effective_area_hint=effective_area_hint,
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

        if area_hint and candidate_area:
            normalized_candidate_area = normalize_text(candidate_area)
            area_tokens = set(tokenize(area_hint))
            candidate_area_tokens = set(tokenize(normalized_candidate_area))
            if area_tokens and area_tokens <= candidate_area_tokens:
                structure_similarity = min(1.0, structure_similarity + 0.25)
                token_similarity = min(1.0, token_similarity + 0.15)
                phonetic_similarity = min(1.0, phonetic_similarity + 0.10)

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

    def _target_token_coverage_similarity(self, utter_tokens: list[str], target_tokens: list[str]) -> float:
        """How completely the target tokens are covered by the utterance.

        This is useful for slot-based conversation phrases like `set timer for`,
        where the utterance may contain extra slot words that should not heavily
        penalize the score.
        """
        if not target_tokens:
            return 0.0
        utter_set = set(utter_tokens)
        target_set = set(target_tokens)
        return len(utter_set & target_set) / len(target_set)

    def _ordered_token_subsequence_similarity(self, utter_tokens: list[str], target_tokens: list[str]) -> float:
        """Reward phrases whose fixed tokens appear in utterance order."""
        if not utter_tokens or not target_tokens:
            return 0.0

        filtered_target = [token for token in target_tokens if token not in {"a", "an", "the", "my"}]
        if not filtered_target:
            return 0.0

        utter_index = 0
        matched = 0
        for target_token in filtered_target:
            while utter_index < len(utter_tokens) and utter_tokens[utter_index] != target_token:
                utter_index += 1
            if utter_index >= len(utter_tokens):
                break
            matched += 1
            utter_index += 1

        return matched / len(filtered_target)

    def _set_similarity(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _action_compatibility(self, action: str | None, capabilities: list[str]) -> float:
        if action is None:
            return 0.6

        capability_set = set(capabilities)
        if action in capability_set:
            return 1.0

        opposites = {
            "turn_on": "turn_off",
            "turn_off": "turn_on",
            "activate": "deactivate",
            "deactivate": "activate",
            "lock": "unlock",
            "unlock": "lock",
            "arm": "disarm",
            "disarm": "arm",
            "open": "close",
            "close": "open",
            "enable": "disable",
            "disable": "enable",
        }
        opposite = opposites.get(action)
        if opposite and opposite in capability_set:
            return 0.0

        if action == "query" and "query" in capability_set:
            return 1.0

        return 0.2

    def _infer_phrase_capabilities(self, phrase: str) -> list[str]:
        """Infer likely action capabilities from a conversation phrase."""
        normalized = normalize_text(phrase)
        tokens = normalized.split()
        capabilities: set[str] = set()

        for trigger_phrase, action in ACTION_MAP.items():
            if normalized.startswith(trigger_phrase):
                capabilities.add(action)

        if " on" in f" {normalized}" or normalized.endswith(" on") or "start" in tokens:
            capabilities.add("turn_on")
        if " off" in f" {normalized}" or normalized.endswith(" off") or "stop" in tokens:
            capabilities.add("turn_off")
        if "toggle" in tokens:
            capabilities.update({"turn_on", "turn_off", "activate", "deactivate"})
        if "set" in tokens or "temperature" in tokens or "temp" in tokens:
            capabilities.add("set")
        if normalized.startswith("what is") or "status" in tokens:
            capabilities.add("query")
        if "lock" in tokens:
            capabilities.add("lock")
        if "unlock" in tokens:
            capabilities.add("unlock")
        if "arm" in tokens:
            capabilities.add("arm")
        if "disarm" in tokens:
            capabilities.add("disarm")
        if "open" in tokens:
            capabilities.add("open")
        if "close" in tokens:
            capabilities.add("close")
        if "enable" in tokens:
            capabilities.add("enable")
        if "disable" in tokens:
            capabilities.add("disable")
        if "activate" in tokens:
            capabilities.add("activate")
        if "deactivate" in tokens:
            capabilities.add("deactivate")

        if not capabilities:
            capabilities.update({"activate", "query", "set"})

        return sorted(capabilities)

    def _structure_similarity(self, utter_tokens: list[str], target_tokens: list[str]) -> float:
        if not utter_tokens or not target_tokens:
            return 0.0
        ratio = min(len(utter_tokens), len(target_tokens)) / max(
            len(utter_tokens), len(target_tokens)
        )
        first_token_bonus = 0.2 if utter_tokens[0] == target_tokens[0] else 0.0
        return min(1.0, ratio + first_token_bonus)

    def _normalize_conversation_phrase_for_scoring(self, phrase: str) -> str:
        """Normalize Home Assistant sentence syntax for fuzzy scoring."""
        working = phrase.lower()

        # Replace slot placeholders before punctuation/brace normalization.
        working = re.sub(r"\{[^}]+\}", " ", working)

        # Remove optional markers while keeping the inner text.
        working = working.replace("[", " ").replace("]", " ")

        # Reduce alternation groups like `(alarm | timer | reminder)` to a simple space-separated form.
        def _replace_alternation(match: re.Match[str]) -> str:
            content = match.group(1)
            options = [normalize_text(part) for part in content.split("|")]
            options = [option for option in options if option]
            if not options:
                return " "
            return " " + " ".join(options) + " "

        working = re.sub(r"\(([^)]+)\)", _replace_alternation, working)
        working = normalize_text(working)

        tokens = [
            token
            for token in tokenize(working)
            if token not in {"a", "an", "the", "my", "when", "amount", "name"}
        ]
        return " ".join(tokens)

    def _expand_conversation_phrase_variants(self, phrase: str) -> list[str]:
        """Expand a HA sentence pattern into simple phrase variants for scoring."""
        variants = [phrase]

        # Expand one level of alternation groups like `(set|start)` or `(alarm | timer)`.
        while True:
            expanded = False
            next_variants: list[str] = []
            for variant in variants:
                match = re.search(r"\(([^()]+)\)", variant)
                if not match:
                    next_variants.append(variant)
                    continue
                expanded = True
                options = [part.strip() for part in match.group(1).split("|") if part.strip()]
                prefix = variant[: match.start()]
                suffix = variant[match.end() :]
                for option in options:
                    next_variants.append(prefix + option + suffix)
            variants = next_variants
            if not expanded:
                break

        # Expand simple optional groups like `[a | an]` or `[my]`.
        final_variants: list[str] = []
        for variant in variants:
            match = re.search(r"\[([^\[\]]+)\]", variant)
            if not match:
                final_variants.append(variant)
                continue

            content = match.group(1)
            options = [part.strip() for part in content.split("|") if part.strip()]
            prefix = variant[: match.start()]
            suffix = variant[match.end() :]
            final_variants.append(prefix + suffix)
            for option in options:
                final_variants.append(prefix + option + suffix)

        cleaned: list[str] = []
        seen: set[str] = set()
        for variant in final_variants or variants:
            normalized = self._normalize_conversation_phrase_for_scoring(variant)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(variant)
        return cleaned

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
