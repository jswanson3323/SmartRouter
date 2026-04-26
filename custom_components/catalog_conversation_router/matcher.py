"""Catalog-grounded fuzzy matcher."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import CandidateScore, CandidateType, Catalog, EntityTarget, MatchResult
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

DOMAIN_HINT_MAP: dict[str, set[str]] = {
    "fan": {"fan", "fans"},
    "light": {"light", "lights", "lamp", "lamps"},
    "switch": {"switch", "switches", "outlet", "outlets", "plug", "plugs"},
    "media_player": {"tv", "television", "speaker", "stereo", "receiver", "media"},
    "cover": {"blind", "blinds", "shade", "shades", "cover", "covers", "curtain", "curtains"},
    "climate": {"thermostat", "heater", "ac", "climate", "temperature"},
}

# Query-style prefixes for generic queries.
QUERY_PREFIXES = (
    "what is",
    "whats",
    "status",
    "what are",
    "tell me",
    "do i have",
    "are there",
    "how much",
    "how long",
    "when does",
    "when is",
)


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

        # Generic query parsing: handle "what is ... on/for/in/at ..." and similar.
        query_match = re.match(r"^(.+?)\s+\b(on|for|in|at)\b\s+(.+)$", normalized)
        if query_match:
            query_prefix = query_match.group(1).strip()
            query_target = query_match.group(3).strip()
            if chosen_action == "query" or any(
                query_prefix.startswith(prefix) for prefix in QUERY_PREFIXES
            ):
                target_tokens = query_target.split()
                while target_tokens and target_tokens[0] in {"the", "a", "an", "my"}:
                    target_tokens.pop(0)
                query_target = " ".join(target_tokens)
                return ParsedUtterance(
                    action="query",
                    action_phrase=query_prefix,
                    target_phrase=query_target,
                    area_hint=None,
                )

        # Remove leading determiners from the extracted target phrase.
        target_tokens = target.split()
        while target_tokens and target_tokens[0] in {"the", "a", "an"}:
            target_tokens.pop(0)
        target = " ".join(target_tokens)

        area_hint = None
        # Parse location qualifiers using word boundaries to avoid splitting inside words
        # (e.g., "great" must not match "at"). Only do this for non-query
        # utterances; for query-style phrases, trailing `on/in/at/for ...` often
        # belongs to the semantic target instead of an area.
        if chosen_action != "query":
            qualifier_match = re.search(r"\b(in|at|on)\b\s+(.+)$", target)
            if qualifier_match:
                before = target[: qualifier_match.start()].strip()
                after = qualifier_match.group(2).strip()
                if before and after:
                    area_tokens = after.split()
                    area_hint = area_tokens[0] if area_tokens and area_tokens[0] not in {"the", "a", "an", "my"} else None
                    target = before

        return ParsedUtterance(
            action=chosen_action,
            action_phrase=chosen_action_phrase,
            target_phrase=target,
            area_hint=area_hint,
        )

    def match(
        self,
        utterance: str,
        catalog: Catalog,
        origin_area: str | None = None,
        origin_super_area: str | None = None,
    ) -> MatchResult:
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
        effective_super_area_hint = self._resolve_effective_super_area_hint(
            area_hint=effective_area_hint,
            entities=catalog.entity_targets,
            explicit_super_area_hint=origin_super_area,
        )
        inferred_domain = self._infer_domain_from_tokens(utter_tokens)
        generic_domain_request = self._is_generic_domain_request(utter_tokens, inferred_domain)
        area_scoped_domain_entity_id = self._resolve_area_scoped_domain_entity(
            utter_tokens=utter_tokens,
            action=parsed.action,
            entities=catalog.entity_targets,
            area_hint=effective_area_hint,
            generic_domain_request=generic_domain_request,
        )
        super_area_scoped_domain_entity_id = None
        if area_scoped_domain_entity_id is None:
            super_area_scoped_domain_entity_id = self._resolve_super_area_scoped_domain_entity(
                utter_tokens=utter_tokens,
                action=parsed.action,
                entities=catalog.entity_targets,
                area_hint=effective_area_hint,
                super_area_hint=effective_super_area_hint,
                generic_domain_request=generic_domain_request,
            )

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
            entity_target_similarity = self._semantic_target_similarity(utter_tokens, target_tokens)
            score_detail["token_similarity"] = max(
                score_detail["token_similarity"],
                entity_target_similarity,
            )
            score_detail["phonetic_similarity"] = max(
                score_detail["phonetic_similarity"],
                self._semantic_target_similarity(
                    list(utter_phonetic),
                    list(target_phonetic),
                ),
            )
            domain_hint_match = self._domain_hint_match(utter_tokens, entity.domain, entity.name, entity.aliases)
            if domain_hint_match > 0:
                score_detail["token_similarity"] = min(1.0, score_detail["token_similarity"] + domain_hint_match)
                score_detail["structure_similarity"] = min(1.0, score_detail["structure_similarity"] + 0.1)

            if generic_domain_request and effective_area_hint and entity.area:
                candidate_area_tokens = set(tokenize(normalize_text(entity.area)))
                hint_tokens = set(tokenize(effective_area_hint))
                area_match = 1.0 if hint_tokens and hint_tokens <= candidate_area_tokens else 0.0
                if area_match > 0 and (entity_target_similarity > 0 or domain_hint_match > 0):
                    score_detail["structure_similarity"] = min(1.0, score_detail["structure_similarity"] + 0.3)
                    score_detail["token_similarity"] = min(1.0, score_detail["token_similarity"] + 0.2)
                    score_detail["phonetic_similarity"] = min(1.0, score_detail["phonetic_similarity"] + 0.1)
                elif area_match == 0 and (entity_target_similarity > 0 or domain_hint_match > 0):
                    score_detail["structure_similarity"] = max(0.0, score_detail["structure_similarity"] - 0.1)

            if area_scoped_domain_entity_id and entity.entity_id == area_scoped_domain_entity_id:
                score_detail["token_similarity"] = min(1.0, score_detail["token_similarity"] + 0.25)
                score_detail["structure_similarity"] = min(1.0, score_detail["structure_similarity"] + 0.25)
                score_detail["phonetic_similarity"] = min(1.0, score_detail["phonetic_similarity"] + 0.10)
                score_detail["area_scoped_domain_resolution"] = 1.0

            area_preference_bonus = self._area_preference_bonus(
                utter_tokens=utter_tokens,
                inferred_domain=inferred_domain,
                effective_area_hint=effective_area_hint,
                entity=entity,
            )
            score_detail["area_preference_bonus"] = area_preference_bonus
            if area_preference_bonus > 0:
                score_detail["token_similarity"] = min(1.0, score_detail["token_similarity"] + area_preference_bonus)
                score_detail["structure_similarity"] = min(1.0, score_detail["structure_similarity"] + area_preference_bonus)
                score_detail["whole_target_similarity"] = min(1.0, score_detail["whole_target_similarity"] + (area_preference_bonus * 0.75))
                score_detail["phonetic_similarity"] = min(1.0, score_detail["phonetic_similarity"] + (area_preference_bonus * 0.5))
            elif area_preference_bonus < 0:
                score_detail["token_similarity"] = max(0.0, score_detail["token_similarity"] + area_preference_bonus)
                score_detail["structure_similarity"] = max(0.0, score_detail["structure_similarity"] + area_preference_bonus)
                score_detail["whole_target_similarity"] = max(0.0, score_detail["whole_target_similarity"] + (area_preference_bonus * 0.75))

            super_area_preference_bonus = self._super_area_preference_bonus(
                utter_tokens=utter_tokens,
                inferred_domain=inferred_domain,
                effective_area_hint=effective_area_hint,
                effective_super_area_hint=effective_super_area_hint,
                entity=entity,
            )
            score_detail["super_area_preference_bonus"] = super_area_preference_bonus
            if super_area_preference_bonus > 0:
                score_detail["token_similarity"] = min(1.0, score_detail["token_similarity"] + super_area_preference_bonus)
                score_detail["structure_similarity"] = min(1.0, score_detail["structure_similarity"] + super_area_preference_bonus)
                score_detail["whole_target_similarity"] = min(1.0, score_detail["whole_target_similarity"] + (super_area_preference_bonus * 0.75))
                score_detail["phonetic_similarity"] = min(1.0, score_detail["phonetic_similarity"] + (super_area_preference_bonus * 0.5))
            elif super_area_preference_bonus < 0:
                score_detail["token_similarity"] = max(0.0, score_detail["token_similarity"] + super_area_preference_bonus)
                score_detail["structure_similarity"] = max(0.0, score_detail["structure_similarity"] + super_area_preference_bonus)
                score_detail["whole_target_similarity"] = max(0.0, score_detail["whole_target_similarity"] + (super_area_preference_bonus * 0.75))

            if super_area_scoped_domain_entity_id and entity.entity_id == super_area_scoped_domain_entity_id:
                score_detail["token_similarity"] = min(1.0, score_detail["token_similarity"] + 0.18)
                score_detail["structure_similarity"] = min(1.0, score_detail["structure_similarity"] + 0.18)
                score_detail["phonetic_similarity"] = min(1.0, score_detail["phonetic_similarity"] + 0.08)
                score_detail["super_area_scoped_domain_resolution"] = 1.0

            final_score = self._weighted_score(score_detail)
            if area_scoped_domain_entity_id and entity.entity_id == area_scoped_domain_entity_id:
                final_score = max(final_score, 0.95)
            if super_area_scoped_domain_entity_id and entity.entity_id == super_area_scoped_domain_entity_id:
                final_score = max(final_score, 0.91)
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
                conversation_target_similarity = self._semantic_target_similarity(
                    utter_tokens,
                    phrase_tokens,
                )
                score_detail["semantic_target_similarity"] = conversation_target_similarity
                if utter_tokens:
                    score_detail["token_similarity"] = max(
                        score_detail["token_similarity"],
                        conversation_target_similarity,
                    )
                    if conversation_target_similarity == 0.0:
                        score_detail["action_compatibility"] = min(
                            score_detail["action_compatibility"],
                            0.0 if parsed.action in {"turn_on", "turn_off", "set", "open", "close", "lock", "unlock"} else 0.15,
                        )
                        score_detail["structure_similarity"] = min(
                            score_detail["structure_similarity"],
                            0.15 if parsed.action in {"turn_on", "turn_off", "set", "open", "close", "lock", "unlock"} else 0.35,
                        )
                        score_detail["whole_target_similarity"] = min(score_detail["whole_target_similarity"], 0.25)
                    if generic_domain_request and conversation_target_similarity == 0.0:
                        score_detail["token_similarity"] = min(score_detail["token_similarity"], 0.18)
                        score_detail["structure_similarity"] = min(score_detail["structure_similarity"], 0.18)
                        score_detail["whole_target_similarity"] = min(score_detail["whole_target_similarity"], 0.18)

                pattern_bonus, matched_slots, total_slots = self._conversation_pattern_bonus(
                    utterance_normalized=normalized,
                    raw_phrase=phrase,
                )
                score_detail["pattern_bonus"] = pattern_bonus
                score_detail["pattern_matched_slots"] = float(matched_slots)
                score_detail["pattern_total_slots"] = float(total_slots)
                if pattern_bonus != 0:
                    score_detail["structure_similarity"] = min(
                        1.0,
                        score_detail["structure_similarity"] + pattern_bonus,
                    )
                    score_detail["token_similarity"] = min(
                        1.0,
                        score_detail["token_similarity"] + max(0.0, pattern_bonus * 0.4),
                    )
                    score_detail["whole_target_similarity"] = min(
                        1.0,
                        score_detail["whole_target_similarity"] + max(0.0, pattern_bonus * 0.3),
                    )
                if total_slots > matched_slots:
                    missing_slots = total_slots - matched_slots
                    slot_penalty = min(0.25, 0.10 * missing_slots)
                    score_detail["token_similarity"] = max(0.0, score_detail["token_similarity"] - slot_penalty)
                    score_detail["structure_similarity"] = max(0.0, score_detail["structure_similarity"] - slot_penalty)
                    score_detail["whole_target_similarity"] = max(0.0, score_detail["whole_target_similarity"] - slot_penalty)

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
            conversation_target_similarity = self._semantic_target_similarity(
                utter_tokens,
                best_target_tokens,
            )
            score_detail["semantic_target_similarity"] = conversation_target_similarity
            if utter_tokens:
                score_detail["token_similarity"] = max(
                    score_detail["token_similarity"],
                    conversation_target_similarity,
                )
                if conversation_target_similarity == 0.0:
                    score_detail["action_compatibility"] = min(
                        score_detail["action_compatibility"],
                        0.15,
                    )
                    score_detail["structure_similarity"] = min(
                        score_detail["structure_similarity"],
                        0.35,
                    )
                    if generic_domain_request:
                        score_detail["token_similarity"] = min(score_detail["token_similarity"], 0.18)
                        score_detail["structure_similarity"] = min(score_detail["structure_similarity"], 0.18)
                        score_detail["whole_target_similarity"] = min(score_detail["whole_target_similarity"], 0.18)

            pattern_bonus, matched_slots, total_slots = self._conversation_pattern_bonus(
                utterance_normalized=normalized,
                raw_phrase=best_phrase_raw,
            )
            score_detail["pattern_bonus"] = pattern_bonus
            score_detail["pattern_matched_slots"] = float(matched_slots)
            score_detail["pattern_total_slots"] = float(total_slots)
            if pattern_bonus != 0:
                score_detail["structure_similarity"] = min(
                    1.0,
                    score_detail["structure_similarity"] + pattern_bonus,
                )
                score_detail["token_similarity"] = min(
                    1.0,
                    score_detail["token_similarity"] + max(0.0, pattern_bonus * 0.4),
                )
                score_detail["whole_target_similarity"] = min(
                    1.0,
                    score_detail["whole_target_similarity"] + max(0.0, pattern_bonus * 0.3),
                )
            if total_slots > matched_slots:
                missing_slots = total_slots - matched_slots
                slot_penalty = min(0.25, 0.10 * missing_slots)
                score_detail["token_similarity"] = max(0.0, score_detail["token_similarity"] - slot_penalty)
                score_detail["structure_similarity"] = max(0.0, score_detail["structure_similarity"] - slot_penalty)
                score_detail["whole_target_similarity"] = max(0.0, score_detail["whole_target_similarity"] - slot_penalty)

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
                        "candidate_family": self._candidate_family(target.target_id),
                        "exact_phrase_match": self._is_exact_conversation_phrase_match(
                            utterance_normalized=normalized,
                            raw_phrase=best_phrase_raw,
                            canonical_phrase=target.canonical_phrase,
                        ),
                        "matched_sample_phrase": best_phrase,
                        "matched_sample_phrase_raw": best_phrase_raw,
                        "matched_sample_phrase_normalized_for_scoring": best_phrase_for_scoring,
                        "parsed_target_before_normalization": parsed_target_before,
                        "parsed_target_after_normalization": parsed_target_after,
                    },
                )
            )

        ranked = sorted(scores, key=lambda item: item.score, reverse=True)
        if area_scoped_domain_entity_id:
            resolved = next((c for c in ranked if c.candidate_id == area_scoped_domain_entity_id), None)
            if resolved:
                remaining = [c for c in ranked if c.candidate_id != area_scoped_domain_entity_id]
                ranked = [resolved, *remaining]
        elif super_area_scoped_domain_entity_id:
            resolved = next((c for c in ranked if c.candidate_id == super_area_scoped_domain_entity_id), None)
            if resolved:
                remaining = [c for c in ranked if c.candidate_id != super_area_scoped_domain_entity_id]
                ranked = [resolved, *remaining]
        ranked = self._dedupe_by_canonical_phrase(ranked)
        top = ranked[:3]
        best = top[0] if top else None
        second = top[1] if len(top) > 1 else None
        best_detail = best.detail if best is not None else {}
        second_detail = second.detail if second is not None else {}

        matched = bool(
            best
            and best.score >= self._fuzzy_threshold
            and (best.score - (second.score if second else 0.0)) >= self._ambiguity_gap
        )
        if (
            best
            and best.candidate_type == CandidateType.CONVERSATION_TARGET
            and best_detail.get("exact_phrase_match") is True
        ):
            matched = True
        if (
            best
            and second
            and best.candidate_type == CandidateType.CONVERSATION_TARGET
            and second.candidate_type == CandidateType.CONVERSATION_TARGET
            and best_detail.get("candidate_family")
            and best_detail.get("candidate_family") == second_detail.get("candidate_family")
        ):
            matched = True
        if area_scoped_domain_entity_id and best and best.candidate_id == area_scoped_domain_entity_id:
            matched = True
        if super_area_scoped_domain_entity_id and best and best.candidate_id == super_area_scoped_domain_entity_id:
            matched = True

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
            effective_super_area_hint=effective_super_area_hint,
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

    def _semantic_target_similarity(self, utter_tokens: list[str], candidate_tokens: list[str]) -> float:
        """Compare target-bearing tokens while ignoring action/filler words."""
        utter_semantic = self._semantic_tokens(utter_tokens)
        candidate_semantic = self._semantic_tokens(candidate_tokens)
        if not utter_semantic or not candidate_semantic:
            return 0.0
        return len(set(utter_semantic) & set(candidate_semantic)) / len(set(utter_semantic) | set(candidate_semantic))

    def _semantic_tokens(self, tokens: list[str]) -> list[str]:
        """Remove action and filler words so noun matching matters more."""
        stopwords = {
            "a",
            "an",
            "the",
            "my",
            "turn",
            "switch",
            "set",
            "what",
            "is",
            "status",
            "activate",
            "deactivate",
            "lock",
            "unlock",
            "arm",
            "disarm",
            "open",
            "close",
            "enable",
            "disable",
            "start",
            "stop",
            "on",
            "off",
            "in",
            "at",
            "for",
            "to",
        }
        return [token for token in tokens if token not in stopwords]




    def _conversation_pattern_bonus(
        self,
        *,
        utterance_normalized: str,
        raw_phrase: str,
    ) -> tuple[float, int, int]:
        """Reward conversation patterns that match fixed tokens and satisfy slots.

        This is grammar-aware and generic. It does not special-case timers,
        reminders, alarms, or any other single intent family.
        """
        utterance_tokens = tokenize(self._strip_leading_polite_prefix(utterance_normalized))
        total_slots = len(re.findall(r"\{[^}]+\}", raw_phrase))
        if total_slots == 0:
            return 0.0, 0, 0

        matched_slots = self._count_supported_slots(utterance_tokens, raw_phrase)

        # Reward patterns that can satisfy more slots. This naturally makes more
        # specific patterns win when the utterance contains enough information.
        bonus = 0.08 * matched_slots
        if matched_slots == total_slots and total_slots > 1:
            bonus += 0.04 * (total_slots - 1)
        elif matched_slots < total_slots:
            bonus -= 0.02 * (total_slots - matched_slots)

        return max(-0.08, min(0.25, bonus)), matched_slots, total_slots

    def _count_supported_slots(self, utterance_tokens: list[str], raw_phrase: str) -> int:
        """Count how many slots in a conversation pattern are supported.

        A slot is considered supported when the utterance contains token span(s)
        between the nearest fixed-token anchors around that slot.
        """
        matches = list(re.finditer(r"\{[^}]+\}", raw_phrase))
        if not matches:
            return 0

        supported = 0
        for match in matches:
            prefix_text = raw_phrase[: match.start()]
            suffix_text = raw_phrase[match.end() :]
            prefix_tokens = tokenize(self._normalize_conversation_phrase_for_scoring(prefix_text))
            suffix_tokens = tokenize(self._normalize_conversation_phrase_for_scoring(suffix_text))
            if self._slot_span_supported(utterance_tokens, prefix_tokens, suffix_tokens):
                supported += 1
        return supported

    def _slot_span_supported(
        self,
        utterance_tokens: list[str],
        prefix_tokens: list[str],
        suffix_tokens: list[str],
    ) -> bool:
        """Return True when utterance has content where the slot should be."""
        prefix_span = self._find_subsequence_span(utterance_tokens, prefix_tokens, 0)
        prefix_end = prefix_span[1] if prefix_span else 0

        if suffix_tokens:
            suffix_span = self._find_subsequence_span(utterance_tokens, suffix_tokens, prefix_end)
            if not suffix_span:
                return False
            return suffix_span[0] > prefix_end

        return prefix_end < len(utterance_tokens)

    def _find_subsequence_span(
        self,
        utterance_tokens: list[str],
        pattern_tokens: list[str],
        start_index: int,
    ) -> tuple[int, int] | None:
        """Find the first ordered subsequence span for pattern tokens."""
        if not pattern_tokens:
            return (start_index, start_index)

        first_index: int | None = None
        current_index = start_index
        for pattern_token in pattern_tokens:
            while current_index < len(utterance_tokens) and utterance_tokens[current_index] != pattern_token:
                current_index += 1
            if current_index >= len(utterance_tokens):
                return None
            if first_index is None:
                first_index = current_index
            current_index += 1

        return (first_index if first_index is not None else start_index, current_index)


    def _strip_leading_polite_prefix(self, text: str) -> str:
        """Remove polite filler at the front of an utterance."""
        normalized = normalize_text(text)
        while True:
            updated = re.sub(
                r"^(?:pretty please|please|will you|would you|could you|can you|kindly|hey|hey assistant)\s+",
                "",
                normalized,
            )
            if updated == normalized:
                return normalized.strip()
            normalized = updated.strip()

    def _domain_hint_match(
        self,
        utter_tokens: list[str],
        domain: str,
        entity_name: str,
        aliases: list[str],
    ) -> float:
        """Detect generic domain nouns like fan/light/tv in the utterance."""
        utter_set = set(utter_tokens)
        domain_terms = DOMAIN_HINT_MAP.get(domain, set())
        if utter_set & domain_terms:
            return 0.25

        names = [normalize_text(entity_name), *[normalize_text(alias) for alias in aliases]]
        for name in names:
            name_tokens = set(tokenize(name))
            if name_tokens & utter_set & domain_terms:
                return 0.2
        return 0.0

    def _resolve_area_scoped_domain_entity(
        self,
        *,
        utter_tokens: list[str],
        action: str | None,
        entities: list[EntityTarget],
        area_hint: str | None,
        generic_domain_request: bool,
    ) -> str | None:
        """Resolve deterministic area-scoped entity for generic domain targets.

        Example: "turn on the light" from "master bedroom" should resolve to
        the only `light` entity in that area, if exactly one exists.
        """
        if not area_hint or not generic_domain_request:
            return None

        inferred_domain = self._infer_domain_from_tokens(utter_tokens)
        if not inferred_domain:
            return None

        hint_tokens = set(tokenize(area_hint))
        if not hint_tokens:
            return None

        scoped_candidates = []
        for entity in entities:
            if entity.domain != inferred_domain or not entity.area:
                continue
            entity_area_tokens = set(tokenize(normalize_text(entity.area)))
            if hint_tokens <= entity_area_tokens:
                if self._action_compatibility(action, entity.capabilities) > 0.0:
                    scoped_candidates.append(entity)

        if len(scoped_candidates) == 1:
            return scoped_candidates[0].entity_id
        return None

    def _resolve_super_area_scoped_domain_entity(
        self,
        *,
        utter_tokens: list[str],
        action: str | None,
        entities: list[EntityTarget],
        area_hint: str | None,
        super_area_hint: str | None,
        generic_domain_request: bool,
    ) -> str | None:
        """Resolve a unique same-super-area entity when area match is unavailable."""
        if not super_area_hint or not generic_domain_request:
            return None

        inferred_domain = self._infer_domain_from_tokens(utter_tokens)
        if not inferred_domain:
            return None

        super_area_tokens = set(tokenize(super_area_hint))
        area_tokens = set(tokenize(area_hint)) if area_hint else set()
        if not super_area_tokens:
            return None

        scoped_candidates = []
        for entity in entities:
            if entity.domain != inferred_domain or not entity.super_area:
                continue
            entity_super_area_tokens = set(tokenize(normalize_text(entity.super_area)))
            entity_area_tokens = set(tokenize(normalize_text(entity.area))) if entity.area else set()
            if area_tokens and area_tokens <= entity_area_tokens:
                continue
            if super_area_tokens <= entity_super_area_tokens:
                if self._action_compatibility(action, entity.capabilities) > 0.0:
                    scoped_candidates.append(entity)

        if len(scoped_candidates) == 1:
            return scoped_candidates[0].entity_id
        return None

    def _infer_domain_from_tokens(self, utter_tokens: list[str]) -> str | None:
        utter_set = set(utter_tokens)
        if not utter_set:
            return None
        for domain, terms in DOMAIN_HINT_MAP.items():
            if utter_set & terms:
                return domain
        return None

    def _is_generic_domain_request(self, utter_tokens: list[str], inferred_domain: str | None) -> bool:
        """Return True for generic commands like `turn on the light`."""
        if not inferred_domain:
            return False
        semantic_tokens = self._semantic_tokens(utter_tokens)
        if not semantic_tokens:
            return False
        domain_terms = DOMAIN_HINT_MAP.get(inferred_domain, set())
        return all(token in domain_terms for token in semantic_tokens)

    def _area_preference_bonus(
        self,
        *,
        utter_tokens: list[str],
        inferred_domain: str | None,
        effective_area_hint: str | None,
        entity: EntityTarget,
    ) -> float:
        """Prefer same-area entities for generic domain requests.

        This is intentionally generic. It applies to commands such as:
        - turn on the light
        - turn off the fan
        - turn on the tv

        when the origin area is known and the utterance does not otherwise
        uniquely identify a target.
        """
        if not effective_area_hint or not entity.area or not inferred_domain:
            return 0.0
        if entity.domain != inferred_domain:
            return 0.0
        if not self._is_generic_domain_request(utter_tokens, inferred_domain):
            return 0.0

        hint_tokens = set(tokenize(effective_area_hint))
        entity_area_tokens = set(tokenize(normalize_text(entity.area)))
        if not hint_tokens or not entity_area_tokens:
            return 0.0

        if hint_tokens <= entity_area_tokens:
            return 0.35
        return -0.12

    def _super_area_preference_bonus(
        self,
        *,
        utter_tokens: list[str],
        inferred_domain: str | None,
        effective_area_hint: str | None,
        effective_super_area_hint: str | None,
        entity: EntityTarget,
    ) -> float:
        """Prefer same-super-area entities after direct area matching."""
        if not effective_super_area_hint or not entity.super_area or not inferred_domain:
            return 0.0
        if entity.domain != inferred_domain:
            return 0.0
        if not self._is_generic_domain_request(utter_tokens, inferred_domain):
            return 0.0

        if effective_area_hint and entity.area:
            area_tokens = set(tokenize(effective_area_hint))
            entity_area_tokens = set(tokenize(normalize_text(entity.area)))
            if area_tokens and area_tokens <= entity_area_tokens:
                return 0.0

        super_area_tokens = set(tokenize(effective_super_area_hint))
        entity_super_area_tokens = set(tokenize(normalize_text(entity.super_area)))
        if not super_area_tokens or not entity_super_area_tokens:
            return 0.0

        if super_area_tokens <= entity_super_area_tokens:
            return 0.18
        return -0.06

    def _resolve_effective_super_area_hint(
        self,
        *,
        area_hint: str | None,
        entities: list[EntityTarget],
        explicit_super_area_hint: str | None = None,
    ) -> str | None:
        """Infer the SuperArea for the active area from catalog entities."""
        if explicit_super_area_hint:
            return normalize_text(explicit_super_area_hint)
        if not area_hint:
            return None

        area_tokens = set(tokenize(area_hint))
        if not area_tokens:
            return None

        matches = {
            entity.super_area.strip()
            for entity in entities
            if entity.area
            and entity.super_area
            and area_tokens <= set(tokenize(normalize_text(entity.area)))
        }
        if len(matches) == 1:
            return next(iter(matches))
        return None

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

    def _candidate_family(self, candidate_id: str) -> str | None:
        if not candidate_id or ":" not in candidate_id:
            return None
        return candidate_id.rsplit(":", 1)[0]

    def _is_exact_conversation_phrase_match(
        self,
        *,
        utterance_normalized: str,
        raw_phrase: str,
        canonical_phrase: str,
    ) -> bool:
        utterance_text = normalize_text(utterance_normalized)
        return (
            utterance_text == normalize_text(raw_phrase)
            or utterance_text == normalize_text(canonical_phrase)
        )

    def _normalize_asr_target_tokens(self, target_phrase: str) -> list[str]:
        """Normalize likely ASR noun substitutions before scoring."""
        tokens = tokenize(target_phrase)
        if not tokens:
            return tokens

        all_domain_terms = sorted(
            {term for terms in DOMAIN_HINT_MAP.values() for term in terms},
            key=len,
            reverse=True,
        )
        corrected: list[str] = []
        for token in tokens:
            best = token
            best_ratio = 0.0
            for term in all_domain_terms:
                ratio = SequenceMatcher(a=token, b=term).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = term
            corrected.append(best if best_ratio >= 0.84 else token)
        return corrected

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
