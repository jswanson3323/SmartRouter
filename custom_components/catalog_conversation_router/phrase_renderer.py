"""Generic Home Assistant sentence pattern renderer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .phonetics import normalize_text, tokenize

ALT_RE = re.compile(r"\(([^()]+)\)")
OPT_RE = re.compile(r"\[([^\[\]]+)\]")
SLOT_RE = re.compile(r"\{([^}]+)\}")


@dataclass(slots=True)
class RenderedPhrase:
    """Rendered result for a conversation pattern."""

    text: str
    rendered_from_pattern: bool
    slots: dict[str, str]


def render_conversation_pattern(original_utterance: str, pattern: str) -> RenderedPhrase:
    """Render a concrete Assist phrase from HA sentence grammar."""
    utterance = normalize_text(original_utterance)
    utterance_tokens = tokenize(utterance)

    with_alternatives = _replace_alternatives(pattern, utterance_tokens)
    with_optionals = _replace_optionals(with_alternatives, utterance_tokens)
    cleaned = re.sub(r"\s+", " ", with_optionals).strip()

    if not SLOT_RE.search(cleaned):
        rendered = normalize_text(cleaned)
        return RenderedPhrase(text=rendered, rendered_from_pattern=True, slots={})

    slots = _extract_slots(cleaned, utterance_tokens)
    rendered = cleaned
    for slot_name, value in slots.items():
        rendered = re.sub(rf"\{{\s*{re.escape(slot_name)}\s*\}}", value, rendered)

    rendered = SLOT_RE.sub("", rendered)
    rendered = normalize_text(rendered)
    return RenderedPhrase(text=rendered, rendered_from_pattern=True, slots=slots)


def _replace_alternatives(pattern: str, utterance_tokens: list[str]) -> str:
    def _pick(match: re.Match[str]) -> str:
        options = [normalize_text(option) for option in match.group(1).split("|")]
        options = [option for option in options if option]
        if not options:
            return ""

        best_option = options[0]
        best_score = -1.0
        utter_set = set(utterance_tokens)
        for option in options:
            option_tokens = tokenize(option)
            if not option_tokens:
                continue
            overlap = len(set(option_tokens) & utter_set) / len(set(option_tokens))
            if overlap > best_score:
                best_score = overlap
                best_option = option
        return best_option

    updated = pattern
    while True:
        next_value = ALT_RE.sub(_pick, updated)
        if next_value == updated:
            return updated
        updated = next_value


def _replace_optionals(pattern: str, utterance_tokens: list[str]) -> str:
    utter_set = set(utterance_tokens)

    def _resolve_optional_content(content: str) -> str:
        parts = [normalize_text(part) for part in content.split("|")]
        parts = [part for part in parts if part]
        if not parts:
            return ""

        best = parts[0]
        best_score = -1.0
        for part in parts:
            part_tokens = tokenize(part)
            if not part_tokens:
                continue
            score = len(set(part_tokens) & utter_set) / len(set(part_tokens))
            if score > best_score:
                best_score = score
                best = part

        return best if best_score > 0 else ""

    def _pick(match: re.Match[str]) -> str:
        content = _resolve_optional_content(match.group(1))
        return content

    updated = pattern
    while True:
        next_value = OPT_RE.sub(_pick, updated)
        if next_value == updated:
            return updated
        updated = next_value


def _extract_slots(pattern: str, utterance_tokens: list[str]) -> dict[str, str]:
    parts = re.split(r"(\{[^}]+\})", pattern)
    slot_indices = [idx for idx, part in enumerate(parts) if SLOT_RE.fullmatch(part)]
    if not slot_indices:
        return {}

    cursor = 0
    slots: dict[str, str] = {}

    for slot_idx in slot_indices:
        slot_name_match = SLOT_RE.fullmatch(parts[slot_idx])
        if slot_name_match is None:
            continue
        slot_name = normalize_text(slot_name_match.group(1))

        prev_literal = ""
        for part in reversed(parts[:slot_idx]):
            if SLOT_RE.fullmatch(part):
                continue
            if part.strip():
                prev_literal = part
                break

        next_literal = ""
        for part in parts[slot_idx + 1 :]:
            if SLOT_RE.fullmatch(part):
                continue
            if part.strip():
                next_literal = part
                break

        prev_tokens = tokenize(prev_literal)
        next_tokens = tokenize(next_literal)

        if prev_tokens:
            prev_span = _find_subsequence(utterance_tokens, prev_tokens, cursor)
            if prev_span is not None:
                slot_start = prev_span[1]
            else:
                slot_start = cursor
        else:
            slot_start = cursor

        if next_tokens:
            next_span = _find_subsequence(utterance_tokens, next_tokens, slot_start)
            slot_end = next_span[0] if next_span is not None else len(utterance_tokens)
        else:
            slot_end = len(utterance_tokens)

        if slot_end < slot_start:
            slot_end = slot_start

        slot_value_tokens = utterance_tokens[slot_start:slot_end]
        if slot_value_tokens:
            slots[slot_name] = " ".join(slot_value_tokens)
            cursor = slot_end
        else:
            slots[slot_name] = ""
            cursor = slot_start

    return slots


def _find_subsequence(tokens: list[str], pattern: list[str], start_index: int) -> tuple[int, int] | None:
    if not pattern:
        return (start_index, start_index)

    idx = start_index
    first = None
    for token in pattern:
        while idx < len(tokens) and tokens[idx] != token:
            idx += 1
        if idx >= len(tokens):
            return None
        if first is None:
            first = idx
        idx += 1

    return (first if first is not None else start_index, idx)
