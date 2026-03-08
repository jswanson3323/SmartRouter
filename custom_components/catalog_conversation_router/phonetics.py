"""Text normalization and lightweight phonetic helpers."""

from __future__ import annotations

import re
import unicodedata

COMMON_CONFUSIONS: dict[str, str] = {
    "line": "light",
    "lite": "light",
    "haul": "hall",
    "moat": "mode",
    "there": "their",
    "their": "there",
    "four": "for",
}

STOP_WORDS = {
    "the",
    "a",
    "an",
    "please",
    "to",
    "in",
    "at",
    "of",
    "my",
}


def normalize_text(text: str) -> str:
    """Normalize text for deterministic matching."""
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def apply_confusions(tokens: list[str]) -> list[str]:
    """Apply known ASR substitutions conservatively."""
    return [COMMON_CONFUSIONS.get(token, token) for token in tokens]


def tokenize(text: str, *, drop_stop_words: bool = True) -> list[str]:
    """Tokenize normalized text."""
    tokens = normalize_text(text).split()
    if drop_stop_words:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    return apply_confusions(tokens)


def phonetic_key(word: str) -> str:
    """Generate a tiny Soundex-like phonetic key."""
    word = normalize_text(word).replace(" ", "")
    if not word:
        return ""

    mapping = {
        **dict.fromkeys(list("bfpv"), "1"),
        **dict.fromkeys(list("cgjkqsxz"), "2"),
        **dict.fromkeys(list("dt"), "3"),
        "l": "4",
        **dict.fromkeys(list("mn"), "5"),
        "r": "6",
    }

    first = word[0]
    encoded: list[str] = [first]
    prev = mapping.get(first, "")
    for ch in word[1:]:
        code = mapping.get(ch, "")
        if code != prev and code:
            encoded.append(code)
        prev = code

    key = "".join(encoded)
    return (key + "000")[:4]


def phonetic_tokens(tokens: list[str]) -> list[str]:
    """Phonetic transform for each token."""
    return [phonetic_key(token) for token in tokens if token]
