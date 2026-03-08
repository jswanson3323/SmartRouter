"""Safety checks for fuzzy and translated actions."""

from __future__ import annotations

from .models import RiskTier, SafetyDecision

HIGH_RISK_ACTIONS = {
    "unlock",
    "disarm",
    "disable",
    "open_garage",
    "open",
}

OPPOSITE_VERB_PAIRS = {
    "lock": "unlock",
    "unlock": "lock",
    "arm": "disarm",
    "disarm": "arm",
    "open": "close",
    "close": "open",
    "enable": "disable",
    "disable": "enable",
}


def classify_risk(action: str | None, canonical_phrase: str | None = None) -> RiskTier:
    """Classify risk from action and phrase hints."""
    if not action:
        return RiskTier.LOW

    normalized = action.lower().strip()
    if normalized in HIGH_RISK_ACTIONS:
        return RiskTier.HIGH

    phrase = (canonical_phrase or "").lower()
    if "garage" in phrase and normalized in {"open", "close"}:
        return RiskTier.HIGH
    if "alarm" in phrase and normalized in {"arm", "disarm", "disable", "enable"}:
        return RiskTier.HIGH

    return RiskTier.LOW


def is_opposite_action(source_action: str | None, target_action: str | None) -> bool:
    """Detect dangerous opposite action rewrites."""
    if not source_action or not target_action:
        return False
    return OPPOSITE_VERB_PAIRS.get(source_action) == target_action


def validate_fuzzy_execution(
    *,
    inferred_action: str | None,
    candidate_action: str | None,
    canonical_phrase: str,
    best_score: float,
    second_score: float,
    fuzzy_threshold: float,
    ambiguity_gap: float,
    high_risk_threshold: float,
) -> SafetyDecision:
    """Apply confidence and ambiguity safety gates."""
    if is_opposite_action(inferred_action, candidate_action):
        return SafetyDecision(
            allowed=False,
            reason="opposite_verb_protection",
            risk_tier=RiskTier.HIGH,
        )

    risk = classify_risk(candidate_action, canonical_phrase)
    threshold = fuzzy_threshold if risk == RiskTier.LOW else max(high_risk_threshold, fuzzy_threshold)
    gap_required = ambiguity_gap if risk == RiskTier.LOW else max(ambiguity_gap, 0.12)

    if best_score < threshold:
        return SafetyDecision(
            allowed=False,
            reason="confidence_below_threshold",
            risk_tier=risk,
        )

    if best_score - second_score < gap_required:
        return SafetyDecision(
            allowed=False,
            reason="ambiguity_gap_too_small",
            risk_tier=risk,
        )

    return SafetyDecision(allowed=True, reason="allowed", risk_tier=risk)
