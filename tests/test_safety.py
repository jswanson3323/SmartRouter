"""Safety tests."""

from custom_components.catalog_conversation_router.safety import validate_fuzzy_execution


def test_low_risk_allowed() -> None:
    decision = validate_fuzzy_execution(
        inferred_action="turn_on",
        candidate_action="turn_on",
        canonical_phrase="turn on kitchen light",
        best_score=0.9,
        second_score=0.6,
        fuzzy_threshold=0.84,
        ambiguity_gap=0.08,
        high_risk_threshold=0.96,
    )
    assert decision.allowed is True


def test_opposite_verb_rejected() -> None:
    decision = validate_fuzzy_execution(
        inferred_action="lock",
        candidate_action="unlock",
        canonical_phrase="unlock front door",
        best_score=0.99,
        second_score=0.2,
        fuzzy_threshold=0.84,
        ambiguity_gap=0.08,
        high_risk_threshold=0.96,
    )
    assert decision.allowed is False
    assert decision.reason == "opposite_verb_protection"


def test_high_risk_threshold_applied() -> None:
    decision = validate_fuzzy_execution(
        inferred_action="unlock",
        candidate_action="unlock",
        canonical_phrase="unlock front door",
        best_score=0.92,
        second_score=0.1,
        fuzzy_threshold=0.84,
        ambiguity_gap=0.08,
        high_risk_threshold=0.96,
    )
    assert decision.allowed is False
    assert decision.reason == "confidence_below_threshold"
