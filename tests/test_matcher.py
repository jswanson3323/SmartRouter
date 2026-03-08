"""Matcher tests."""

from custom_components.catalog_conversation_router.matcher import FuzzyMatcher
from custom_components.catalog_conversation_router.models import (
    Catalog,
    CatalogMetadata,
    ConversationTarget,
    EntityTarget,
)


def _catalog() -> Catalog:
    entity = EntityTarget(
        entity_id="light.kitchen_light",
        name="kitchen light",
        normalized_name="kitchen light",
        aliases=["kitchen lamp"],
        domain="light",
        area="Kitchen",
        floor=None,
        device_name="Kitchen Lights",
        exposed=True,
        capabilities=["turn_on", "turn_off"],
        tokens=["kitchen", "light", "lamp"],
        phonetic_tokens=["K325", "L230", "L510"],
    )
    conv = ConversationTarget(
        target_id="manual:movie",
        type="manual",
        display_name="movie mode",
        normalized_name="movie mode",
        sample_phrases=["movie time"],
        canonical_phrase="activate movie mode",
        source="manual",
        slots=[],
        tokens=["movie", "mode", "time"],
        phonetic_tokens=["M100", "M300", "T500"],
        aliases=["movie moat"],
    )
    return Catalog(
        metadata=CatalogMetadata(
            revision="r1",
            last_refreshed="now",
            language="en",
            entity_count=1,
            conversation_target_count=1,
        ),
        entity_targets=[entity],
        conversation_targets=[conv],
    )


def test_token_correction_matches_entity() -> None:
    matcher = FuzzyMatcher(fuzzy_threshold=0.5, ambiguity_gap=0.05)
    result = matcher.match("turn on kitchen line", _catalog())
    assert result.best is not None
    assert result.best.candidate_id == "light.kitchen_light"


def test_phonetic_like_correction_matches_conversation_target() -> None:
    matcher = FuzzyMatcher(fuzzy_threshold=0.45, ambiguity_gap=0.05)
    result = matcher.match("activate movie moat", _catalog())
    assert result.best is not None
    assert result.best.candidate_id == "manual:movie"


def test_threshold_rejection() -> None:
    matcher = FuzzyMatcher(fuzzy_threshold=0.95, ambiguity_gap=0.05)
    result = matcher.match("random unrelated command", _catalog())
    assert result.matched is False


def test_ambiguity_rejection() -> None:
    catalog = _catalog()
    catalog.entity_targets.append(
        EntityTarget(
            entity_id="light.kitchen_strip",
            name="kitchen strip",
            normalized_name="kitchen strip",
            aliases=[],
            domain="light",
            area="Kitchen",
            floor=None,
            device_name=None,
            exposed=True,
            capabilities=["turn_on", "turn_off"],
            tokens=["kitchen", "strip"],
            phonetic_tokens=["K325", "S361"],
        )
    )
    matcher = FuzzyMatcher(fuzzy_threshold=0.4, ambiguity_gap=0.20)
    result = matcher.match("turn on kitchen", catalog)
    assert result.matched is False


def test_same_canonical_phrase_not_counted_as_ambiguity() -> None:
    catalog = _catalog()
    catalog.conversation_targets.append(
        ConversationTarget(
            target_id="manual:kitchen_light_alias",
            type="manual",
            display_name="kitchen main light",
            normalized_name="kitchen main light",
            sample_phrases=["kitchen light"],
            canonical_phrase="turn on kitchen light",
            source="manual",
            slots=[],
            tokens=["kitchen", "light", "main"],
            phonetic_tokens=["K325", "L230", "M500"],
            aliases=["kitchen line"],
        )
    )
    matcher = FuzzyMatcher(fuzzy_threshold=0.4, ambiguity_gap=0.20)
    result = matcher.match("turn on kitchen line", catalog)
    assert result.best is not None
    assert result.best.canonical_phrase == "turn on kitchen light"
    assert result.matched is True
    assert len({c.canonical_phrase for c in result.top_candidates}) == len(result.top_candidates)
