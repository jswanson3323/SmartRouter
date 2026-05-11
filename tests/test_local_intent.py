"""Local intent resolver tests."""

from custom_components.catalog_conversation_router.local_intent import LocalIntentResolver
from custom_components.catalog_conversation_router.models import (
    Catalog,
    CatalogMetadata,
    ConversationTarget,
    EntityTarget,
)


def _conversation_target(target_id: str, pattern: str) -> ConversationTarget:
    return ConversationTarget(
        target_id=target_id,
        type="manual",
        display_name=pattern,
        normalized_name=pattern.lower(),
        sample_phrases=[],
        canonical_phrase=pattern,
        source="manual",
        slots=[],
        tokens=[],
        phonetic_tokens=[],
    )


def _entity_target(
    entity_id: str,
    name: str,
    *,
    domain: str,
    area: str | None = None,
    super_area: str | None = None,
    capabilities: list[str] | None = None,
) -> EntityTarget:
    normalized_name = name.lower()
    return EntityTarget(
        entity_id=entity_id,
        name=name,
        normalized_name=normalized_name,
        aliases=[],
        domain=domain,
        area=area,
        super_area=super_area,
        floor=None,
        device_name=None,
        exposed=True,
        capabilities=capabilities or [],
        tokens=normalized_name.split(),
        phonetic_tokens=normalized_name.split(),
    )


def _catalog() -> Catalog:
    return Catalog(
        metadata=CatalogMetadata(
            revision="r1",
            last_refreshed="now",
            language="en",
            entity_count=3,
            conversation_target_count=3,
        ),
        entity_targets=[
            _entity_target(
                "light.office_desk",
                "Office Desk Light",
                domain="light",
                area="Office",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
            _entity_target(
                "light.office_hall",
                "Office Hall Light",
                domain="light",
                area="Office",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
            _entity_target(
                "fan.gym",
                "Gym Fan",
                domain="fan",
                area="Gym",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
        ],
        conversation_targets=[
            _conversation_target("manual:timer", "set [a] timer for {when}"),
            _conversation_target("manual:weather", "what's the forecast for {day}"),
            _conversation_target(
                "manual:timer-query",
                "when does my (timer | alarm | reminder) go off",
            ),
        ],
    )


def test_phrase_match_optional_words_render_concrete_phrase() -> None:
    result = LocalIntentResolver().resolve(
        utterance="set timer for five minutes",
        catalog=_catalog(),
    )

    assert result.valid is True
    assert result.source == "phrase_matcher"
    assert result.intent_family == "manual:timer"
    assert result.canonical_text == "set timer for five minutes"


def test_phrase_match_requires_day_slot() -> None:
    result = LocalIntentResolver().resolve(
        utterance="what's the forecast",
        catalog=_catalog(),
    )

    assert result.valid is False
    assert result.canonical_text is None


def test_phrase_match_requires_semantic_choice() -> None:
    result = LocalIntentResolver().resolve(
        utterance="when does my thing go off",
        catalog=_catalog(),
    )

    assert result.valid is False
    assert result.canonical_text is None


def test_entity_builder_builds_synthetic_area_command() -> None:
    result = LocalIntentResolver().resolve(
        utterance="turn on the office lights",
        catalog=_catalog(),
    )

    assert result.valid is True
    assert result.source == "entity_builder"
    assert result.canonical_text == "turn on office lights"


def test_entity_builder_declines_ambiguous_partial_command() -> None:
    result = LocalIntentResolver().resolve(
        utterance="turn on the light",
        catalog=_catalog(),
    )

    assert result.valid is False
    assert result.canonical_text is None


def test_entity_builder_uses_intent_grammar_for_illuminate() -> None:
    result = LocalIntentResolver().resolve(
        utterance="illuminate the office",
        catalog=_catalog(),
    )

    assert result.valid is True
    assert result.source == "entity_builder"
    assert result.canonical_text == "turn on office lights"


def test_entity_builder_uses_origin_area_for_generic_kill_the_fan() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r2",
            last_refreshed="now",
            language="en",
            entity_count=1,
            conversation_target_count=0,
        ),
        entity_targets=[
            _entity_target(
                "fan.office",
                "Office Fan",
                domain="fan",
                area="Office",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
        ],
        conversation_targets=[],
    )

    result = LocalIntentResolver().resolve(
        utterance="kill the fan",
        catalog=catalog,
        origin_area="Office",
    )

    assert result.valid is True
    assert result.source == "entity_builder"
    assert result.canonical_text == "turn off office fan"


def test_entity_builder_uses_plural_area_group_for_kill_the_fans() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r3",
            last_refreshed="now",
            language="en",
            entity_count=2,
            conversation_target_count=0,
        ),
        entity_targets=[
            _entity_target(
                "fan.office_left",
                "Office Left Fan",
                domain="fan",
                area="Office",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
            _entity_target(
                "fan.office_right",
                "Office Right Fan",
                domain="fan",
                area="Office",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
        ],
        conversation_targets=[],
    )

    result = LocalIntentResolver().resolve(
        utterance="kill the fans",
        catalog=catalog,
        origin_area="Office",
    )

    assert result.valid is True
    assert result.source == "entity_builder"
    assert result.canonical_text == "turn off office fans"


def test_entity_builder_declines_singular_light_when_only_plural_group_is_clear() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r4",
            last_refreshed="now",
            language="en",
            entity_count=2,
            conversation_target_count=0,
        ),
        entity_targets=[
            _entity_target(
                "light.office_desk",
                "Office Desk Light",
                domain="light",
                area="Office",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
            _entity_target(
                "light.office_hall",
                "Office Hall Light",
                domain="light",
                area="Office",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
        ],
        conversation_targets=[],
    )

    result = LocalIntentResolver().resolve(
        utterance="kill the light",
        catalog=catalog,
        origin_area="Office",
    )

    assert result.valid is False
    assert result.canonical_text is None
