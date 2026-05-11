"""Local intent resolver tests."""

from custom_components.catalog_conversation_router.local_intent import LocalIntentResolver
from custom_components.catalog_conversation_router.models import (
    Catalog,
    CatalogMetadata,
    ConversationTarget,
    EntityTarget,
)
from custom_components.catalog_conversation_router.semantic_intent import (
    SemanticEntityCandidate,
    SemanticPhraseCandidate,
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


class _FakeSemanticRanker:
    def __init__(self, *, phrase=None, entity=None):
        self._phrase = phrase or []
        self._entity = entity or []

    def available(self) -> bool:
        return True

    def unavailable_reason(self):
        return None

    def rank_phrase_candidates(self, **kwargs):
        return list(self._phrase)

    def rank_entity_commands(self, **kwargs):
        return list(self._entity)


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
    result = LocalIntentResolver(
        semantic_ranker=_FakeSemanticRanker(
            entity=[
                SemanticEntityCandidate(
                    action="turn_on",
                    target_name="office lights",
                    tool_group="lighting",
                    score=0.92,
                    synthetic=True,
                    singular=False,
                )
            ]
        )
    ).resolve(
        utterance="illuminate the office",
        catalog=_catalog(),
    )

    assert result.valid is True
    assert result.source == "semantic_entity_matcher"
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

    result = LocalIntentResolver(
        semantic_ranker=_FakeSemanticRanker(
            entity=[
                SemanticEntityCandidate(
                    action="turn_off",
                    target_name="office fan",
                    tool_group="fan",
                    score=0.9,
                    synthetic=False,
                    singular=True,
                )
            ]
        )
    ).resolve(
        utterance="kill the fan",
        catalog=catalog,
        origin_area="Office",
    )

    assert result.valid is True
    assert result.source == "semantic_entity_matcher"
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

    result = LocalIntentResolver(
        semantic_ranker=_FakeSemanticRanker(
            entity=[
                SemanticEntityCandidate(
                    action="turn_off",
                    target_name="office fans",
                    tool_group="fan",
                    score=0.9,
                    synthetic=True,
                    singular=False,
                )
            ]
        )
    ).resolve(
        utterance="kill the fans",
        catalog=catalog,
        origin_area="Office",
    )

    assert result.valid is True
    assert result.source == "semantic_entity_matcher"
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


def test_semantic_phrase_match_can_resolve_timer_status_by_meaning() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r5",
            last_refreshed="now",
            language="en",
            entity_count=0,
            conversation_target_count=1,
        ),
        entity_targets=[],
        conversation_targets=[
            _conversation_target("manual:timer-status", "do i have a timer running"),
        ],
    )
    ranker = _FakeSemanticRanker(
        phrase=[
            SemanticPhraseCandidate(
                target_id="manual:timer-status",
                example_text="do i have a timer running",
                raw_text="do i have a timer running",
                score=0.91,
                concrete=True,
                has_slots=False,
                tool_group="timers",
            )
        ]
    )

    result = LocalIntentResolver(semantic_ranker=ranker).resolve(
        utterance="did i set a timer",
        catalog=catalog,
    )

    assert result.valid is True
    assert result.source == "semantic_phrase_matcher"
    assert result.canonical_text == "do i have a timer running"


def test_semantic_entity_match_can_resolve_office_fan_by_meaning() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r6",
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
    ranker = _FakeSemanticRanker(
        entity=[
            SemanticEntityCandidate(
                action="turn_off",
                target_name="office fan",
                tool_group="fan",
                score=0.9,
                synthetic=False,
                singular=True,
            )
        ]
    )

    result = LocalIntentResolver(semantic_ranker=ranker).resolve(
        utterance="kill the fan",
        catalog=catalog,
        origin_area="Office",
    )

    assert result.valid is True
    assert result.source == "semantic_entity_matcher"
    assert result.canonical_text == "turn off office fan"


def test_semantic_entity_match_uses_origin_area_to_break_plural_tie() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r7",
            last_refreshed="now",
            language="en",
            entity_count=3,
            conversation_target_count=0,
        ),
        entity_targets=[
            _entity_target(
                "light.office_main",
                "Office Main Light",
                domain="light",
                area="Office",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
            _entity_target(
                "light.great_room_main",
                "Great Room Light",
                domain="light",
                area="Great Room",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
            _entity_target(
                "light.kitchen_main",
                "Kitchen Light",
                domain="light",
                area="Kitchen",
                capabilities=["turn_on", "turn_off", "set", "query"],
            ),
        ],
        conversation_targets=[],
    )
    ranker = _FakeSemanticRanker(
        entity=[
            SemanticEntityCandidate(
                action="turn_off",
                target_name="great room lights",
                tool_group="lighting",
                score=0.8534,
                synthetic=True,
                singular=False,
            ),
            SemanticEntityCandidate(
                action="turn_off",
                target_name="kitchen lights",
                tool_group="lighting",
                score=0.8467,
                synthetic=True,
                singular=False,
            ),
            SemanticEntityCandidate(
                action="turn_off",
                target_name="office lights",
                tool_group="lighting",
                score=0.834,
                synthetic=True,
                singular=False,
            ),
        ]
    )

    result = LocalIntentResolver(semantic_ranker=ranker).resolve(
        utterance="kill the lights",
        catalog=catalog,
        origin_area="Office",
    )

    assert result.valid is True
    assert result.source == "semantic_entity_matcher"
    assert result.canonical_text == "turn off office lights"


def test_semantic_direct_phrase_match_skips_device_control_phrase_noise() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r8",
            last_refreshed="now",
            language="en",
            entity_count=0,
            conversation_target_count=1,
        ),
        entity_targets=[],
        conversation_targets=[
            _conversation_target("manual:spa-lights-off", "turn off the spa lights"),
        ],
    )
    ranker = _FakeSemanticRanker(
        phrase=[
            SemanticPhraseCandidate(
                target_id="manual:spa-lights-off",
                example_text="turn off the spa lights",
                raw_text="turn off the spa lights",
                score=0.91,
                concrete=True,
                has_slots=False,
                tool_group="lighting",
            )
        ]
    )

    result = LocalIntentResolver(semantic_ranker=ranker).resolve(
        utterance="kill the lights",
        catalog=catalog,
    )

    assert result.valid is False
    assert result.source is None
