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
        super_area=None,
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


def _room_catalog() -> Catalog:
    return Catalog(
        metadata=CatalogMetadata(
            revision="r2",
            last_refreshed="now",
            language="en",
            entity_count=3,
            conversation_target_count=0,
        ),
        entity_targets=[
            EntityTarget(
                entity_id="light.office_light",
                name="Office Light",
                normalized_name="office light",
                aliases=[],
                domain="light",
                area="Office",
                super_area=None,
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["office", "light"],
                phonetic_tokens=["O120", "L230"],
            ),
            EntityTarget(
                entity_id="fan.office_fan",
                name="Office Fan",
                normalized_name="office fan",
                aliases=[],
                domain="fan",
                area="Office",
                super_area=None,
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["office", "fan"],
                phonetic_tokens=["O120", "F500"],
            ),
            EntityTarget(
                entity_id="fan.great_room_fan",
                name="Great Room Fan",
                normalized_name="great room fan",
                aliases=["break room fan"],
                domain="fan",
                area="Great Room",
                super_area=None,
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["great", "room", "fan", "break"],
                phonetic_tokens=["G630", "R500", "F500", "B620"],
            ),
        ],
        conversation_targets=[],
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
            super_area=None,
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


def test_turn_on_office_life_maps_to_office_light() -> None:
    matcher = FuzzyMatcher(fuzzy_threshold=0.5, ambiguity_gap=0.05)
    result = matcher.match("turn on the office life", _room_catalog())
    assert result.best is not None
    assert result.best.candidate_id == "light.office_light"
    assert result.best.canonical_phrase.lower() == "turn on office light"
    assert result.parsed_target_before_normalization == "office life"
    assert result.parsed_target_after_normalization == "office light"


def test_turn_on_kitchen_line_maps_to_kitchen_light() -> None:
    matcher = FuzzyMatcher(fuzzy_threshold=0.5, ambiguity_gap=0.05)
    result = matcher.match("turn on the kitchen line", _catalog())
    assert result.best is not None
    assert result.best.canonical_phrase.lower() == "turn on kitchen light"
    assert result.parsed_target_after_normalization == "kitchen light"


def test_turn_off_break_room_fam_maps_to_great_room_fan() -> None:
    matcher = FuzzyMatcher(fuzzy_threshold=0.45, ambiguity_gap=0.05)
    result = matcher.match("turn off the break room fam", _room_catalog())
    assert result.best is not None
    assert result.best.candidate_id == "fan.great_room_fan"
    assert result.best.canonical_phrase.lower() == "turn off great room fan"
    assert result.parsed_target_after_normalization == "break room fan"


def test_parsed_target_extraction_great_room_fan() -> None:
    matcher = FuzzyMatcher(fuzzy_threshold=0.45, ambiguity_gap=0.05)
    result = matcher.match("turn on the great room fan", _room_catalog())
    assert result.parsed_target_before_normalization == "great room fan"


def test_parsed_target_extraction_office_life() -> None:
    matcher = FuzzyMatcher(fuzzy_threshold=0.45, ambiguity_gap=0.05)
    result = matcher.match("turn on the office life", _room_catalog())
    assert result.parsed_target_before_normalization == "office life"


def test_parsed_target_extraction_break_room_fam() -> None:
    matcher = FuzzyMatcher(fuzzy_threshold=0.45, ambiguity_gap=0.05)
    result = matcher.match("turn off the break room fam", _room_catalog())
    assert result.parsed_target_before_normalization == "break room fam"


def test_area_scoped_generic_light_resolves_to_master_bedroom() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r3",
            last_refreshed="now",
            language="en",
            entity_count=2,
            conversation_target_count=0,
        ),
        entity_targets=[
            EntityTarget(
                entity_id="light.master_bedroom_light",
                name="Master Bedroom Light",
                normalized_name="master bedroom light",
                aliases=[],
                domain="light",
                area="Master Bedroom",
                super_area="Upstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["master", "bedroom", "light"],
                phonetic_tokens=["M236", "B365", "L230"],
            ),
            EntityTarget(
                entity_id="light.gym_light",
                name="Gym Light",
                normalized_name="gym light",
                aliases=[],
                domain="light",
                area="Gym",
                super_area="Downstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["gym", "light"],
                phonetic_tokens=["J500", "L230"],
            ),
        ],
        conversation_targets=[],
    )
    matcher = FuzzyMatcher(fuzzy_threshold=0.5, ambiguity_gap=0.05)
    result = matcher.match("turn on the light", catalog, origin_area="Master Bedroom")
    assert result.best is not None
    assert result.best.candidate_id == "light.master_bedroom_light"
    assert result.best.canonical_phrase.lower() == "turn on master bedroom light"
    assert result.matched is True


def test_area_scoped_generic_ligh_typo_resolves_to_master_bedroom() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r4",
            last_refreshed="now",
            language="en",
            entity_count=2,
            conversation_target_count=0,
        ),
        entity_targets=[
            EntityTarget(
                entity_id="light.master_bedroom_light",
                name="Master Bedroom Light",
                normalized_name="master bedroom light",
                aliases=[],
                domain="light",
                area="Master Bedroom",
                super_area="Upstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["master", "bedroom", "light"],
                phonetic_tokens=["M236", "B365", "L230"],
            ),
            EntityTarget(
                entity_id="light.gym_light",
                name="Gym Light",
                normalized_name="gym light",
                aliases=[],
                domain="light",
                area="Gym",
                super_area="Downstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["gym", "light"],
                phonetic_tokens=["J500", "L230"],
            ),
        ],
        conversation_targets=[],
    )
    matcher = FuzzyMatcher(fuzzy_threshold=0.5, ambiguity_gap=0.05)
    result = matcher.match("turn on the ligh", catalog, origin_area="Master Bedroom")
    assert result.best is not None
    assert result.best.candidate_id == "light.master_bedroom_light"
    assert result.best.canonical_phrase.lower() == "turn on master bedroom light"
    assert result.parsed_target_after_normalization == "light"
    assert result.matched is True


def test_super_area_second_pass_resolves_when_area_has_no_match() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r5",
            last_refreshed="now",
            language="en",
            entity_count=2,
            conversation_target_count=0,
        ),
        entity_targets=[
            EntityTarget(
                entity_id="light.hall_light",
                name="Hall Light",
                normalized_name="hall light",
                aliases=[],
                domain="light",
                area="Hall",
                super_area="Upstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["hall", "light"],
                phonetic_tokens=["H400", "L230"],
            ),
            EntityTarget(
                entity_id="light.gym_light",
                name="Gym Light",
                normalized_name="gym light",
                aliases=[],
                domain="light",
                area="Gym",
                super_area="Downstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["gym", "light"],
                phonetic_tokens=["J500", "L230"],
            ),
        ],
        conversation_targets=[],
    )
    matcher = FuzzyMatcher(fuzzy_threshold=0.5, ambiguity_gap=0.05)
    result = matcher.match("turn on the light", catalog, origin_area="Master Bedroom")
    assert result.best is not None
    assert result.best.candidate_id == "light.hall_light"
    assert result.best.canonical_phrase.lower() == "turn on hall light"
    assert result.effective_super_area_hint == "Upstairs"
    assert result.matched is True


def test_area_match_beats_super_area_match() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r6",
            last_refreshed="now",
            language="en",
            entity_count=2,
            conversation_target_count=0,
        ),
        entity_targets=[
            EntityTarget(
                entity_id="light.master_bedroom_light",
                name="Master Bedroom Light",
                normalized_name="master bedroom light",
                aliases=[],
                domain="light",
                area="Master Bedroom",
                super_area="Upstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["master", "bedroom", "light"],
                phonetic_tokens=["M236", "B365", "L230"],
            ),
            EntityTarget(
                entity_id="light.hall_light",
                name="Hall Light",
                normalized_name="hall light",
                aliases=[],
                domain="light",
                area="Hall",
                super_area="Upstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["hall", "light"],
                phonetic_tokens=["H400", "L230"],
            ),
        ],
        conversation_targets=[],
    )
    matcher = FuzzyMatcher(fuzzy_threshold=0.5, ambiguity_gap=0.05)
    result = matcher.match("turn on the light", catalog, origin_area="Master Bedroom")
    assert result.best is not None
    assert result.best.candidate_id == "light.master_bedroom_light"
    assert result.best.canonical_phrase.lower() == "turn on master bedroom light"


def test_explicit_other_room_target_is_not_overridden_by_origin_area() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r7",
            last_refreshed="now",
            language="en",
            entity_count=2,
            conversation_target_count=0,
        ),
        entity_targets=[
            EntityTarget(
                entity_id="fan.guest_room_fan",
                name="Guest Room Fan",
                normalized_name="guest room fan",
                aliases=[],
                domain="fan",
                area="Guest Room",
                super_area="Upstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["guest", "room", "fan"],
                phonetic_tokens=["G230", "R500", "F500"],
            ),
            EntityTarget(
                entity_id="fan.gym_fan",
                name="Gym Fan",
                normalized_name="gym fan",
                aliases=[],
                domain="fan",
                area="Gym",
                super_area="Downstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["gym", "fan"],
                phonetic_tokens=["J500", "F500"],
            ),
        ],
        conversation_targets=[],
    )
    matcher = FuzzyMatcher(fuzzy_threshold=0.5, ambiguity_gap=0.05)
    result = matcher.match("turn off the gym fan", catalog, origin_area="Guest Room")
    assert result.best is not None
    assert result.best.candidate_id == "fan.gym_fan"
    assert result.best.canonical_phrase.lower() == "turn off gym fan"


def test_generic_fan_request_prefers_entity_over_spa_conversation_target() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r8",
            last_refreshed="now",
            language="en",
            entity_count=1,
            conversation_target_count=1,
        ),
        entity_targets=[
            EntityTarget(
                entity_id="fan.great_room_fan",
                name="Great Room Fan",
                normalized_name="great room fan",
                aliases=[],
                domain="fan",
                area="Great Room",
                super_area="Great Room",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["great", "room", "fan"],
                phonetic_tokens=["G630", "R500", "F500"],
            )
        ],
        conversation_targets=[
            ConversationTarget(
                target_id="manual:spa",
                type="manual",
                display_name="Spa Control",
                normalized_name="spa control",
                sample_phrases=["turn on the spa", "turn off the spa"],
                canonical_phrase="turn on the spa",
                source="manual",
                slots=[],
                tokens=["turn", "on", "spa"],
                phonetic_tokens=["T650", "O500", "S100"],
            )
        ],
    )
    matcher = FuzzyMatcher(fuzzy_threshold=0.5, ambiguity_gap=0.05)
    result = matcher.match(
        "turn on the fan",
        catalog,
        origin_area="Kitchen",
        origin_super_area="Great Room",
    )
    assert result.best is not None
    assert result.best.candidate_id == "fan.great_room_fan"
    assert result.best.canonical_phrase.lower() == "turn on great room fan"


def test_exact_conversation_match_wins_over_same_family_slot_variant() -> None:
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r9",
            last_refreshed="now",
            language="en",
            entity_count=0,
            conversation_target_count=2,
        ),
        entity_targets=[],
        conversation_targets=[
            ConversationTarget(
                target_id="automation_blueprint:custom/blueprint-spa.yaml:spa_on_ask",
                type="automation_blueprint",
                display_name="Spa Control",
                normalized_name="spa control",
                sample_phrases=["turn on the spa"],
                canonical_phrase="turn on the spa",
                source="automation_blueprint",
                slots=[],
                tokens=["turn", "on", "spa"],
                phonetic_tokens=["T650", "O500", "S100"],
            ),
            ConversationTarget(
                target_id="automation_blueprint:custom/blueprint-spa.yaml:spa_set_temp",
                type="automation_blueprint",
                display_name="Spa Control",
                normalized_name="spa control",
                sample_phrases=["turn on the spa to {amount}"],
                canonical_phrase="set the spa to {amount}",
                source="automation_blueprint",
                slots=["amount"],
                tokens=["turn", "on", "spa", "amount"],
                phonetic_tokens=["T650", "O500", "S100", "A553"],
            ),
        ],
    )
    matcher = FuzzyMatcher(fuzzy_threshold=0.63, ambiguity_gap=0.025)
    result = matcher.match("turn on the spa", catalog, origin_area="Office")
    assert result.best is not None
    assert result.best.candidate_id == "automation_blueprint:custom/blueprint-spa.yaml:spa_on_ask"
    assert result.best.canonical_phrase == "turn on the spa"
    assert result.matched is True
