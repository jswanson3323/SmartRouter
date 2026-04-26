"""LLM translation parser tests."""

from custom_components.catalog_conversation_router.llm_adapter import LLMAdapter
from custom_components.catalog_conversation_router.models import Catalog, CatalogMetadata, ConversationTarget, EntityTarget


class _FakeAgentAdapter:
    async def async_process(self, **kwargs):
        raise NotImplementedError


def test_invalid_llm_json_is_rejected() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    result = adapter._parse_translation_json("this is not json")
    assert result.valid is False
    assert result.notes == "no_json_found"


def test_valid_llm_json_translation() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    text = '{"mode": "translate_for_local", "canonical_text": "turn on kitchen light", "confidence": 0.93, "target_type": "entity_command", "notes": "mapped line to light"}'
    result = adapter._parse_translation_json(text)
    assert result.valid is True
    assert result.canonical_text == "turn on kitchen light"


def test_translation_prompt_includes_origin_area_context() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r1",
            last_refreshed="now",
            language="en",
            entity_count=3,
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
    prompt = adapter._build_translation_prompt(
        utterance="turn on the light",
        language="en",
        catalog=catalog,
        max_candidates=20,
        origin_area="Master Bedroom",
    )
    assert "Origin area: 'Master Bedroom'" in prompt
    assert "Origin SuperArea: 'Upstairs'" in prompt
    assert "Origin-area entity targets: ['Master Bedroom Light']" in prompt
    assert "Origin-SuperArea entity targets: ['Hall Light']" in prompt
    assert "Never return tool names" in prompt
    assert "HassTurnOn" in prompt


def test_invalid_llm_canonical_text_is_rejected_against_catalog() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    catalog = Catalog(
        metadata=CatalogMetadata(
            revision="r1",
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
                target_id="manual:movie",
                type="manual",
                display_name="movie mode",
                normalized_name="movie mode",
                sample_phrases=[],
                canonical_phrase="activate movie mode",
                source="manual",
                slots=[],
                tokens=["movie", "mode"],
                phonetic_tokens=["M100", "M300"],
            )
        ],
    )
    result = adapter._validate_translation_result(
        adapter._parse_translation_json(
            '{"mode":"translate_for_local","canonical_text":"HassTurnOn","confidence":0.9,"target_type":"entity_command","notes":"bad"}'
        ),
        catalog,
    )
    assert result.valid is False
    assert result.canonical_text is None
    assert result.notes == "invalid_canonical_text"
