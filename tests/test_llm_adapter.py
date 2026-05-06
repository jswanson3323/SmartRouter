"""LLM translation parser tests."""

from custom_components.catalog_conversation_router.llm_adapter import LLMAdapter
from custom_components.catalog_conversation_router.models import Catalog, CatalogMetadata, ConversationTarget, EntityTarget


class _FakeAgentAdapter:
    async def async_process(self, **kwargs):
        raise NotImplementedError


def _catalog() -> Catalog:
    return Catalog(
        metadata=CatalogMetadata(
            revision="r1",
            last_refreshed="now",
            language="en",
            entity_count=3,
            conversation_target_count=2,
        ),
        entity_targets=[
            EntityTarget(
                entity_id="light.kitchen_light",
                name="Kitchen Light",
                normalized_name="kitchen light",
                aliases=[],
                domain="light",
                area="Kitchen",
                super_area="Downstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["kitchen", "light"],
                phonetic_tokens=["K325", "L230"],
            ),
            EntityTarget(
                entity_id="climate.downstairs",
                name="Downstairs Thermostat",
                normalized_name="downstairs thermostat",
                aliases=[],
                domain="climate",
                area="Downstairs",
                super_area="Downstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["set"],
                tokens=["downstairs", "thermostat"],
                phonetic_tokens=["D523", "T652"],
            ),
            EntityTarget(
                entity_id="fan.office",
                name="Office Fan",
                normalized_name="office fan",
                aliases=[],
                domain="fan",
                area="Office",
                super_area="Downstairs",
                floor=None,
                device_name=None,
                exposed=True,
                capabilities=["turn_on", "turn_off"],
                tokens=["office", "fan"],
                phonetic_tokens=["O120", "F500"],
            ),
        ],
        conversation_targets=[
            ConversationTarget(
                target_id="manual:timers",
                type="manual",
                display_name="timers",
                normalized_name="timers",
                sample_phrases=[],
                canonical_phrase="what timers do I have",
                source="manual",
                slots=[],
                tokens=["what", "timers", "do", "i", "have"],
                phonetic_tokens=["W300"],
            ),
            ConversationTarget(
                target_id="manual:movie",
                type="manual",
                display_name="movie mode",
                normalized_name="movie mode",
                sample_phrases=[],
                canonical_phrase="activate movie mode",
                source="manual",
                slots=[],
                tokens=["activate", "movie", "mode"],
                phonetic_tokens=["A231"],
            ),
        ],
    )


def test_invalid_llm_json_is_rejected() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    result = adapter._parse_translation_json("this is not json")
    assert result.valid is False
    assert result.notes == "no_json_found"


def test_valid_llm_json_translation() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    text = '{"mode":"translate","tool_group":"lighting","translated_text":"turn on Kitchen Light","confidence":0.93}'
    result = adapter._validate_translation_result(adapter._parse_translation_json(text), _catalog())
    assert result.valid is True
    assert result.canonical_text == "turn on Kitchen Light"
    assert result.tool_group == "lighting"


def test_translation_prompt_includes_full_catalog_lines() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    prompt = adapter._build_translation_prompt(
        utterance="turn on the light",
        language="en",
        catalog=_catalog(),
        max_candidates=20,
        origin_area="Kitchen",
    )
    assert "You are a translation/classification layer for a smart home AI system." in prompt
    assert "Catalog phrases:" in prompt
    assert "[lighting] turn on Kitchen Light" in prompt
    assert "[timers] what timers do I have" in prompt
    assert 'Return format:\n{"mode":"translate|state|control|general"' in prompt


def test_translation_prompt_dedupes_catalog_phrases() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    catalog = _catalog()
    catalog.conversation_targets.append(
        ConversationTarget(
            target_id="manual:timers-duplicate",
            type="manual",
            display_name="timers duplicate",
            normalized_name="timers duplicate",
            sample_phrases=[],
            canonical_phrase="  what   timers do I have  ",
            source="manual",
            slots=[],
            tokens=["what", "timers", "do", "i", "have"],
            phonetic_tokens=["W300"],
        )
    )
    lines = adapter._render_catalog_phrases(catalog)
    assert lines.count("[timers] what timers do I have") == 1


def test_non_translate_mode_is_parsed_but_not_usable_for_translation() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    result = adapter._validate_translation_result(
        adapter._parse_translation_json(
            '{"mode":"state","tool_group":"lighting","translated_text":null,"confidence":0.8}'
        ),
        _catalog(),
    )
    assert result.valid is True
    assert result.mode == "state"
    assert result.canonical_text is None


def test_invalid_llm_canonical_text_is_rejected_against_catalog() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    result = adapter._validate_translation_result(
        adapter._parse_translation_json(
            '{"mode":"translate","tool_group":"lighting","translated_text":"HassTurnOn","confidence":0.9}'
        ),
        _catalog(),
    )
    assert result.valid is False
    assert result.canonical_text is None
    assert result.notes == "invalid_canonical_text"


def test_translation_validation_normalizes_case_and_whitespace() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    result = adapter._validate_translation_result(
        adapter._parse_translation_json(
            '{"mode":"translate","tool_group":"lighting","translated_text":"  turn off   kitchen light ","confidence":0.9}'
        ),
        _catalog(),
    )
    assert result.valid is True
    assert result.canonical_text == "turn off Kitchen Light"


def test_translation_validation_infers_tool_group_from_catalog_line() -> None:
    adapter = LLMAdapter(_FakeAgentAdapter())
    result = adapter._validate_translation_result(
        adapter._parse_translation_json(
            '{"mode":"translate","tool_group":"not-real","translated_text":"what timers do I have","confidence":0.9}'
        ),
        _catalog(),
    )
    assert result.valid is True
    assert result.tool_group == "timers"
    assert result.notes == "tool_group_inferred"
