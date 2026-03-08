"""LLM translation parser tests."""

from custom_components.catalog_conversation_router.llm_adapter import LLMAdapter


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
