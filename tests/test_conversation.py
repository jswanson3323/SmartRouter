"""Conversation streaming helper tests."""

from custom_components.catalog_conversation_router.conversation import _should_flush_stream_delta


def test_stream_delta_waits_for_meaningful_chunk() -> None:
    assert _should_flush_stream_delta("Un") is False
    assert _should_flush_stream_delta("Unicorns are mythical") is False


def test_stream_delta_flushes_on_sentence_boundary_or_size() -> None:
    assert _should_flush_stream_delta("Unicorns are mythical creatures.") is True
    assert _should_flush_stream_delta("Unicorns are mythical creatures with one horn") is True
