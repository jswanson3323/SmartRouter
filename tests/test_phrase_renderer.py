"""Generic conversation pattern rendering tests."""

from custom_components.catalog_conversation_router.phrase_renderer import render_conversation_pattern


def test_render_time_left_pattern() -> None:
    rendered = render_conversation_pattern(
        "How much time is left on the test timer?",
        "How much time is left on my {name} (alarm | timer | reminder)",
    )
    assert rendered.text == "how much time is left on my test timer"


def test_render_named_timer_pattern() -> None:
    rendered = render_conversation_pattern(
        "set a test timer for five minutes",
        "(set|start|create|begin) [a | an] {name} timer for {when}",
    )
    assert rendered.text == "set a test timer for five minutes"


def test_render_simple_timer_pattern() -> None:
    rendered = render_conversation_pattern(
        "set a timer for five minutes",
        "set [a] timer for {when}",
    )
    assert rendered.text == "set a timer for five minutes"


def test_render_named_alarm_pattern() -> None:
    rendered = render_conversation_pattern(
        "set a laundry alarm for 7 am",
        "set a {name} alarm for {when}",
    )
    assert rendered.text == "set a laundry alarm for 7 am"
