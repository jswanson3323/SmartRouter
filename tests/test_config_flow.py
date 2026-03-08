"""Config flow helper tests."""

import pytest

pytest.importorskip("homeassistant")

from custom_components.catalog_conversation_router.config_flow import (
    build_router_config,
    parse_manual_targets,
)


def test_build_router_config_defaults() -> None:
    payload = build_router_config({"local_agent_id": "local", "llm_agent_id": "llm"})
    assert payload["fuzzy_enabled"] is True
    assert payload["fuzzy_threshold"] == 0.84


def test_parse_manual_targets() -> None:
    targets = parse_manual_targets('[{"display_name": "Movie", "canonical_phrase": "activate movie mode"}]')
    assert isinstance(targets, list)
    assert targets[0]["display_name"] == "Movie"
