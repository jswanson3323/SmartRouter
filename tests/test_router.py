"""Router pipeline tests."""

import asyncio
import sys
import types

from custom_components.catalog_conversation_router.agent_router import AgentRouter
from custom_components.catalog_conversation_router.models import (
    CandidateType,
    Catalog,
    EntityTarget,
    CatalogMetadata,
    LocalAgentOutcome,
    RouterConfig,
)


class _FakeCatalogManager:
    def __init__(self):
        self._catalog = Catalog(
            metadata=CatalogMetadata(
                revision="r1",
                last_refreshed="now",
                language="en",
                entity_count=0,
                conversation_target_count=0,
            )
        )

    def get_catalog(self):
        return self._catalog


class _FakeMatcher:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def match(self, text, catalog, origin_area=None, origin_super_area=None):
        self.calls.append(
            {
                "text": text,
                "origin_area": origin_area,
                "origin_super_area": origin_super_area,
            }
        )
        return self._result


class _FakeAgentAdapter:
    def __init__(self, outcomes):
        self._outcomes = outcomes
        self.calls = []

    async def async_process(self, **kwargs):
        self.calls.append(kwargs)
        return self._outcomes.pop(0)


class _FakeLLMAdapter:
    def __init__(self, translation, fallback_outcome):
        self._translation = translation
        self._fallback_outcome = fallback_outcome
        self.translate_calls = []
        self.fallback_calls = []

    async def async_translate_for_local(self, **kwargs):
        self.translate_calls.append(kwargs)
        return self._translation

    async def async_final_fallback(self, **kwargs):
        self.fallback_calls.append(kwargs)
        return self._fallback_outcome


class _MatchResult:
    def __init__(self, best=None, top=None, matched=False):
        self.best = best
        self.top_candidates = top or []
        self.matched = matched
        self.inferred_action = "turn_on"
        self.normalized_utterance = "turn on kitchen line"
        self.effective_area_hint = None
        self.effective_super_area_hint = None


class _FakeStates:
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, entity_id):
        return self._mapping.get(entity_id)


class _Candidate:
    def __init__(self, phrase, score=0.91, candidate_type=CandidateType.ENTITY_COMMAND, detail=None):
        self.candidate_id = "light.kitchen"
        self.candidate_type = candidate_type
        self.canonical_phrase = phrase
        self.score = score
        self.action = "turn_on"
        self.target_name = "kitchen light"
        self.detail = detail or {}


class _Translation:
    def __init__(self, valid, canonical_text):
        self.mode = "translate_for_local" if valid else "fallback_answer"
        self.canonical_text = canonical_text
        self.confidence = 0.9
        self.target_type = CandidateType.ENTITY_COMMAND
        self.notes = ""
        self.valid = valid


def _config() -> RouterConfig:
    return RouterConfig(
        local_agent_id="local",
        llm_agent_id="llm",
        language="en",
        fuzzy_enabled=True,
        fuzzy_threshold=0.84,
        ambiguity_gap=0.08,
        llm_translate_enabled=True,
        llm_fallback_enabled=True,
        debug_enabled=False,
        catalog_auto_refresh_enabled=True,
        high_risk_threshold=0.96,
        max_llm_candidates=20,
    )


def _entity_target(
    entity_id: str,
    name: str,
    *,
    domain: str,
    area: str | None = None,
    super_area: str | None = None,
    aliases: list[str] | None = None,
):
    normalized_name = name.lower()
    tokens = normalized_name.split()
    return EntityTarget(
        entity_id=entity_id,
        name=name,
        normalized_name=normalized_name,
        aliases=aliases or [],
        domain=domain,
        area=area,
        super_area=super_area,
        floor=None,
        device_name=None,
        exposed=True,
        capabilities=[],
        tokens=tokens,
        phonetic_tokens=tokens,
    )


def _outcome(success, text="ok"):
    return LocalAgentOutcome(
        success=success,
        response={"text": text},
        response_text=text,
        failure_category=None,
        raw={},
    )


def _conversation_result_outcome(
    success,
    text="ok",
    *,
    continue_conversation=False,
    conversation_id="downstream-conv-1",
):
    response = types.SimpleNamespace(
        conversation_id=conversation_id,
        continue_conversation=continue_conversation,
        response=types.SimpleNamespace(response_type="query_answer"),
    )
    return LocalAgentOutcome(
        success=success,
        response=response,
        response_text=text,
        failure_category=None,
        raw=response,
        processed_locally=False,
        conversation_id=conversation_id,
        continue_conversation=continue_conversation,
    )


def _local_conversation_result_outcome(
    success,
    text="ok",
    *,
    continue_conversation=False,
    conversation_id="downstream-conv-2",
):
    response = types.SimpleNamespace(
        conversation_id=conversation_id,
        continue_conversation=continue_conversation,
        response=types.SimpleNamespace(response_type="query_answer"),
        processed_locally=True,
    )
    return LocalAgentOutcome(
        success=success,
        response=response,
        response_text=text,
        failure_category=None,
        raw=response,
        processed_locally=True,
        conversation_id=conversation_id,
        continue_conversation=continue_conversation,
    )


def test_exact_local_success() -> None:
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=_FakeAgentAdapter([_outcome(True)]),
        llm_adapter=_FakeLLMAdapter(_Translation(False, None), _outcome(True)),
        hass=None,
    )
    result = asyncio.run(
        router.async_route(
            text="turn on kitchen light",
            language="en",
            conversation_id=None,
            context=None,
        )
    )
    assert result.path.value == "exact_local"


def test_fuzzy_path_success() -> None:
    match = _MatchResult(best=_Candidate("turn on kitchen light"), top=[_Candidate("turn on kitchen light")])
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(False), _outcome(True)]),
        llm_adapter=_FakeLLMAdapter(_Translation(False, None), _outcome(True)),
        hass=None,
    )
    result = asyncio.run(
        router.async_route(
            text="turn on kitchen line",
            language="en",
            conversation_id=None,
            context=None,
        )
    )
    assert result.path.value == "fuzzy_local"
    assert result.trace.fuzzy_match_duration_ms is not None
    assert result.trace.route_duration_ms is not None


def test_debug_collection_skips_llm_translation_after_fuzzy_match() -> None:
    match = _MatchResult(best=_Candidate("turn on kitchen light"), top=[_Candidate("turn on kitchen light")])
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(True)]),
        llm_adapter=_FakeLLMAdapter(_Translation(True, "turn on gym light"), _outcome(True)),
        hass=None,
    )
    result = asyncio.run(
        router.async_route(
            text="turn on kitchen line",
            language="en",
            conversation_id=None,
            context=None,
            dry_run=True,
            debug_collect_all=True,
        )
    )
    assert result.path.value == "fuzzy_local"
    assert result.trace.llm_translation_summary is not None
    assert result.trace.llm_translation_summary["mode"] == "skipped"
    assert result.trace.llm_translation_summary["notes"] == "skipped_due_to_fuzzy_match"
    assert result.trace.llm_translated_local_executed is False


def test_llm_translated_local_success() -> None:
    match = _MatchResult(best=None, top=[])
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(False), _outcome(True)]),
        llm_adapter=_FakeLLMAdapter(_Translation(True, "turn on kitchen light"), _outcome(True)),
        hass=None,
    )
    result = asyncio.run(
        router.async_route(
            text="turn on kitchen line",
            language="en",
            conversation_id=None,
            context=None,
        )
    )
    assert result.path.value == "llm_translated_local"


def test_area_scoped_ambiguity_still_executes_locally() -> None:
    best = _Candidate("turn on master bedroom light", score=0.90)
    best.detail = {"area_scoped_domain_resolution": 1.0}
    second = _Candidate("turn on gym light", score=0.89)
    match = _MatchResult(best=best, top=[best, second])
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(True)]),
        llm_adapter=_FakeLLMAdapter(_Translation(True, "turn on gym light"), _outcome(True)),
        hass=None,
    )
    result = asyncio.run(
        router.async_route(
            text="turn on the ligh",
            language="en",
            conversation_id=None,
            context=None,
        )
    )
    assert result.path.value == "fuzzy_local"
    assert result.trace.assist_pipeline_input == "turn on master bedroom light"


def test_final_fallback_path() -> None:
    match = _MatchResult(best=None, top=[])
    llm_adapter = _FakeLLMAdapter(_Translation(False, None), _outcome(True, "llm answer"))
    context = object()
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(False)]),
        llm_adapter=llm_adapter,
        hass=None,
    )
    result = asyncio.run(
        router.async_route(
            text="complex question",
            language="en",
            conversation_id="conv-1",
            context=context,
            device_id="device-123",
            satellite_id="assist_satellite.kitchen",
            extra_system_prompt="You are assisting from the kitchen satellite.",
        )
    )
    assert result.path.value == "llm_fallback"
    assert llm_adapter.fallback_calls[-1]["llm_agent_id"] == "llm"
    assert llm_adapter.fallback_calls[-1]["language"] == "en"
    assert llm_adapter.fallback_calls[-1]["conversation_id"] == "conv-1"
    assert llm_adapter.fallback_calls[-1]["context"] is context
    assert llm_adapter.fallback_calls[-1]["device_id"] == "device-123"
    assert llm_adapter.fallback_calls[-1]["satellite_id"] == "assist_satellite.kitchen"
    assert llm_adapter.fallback_calls[-1]["extra_system_prompt"] is None
    assert result.trace.llm_fallback_upstream_prompt_suppressed is True
    assert result.trace.llm_fallback_upstream_prompt_chars == len("You are assisting from the kitchen satellite.")
    assert result.trace.llm_fallback_prompt_chars == 0
    assert result.trace.llm_fallback_duration_ms is not None
    assert result.trace.route_duration_ms is not None


def test_final_fallback_suppresses_large_upstream_prompt() -> None:
    match = _MatchResult(best=None, top=[])
    llm_adapter = _FakeLLMAdapter(_Translation(False, None), _outcome(True, "llm answer"))
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(False)]),
        llm_adapter=llm_adapter,
        hass=None,
    )
    upstream_prompt = "large upstream prompt " * 1000

    result = asyncio.run(
        router.async_route(
            text="complex question",
            language="en",
            conversation_id="conv-large",
            context=None,
            extra_system_prompt=upstream_prompt,
        )
    )

    assert result.path.value == "llm_fallback"
    assert llm_adapter.fallback_calls[-1]["extra_system_prompt"] is None
    assert result.trace.llm_fallback_upstream_prompt_suppressed is True
    assert result.trace.llm_fallback_upstream_prompt_chars == len(upstream_prompt.strip())
    assert result.trace.llm_fallback_prompt_chars == 0
    assert result.trace.llm_state_enrichment_prompt is None


def test_active_llm_conversation_bypasses_fuzzy_and_returns_to_llm() -> None:
    match = _MatchResult(best=_Candidate("turn on kitchen light"), top=[_Candidate("turn on kitchen light")])
    matcher = _FakeMatcher(match)
    llm_adapter = _FakeLLMAdapter(
        _Translation(False, None),
        _conversation_result_outcome(True, "For how long?", continue_conversation=True),
    )
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=matcher,
        agent_adapter=_FakeAgentAdapter([_outcome(False)]),
        llm_adapter=llm_adapter,
        hass=None,
    )

    first = asyncio.run(
        router.async_route(
            text="set a timer",
            language="en",
            conversation_id="outer-conv-1",
            context=None,
        )
    )
    assert first.path.value == "llm_fallback"
    assert matcher.calls

    matcher.calls.clear()
    llm_adapter._fallback_outcome = _conversation_result_outcome(
        True, "Timer set for ten minutes.", continue_conversation=False
    )
    second = asyncio.run(
        router.async_route(
            text="ten minutes",
            language="en",
            conversation_id="outer-conv-1",
            context=None,
            device_id="device-123",
            satellite_id="assist_satellite.kitchen",
            extra_system_prompt="You are assisting from the kitchen satellite.",
        )
    )

    assert second.path.value == "llm_fallback"
    assert matcher.calls == []
    assert llm_adapter.fallback_calls[-1]["conversation_id"] == "downstream-conv-1"
    assert llm_adapter.fallback_calls[-1]["device_id"] == "device-123"
    assert llm_adapter.fallback_calls[-1]["satellite_id"] == "assist_satellite.kitchen"
    assert llm_adapter.fallback_calls[-1]["extra_system_prompt"] is None
    assert second.trace.llm_fallback_upstream_prompt_suppressed is True
    assert second.trace.llm_fallback_upstream_prompt_chars == len("You are assisting from the kitchen satellite.")
    assert second.trace.llm_fallback_prompt_chars == 0
    assert second.trace.llm_translation_summary is not None
    assert second.trace.llm_translation_summary["notes"] == "active_llm_conversation"
    assert second.trace.downstream_conversation_id == "downstream-conv-1"


def test_active_llm_conversation_adds_targeted_state_enrichment() -> None:
    catalog_manager = _FakeCatalogManager()
    catalog_manager._catalog.entity_targets = [
        _entity_target("light.office", "Office Light", domain="light", area="Office"),
        _entity_target("light.teen_room", "Teen Room Light", domain="light", area="Teen Room"),
    ]
    llm_adapter = _FakeLLMAdapter(
        _Translation(False, None),
        _conversation_result_outcome(True, "The Office Light is on.", continue_conversation=True),
    )
    hass = types.SimpleNamespace(
        states=_FakeStates(
            {
                "light.office": types.SimpleNamespace(state="on", attributes={}),
                "light.teen_room": types.SimpleNamespace(state="off", attributes={}),
            }
        )
    )
    router = AgentRouter(
        config=_config(),
        catalog_manager=catalog_manager,
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=_FakeAgentAdapter([_outcome(False)]),
        llm_adapter=llm_adapter,
        hass=hass,
    )

    first = asyncio.run(
        router.async_route(
            text="ask me a question",
            language="en",
            conversation_id="outer-enrich-1",
            context=None,
        )
    )
    assert first.path.value == "llm_fallback"

    second = asyncio.run(
        router.async_route(
            text="the office light and the teen room light",
            language="en",
            conversation_id="outer-enrich-1",
            context=None,
            extra_system_prompt="large upstream prompt",
        )
    )

    assert second.path.value == "llm_fallback"
    prompt = llm_adapter.fallback_calls[-1]["extra_system_prompt"]
    assert "Router-resolved live state for this turn" in prompt
    assert "Office Light: on" in prompt
    assert "Teen Room Light: off" in prompt
    assert "upstream prompt" not in prompt
    assert second.trace.llm_state_enrichment_applied is True
    assert second.trace.llm_state_enrichment_targets == ["Office Light", "Teen Room Light"]
    assert second.trace.llm_fallback_upstream_prompt_suppressed is True
    assert second.trace.llm_fallback_upstream_prompt_chars == len("large upstream prompt")
    assert second.trace.llm_fallback_prompt_chars == len(prompt)


def test_active_llm_state_enrichment_prefers_origin_area() -> None:
    catalog_manager = _FakeCatalogManager()
    catalog_manager._catalog.entity_targets = [
        _entity_target(
            "light.office_hall",
            "Hall Light",
            domain="light",
            area="Office",
            super_area="Great Room",
        ),
        _entity_target(
            "light.great_room_hall",
            "Hall Light",
            domain="light",
            area="Great Room",
            super_area="Great Room",
        ),
    ]
    llm_adapter = _FakeLLMAdapter(
        _Translation(False, None),
        _conversation_result_outcome(True, "The Hall Light is on.", continue_conversation=True),
    )
    hass = types.SimpleNamespace(
        states=_FakeStates(
            {
                "light.office_hall": types.SimpleNamespace(state="on", attributes={}),
                "light.great_room_hall": types.SimpleNamespace(state="off", attributes={}),
            }
        )
    )
    router = AgentRouter(
        config=_config(),
        catalog_manager=catalog_manager,
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=_FakeAgentAdapter([_outcome(False)]),
        llm_adapter=llm_adapter,
        hass=hass,
    )
    router._active_conversations["outer-enrich-2"] = types.SimpleNamespace(
        outer_conversation_id="outer-enrich-2",
        executor_type="llm",
        agent_id="llm",
        downstream_conversation_id="downstream-conv-1",
    )

    result = asyncio.run(
        router.async_route(
            text="the hall light",
            language="en",
            conversation_id="outer-enrich-2",
            context=None,
            origin_area="Office",
        )
    )

    assert result.path.value == "llm_fallback"
    prompt = llm_adapter.fallback_calls[-1]["extra_system_prompt"]
    assert "Hall Light: on" in prompt
    assert "Hall Light: off" not in prompt


def test_first_turn_llm_state_enrichment_applies_for_temperature_query() -> None:
    catalog_manager = _FakeCatalogManager()
    catalog_manager._catalog.entity_targets = [
        _entity_target(
            "climate.great_room_thermostat",
            "Great Room Thermostat",
            domain="climate",
            area="Great Room",
        ),
        _entity_target(
            "sensor.great_room_temperature",
            "Great Room Thermostat Temperature",
            domain="sensor",
            area="Great Room",
        ),
    ]
    llm_adapter = _FakeLLMAdapter(_Translation(False, None), _outcome(True, "The temperature is 76 F."))
    hass = types.SimpleNamespace(
        states=_FakeStates(
            {
                "climate.great_room_thermostat": types.SimpleNamespace(
                    state="heat_cool",
                    attributes={"current_temperature": 76, "temperature_unit": "°F"},
                ),
                "sensor.great_room_temperature": types.SimpleNamespace(
                    state="76.1",
                    attributes={"unit_of_measurement": "°F"},
                ),
            }
        )
    )
    router = AgentRouter(
        config=_config(),
        catalog_manager=catalog_manager,
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=_FakeAgentAdapter([_outcome(True, "The temperature is 76 F.")]),
        llm_adapter=llm_adapter,
        hass=hass,
    )

    result = asyncio.run(
        router.async_route(
            text="What is the tempurature in the great room?",
            language="en",
            conversation_id=None,
            context=None,
        )
    )

    assert result.path.value == "local_state_query"
    assert router._agent_adapter.calls[-1]["text"] == "what is the current temperature in Great Room"
    assert result.trace.state_query_detected is True
    assert result.trace.state_query_kind == "temperature"
    assert result.trace.state_query_fuzzy_match_target == "Great Room"


def test_active_llm_state_enrichment_handles_binary_state_query() -> None:
    catalog_manager = _FakeCatalogManager()
    catalog_manager._catalog.entity_targets = [
        _entity_target("light.office", "Office Light", domain="light", area="Office"),
    ]
    llm_adapter = _FakeLLMAdapter(
        _Translation(False, None),
        _conversation_result_outcome(True, "The office light is on.", continue_conversation=True),
    )
    hass = types.SimpleNamespace(
        states=_FakeStates(
            {
                "light.office": types.SimpleNamespace(state="on", attributes={}),
            }
        )
    )
    router = AgentRouter(
        config=_config(),
        catalog_manager=catalog_manager,
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=_FakeAgentAdapter([_outcome(True, "The office light is on.")]),
        llm_adapter=llm_adapter,
        hass=hass,
    )
    router._active_conversations["outer-binary-1"] = types.SimpleNamespace(
        outer_conversation_id="outer-binary-1",
        executor_type="llm",
        agent_id="llm",
        downstream_conversation_id="downstream-conv-1",
    )

    result = asyncio.run(
        router.async_route(
            text="Is the office light on?",
            language="en",
            conversation_id="outer-binary-1",
            context=None,
        )
    )

    assert result.path.value == "local_state_query"
    assert router._agent_adapter.calls[-1]["text"] == "what is the state of Office Light"
    assert llm_adapter.fallback_calls == []
    assert result.trace.state_query_detected is True
    assert result.trace.state_query_kind == "binary"
    assert result.trace.state_query_fuzzy_match_target == "Office Light"
    assert result.trace.state_query_intercepted_during_llm is True


def test_fuzzy_state_query_recovers_line_to_light() -> None:
    catalog_manager = _FakeCatalogManager()
    catalog_manager._catalog.entity_targets = [
        _entity_target("light.office", "Office Light", domain="light", area="Office"),
    ]
    agent_adapter = _FakeAgentAdapter([_outcome(True, "The office light is on.")])
    router = AgentRouter(
        config=_config(),
        catalog_manager=catalog_manager,
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=agent_adapter,
        llm_adapter=_FakeLLMAdapter(_Translation(False, None), _outcome(True)),
        hass=types.SimpleNamespace(
            states=_FakeStates(
                {
                    "light.office": types.SimpleNamespace(state="on", attributes={}),
                }
            )
        ),
    )

    result = asyncio.run(
        router.async_route(
            text="Is the office line on?",
            language="en",
            conversation_id=None,
            context=None,
        )
    )

    assert result.path.value == "local_state_query"
    assert agent_adapter.calls[-1]["text"] == "what is the state of Office Light"
    assert result.trace.state_query_fuzzy_match_target == "Office Light"


def test_active_llm_owner_preserved_after_local_state_query() -> None:
    catalog_manager = _FakeCatalogManager()
    catalog_manager._catalog.entity_targets = [
        _entity_target("light.office", "Office Light", domain="light", area="Office"),
    ]
    agent_adapter = _FakeAgentAdapter([_outcome(True, "The office light is on.")])
    llm_adapter = _FakeLLMAdapter(
        _Translation(False, None),
        _conversation_result_outcome(True, "Because it changed since earlier.", continue_conversation=False),
    )
    router = AgentRouter(
        config=_config(),
        catalog_manager=catalog_manager,
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=agent_adapter,
        llm_adapter=llm_adapter,
        hass=types.SimpleNamespace(
            states=_FakeStates(
                {
                    "light.office": types.SimpleNamespace(state="on", attributes={}),
                }
            )
        ),
    )
    router._active_conversations["outer-owner-1"] = types.SimpleNamespace(
        outer_conversation_id="outer-owner-1",
        executor_type="llm",
        agent_id="llm",
        downstream_conversation_id="downstream-conv-1",
    )

    first = asyncio.run(
        router.async_route(
            text="Is the office light on?",
            language="en",
            conversation_id="outer-owner-1",
            context=None,
        )
    )
    assert first.path.value == "local_state_query"

    second = asyncio.run(
        router.async_route(
            text="Why was that different earlier?",
            language="en",
            conversation_id="outer-owner-1",
            context=None,
        )
    )
    assert second.path.value == "llm_fallback"
    assert llm_adapter.fallback_calls[-1]["conversation_id"] == "downstream-conv-1"


def test_active_llm_explicit_local_action_is_intercepted() -> None:
    catalog_manager = _FakeCatalogManager()
    catalog_manager._catalog.entity_targets = [
        _entity_target("light.office_drum_light", "Office Drum Light", domain="light", area="Office"),
    ]
    agent_adapter = _FakeAgentAdapter([_outcome(True, "Turned on the Office Drum Light")])
    llm_adapter = _FakeLLMAdapter(
        _Translation(False, None),
        _conversation_result_outcome(True, "LLM should not answer this turn.", continue_conversation=False),
    )
    router = AgentRouter(
        config=_config(),
        catalog_manager=catalog_manager,
        matcher=_FakeMatcher(
            _MatchResult(
                best=_Candidate("turn on Office Drum Light"),
                top=[_Candidate("turn on Office Drum Light")],
                matched=True,
            )
        ),
        agent_adapter=agent_adapter,
        llm_adapter=llm_adapter,
        hass=None,
    )
    router._active_conversations["outer-action-1"] = types.SimpleNamespace(
        outer_conversation_id="outer-action-1",
        executor_type="llm",
        agent_id="llm",
        downstream_conversation_id="downstream-conv-1",
    )

    result = asyncio.run(
        router.async_route(
            text="turn on the office drum light",
            language="en",
            conversation_id="outer-action-1",
            context=None,
        )
    )

    assert result.path.value == "fuzzy_local"
    assert agent_adapter.calls[-1]["text"] == "turn on Office Drum Light"
    assert agent_adapter.calls[-1]["conversation_id"] is None
    assert llm_adapter.fallback_calls == []
    assert result.trace.local_action_detected is True
    assert result.trace.local_action_intercepted_during_llm is True


def test_area_domain_state_query_intercepts_to_local_agent() -> None:
    agent_adapter = _FakeAgentAdapter([_outcome(True, "Cabinet Lights and Kitchen Sink Light")])
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=agent_adapter,
        llm_adapter=_FakeLLMAdapter(_Translation(False, None), _outcome(True)),
        hass=None,
    )

    result = asyncio.run(
        router.async_route(
            text="What lights are on in the kitchen?",
            language="en",
            conversation_id=None,
            context=None,
        )
    )

    assert result.path.value == "local_state_query"
    assert agent_adapter.calls[-1]["text"] == "what lights are on in kitchen"
    assert result.trace.state_query_detected is True
    assert result.trace.state_query_kind == "domain_state"
    assert result.trace.state_query_target_phrase == "kitchen"


def test_active_local_conversation_bypasses_fuzzy_and_returns_to_local() -> None:
    match = _MatchResult(best=_Candidate("turn on gym fan"), top=[_Candidate("turn on gym fan")])
    matcher = _FakeMatcher(match)
    agent_adapter = _FakeAgentAdapter(
        [
            _local_conversation_result_outcome(True, "What temp?", continue_conversation=True),
            _local_conversation_result_outcome(True, "Hot tub set to 99.", continue_conversation=False),
        ]
    )
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=matcher,
        agent_adapter=agent_adapter,
        llm_adapter=_FakeLLMAdapter(_Translation(False, None), _outcome(True)),
        hass=None,
    )

    first = asyncio.run(
        router.async_route(
            text="turn on the hot tub",
            language="en",
            conversation_id="outer-conv-2",
            context=None,
        )
    )
    assert first.path.value == "exact_local"
    assert matcher.calls

    matcher.calls.clear()
    second = asyncio.run(
        router.async_route(
            text="99",
            language="en",
            conversation_id="outer-conv-2",
            context=None,
            device_id="device-123",
            satellite_id="assist_satellite.spa",
            extra_system_prompt="You are assisting from the spa satellite.",
        )
    )

    assert second.path.value == "exact_local"
    assert matcher.calls == []
    assert agent_adapter.calls[-1]["conversation_id"] == "downstream-conv-2"
    assert agent_adapter.calls[-1]["device_id"] == "device-123"
    assert agent_adapter.calls[-1]["satellite_id"] == "assist_satellite.spa"
    assert agent_adapter.calls[-1]["extra_system_prompt"] == "You are assisting from the spa satellite."
    assert second.trace.llm_translation_summary is not None
    assert second.trace.llm_translation_summary["notes"] == "active_local_conversation"
    assert second.trace.downstream_conversation_id == "downstream-conv-2"


def test_conversation_pattern_is_rendered_for_assist_input() -> None:
    pattern = "How much time is left on my {name} (alarm | timer | reminder)"
    match = _MatchResult(
        best=_Candidate(
            pattern,
            candidate_type=CandidateType.CONVERSATION_TARGET,
            detail={
                "matched_sample_phrase_raw": pattern,
                "matched_sample_phrase_normalized_for_scoring": "how much time is left on my timer",
            },
        ),
        top=[
            _Candidate(
                pattern,
                candidate_type=CandidateType.CONVERSATION_TARGET,
                detail={"matched_sample_phrase_raw": pattern},
            )
        ],
    )
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(True)]),
        llm_adapter=_FakeLLMAdapter(_Translation(False, None), _outcome(True)),
        hass=None,
    )
    result = asyncio.run(
        router.async_route(
            text="How much time is left on the test timer?",
            language="en",
            conversation_id=None,
            context=None,
        )
    )
    assert result.path.value == "fuzzy_local"
    assert result.trace.chosen_canonical_phrase == pattern
    assert result.trace.assist_pipeline_input == "how much time is left on my test timer"


def test_origin_super_area_resolves_from_area_label_ids(monkeypatch) -> None:
    area = types.SimpleNamespace(
        id="area_kitchen",
        name="Kitchen",
        label_ids=["label_great_room"],
    )
    area_reg = types.SimpleNamespace(
        _areas={"area_kitchen": area},
        async_get_area=lambda area_id: area if area_id == "area_kitchen" else None,
    )
    device_reg = types.SimpleNamespace(
        async_get=lambda device_id: types.SimpleNamespace(area_id="area_kitchen")
        if device_id == "device1"
        else None
    )
    entity_reg = types.SimpleNamespace(async_get=lambda entity_id: None)
    label_reg = types.SimpleNamespace(
        labels={"label_great_room": types.SimpleNamespace(name="SuperArea: Great Room")}
    )

    ar_mod = types.ModuleType("homeassistant.helpers.area_registry")
    ar_mod.async_get = lambda _hass: area_reg
    dr_mod = types.ModuleType("homeassistant.helpers.device_registry")
    dr_mod.async_get = lambda _hass: device_reg
    er_mod = types.ModuleType("homeassistant.helpers.entity_registry")
    er_mod.async_get = lambda _hass: entity_reg
    lr_mod = types.ModuleType("homeassistant.helpers.label_registry")
    lr_mod.async_get = lambda _hass: label_reg

    monkeypatch.setitem(sys.modules, "homeassistant.helpers.area_registry", ar_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.device_registry", dr_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.entity_registry", er_mod)
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.label_registry", lr_mod)

    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=_FakeAgentAdapter([_outcome(True)]),
        llm_adapter=_FakeLLMAdapter(_Translation(False, None), _outcome(True)),
        hass=object(),
    )

    result = router._resolve_origin_super_area(
        origin_area="Kitchen",
        device_id="device1",
        satellite_id=None,
        context=types.SimpleNamespace(device_id=None),
    )
    assert result == "Great Room"


def test_open_domain_request_skips_translation_and_exact_local() -> None:
    llm_adapter = _FakeLLMAdapter(_Translation(False, None), _outcome(True, text="boil eggs like this"))
    agent_adapter = _FakeAgentAdapter([])
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=agent_adapter,
        llm_adapter=llm_adapter,
        hass=None,
    )

    result = asyncio.run(
        router.async_route(
            text="how do i boil eggs?",
            language="en",
            conversation_id=None,
            context=None,
        )
    )

    assert result.path.value == "llm_fallback"
    assert llm_adapter.translate_calls == []
    assert len(llm_adapter.fallback_calls) == 1
    assert agent_adapter.calls == []
    assert result.trace.llm_translation_summary is not None
    assert result.trace.llm_translation_summary["notes"] == "direct_llm_only"


def test_open_domain_request_returns_streaming_fallback_request() -> None:
    llm_adapter = _FakeLLMAdapter(_Translation(False, None), _outcome(True, text="boil eggs like this"))
    agent_adapter = _FakeAgentAdapter([])
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(_MatchResult()),
        agent_adapter=agent_adapter,
        llm_adapter=llm_adapter,
        hass=None,
    )

    result = asyncio.run(
        router.async_route(
            text="how do i boil eggs?",
            language="en",
            conversation_id="outer-conv-1",
            context=None,
            allow_streaming_llm_fallback=True,
            extra_system_prompt="streaming upstream prompt",
        )
    )

    assert result.path.value == "llm_fallback"
    assert result.streaming_request is not None
    assert result.streaming_request.utterance == "how do i boil eggs?"
    assert result.streaming_request.conversation_id == "outer-conv-1"
    assert result.streaming_request.extra_system_prompt is None
    assert llm_adapter.translate_calls == []
    assert llm_adapter.fallback_calls == []
    assert agent_adapter.calls == []
    assert result.trace.llm_fallback_stream_attempted is True
    assert result.trace.llm_fallback_upstream_prompt_suppressed is True
    assert result.trace.llm_fallback_upstream_prompt_chars == len("streaming upstream prompt")
    assert result.trace.llm_fallback_prompt_chars == 0
    assert result.trace.route_duration_ms is not None
