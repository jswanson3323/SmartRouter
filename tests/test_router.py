"""Router pipeline tests."""

import asyncio
import sys
import types

from custom_components.catalog_conversation_router.agent_router import AgentRouter
from custom_components.catalog_conversation_router.models import (
    CandidateType,
    Catalog,
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

    def match(self, text, catalog, origin_area=None, origin_super_area=None):
        return self._result


class _FakeAgentAdapter:
    def __init__(self, outcomes):
        self._outcomes = outcomes

    async def async_process(self, **kwargs):
        return self._outcomes.pop(0)


class _FakeLLMAdapter:
    def __init__(self, translation, fallback_outcome):
        self._translation = translation
        self._fallback_outcome = fallback_outcome

    async def async_translate_for_local(self, **kwargs):
        return self._translation

    async def async_final_fallback(self, **kwargs):
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


def _outcome(success, text="ok"):
    return LocalAgentOutcome(
        success=success,
        response={"text": text},
        response_text=text,
        failure_category=None,
        raw={},
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
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(False)]),
        llm_adapter=_FakeLLMAdapter(_Translation(False, None), _outcome(True, "llm answer")),
        hass=None,
    )
    result = asyncio.run(
        router.async_route(
            text="complex question",
            language="en",
            conversation_id=None,
            context=None,
        )
    )
    assert result.path.value == "llm_fallback"


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
