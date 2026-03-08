"""Router pipeline tests."""

import asyncio

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

    def match(self, text, catalog):
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


class _Candidate:
    def __init__(self, phrase, score=0.91):
        self.candidate_id = "light.kitchen"
        self.candidate_type = CandidateType.ENTITY_COMMAND
        self.canonical_phrase = phrase
        self.score = score
        self.action = "turn_on"
        self.target_name = "kitchen light"
        self.detail = {}


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


def test_llm_translated_local_success() -> None:
    match = _MatchResult(best=None, top=[])
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(False), _outcome(True)]),
        llm_adapter=_FakeLLMAdapter(_Translation(True, "turn on kitchen light"), _outcome(True)),
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


def test_final_fallback_path() -> None:
    match = _MatchResult(best=None, top=[])
    router = AgentRouter(
        config=_config(),
        catalog_manager=_FakeCatalogManager(),
        matcher=_FakeMatcher(match),
        agent_adapter=_FakeAgentAdapter([_outcome(False)]),
        llm_adapter=_FakeLLMAdapter(_Translation(False, None), _outcome(True, "llm answer")),
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
