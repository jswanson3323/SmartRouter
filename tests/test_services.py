"""Service registration tests."""

import asyncio

from custom_components.catalog_conversation_router.const import (
    ATTR_EXECUTE_BRANCHES,
    ATTR_PATH,
    ATTR_TEXT,
    DOMAIN,
    SERVICE_GET_CATALOG_STATS,
    SERVICE_REBUILD_CATALOG,
    SERVICE_TEST_UTTERANCE,
    SERVICE_TEST_UTTERANCE_TO_FILE,
)
from custom_components.catalog_conversation_router.services import async_register_services


class _FakeCatalogManager:
    def __init__(self):
        self.rebuild_calls = 0

    async def async_rebuild(self):
        self.rebuild_calls += 1

    def stats(self):
        return {"revision": "r1"}


class _FakeTrace:
    def as_dict(self):
        return {"trace": True}


class _FakeRouteResult:
    def __init__(self):
        self.path = type("Path", (), {"value": "exact_local"})
        self.trace = _FakeTrace()


class _FakeRouter:
    def __init__(self):
        self.calls = []

    async def async_route(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeRouteResult()


class _FakeRuntime:
    def __init__(self):
        self.catalog_manager = _FakeCatalogManager()
        self.router = _FakeRouter()
        self.config = type("Cfg", (), {"language": "en"})


class _FakeServices:
    def __init__(self):
        self._handlers = {}

    def has_service(self, domain, service):
        return (domain, service) in self._handlers

    def async_register(self, domain, service, handler, **kwargs):
        self._handlers[(domain, service)] = handler

    def async_remove(self, domain, service):
        self._handlers.pop((domain, service), None)


class _FakeHass:
    def __init__(self):
        self.data = {DOMAIN: {"entry1": _FakeRuntime()}}
        self.services = _FakeServices()
        self.executor_jobs = []

    async def async_add_executor_job(self, func, *args):
        self.executor_jobs.append((func, args))
        return func(*args)


class _Call:
    def __init__(self, data=None):
        self.data = data or {}
        self.context = None


def test_services_handlers() -> None:
    hass = _FakeHass()
    asyncio.run(async_register_services(hass))

    rebuild = hass.services._handlers[(DOMAIN, SERVICE_REBUILD_CATALOG)]
    stats = hass.services._handlers[(DOMAIN, SERVICE_GET_CATALOG_STATS)]
    test_utterance = hass.services._handlers[(DOMAIN, SERVICE_TEST_UTTERANCE)]
    test_utterance_to_file = hass.services._handlers[(DOMAIN, SERVICE_TEST_UTTERANCE_TO_FILE)]

    asyncio.run(rebuild(_Call()))
    assert hass.data[DOMAIN]["entry1"].catalog_manager.rebuild_calls == 1

    stats_result = asyncio.run(stats(_Call()))
    assert "entries" in stats_result

    test_result = asyncio.run(test_utterance(_Call({ATTR_TEXT: "turn on kitchen light"})))
    assert test_result["entries"]["entry1"]["path"] == "exact_local"
    assert test_result["execute_branches"] is False
    router_call = hass.data[DOMAIN]["entry1"].router.calls[-1]
    assert router_call["debug_collect_all"] is True
    assert router_call["execute_debug_branches"] is False

    file_result = asyncio.run(
        test_utterance_to_file(
            _Call(
                {
                    ATTR_TEXT: "turn on kitchen light",
                    ATTR_PATH: "router_debug.json",
                    ATTR_EXECUTE_BRANCHES: True,
                }
            )
        )
    )
    assert file_result["path"] == "/config/router_debug.json"
    router_call = hass.data[DOMAIN]["entry1"].router.calls[-1]
    assert router_call["execute_debug_branches"] is True
