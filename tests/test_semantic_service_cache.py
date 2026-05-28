"""Tests for semantic service document vector caching."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load_server_module():
    module_name = "test_semantic_service_server"
    if module_name in sys.modules:
        return sys.modules[module_name]

    fake_fastapi = types.ModuleType("fastapi")

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def post(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    fake_fastapi.FastAPI = _FastAPI
    sys.modules["fastapi"] = fake_fastapi

    fake_fastembed = types.ModuleType("fastembed")

    class _TextEmbedding:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, texts):
            for text in texts:
                yield [float(len(text))]

    fake_fastembed.TextEmbedding = _TextEmbedding
    sys.modules["fastembed"] = fake_fastembed

    fake_pydantic = types.ModuleType("pydantic")
    fake_pydantic.BaseModel = object

    def _field(*, default_factory=None, default=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    fake_pydantic.Field = _field
    sys.modules["pydantic"] = fake_pydantic

    server_path = (
        Path(__file__).resolve().parents[1]
        / "addon"
        / "catalog-router-semantic"
        / "app"
        / "server.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, server_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_semantic_service_only_keeps_one_doc_vector_cache() -> None:
    module = _load_server_module()
    service = module.SemanticService()

    first_docs = ["turn on kitchen light"]
    second_docs = ["what is office fan"]

    first_key = service._doc_cache_key(cache_namespace="phrase", docs=first_docs)  # noqa: SLF001
    second_key = service._doc_cache_key(cache_namespace="entity", docs=second_docs)  # noqa: SLF001

    service._get_cached_doc_vectors(cache_namespace="phrase", docs=first_docs)  # noqa: SLF001
    assert list(service._doc_vector_caches) == [first_key]  # noqa: SLF001

    service._get_cached_doc_vectors(cache_namespace="entity", docs=second_docs)  # noqa: SLF001
    assert list(service._doc_vector_caches) == [second_key]  # noqa: SLF001
