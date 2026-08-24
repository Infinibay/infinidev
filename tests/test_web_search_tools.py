"""Regression tests for web-search timeout and cache behavior."""

from __future__ import annotations

import json
import sys
import threading
import time
import types

import pytest

from infinidev.config.settings import settings
from infinidev.tools.web import backends
from infinidev.tools.web import code_search_web_tool as code_search_module
from infinidev.tools.web.code_search_web_tool import CodeSearchWebTool


@pytest.fixture(autouse=True)
def _clear_code_search_cache():
    code_search_module._cache.clear()
    yield
    code_search_module._cache.clear()


def test_search_ddg_timeout_returns_without_waiting_for_worker(monkeypatch):
    """The wall-clock deadline must not wait for a blocked DDGS worker."""
    release = threading.Event()

    class BlockingDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(self, _query, *, max_results):
            release.wait(timeout=1.0)
            return [{"title": "late", "href": "https://example.com", "body": "late"}]

    ddgs_module = types.ModuleType("ddgs")
    ddgs_module.DDGS = BlockingDDGS
    monkeypatch.setitem(sys.modules, "ddgs", ddgs_module)
    monkeypatch.setattr(backends.web_rate_limiter, "acquire", lambda: None)
    monkeypatch.setattr(settings, "WEB_TIMEOUT", 0.02)

    started = time.monotonic()
    try:
        result = backends.search_ddg("blocked request", num_results=1)
    finally:
        release.set()

    assert result == []
    assert time.monotonic() - started < 0.5


def test_search_ddg_caps_timed_out_workers_and_recovers(monkeypatch):
    release = threading.Event()
    workers_finished = threading.Event()
    counter_lock = threading.Lock()
    started = 0
    finished = 0

    class BlockingDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(self, _query, *, max_results):
            nonlocal started, finished
            with counter_lock:
                started += 1
            release.wait(timeout=1.0)
            with counter_lock:
                finished += 1
                if finished >= 2:
                    workers_finished.set()
            return [
                {
                    "title": f"result-{max_results}",
                    "href": "https://example.com",
                    "body": "example",
                }
            ]

    ddgs_module = types.ModuleType("ddgs")
    ddgs_module.DDGS = BlockingDDGS
    monkeypatch.setitem(sys.modules, "ddgs", ddgs_module)
    monkeypatch.setattr(backends.web_rate_limiter, "acquire", lambda: None)
    monkeypatch.setattr(settings, "WEB_TIMEOUT", 0.02)
    monkeypatch.setattr(
        backends,
        "_ddg_worker_slots",
        threading.BoundedSemaphore(2),
        raising=False,
    )
    monkeypatch.setattr(backends, "_DDG_MAX_IN_FLIGHT", 2, raising=False)

    try:
        assert backends.search_ddg("first blocked request", num_results=1) == []
        assert backends.search_ddg("second blocked request", num_results=1) == []
        assert backends.search_ddg("saturated request", num_results=1) == []
        with counter_lock:
            assert started == 2
    finally:
        release.set()

    assert workers_finished.wait(timeout=0.5)
    assert backends.search_ddg("recovered request", num_results=1) == [
        {
            "title": "result-1",
            "url": "https://example.com",
            "snippet": "example",
        }
    ]
    with counter_lock:
        assert started == 3


def test_code_search_cache_expires_after_ttl(monkeypatch):
    now = [100.0]
    calls = []

    def fake_search(query, *, num_results):
        calls.append((query, num_results))
        return [
            {
                "title": f"result-{len(calls)}",
                "url": "https://example.com",
                "snippet": "example",
            }
        ]

    monkeypatch.setattr(code_search_module, "search_ddg", fake_search)
    monkeypatch.setattr(code_search_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(settings, "WEB_CACHE_TTL_SECONDS", 60)
    tool = CodeSearchWebTool()

    first = tool._run("python cache")
    now[0] += 59
    assert tool._run("python cache") == first
    assert len(calls) == 1

    now[0] += 2
    refreshed = tool._run("python cache")
    assert refreshed != first
    assert len(calls) == 2


def test_code_search_cache_uses_lru_eviction(monkeypatch):
    calls = []

    def fake_search(query, *, num_results):
        calls.append((query, num_results))
        return [{"title": query, "url": "https://example.com", "snippet": "example"}]

    monkeypatch.setattr(code_search_module, "search_ddg", fake_search)
    monkeypatch.setattr(code_search_module, "_CACHE_MAX", 2)
    tool = CodeSearchWebTool()

    first = tool._run("first")
    tool._run("second")
    assert tool._run("first") == first  # Touch the oldest entry.
    tool._run("third")
    assert tool._run("first") == first
    assert len(calls) == 3

    tool._run("second")
    assert len(calls) == 4


def test_code_search_converts_backend_exception_to_tool_error(monkeypatch):
    def failing_search(_query, *, num_results):
        raise RuntimeError(f"network unavailable ({num_results})")

    monkeypatch.setattr(code_search_module, "search_ddg", failing_search)

    result = json.loads(CodeSearchWebTool()._run("python retries", num_results=3))

    assert result == {"error": "Code search failed: network unavailable (3)"}
