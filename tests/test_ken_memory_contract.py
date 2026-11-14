"""Offline regressions for structured Ken memory and declared input validity."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from infinidev.engine.ken_client import KenClient
from infinidev.engine.mcp_client import McpTool
from infinidev.engine.task_runtime import TaskRuntime


class MemoryManager:
    """A schema-publishing transport with no daemon or embedding dependency."""

    def __init__(self, data, properties=None):
        self.data = data
        self.properties = properties or {}
        self.calls = []

    def all_names(self):
        return ["ken"]

    def list_tools(self, server=None):
        return [
            McpTool(
                server="ken", name=name, description="",
                input_schema={"properties": self.properties},
            )
            for name in ("ken_recall", "ken_remember", "ken_rank")
        ]

    def try_call(self, server, tool, arguments):
        self.calls.append((tool, deepcopy(arguments)))
        return SimpleNamespace(data=deepcopy(self.data))


@pytest.fixture
def memory_row():
    # Source-derived full-list contract agreed with the Ken agent, not a wire capture.
    return {
        "topic": "cache identity", "content": "Include rules in the cache key.",
        "score": 0.9, "tags": ["cache"],
        "justification": {
            "version": 1, "kind": "decision", "rationale": "Keep scope.",
            "evidence": [{"path": "src/cache.py", "note": "Observation"}],
            "dependencies": [{"kind": "file", "path": "src/cache.py", "sha256": "abc"}],
            "assumptions": ["Configuration matches."], "recheck": "Check again.",
            "check_runs": ["receipt-1"], "extension": {"nested": [1, 2]},
        },
        "validity": {
            "state": "unchanged", "issues": [],
            "scope": "declared dependencies only; not a truth or coverage check",
            "checks": [{"run_id": "receipt-1", "observations": {"extra": True}}],
        },
    }


@pytest.mark.parametrize("name", ["ken_recall", "ken_remember", "ken_rank"])
def test_facade_requests_native_json_only_when_schema_accepts_it(name):
    manager = MemoryManager([], {"output_format": {"type": "string", "enum": ["compact", "json"]}})
    client = KenClient(manager)
    client._call(name, {"query": "cache"})
    assert manager.calls == [(name, {"query": "cache", "output_format": "json"})]


@pytest.mark.parametrize("properties", [{}, {"output_format": {"type": "string", "enum": ["compact"]}}])
def test_facade_preserves_arguments_for_legacy_or_compact_only_schema(properties):
    manager = MemoryManager([], properties)
    client = KenClient(manager)
    client._call("ken_recall", {"query": "cache"})
    assert manager.calls == [("ken_recall", {"query": "cache"})]


@pytest.mark.parametrize("state", ["unchanged", "stale", "unknown", "untracked"])
def test_full_memory_retains_nested_metadata_and_renders_its_scope(memory_row, state):
    memory_row["validity"]["state"] = state
    client = KenClient(MemoryManager([memory_row]))
    hit = client.recall("cache")[0]
    assert hit.justification == memory_row["justification"]
    assert hit.validity == memory_row["validity"]
    rendered = hit.render()
    assert json.dumps(memory_row["justification"], ensure_ascii=False) in rendered
    assert json.dumps(memory_row["validity"], ensure_ascii=False) in rendered
    assert "does not certify" in rendered


def test_legacy_memory_does_not_acquire_fabricated_validity():
    client = KenClient(MemoryManager([{"topic": "old", "content": "Legacy note."}]))
    hit = client.recall("old")[0]
    assert hit.justification is None
    assert hit.validity is None
    assert "unchanged" not in hit.render()
    assert "not supplied" in hit.render()


def test_task_recall_preserves_metadata_and_full_content(memory_row, monkeypatch):
    memory_row["content"] = "An important conclusion. " * 30
    memory_row["validity"]["state"] = "stale"
    client = KenClient(MemoryManager([memory_row]))
    monkeypatch.setattr("infinidev.engine.ken_client.get_ken_client", lambda: client)
    runtime = TaskRuntime(persist_events=False)
    recalled = runtime.recall("cache")[0]["snippet"]
    assert memory_row["content"] in recalled
    assert json.dumps(memory_row["validity"], ensure_ascii=False) in recalled
    assert "Configuration matches." in recalled
    assert "Check again." in recalled
    assert any(recalled in entry.content for entry in runtime.state.memory)


@pytest.mark.parametrize("response, expected", [
    ({"ok": True, "topic": "cache"}, True),
    ({"ok": False, "error": "dependency changed"}, False),
    ({"topic": "cache"}, False),
    (None, False),
])
def test_remember_reports_only_explicit_success(response, expected):
    client = KenClient(MemoryManager(response))
    assert client.remember("cache", "Include rules.") is expected


def test_remember_forwards_explicit_justification_without_inventing_one(memory_row):
    manager = MemoryManager({"ok": True})
    client = KenClient(manager)
    assert client.remember("cache", "Include rules.", justification=memory_row["justification"])
    assert manager.calls[-1][1]["justification"] == memory_row["justification"]
    assert client.remember("cache", "Plain overwrite.")
    assert "justification" not in manager.calls[-1][1]
