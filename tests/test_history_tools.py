"""Tests for the history_search / history_read / history_trace tools."""

from __future__ import annotations

import json

import pytest

from infinidev.engine.history import events as ev
from infinidev.engine.history import store
from infinidev.tools.history import (
    HistoryReadTool,
    HistorySearchTool,
    HistoryTraceTool,
)


@pytest.fixture
def seeded_run(temp_db, tool_context):
    """One run with a small causal chain, bound to session 'test-session'."""
    run_id = store.create_run(
        session_id="test-session", engine="staged", mode="auto",
        goal_title="Add JWT", goal_request="Add JWT to all endpoints",
    )
    e1 = store.append_event(run_id, "test-session", ev.RUN_STARTED,
                            {"mode": "auto", "engine": "staged"})
    e2 = store.append_event(run_id, "test-session", ev.ENGINE_SELECTED,
                            {"engine": "staged", "confidence": 0.9},
                            parent_event_id=e1)
    e3 = store.append_event(run_id, "test-session", ev.TASK_CLOSED,
                            {"title": "Add JWT middleware", "status": "completed"},
                            node_id="task-1", parent_event_id=e2)
    store.append_event(run_id, "test-session", ev.DIGEST_CREATED,
                       {"status": "completed"},
                       visibility=ev.VISIBILITY_ARCHIVE_ONLY, parent_event_id=e3)
    store.finish_run(run_id, "completed", digest={"status": "completed"})
    return run_id, {"run_started": e1, "engine_selected": e2, "task_closed": e3}


def _payload(result: str) -> dict:
    return json.loads(result)


class TestHistorySearch:
    def test_full_text_hit(self, bound_tool, seeded_run):
        run_id, ids = seeded_run
        tool = bound_tool(HistorySearchTool)
        result = _payload(tool._run(query="JWT middleware"))
        assert result["count"] == 1
        hit = result["results"][0]
        assert hit["event_id"] == ids["task_closed"]
        assert hit["node_id"] == "task-1"
        assert "payload_snippet" in hit

    def test_event_type_filter(self, bound_tool, seeded_run):
        tool = bound_tool(HistorySearchTool)
        result = _payload(tool._run(event_type=ev.ENGINE_SELECTED))
        assert result["count"] == 1
        assert result["results"][0]["event_type"] == ev.ENGINE_SELECTED

    def test_archive_hidden_by_default(self, bound_tool, seeded_run):
        tool = bound_tool(HistorySearchTool)
        hidden = _payload(tool._run(event_type=ev.DIGEST_CREATED))
        assert hidden["count"] == 0
        shown = _payload(tool._run(event_type=ev.DIGEST_CREATED,
                                   include_archive=True))
        assert shown["count"] == 1

    def test_context_window(self, bound_tool, seeded_run):
        _, ids = seeded_run
        tool = bound_tool(HistorySearchTool)
        result = _payload(tool._run(query="JWT middleware", context_window=1))
        context_ids = {c["event_id"] for c in result["results"][0]["context"]}
        assert ids["engine_selected"] in context_ids


class TestHistoryRead:
    def test_read_by_ids_with_window(self, bound_tool, seeded_run):
        _, ids = seeded_run
        tool = bound_tool(HistoryReadTool)
        result = _payload(tool._run(event_ids=[ids["task_closed"]],
                                    window_before=1))
        returned_ids = {e["event_id"] for e in result["events"]}
        assert ids["task_closed"] in returned_ids
        assert ids["engine_selected"] in returned_ids
        target = next(e for e in result["events"]
                      if e["event_id"] == ids["task_closed"])
        assert target["payload"]["title"] == "Add JWT middleware"

    def test_read_run_timeline(self, bound_tool, seeded_run):
        run_id, _ = seeded_run
        tool = bound_tool(HistoryReadTool)
        result = _payload(tool._run(run_id=run_id))
        assert result["count"] >= 3
        assert all(e["run_id"] == run_id for e in result["events"])

    def test_requires_target(self, bound_tool, seeded_run):
        tool = bound_tool(HistoryReadTool)
        result = tool._run()
        assert "error" in result.lower() or "Provide" in result


class TestHistoryTrace:
    def test_trace_chain_from_leaf(self, bound_tool, seeded_run):
        _, ids = seeded_run
        tool = bound_tool(HistoryTraceTool)
        result = _payload(tool._run(event_id=ids["task_closed"]))
        chain_ids = [c["event_id"] for c in result["chain"]]
        # Root-first, archive_only digest excluded.
        assert chain_ids == [ids["run_started"], ids["engine_selected"],
                             ids["task_closed"]]

    def test_trace_run_timeline(self, bound_tool, seeded_run):
        run_id, _ = seeded_run
        tool = bound_tool(HistoryTraceTool)
        result = _payload(tool._run(run_id=run_id))
        assert result["engine"] == "staged"
        assert result["status"] == "completed"
        assert result["count"] >= 3

    def test_trace_missing_event(self, bound_tool, seeded_run):
        tool = bound_tool(HistoryTraceTool)
        result = _payload(tool._run(event_id="evt_doesnotexist"))
        assert result["count"] == 0
