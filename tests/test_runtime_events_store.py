"""Regression tests for the durable TaskRuntime event log."""

from __future__ import annotations

from infinidev.engine.runtime_events_store import list_events, store_event
from infinidev.engine.task_runtime import TaskRuntime


def test_list_events_limit_returns_most_recent_in_chronological_order(
    temp_db, monkeypatch
):
    from infinidev.engine import runtime_events_store

    timestamps = iter([
        "2026-01-01T00:00:01+00:00",
        "2026-01-01T00:00:02+00:00",
        "2026-01-01T00:00:03+00:00",
    ])
    monkeypatch.setattr(runtime_events_store, "_now", lambda: next(timestamps))
    store_event("session", "first")
    store_event("session", "second")
    store_event("session", "third")

    events = list_events("session", limit=2)

    assert [event["event"] for event in events] == ["second", "third"]


def test_task_event_persists_structured_payload_and_child_id(temp_db):
    runtime = TaskRuntime(task_id="session")
    task = runtime.add_task("Inspect")

    event = next(
        event
        for event in list_events("session")
        if event["event"] == "task_created"
    )

    assert event["task_id"] == task.id
    assert event["payload"]["task"]["id"] == task.id
    assert event["payload"]["task"]["title"] == "Inspect"
