"""Tests for the MCP runtime bridge."""

from __future__ import annotations

from infinidev.engine.mcp_runtime_bridge import McpRuntimeBridge
from infinidev.engine.runtime_state import TaskStatus
from infinidev.engine.task_runtime import TaskRuntime


def test_bridge_records_started_event_in_memory():
    runtime = TaskRuntime()
    bridge = McpRuntimeBridge(runtime)
    bridge({"event": "started", "server": "ken", "pid": 1234})
    assert any("started" in entry.content for entry in runtime.state.memory)


def test_bridge_records_tool_call_and_result():
    runtime = TaskRuntime()
    bridge = McpRuntimeBridge(runtime)
    bridge({"event": "tool_call", "server": "ken", "tool": "search"})
    bridge({"event": "tool_result", "server": "ken", "tool": "search"})
    kinds = [entry.kind for entry in runtime.state.memory]
    assert "mcp_call" in kinds


def test_bridge_blocks_current_task_on_failure():
    events: list[dict] = []
    runtime = TaskRuntime(on_event=events.append, persist_events=False)
    task = runtime.add_task("Work")
    runtime.start_next_task()
    bridge = McpRuntimeBridge(runtime)
    bridge({"event": "failure", "server": "ken", "error": "down", "count": 1})

    assert task.status == TaskStatus.BLOCKED
    assert task.result == "ken failure (1): down"
    assert runtime.state.current_task_id is None
    assert any(event["event"] == "task_blocked" for event in events)
