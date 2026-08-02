"""Foreground tool cancellation keeps the surrounding agent task alive."""

from __future__ import annotations

import json
from unittest.mock import Mock

from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.tool_dispatch import execute_tool_call
from infinidev.ui.app import InfinidevApp


class _CooperativeTool:
    name = "cooperative_tool"

    def _run(self) -> str:
        return json.dumps({"partial_output": "work before cancellation"})


class _Autocomplete:
    visible = False


class _EngineStub:
    def __init__(self, *, tool_active: bool) -> None:
        self.tool_active = tool_active
        self.tool_cancel_calls = 0
        self.task_cancel_calls = 0

    def cancel_active_tool(self) -> bool:
        self.tool_cancel_calls += 1
        return self.tool_active

    def cancel(self) -> None:
        self.task_cancel_calls += 1


def _bare_app(engine: _EngineStub) -> InfinidevApp:
    app = object.__new__(InfinidevApp)
    app.active_dialog = None
    app._autocomplete = _Autocomplete()
    app._engine_running = True
    app.engine = engine
    app._cancel_hold_start = None
    app._cancel_last_escape = 0.0
    app._cancel_watcher_active = True
    app._update_cancel_bar = Mock()
    app.flash_status = Mock()
    app.invalidate = Mock()
    return app


def test_engine_tool_cancel_does_not_cancel_the_task() -> None:
    engine = LoopEngine()

    assert engine.cancel_active_tool() is False
    engine._begin_tool_batch()
    try:
        assert engine.has_active_tool is True
        assert engine.cancel_active_tool() is True
        assert engine._tool_cancel_event.is_set()
        assert engine.is_cancelled is False
    finally:
        engine._finish_tool_batch()

    assert engine.has_active_tool is False
    assert engine._tool_cancel_event.is_set() is False


def test_cancelled_tool_result_explicitly_notifies_the_agent() -> None:
    engine = LoopEngine()
    engine._begin_tool_batch()
    engine._tool_cancel_event.set()
    try:
        result = execute_tool_call(
            {_CooperativeTool.name: _CooperativeTool()},
            _CooperativeTool.name,
            {},
            hook_metadata={"cancel_event": engine._tool_cancel_event},
        )
    finally:
        engine._finish_tool_batch()

    payload = json.loads(result)
    assert payload["cancelled_by_user"] is True
    assert "user stopped this tool" in payload["error"].lower()


def test_double_escape_stops_tool_without_stopping_task(monkeypatch) -> None:
    engine = _EngineStub(tool_active=True)
    app = _bare_app(engine)
    clock = iter((100.0, 100.2))
    monkeypatch.setattr("infinidev.ui.app.time.monotonic", lambda: next(clock))

    app.handle_escape()
    app.handle_escape()

    assert engine.tool_cancel_calls == 1
    assert engine.task_cancel_calls == 0
    assert app._cancel_hold_start is None
    app.flash_status.assert_called_once_with(
        "Stopping current tool — the agent will be notified"
    )


def test_double_escape_without_active_tool_keeps_hold_cancel_available(monkeypatch) -> None:
    engine = _EngineStub(tool_active=False)
    app = _bare_app(engine)
    clock = iter((100.0, 100.2))
    monkeypatch.setattr("infinidev.ui.app.time.monotonic", lambda: next(clock))

    app.handle_escape()
    app.handle_escape()

    assert engine.tool_cancel_calls == 1
    assert engine.task_cancel_calls == 0
    assert app._cancel_hold_start == 100.0
