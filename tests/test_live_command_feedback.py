"""Regression tests for visible, bounded command execution feedback."""

from __future__ import annotations

import json
import threading
import time

from infinidev.engine.hooks.hooks import HookContext, HookEvent
from infinidev.engine.hooks.ui_hooks import _on_pre_tool
from infinidev.engine.tool_progress import tool_progress_context
from infinidev.flows.event_listeners import event_bus
from infinidev.tools.shell.execute_command import ExecuteCommandTool
from infinidev.tools.stdin_prompt import set_stdin_input_handler
from infinidev.ui.controls.chat_history import ChatHistoryControl
from infinidev.ui.event_handler import LIVE_TOOL_OUTPUT_LINES, process_event


class _Control:
    def __init__(self) -> None:
        self.work_label = ""
        self.invalidations = 0

    def invalidate_cache(self) -> None:
        self.invalidations += 1


class _App:
    def __init__(self) -> None:
        self._streaming_tool_name = None
        self._streaming_token_count = 0
        self._actions_text = ""
        self._chat_history_control = _Control()
        self.chat_messages: list[dict] = []
        self.invalidations = 0
        self.token_updates: list[dict] = []
        self.logs: list[str] = []

    def invalidate(self) -> None:
        self.invalidations += 1

    def update_context_tokens(self, **kwargs) -> None:
        self.token_updates.append(kwargs)

    def add_log(self, text: str) -> None:
        self.logs.append(text)


def _event_payload(run_id: str = "run-1") -> dict:
    return {
        "tool_run_id": run_id,
        "tool_name": "execute_command",
        "tool_detail": "python -m pytest",
        "tool_arguments": {"command": "python -m pytest", "timeout": 300},
    }


def _render(control: ChatHistoryControl, width: int = 90) -> str:
    control._line_cache = None
    control._last_rebuild = 0.0
    lines, _, _ = control._build_lines(width)
    return "\n".join("".join(text for _, text in line) for line in lines)


def test_event_lifecycle_keeps_one_row_and_only_the_latest_lines():
    app = _App()
    process_event(app, "loop_tool_start", _event_payload())

    assert len(app.chat_messages) == 1
    message = app.chat_messages[0]
    assert message["running"] is True
    assert message["text"] == "python -m pytest"
    assert "python -m pytest" in app._actions_text

    output = "".join(f"line-{index}\n" for index in range(25)) + "partial-progress"
    process_event(
        app,
        "loop_tool_output",
        {"tool_run_id": "run-1", "chunk": output},
    )

    assert len(message["live_output_tail"]) == LIVE_TOOL_OUTPUT_LINES
    assert message["live_output_tail"][0] == "line-5"
    assert message["live_output_tail"][-1] == "line-24"
    assert message["_live_output_partial"] == "partial-progress"

    result = json.dumps(
        {"exit_code": 0, "stdout": "done\n", "stderr": "", "success": True}
    )
    process_event(
        app,
        "loop_tool_call",
        {
            **_event_payload(),
            "tool_result_full": result,
            "tool_output_preview": "done",
            "tool_error": "",
            "exec_data": json.loads(result),
        },
    )

    assert len(app.chat_messages) == 1
    assert message["running"] is False
    assert message["result"] == result
    assert "live_output_tail" not in message
    assert "_live_output_partial" not in message


def test_running_command_is_visible_and_click_reveals_live_tail():
    message = {
        "type": "tool_call",
        "tool_name": "execute_command",
        "text": "python -m pytest",
        "args": {"command": "python -m pytest"},
        "result": "",
        "error": "",
        "running": True,
        "live_output_tail": ["test_a PASSED", "test_b PASSED"],
        "_live_output_partial": "tests/test_c.py::test_c",
    }
    control = ChatHistoryControl([message])

    collapsed_detail = _render(control)
    assert "python -m pytest" in collapsed_detail
    assert "test_a PASSED" not in collapsed_detail

    control._clickable_lines[1]()
    expanded_detail = _render(control)
    assert "● running" in expanded_detail
    assert "test_a PASSED" in expanded_detail
    assert "tests/test_c.py::test_c" in expanded_detail


def test_pre_tool_event_contains_full_arguments_and_run_id():
    seen: list[tuple[str, dict]] = []

    def subscriber(event_type, project_id, agent_id, data):
        seen.append((event_type, data))

    event_bus.subscribe(subscriber)
    try:
        _on_pre_tool(HookContext(
            event=HookEvent.PRE_TOOL,
            tool_name="execute_command",
            arguments={"command": "uv run pytest tests/test_tools_shell.py"},
            metadata={"tool_run_id": "run-42"},
            project_id=7,
            agent_id="agent-a",
        ))
    finally:
        event_bus.unsubscribe(subscriber)

    assert seen == [(
        "loop_tool_start",
        {
            "agent_id": "agent-a",
            "agent_name": "agent-a",
            "tool_run_id": "run-42",
            "tool_name": "execute_command",
            "tool_detail": "uv run pytest tests/test_tools_shell.py",
            "tool_arguments": {"command": "uv run pytest tests/test_tools_shell.py"},
            "call_num": 0,
            "total_calls": 0,
            "iteration": 0,
        },
    )]


def test_tui_subprocess_path_emits_live_output(
    bound_tool,
    auto_approve_permissions,
):
    seen: list[str] = []

    def subscriber(event_type, project_id, agent_id, data):
        if event_type == "loop_tool_output":
            seen.append(data["chunk"])

    event_bus.subscribe(subscriber)
    set_stdin_input_handler(lambda command, prompt, stdout, stderr: None)
    try:
        tool = bound_tool(ExecuteCommandTool)
        with tool_progress_context("run-shell", 1, "agent-a"):
            result = tool._run(command="printf 'first\\nsecond\\n'")
    finally:
        set_stdin_input_handler(None)
        event_bus.unsubscribe(subscriber)

    assert json.loads(result)["success"] is True
    assert "first\nsecond\n" in "".join(seen)


def test_cancel_event_interrupts_running_command(
    bound_tool,
    auto_approve_permissions,
):
    cancel_event = threading.Event()
    timer = threading.Timer(0.2, cancel_event.set)
    set_stdin_input_handler(lambda command, prompt, stdout, stderr: None)
    timer.start()
    started = time.monotonic()
    try:
        tool = bound_tool(ExecuteCommandTool)
        with tool_progress_context(
            "run-cancel", 1, "agent-a", cancel_event=cancel_event,
        ):
            result = tool._run(command="sleep 30", timeout=30)
    finally:
        timer.cancel()
        set_stdin_input_handler(None)

    data = json.loads(result)
    assert data["success"] is False
    assert data["killed_reason"] == "Command interrupted by user"
    assert time.monotonic() - started < 5


def test_work_label_change_restarts_elapsed_clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "infinidev.ui.controls.chat_history.time.monotonic",
        lambda: now[0],
    )
    control = ChatHistoryControl([])
    control.show_thinking = True

    now[0] = 22_000.0
    control.work_label = "Running execute_command: cargo test"

    assert control._work_started_at == 22_000.0
