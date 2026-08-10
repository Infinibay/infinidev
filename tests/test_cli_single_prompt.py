"""Regression coverage for the non-interactive single-prompt CLI path."""

from __future__ import annotations

import logging

import pytest

from infinidev.engine import orchestration
from infinidev.flows.event_listeners import event_bus


@pytest.fixture
def cli_main():
    """Import the CLI without leaking its process-wide logging setup."""
    root = logging.getLogger()
    infinidev_logger = logging.getLogger("infinidev")
    root_level = root.level
    root_handlers = list(root.handlers)
    infinidev_level = infinidev_logger.level
    try:
        from infinidev.cli import main as module

        yield module
    finally:
        root.handlers[:] = root_handlers
        root.setLevel(root_level)
        infinidev_logger.setLevel(infinidev_level)


def test_single_prompt_subscribes_live_classic_renderer(
    cli_main, monkeypatch, capsys
):
    """Tool starts must be visible while a one-shot prompt is running."""
    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", lambda: None)
    monkeypatch.setattr(cli_main, "_end_ken_sessions", lambda: None)
    monkeypatch.setattr(cli_main, "InfinidevAgent", lambda agent_id: object())
    monkeypatch.setattr(cli_main, "LoopEngine", lambda: object())
    monkeypatch.setattr(cli_main, "ReviewEngine", lambda: object())

    from infinidev.cli import session_resume
    from infinidev.tools import permission

    monkeypatch.setattr(session_resume, "begin_fresh_session", lambda _session_id: None)
    monkeypatch.setattr(permission, "set_permission_handler", lambda _handler: None)
    monkeypatch.setattr(
        permission,
        "make_noninteractive_permission_handler",
        lambda _prompt: lambda *_args, **_kwargs: True,
    )

    def fake_run_task(**_kwargs):
        event_bus.emit("loop_tool_start", 1, "cli_agent", {
            "tool_name": "execute_command",
            "tool_detail": "uv run pytest",
            "call_num": 1,
            "total_calls": 1,
        })
        return "finished"

    monkeypatch.setattr(orchestration, "run_task", fake_run_task)

    cli_main._run_single_prompt("Fix it")

    out = capsys.readouterr().out
    assert "running" in out
    assert "execute_command" in out
    assert "uv run pytest" in out
    assert "finished" in out

    event_bus.emit("loop_tool_start", 1, "cli_agent", {
        "tool_name": "read_file",
        "tool_detail": "should-not-render",
    })
    assert capsys.readouterr().out == ""
