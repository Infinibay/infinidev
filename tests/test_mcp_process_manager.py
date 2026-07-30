"""MCP lifecycle as the *user* sees it: the /mcp panel and event persistence."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from infinidev.engine import mcp_client as mcp_module
from infinidev.engine.mcp_client import McpManager, reset_default_mcp_manager
from infinidev.engine.task_runtime import TaskRuntime

FAKE_SERVER = str(Path(__file__).parent / "mcp_fake_server.py")


def server_config(*flags: str, **extra) -> dict:
    return {"command": sys.executable, "args": [FAKE_SERVER, *flags], **extra}


@pytest.fixture(autouse=True)
def _reset():
    reset_default_mcp_manager()
    yield
    reset_default_mcp_manager()


class _FakeApp:
    """Minimal stand-in for InfinidevApp: records what reached the chat."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str]] = []

    def add_message(self, sender: str, text: str, kind: str = "agent") -> None:
        self.messages.append((sender, text, kind))

    @property
    def last(self) -> str:
        return self.messages[-1][1] if self.messages else ""


def _run_mcp_command(app, *args):
    from infinidev.ui.handlers.commands import _cmd_mcp

    _cmd_mcp(app, ["/mcp", *args])


# ── /mcp panel ────────────────────────────────────────────────────────────


def test_mcp_panel_reports_a_reachable_server(monkeypatch):
    monkeypatch.setattr(
        mcp_module, "resolve_mcp_servers", lambda: {"ken": server_config()}
    )
    manager = mcp_module.get_default_mcp_manager()
    manager.get("ken").list_tools()  # bring it up

    app = _FakeApp()
    _run_mcp_command(app)
    assert "ken:" in app.last
    assert "running" in app.last
    assert "6 tools" in app.last


def test_mcp_panel_explains_an_unreachable_server(monkeypatch):
    monkeypatch.setattr(
        mcp_module,
        "resolve_mcp_servers",
        lambda: {"ken": {"command": "definitely-missing-binary"}},
    )
    app = _FakeApp()
    _run_mcp_command(app)
    assert "unavailable" in app.last
    assert "not found on PATH" in app.last


def test_mcp_panel_surfaces_startup_stderr(monkeypatch):
    monkeypatch.setattr(
        mcp_module,
        "resolve_mcp_servers",
        lambda: {"ken": server_config("--no-handshake", "--noise")},
    )
    manager = mcp_module.get_default_mcp_manager()
    with pytest.raises(Exception):
        manager.get("ken").list_tools()

    app = _FakeApp()
    _run_mcp_command(app)
    assert "initialize refused" in app.last


def test_mcp_restart_reports_success(monkeypatch):
    monkeypatch.setattr(
        mcp_module, "resolve_mcp_servers", lambda: {"ken": server_config()}
    )
    app = _FakeApp()
    _run_mcp_command(app, "restart", "ken")
    assert "restarted" in app.last


def test_mcp_stop_reports_unknown_server(monkeypatch):
    monkeypatch.setattr(mcp_module, "resolve_mcp_servers", lambda: {})
    app = _FakeApp()
    _run_mcp_command(app, "stop", "nope")
    assert "not found" in app.last


def test_mcp_action_without_a_name_shows_usage(monkeypatch):
    monkeypatch.setattr(mcp_module, "resolve_mcp_servers", lambda: {})
    app = _FakeApp()
    _run_mcp_command(app, "restart")
    assert "Usage:" in app.last


# ── manager surface ───────────────────────────────────────────────────────


def test_manager_restart_replaces_the_process():
    manager = McpManager({"ken": server_config()})
    assert manager.start("ken") is True
    first_pid = manager.get("ken")._process.pid
    assert manager.restart("ken") is True
    assert manager.get("ken")._process.pid != first_pid
    manager.close()


def test_close_terminates_every_server():
    manager = McpManager({"a": server_config(), "b": server_config()})
    manager.start("a")
    manager.start("b")
    manager.close()
    assert manager.all_names() == []


# ── runtime event persistence ─────────────────────────────────────────────


def test_runtime_persists_events_to_db(tmp_path, monkeypatch):
    from infinidev.code_intel import _db as ci_db
    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "DB_PATH", str(tmp_path / "events.db"))
    ci_db._conn_cache.__dict__.clear()

    runtime = TaskRuntime(task_id="session-1")
    runtime.add_task("Inspect")
    runtime.start_next_task()
    runtime.complete_current_task("done")

    from infinidev.engine.runtime_events_store import list_events

    kinds = [event["event"] for event in list_events("session-1")]
    assert "task_started" in kinds
    assert "task_completed" in kinds
