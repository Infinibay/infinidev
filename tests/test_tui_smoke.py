"""End-to-end boot test: the real Application, driven by real keystrokes.

Every other UI test renders containers directly. This one runs the actual
prompt_toolkit event loop, so it catches the class of breakage rendering
tests cannot: a key binding that raises on construction, a layout whose
focus target does not exist, a startup thread that deadlocks the loop.
"""

from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from infinidev.config.settings import settings


@pytest.fixture(autouse=True)
def _isolated_ui_settings(tmp_path, monkeypatch):
    """Keep panel persistence from leaking across tests or into the real home."""
    original_sidebar = settings.UI_SIDEBAR_VISIBLE
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    settings.UI_SIDEBAR_VISIBLE = False
    yield
    settings.UI_SIDEBAR_VISIBLE = original_sidebar


class _SizedOutput(DummyOutput):
    def get_size(self) -> Size:
        return Size(rows=30, columns=100)


def _run(keys: str, timeout: float = 25.0):
    """Boot the app, feed *keys*, return it once the loop exits."""

    async def _main():
        with create_pipe_input() as pipe, create_app_session(
            input=pipe, output=_SizedOutput()
        ):
            from infinidev.ui.app import InfinidevApp

            app = InfinidevApp()
            pipe.send_text(keys)
            await asyncio.wait_for(app.app.run_async(), timeout=timeout)
            return app

    return asyncio.run(_main())


def _texts(app) -> str:
    return "\n".join(str(m.get("text") or "") for m in app.chat_messages)


def test_app_boots_and_exits_cleanly():
    app = _run("/exit\r")
    assert app.chat_messages, "the banner must be on screen"
    assert app.chat_messages[0]["type"] == "banner"


def test_question_mark_shows_help_in_the_running_app():
    app = _run("?/exit\r")
    body = _texts(app)
    assert "PANELS" in body and "COMMANDS" in body


def test_sidebar_toggles_in_the_running_app():
    # Alt+. arrives as ESC then '.'
    app = _run("\x1b./exit\r")
    assert app.sidebar_visible is True


def test_startup_does_not_announce_indexing():
    """Ken owns the index; the transcript must not narrate it."""
    app = _run("/exit\r")
    body = _texts(app).lower()
    assert "indexing workspace" not in body
    assert "index ready" not in body


def test_mcp_panel_opens_from_the_running_app():
    app = _run("/mcp\r/exit\r")
    assert "MCP servers" in _texts(app) or "No MCP servers" in _texts(app)
