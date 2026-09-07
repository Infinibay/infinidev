"""Background command aliases, selectable tasks, and live output tabs."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from infinidev.tools.shell.background_manager import BackgroundTaskManager


@pytest.fixture
def manager(monkeypatch):
    manager = BackgroundTaskManager()
    monkeypatch.setattr("infinidev.tools.shell.background_manager._manager", manager)
    yield manager
    manager.shutdown()


@pytest.mark.parametrize("command", ["/ps", "/bg", "/tasks"])
def test_background_command_aliases(command, monkeypatch):
    from infinidev.cli.commands import handle_command as classic_command
    from infinidev.ui.controls.autocomplete import AutocompleteState
    from infinidev.ui.handlers.commands import handle_command

    app = MagicMock()
    handle_command(app, command)
    app.dialog_manager.open_background_tasks.assert_called_once_with()
    handle_command(app, f"{command} bg-1")
    app.open_background_task_tab.assert_called_once_with("bg-1")
    render = MagicMock()
    monkeypatch.setattr("infinidev.cli.commands._render_background_tasks_classic", render)
    assert classic_command(f"{command} bg-1") is True
    render.assert_called_once_with("bg-1")
    autocomplete = AutocompleteState()
    autocomplete.update(command)
    assert command in [cmd for cmd, _ in autocomplete.matches]


def test_task_selector_opens_selected_task(manager, tmp_path):
    from infinidev.ui.dialogs.background_tasks_browser import BackgroundTasksControl

    first = manager.start("true", "first task", str(tmp_path))
    second = manager.start("true", "second task", str(tmp_path))
    opened = []
    control = BackgroundTasksControl(on_open=opened.append)
    control.create_content(80, 24)
    control.select_next()
    control.open_selected()
    assert opened == [second.id]
    control.select_prev()
    control.open_selected()
    assert opened[-1] == first.id


async def _until(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def test_combined_output_keeps_capture_order_and_is_bounded(manager, tmp_path, monkeypatch):
    monkeypatch.setattr("infinidev.tools.shell.background_manager._MAX_BUFFER_BYTES", 64)
    task = manager.start(
        "printf 'first\\n'; while [ ! -f continue ]; do sleep 0.02; done; "
        "printf 'second\\n' >&2; while [ ! -f finish ]; do sleep 0.02; done; "
        "printf '%080d\\n' 0", "ordered output", str(tmp_path),
    )
    await _until(lambda: "first" in task.combined_output()[0])
    (tmp_path / "continue").touch()
    await _until(lambda: "second" in task.combined_output()[0])
    text, discarded = task.combined_output()
    assert text == "first\nsecond\n"
    assert discarded == 0
    assert task.is_running
    (tmp_path / "finish").touch()
    await _until(lambda: not task._reader.is_alive())
    text, discarded = task.combined_output()
    assert len(text.encode()) == 64
    assert discarded > 0


def test_output_control_preserves_scrollback_and_resumes_tail():
    from infinidev.ui.controls.background_output import BackgroundTaskOutputControl

    task = MagicMock()
    task.combined_output.return_value = ("\n".join(f"line {i}" for i in range(30)), 0)
    control = BackgroundTaskOutputControl(task)
    control.create_content(80, 5)
    control.page_up(10)
    before = control.create_content(80, 5).cursor_position.y
    task.combined_output.return_value = (task.combined_output()[0] + "\nnew line", 0)
    content = control.create_content(80, 5)
    assert content.cursor_position.y == before
    control.scroll_end()
    content = control.create_content(80, 5)
    assert content.cursor_position.y == content.line_count - 1
    assert control._follow_tail


async def test_live_tab_opens_refreshes_and_closes_without_stopping_task(
    manager, tmp_path, monkeypatch,
):
    from infinidev.ui.app import InfinidevApp

    monkeypatch.setattr(InfinidevApp, "_start_background_index", lambda self: None)
    monkeypatch.setattr(InfinidevApp, "_persist_session_message", lambda *args: None)
    monkeypatch.setattr(InfinidevApp, "_persist_runtime_state", lambda *args: None)
    task = manager.start(
        "printf 'ready-live\\n'; while [ ! -f continue ]; do sleep 0.02; done; "
        "printf 'new-output-live\\n' >&2; while [ ! -f finish ]; do sleep 0.02; done; exit 7",
        "live demo", str(tmp_path),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        app = InfinidevApp()
        running = asyncio.create_task(app.app.run_async())
        try:
            await _until(lambda: app.app.is_running)
            pipe.send_text("/ps\r")
            await _until(lambda: app.active_dialog == "background_tasks")
            pipe.send_text("\r")
            tab_id = f"background:{task.id}"
            await _until(lambda: app.active_tab == tab_id)
            assert app.active_dialog is None
            control = app._background_tab_controls[tab_id]
            assert app.app.layout.current_control is control
            assert not app._engine_running
            await _until(lambda: "ready-live" in _rendered_text(app))
            (tmp_path / "continue").touch()
            await _until(lambda: "new-output-live" in _rendered_text(app))
            pipe.send_text("\x1bOQ")  # F2 returns to chat while keeping the log tab.
            await _until(lambda: app.active_tab == "chat")
            assert tab_id in app._background_tab_controls
            app.open_background_task_tab(task.id)
            assert len(app._background_tab_controls) == 1
            pipe.send_text("\x17")  # Ctrl+W closes the view, not the process.
            await _until(lambda: app.active_tab == "chat")
            assert task.is_running
            assert not app._background_tab_controls
            pipe.send_text(f"/bg {task.id}\r")
            await _until(lambda: app.active_tab == tab_id)
            (tmp_path / "finish").touch()
            await _until(lambda: "FAILED (exit 7)" in _rendered_text(app))
            app.open_background_task_tab("bg-missing")
            assert app.active_tab == tab_id
        finally:
            if app.app.is_running:
                app.app.exit()
            await asyncio.wait_for(running, 5)


def _rendered_text(app) -> str:
    screen = app.app.renderer.last_rendered_screen
    if screen is None:
        return ""
    return "\n".join(
        "".join(row[x].char for x in sorted(row))
        for _, row in sorted(screen.data_buffer.items())
    )
