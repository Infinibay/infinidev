"""Tests for the background-tasks explorer (TUI dialog + classic /tasks).

These cover the wiring added for the Ctrl+B / /tasks feature: the status-line
verdict shared across renderers, the TUI control's rendering, the classic
plain-text report, and command/keybinding registration.
"""

import time

from infinidev.tools.shell.background_manager import (
    BackgroundTaskManager,
    get_background_manager,
)


def _wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestStatusLine:
    def test_running_task(self, tmp_path):
        mgr = BackgroundTaskManager()
        task = mgr.start("sleep 30", "long sleep", str(tmp_path))
        assert task.status_line().startswith("running for")
        task.stop(force=True)

    def test_exited_ok(self, tmp_path):
        mgr = BackgroundTaskManager()
        task = mgr.start("true", "noop", str(tmp_path))
        assert _wait_for(lambda: not task.is_running)
        assert "exited ok" in task.status_line()

    def test_failed_task(self, tmp_path):
        mgr = BackgroundTaskManager()
        task = mgr.start("exit 7", "failing", str(tmp_path))
        assert _wait_for(lambda: not task.is_running)
        assert "FAILED (exit 7)" in task.status_line()


class TestTuiControl:
    def test_renders_empty_state(self):
        from infinidev.ui.dialogs.background_tasks_browser import (
            BackgroundTasksControl,
        )
        # No reliance on global state being empty — just that it never raises
        # and produces some content lines.
        content = BackgroundTasksControl().create_content(80, 24)
        assert content.line_count >= 1

    def test_renders_task_with_output(self, tmp_path):
        from infinidev.ui.dialogs.background_tasks_browser import (
            BackgroundTasksControl,
        )
        mgr = get_background_manager()  # control reads the singleton
        task = mgr.start("echo marker-line-xyz", "echo demo", str(tmp_path))
        assert _wait_for(lambda: not task.is_running)
        content = BackgroundTasksControl().create_content(80, 24)
        rendered = "\n".join(
            "".join(frag[1] for frag in content.get_line(i))
            for i in range(content.line_count)
        )
        assert task.id in rendered
        assert "echo demo" in rendered
        assert "marker-line-xyz" in rendered

    def test_scroll_clamps(self):
        from infinidev.ui.dialogs.background_tasks_browser import (
            BackgroundTasksControl,
        )
        ctrl = BackgroundTasksControl()
        ctrl.create_content(80, 24)
        ctrl.scroll_up()  # already at top
        assert ctrl._scroll == 0


class TestClassicRenderer:
    def test_list_and_detail(self, tmp_path, capsys):
        from infinidev.cli.commands import _render_background_tasks_classic
        mgr = get_background_manager()
        task = mgr.start("echo classic-xyz", "classic demo", str(tmp_path))
        assert _wait_for(lambda: not task.is_running)

        _render_background_tasks_classic()
        listing = capsys.readouterr().out
        assert task.id in listing
        assert "classic demo" in listing

        _render_background_tasks_classic(task.id)
        detail = capsys.readouterr().out
        assert "classic-xyz" in detail

    def test_unknown_id(self, capsys):
        from infinidev.cli.commands import _render_background_tasks_classic
        _render_background_tasks_classic("bg-nope-9999")
        assert "No background task" in capsys.readouterr().out


class TestRegistration:
    def test_tasks_in_tui_command_table(self):
        from infinidev.ui.handlers.commands import _COMMAND_TABLE
        assert "/tasks" in _COMMAND_TABLE

    def test_ctrl_b_footer_hint(self):
        from infinidev.ui.keybindings import FOOTER_HINTS
        assert any(key == "Ctrl+B" for key, _label, _ctx in FOOTER_HINTS)
