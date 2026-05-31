"""Tests for the background command tools and their manager."""

import json
import time

import pytest

from infinidev.tools.shell.background_manager import (
    BackgroundTaskManager,
    get_background_manager,
)
from infinidev.tools.shell.run_in_background import RunInBackgroundTool
from infinidev.tools.shell.background_status import BackgroundStatusTool
from infinidev.tools.shell.stop_background_task import StopBackgroundTaskTool


def _wait_for(predicate, timeout=5.0, interval=0.05):
    """Poll ``predicate`` until it returns truthy or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestBackgroundManager:
    """Direct tests of the registry, independent of the tool layer."""

    def test_start_captures_stdout_and_exit(self, tmp_path):
        mgr = BackgroundTaskManager()
        task = mgr.start("echo hello-bg", "echo test", str(tmp_path))
        assert _wait_for(lambda: not task.is_running)
        out, _ = task.output()
        assert "hello-bg" in out
        assert task.status == "exited"
        assert task.exit_code == 0

    def test_runtime_is_positive(self, tmp_path):
        mgr = BackgroundTaskManager()
        task = mgr.start("sleep 0.2", "short sleep", str(tmp_path))
        assert task.runtime_seconds() >= 0.0
        assert _wait_for(lambda: not task.is_running)
        assert task.runtime_seconds() >= 0.2

    def test_stop_kills_running_task(self, tmp_path):
        mgr = BackgroundTaskManager()
        task = mgr.start("sleep 30", "long sleep", str(tmp_path))
        assert task.is_running
        assert task.stop(force=True) is True
        assert _wait_for(lambda: not task.is_running)
        assert task.status == "killed"

    def test_failed_command_reports_nonzero_exit(self, tmp_path):
        mgr = BackgroundTaskManager()
        task = mgr.start("exit 3", "failing cmd", str(tmp_path))
        assert _wait_for(lambda: not task.is_running)
        assert task.exit_code == 3
        assert task.status == "exited"


class TestBackgroundTools:
    """Tests through the tool interface the LLM actually calls."""

    def test_run_then_status_then_stop(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        run = bound_tool(RunInBackgroundTool)
        result = json.loads(
            run._run(command="sleep 30", description="long sleep", cwd=str(tmp_path))
        )
        task_id = result["id"]
        assert task_id.startswith("bg-")
        assert result["status"] == "running"

        status = bound_tool(BackgroundStatusTool)
        sdata = json.loads(status._run(task_id=task_id))
        assert sdata["status"] == "running"
        assert "runtime_seconds" in sdata

        stop = bound_tool(StopBackgroundTaskTool)
        kdata = json.loads(stop._run(task_id=task_id, force=True))
        assert kdata["status"] == "killed"

    def test_status_lists_all_tasks(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        run = bound_tool(RunInBackgroundTool)
        run._run(command="echo a", description="task a", cwd=str(tmp_path))
        status = bound_tool(BackgroundStatusTool)
        listing = json.loads(status._run())
        assert "tasks" in listing
        assert any(t["description"] == "task a" for t in listing["tasks"])

    def test_status_unknown_id_errors(self, bound_tool):
        status = bound_tool(BackgroundStatusTool)
        data = json.loads(status._run(task_id="bg-does-not-exist"))
        assert "error" in data

    def test_run_empty_command_errors(self, bound_tool, auto_approve_permissions):
        run = bound_tool(RunInBackgroundTool)
        data = json.loads(run._run(command="   ", description="nothing"))
        assert "error" in data

    def test_run_captures_output_via_status(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        run = bound_tool(RunInBackgroundTool)
        result = json.loads(
            run._run(command="echo captured-line", description="echo", cwd=str(tmp_path))
        )
        task_id = result["id"]
        manager = get_background_manager()
        task = manager.get(task_id)
        assert _wait_for(lambda: not task.is_running)
        status = bound_tool(BackgroundStatusTool)
        sdata = json.loads(status._run(task_id=task_id))
        assert "captured-line" in sdata["stdout"]

    def test_status_tool_is_read_only(self):
        assert BackgroundStatusTool().is_read_only is True

    def test_run_tool_is_not_read_only(self):
        assert RunInBackgroundTool().is_read_only is False
