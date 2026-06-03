"""Tests for the background command tools and their manager."""

import json
import time

import pytest

from infinidev.tools.shell.background_manager import (
    BackgroundTaskManager,
    acknowledge_completion,
    drain_completed_notifications,
    get_background_manager,
)
from infinidev.tools.shell.run_in_background import RunInBackgroundTool
from infinidev.tools.shell.background_status import BackgroundStatusTool
from infinidev.tools.shell.stop_background_task import StopBackgroundTaskTool
from infinidev.tools.shell.wait_for_background_task import WaitForBackgroundTaskTool


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


class TestBackgroundWait:
    """The blocking wait() primitive on BackgroundTask."""

    def test_wait_returns_exited_when_process_finishes(self, tmp_path):
        mgr = BackgroundTaskManager()
        task = mgr.start("sleep 0.2", "short sleep", str(tmp_path))
        assert task.wait(timeout=5) == "exited"
        assert not task.is_running

    def test_wait_times_out_on_long_command(self, tmp_path):
        mgr = BackgroundTaskManager()
        task = mgr.start("sleep 30", "long sleep", str(tmp_path))
        assert task.wait(timeout=0.3) == "timeout"
        assert task.is_running  # still running — wait must NOT stop it
        task.stop(force=True)

    def test_wait_matches_until_text(self, tmp_path):
        mgr = BackgroundTaskManager()
        # Emits the marker, then stays alive so the match (not exit) ends the wait.
        task = mgr.start("echo READY; sleep 30", "server", str(tmp_path))
        assert task.wait(timeout=5, until_text="READY") == "matched"
        assert task.is_running
        task.stop(force=True)

    def test_wait_until_text_falls_back_to_exit(self, tmp_path):
        mgr = BackgroundTaskManager()
        # Marker never appears; the wait should end on exit, not hang.
        task = mgr.start("echo nope", "quick", str(tmp_path))
        assert task.wait(timeout=5, until_text="NEVER") == "exited"


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

    def test_wait_tool_blocks_until_exit(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        run = bound_tool(RunInBackgroundTool)
        rid = json.loads(
            run._run(command="sleep 0.3", description="short sleep", cwd=str(tmp_path))
        )["id"]
        wait = bound_tool(WaitForBackgroundTaskTool)
        data = json.loads(wait._run(task_id=rid, timeout=5))
        assert data["wait_result"] == "exited"
        assert data["timed_out"] is False
        assert data["status"] == "exited"
        assert data["exit_code"] == 0

    def test_wait_tool_reports_timeout_without_stopping(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        run = bound_tool(RunInBackgroundTool)
        rid = json.loads(
            run._run(command="sleep 30", description="long sleep", cwd=str(tmp_path))
        )["id"]
        wait = bound_tool(WaitForBackgroundTaskTool)
        data = json.loads(wait._run(task_id=rid, timeout=1))
        assert data["timed_out"] is True
        assert data["status"] == "running"  # NOT stopped
        get_background_manager().stop(rid, force=True)

    def test_wait_tool_unknown_id_errors(self, bound_tool):
        wait = bound_tool(WaitForBackgroundTaskTool)
        data = json.loads(wait._run(task_id="bg-nope"))
        assert "error" in data

    def test_wait_tool_is_read_only(self):
        assert WaitForBackgroundTaskTool().is_read_only is True


class TestCompletionNotifications:
    """The pump-thread completion queue that surfaces finished tasks."""

    def _drain_clear(self):
        """Empty the global queue so each test starts from a clean slate."""
        drain_completed_notifications()

    def test_natural_completion_is_queued(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        self._drain_clear()
        run = bound_tool(RunInBackgroundTool)
        rid = json.loads(
            run._run(command="echo done", description="quick echo", cwd=str(tmp_path))
        )["id"]
        task = get_background_manager().get(rid)
        assert _wait_for(lambda: not task.is_running)
        # The pump runs on a daemon thread; give it a beat to queue.
        assert _wait_for(lambda: any(t.id == rid for t in drain_completed_notifications()))

    def test_agent_stopped_task_is_not_queued(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        self._drain_clear()
        run = bound_tool(RunInBackgroundTool)
        rid = json.loads(
            run._run(command="sleep 30", description="long sleep", cwd=str(tmp_path))
        )["id"]
        get_background_manager().stop(rid, force=True)
        task = get_background_manager().get(rid)
        assert _wait_for(lambda: not task.is_running)
        # Agent stopped it itself — it must NOT show up as a surprise completion.
        time.sleep(0.3)
        assert all(t.id != rid for t in drain_completed_notifications())

    def test_acknowledge_removes_from_queue(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        self._drain_clear()
        run = bound_tool(RunInBackgroundTool)
        rid = json.loads(
            run._run(command="echo hi", description="quick echo", cwd=str(tmp_path))
        )["id"]
        task = get_background_manager().get(rid)
        assert _wait_for(lambda: not task.is_running)
        assert _wait_for(lambda: _is_queued(rid))
        acknowledge_completion(rid)
        assert all(t.id != rid for t in drain_completed_notifications())

    def test_wait_tool_acknowledges_completion(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        self._drain_clear()
        run = bound_tool(RunInBackgroundTool)
        rid = json.loads(
            run._run(command="sleep 0.2", description="short sleep", cwd=str(tmp_path))
        )["id"]
        wait = bound_tool(WaitForBackgroundTaskTool)
        json.loads(wait._run(task_id=rid, timeout=5))
        # Having explicitly waited, the agent already knows — no surprise nudge.
        assert all(t.id != rid for t in drain_completed_notifications())


def _is_queued(task_id: str) -> bool:
    """Non-consuming check that ``task_id`` is in the completion queue."""
    from infinidev.tools.shell import background_manager as bm
    with bm._completions_lock:
        return task_id in bm._pending_completions
