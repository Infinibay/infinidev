"""Tests for the task runtime and its orchestration integration."""

from infinidev.engine.runtime_state import TaskStatus
from infinidev.engine.task_runtime import TaskRuntime


def test_runtime_tracks_dependencies_and_compacts_only_memory() -> None:
    events: list[dict] = []
    runtime = TaskRuntime(task_id="session", on_event=events.append)
    first = runtime.add_task("Inspect")
    second = runtime.add_task("Implement", depends_on=[first.id])
    runtime.append_chat("user", "Do the work")
    runtime.remember("old detail", importance=0.1)
    active = runtime.start_next_task()
    assert active is first
    runtime.record_step("inspected files")
    runtime.complete_current_task("inspection complete")
    assert first.status == TaskStatus.COMPLETED
    assert second.status == TaskStatus.PENDING
    assert runtime.state.chat_history == [{"role": "user", "content": "Do the work"}]
    assert any(event["event"] == "task_completed" for event in events)
    assert runtime.start_next_task() is second


def test_runtime_cancel_marks_active_task() -> None:
    runtime = TaskRuntime(task_id="session")
    task = runtime.add_task("Work")
    assert runtime.start_next_task() is task
    runtime.cancel()
    assert runtime.state.cancelled is True
    assert task.status == TaskStatus.CANCELLED


def test_runtime_block_is_terminal_but_not_complete() -> None:
    runtime = TaskRuntime(task_id="session")
    task = runtime.add_task("Need user authority")
    runtime.start_next_task()

    runtime.block_current_task("Choose the deployment target")

    assert task.status == TaskStatus.BLOCKED
    assert task.result == "Choose the deployment target"
    assert runtime.state.is_finished() is True
