"""Tests for the task runtime and its orchestration integration."""

import pytest

from infinidev.engine.runtime_state import MemoryEntry, TaskStatus
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


def test_runtime_does_not_start_a_second_active_task() -> None:
    runtime = TaskRuntime(task_id="session", persist_events=False)
    first = runtime.add_task("First")
    second = runtime.add_task("Second")

    assert runtime.start_next_task() is first
    assert runtime.start_next_task() is None
    assert runtime.state.current_task_id == first.id
    assert first.status == TaskStatus.ACTIVE
    assert second.status == TaskStatus.PENDING
    assert second.attempts == 0


def test_runtime_cancel_closes_all_pending_work_and_is_idempotent() -> None:
    events: list[dict] = []
    runtime = TaskRuntime(
        task_id="session", on_event=events.append, persist_events=False
    )
    first = runtime.add_task("First")
    second = runtime.add_task("Second")
    runtime.start_next_task()

    runtime.cancel()
    runtime.cancel()

    assert runtime.state.cancelled is True
    assert runtime.state.current_task_id is None
    assert first.status == TaskStatus.CANCELLED
    assert second.status == TaskStatus.CANCELLED
    assert runtime.start_next_task() is None
    assert runtime.state.is_finished() is True
    assert [event["event"] for event in events].count("runtime_cancelled") == 1


def test_runtime_rejects_unknown_dependencies_and_new_work_after_cancel() -> None:
    runtime = TaskRuntime(task_id="session", persist_events=False)

    with pytest.raises(ValueError, match="unknown task dependencies"):
        runtime.add_task("Impossible", depends_on=["missing"])

    runtime.cancel()
    assert runtime.state.is_finished() is True
    with pytest.raises(RuntimeError, match="cancelled runtime"):
        runtime.add_task("Too late")


def test_runtime_success_requires_nonempty_all_completed_work() -> None:
    runtime = TaskRuntime(task_id="session", persist_events=False)
    assert runtime.state.is_finished() is False
    assert runtime.state.is_successful() is False

    runtime.add_task("Work")
    runtime.start_next_task()
    runtime.block_current_task("external dependency")

    assert runtime.state.is_finished() is True
    assert runtime.state.is_successful() is False


def test_task_result_memory_keeps_owning_task_id() -> None:
    runtime = TaskRuntime(task_id="session", persist_events=False)
    task = runtime.add_task("Work")
    runtime.start_next_task()

    runtime.complete_current_task("done")

    result = next(entry for entry in runtime.state.memory if entry.kind == "task_result")
    assert result.task_id == task.id


def test_memory_compaction_prefers_recent_entries_at_equal_importance() -> None:
    runtime = TaskRuntime(task_id="session", persist_events=False)
    runtime.state.memory = [
        MemoryEntry(id="z-old", kind="fact", content="old", importance=0.5),
        MemoryEntry(id="y-middle", kind="fact", content="middle", importance=0.5),
        MemoryEntry(id="a-new", kind="fact", content="new", importance=0.5),
    ]

    runtime.state.compact_memory(keep=2)

    assert runtime.state.working_memory() == ["middle", "new"]


def test_task_events_use_consistent_runtime_and_child_ids() -> None:
    events: list[dict] = []
    runtime = TaskRuntime(
        task_id="session", on_event=events.append, persist_events=False
    )
    task = runtime.add_task("Work")
    runtime.start_next_task()
    runtime.complete_current_task("done")

    task_events = [event for event in events if event["event"].startswith("task_")]
    assert task_events
    assert {event["runtime_id"] for event in task_events} == {"session"}
    assert {event["task_id"] for event in task_events} == {task.id}

def test_terminal_dependency_blocks_dependents_but_not_independent_work() -> None:
    events: list[dict] = []
    runtime = TaskRuntime(
        task_id="session",
        on_event=events.append,
        persist_events=False,
    )
    root = runtime.add_task("Compile schema")
    child = runtime.add_task("Generate client", depends_on=[root.id])
    grandchild = runtime.add_task("Publish client", depends_on=[child.id])
    independent = runtime.add_task("Update docs")

    assert runtime.start_next_task() is root
    runtime.fail_current_task("schema compiler failed")

    assert root.status == TaskStatus.FAILED
    assert child.status == TaskStatus.BLOCKED
    assert grandchild.status == TaskStatus.BLOCKED
    assert root.id in child.result
    assert child.id in grandchild.result
    assert independent.status == TaskStatus.PENDING
    assert runtime.state.is_finished() is False

    assert runtime.start_next_task() is independent
    runtime.complete_current_task("docs updated")

    blocked_ids = {
        event["task_id"]
        for event in events
        if event["event"] == "task_blocked"
    }
    assert blocked_ids == {child.id, grandchild.id}
    assert runtime.state.is_finished() is True
    assert runtime.state.is_successful() is False


def test_runtime_rejects_invalid_titles_and_illegal_completion() -> None:
    runtime = TaskRuntime(task_id="session", persist_events=False)

    with pytest.raises(ValueError, match="non-empty"):
        runtime.add_task("   ")

    task = runtime.add_task("Work")
    with pytest.raises(RuntimeError, match="active current task"):
        runtime.state.finish_task(task.id, "too early")
    with pytest.raises(KeyError, match="unknown task"):
        runtime.state.finish_task("missing", "impossible")
