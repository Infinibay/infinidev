"""Durable task state and bounded working-memory policy for agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    """Lifecycle state exposed to the UI and persistence layer."""

    PENDING = "pending"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_TASK_STATUSES = frozenset({
    TaskStatus.BLOCKED,
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
})


@dataclass(slots=True)
class TaskItem:
    """A user-visible unit of work with explicit lifecycle metadata."""

    id: str
    title: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    result: str = ""
    error: str = ""
    attempts: int = 0


@dataclass(slots=True)
class MemoryEntry:
    """A searchable working-memory entry, independent from chat history."""

    id: str
    kind: str
    content: str
    task_id: str | None = None
    step_id: str | None = None
    importance: float = 0.5
    active: bool = True


@dataclass(slots=True)
class RuntimeState:
    """Single source of truth for task progress and model working memory."""

    task_id: str
    tasks: list[TaskItem] = field(default_factory=list)
    memory: list[MemoryEntry] = field(default_factory=list)
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    current_task_id: str | None = None
    turn_count: int = 0
    tool_call_count: int = 0
    cancelled: bool = False

    def next_task(self) -> TaskItem | None:
        """Return the first runnable task and mark it active."""
        if (
            self.cancelled
            or self.current_task_id is not None
            or any(task.status is TaskStatus.ACTIVE for task in self.tasks)
        ):
            return None

        completed = {
            task.id for task in self.tasks if task.status == TaskStatus.COMPLETED
        }
        for task in self.tasks:
            if task.status == TaskStatus.PENDING and all(
                dep in completed for dep in task.depends_on
            ):
                task.status = TaskStatus.ACTIVE
                task.attempts += 1
                self.current_task_id = task.id
                return task
        return None

    def finish_task(self, task_id: str, result: str = "") -> None:
        """Mark the active current task complete and release its dependants."""
        task = next((item for item in self.tasks if item.id == task_id), None)
        if task is None:
            raise KeyError(f"unknown task {task_id!r}")
        if (
            task.status is not TaskStatus.ACTIVE
            or self.current_task_id != task_id
        ):
            raise RuntimeError(
                f"task {task_id!r} is not the active current task"
            )
        task.status = TaskStatus.COMPLETED
        task.result = result
        self.current_task_id = None

    def compact_memory(self, keep: int = 12) -> None:
        """Retain high-value recent memory while leaving chat immutable."""
        active = [entry for entry in self.memory if entry.active]
        ranked = sorted(
            enumerate(active),
            key=lambda item: (item[1].importance, item[0]),
            reverse=True,
        )
        retained = {entry.id for _, entry in ranked[:max(0, keep)]}
        for entry in self.memory:
            if entry.active and entry.id not in retained:
                entry.active = False

    def working_memory(self) -> list[str]:
        """Return active memory text for prompt construction."""
        return [entry.content for entry in self.memory if entry.active]

    def is_finished(self) -> bool:
        """Whether the runtime was cancelled or all scheduled work is terminal."""
        return self.cancelled or (
            bool(self.tasks)
            and all(task.status in TERMINAL_TASK_STATUSES for task in self.tasks)
        )

    def is_successful(self) -> bool:
        """Whether every task completed successfully, not merely terminally."""
        return bool(self.tasks) and all(
            task.status is TaskStatus.COMPLETED for task in self.tasks
        )
