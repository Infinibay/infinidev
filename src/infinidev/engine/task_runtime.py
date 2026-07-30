"""Task-oriented runtime facade over the existing tool and LLM orchestration."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from infinidev.engine.runtime_state import (
    MemoryEntry,
    RuntimeState,
    TaskItem,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class TaskRuntime:
    """Coordinate durable tasks, unified chat history, and bounded memory."""

    def __init__(
        self,
        task_id: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        persist_events: bool = True,
    ) -> None:
        self.state = RuntimeState(task_id=task_id or str(uuid.uuid4()))
        self._on_event = on_event
        self._persist_events = persist_events

    def add_task(self, title: str, depends_on: list[str] | None = None) -> TaskItem:
        """Create a task and emit its initial state."""
        task = TaskItem(
            id=str(uuid.uuid4()),
            title=title,
            depends_on=list(depends_on or []),
        )
        self.state.tasks.append(task)
        self._emit("task_created", task=task)
        return task

    def start_next_task(self) -> TaskItem | None:
        """Start the next dependency-ready task and emit its state."""
        task = self.state.next_task()
        if task is not None:
            self._emit("task_started", task=task)
        return task

    def record_step(self, summary: str, step_id: str | None = None) -> None:
        """Record a step in working memory and advance runtime counters."""
        self.state.turn_count += 1
        self.remember(summary, kind="step_summary", importance=0.7)
        self._emit("step_completed", step_id=step_id, summary=summary)
        self.state.compact_memory()

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Recall relevant prior context using the Ken MCP client or fallback."""
        from infinidev.engine.ken_client import get_ken_client

        client = get_ken_client()
        hits = client.memory_search(query, limit=limit)
        results = [
            {
                "target": hit.target,
                "snippet": hit.snippet,
                "score": hit.score,
                "source": hit.source,
            }
            for hit in hits
        ]
        for hit in results:
            self.remember(
                f"{hit['target']}::{hit['snippet']}",
                kind="recalled",
                importance=min(1.0, float(hit["score"]) / 5.0 or 0.1),
            )
        return results

    def append_chat(self, role: str, content: str, **metadata: Any) -> None:
        """Append an immutable public message to the main conversation."""
        message: dict[str, Any] = {"role": role, "content": content, **metadata}
        self.state.chat_history.append(message)
        self._emit("chat_message", message=message)

    def remember(
        self, content: str, kind: str = "fact", importance: float = 0.5
    ) -> MemoryEntry:
        """Add searchable working memory without duplicating the chat transcript."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            kind=kind,
            content=content,
            task_id=self.state.current_task_id,
            importance=max(0.0, min(1.0, importance)),
        )
        self.state.memory.append(entry)
        self._emit("memory_added", entry=entry)
        return entry

    def complete_current_task(self, result: str = "") -> None:
        """Complete the active task and compact only model-facing memory."""
        task_id = self.state.current_task_id
        if task_id is None:
            return
        self.state.finish_task(task_id, result)
        if result:
            self.remember(result, kind="task_result", importance=0.9)
        self.state.compact_memory()
        self._emit("task_completed", task_id=task_id, result=result)

    def cancel(self) -> None:
        """Cancel the runtime and its active task."""
        self.state.cancelled = True
        if self.state.current_task_id:
            for task in self.state.tasks:
                if task.id == self.state.current_task_id:
                    task.status = TaskStatus.CANCELLED
                    break
        self._emit("runtime_cancelled")

    def _emit(self, event: str, **payload: Any) -> None:
        record = {"event": event, "task_id": self.state.task_id, **payload}
        if self._on_event is not None:
            self._on_event(record)
        if self._persist_events:
            from infinidev.engine.runtime_events_store import store_event

            store_event(
                self.state.task_id,
                event,
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"event", "task_id"}
                },
                task_id=record.get("task_id") if event.startswith("task_") else None,
            )
