"""Translate shared orchestration callbacks into durable browser messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from infinidev.engine.orchestration.hooks import NoOpHooks

if TYPE_CHECKING:
    from infinidev.server.runtime import WebSession


class ServerHooks(NoOpHooks):
    """The web renderer uses the same task pipeline as the terminal."""

    def __init__(self, session: WebSession) -> None:
        self.session = session

    def on_phase(self, phase: str) -> None:
        if self.session.cancelled.is_set() and self.session.engine:
            self.session.engine.cancel()
        self.session.state(phase=phase)

    def on_status(self, level: str, msg: str) -> None:
        self.session.add_message("System", msg, "status", level=level)

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        self.session.end_stream(speaker, kind)
        self.session.add_message(speaker, msg, kind)

    def notify_error(self, speaker: str, msg: str, traceback_text: str) -> None:
        self.session.add_message(speaker, msg, "error", traceback=traceback_text)

    def notify_stream_chunk(self, speaker: str, chunk: str, kind: str = "agent") -> None:
        self.session.stream(speaker, chunk, kind)

    def notify_stream_end(self, speaker: str, kind: str = "agent") -> None:
        self.session.end_stream(speaker, kind)

    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        return self.session.ask(prompt, kind)

    def on_step_start(self, step_num: int, total: int, all_steps: list[dict],
                      completed: list[int]) -> None:
        self.session.state(steps={"current": step_num, "total": total,
                                  "steps": all_steps, "completed": completed})

    def on_file_change(self, path: str) -> None:
        self.session.emit({"type": "refresh", "path": path})

    def on_stage_update(self, snapshot: dict[str, Any]) -> None:
        self.session.state(steps=snapshot)
