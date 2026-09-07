"""Background-tasks explorer — single-panel scrollable dialog.

Lists every command started with ``run_in_background`` and shows, per task,
its label, live status verdict, and a short tail of recent output. The
control reads the process-global background manager *live* on each render, so
the panel reflects current state every time the screen redraws (no frozen
snapshot) — a watcher that just exited or a server that just printed its
readiness line shows up the next frame.
"""

from __future__ import annotations

from typing import Callable

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from infinidev.ui.theme import PRIMARY, ACCENT, TEXT, TEXT_MUTED

# Keep selection compact; opening a task exposes the full retained output.
_TAIL_LINES = 6


class BackgroundTasksControl(UIControl):
    """Selectable view of background tasks with live output previews."""

    def __init__(self, on_open: Callable[[str], None] | None = None) -> None:
        self._scroll: int = 0
        self._line_count: int = 0
        self.selected_index = 0
        self._on_open = on_open
        self._task_rows: dict[int, int] = {}

    def is_focusable(self) -> bool:
        return True

    def scroll_up(self) -> None:
        self.select_prev()

    def scroll_down(self) -> None:
        self.select_next()

    def select_prev(self) -> None:
        self.selected_index = max(0, self.selected_index - 1)

    def select_next(self) -> None:
        from infinidev.tools.shell.background_manager import get_background_manager

        count = len(get_background_manager().list())
        self.selected_index = min(max(0, count - 1), self.selected_index + 1)

    def open_selected(self) -> None:
        from infinidev.tools.shell.background_manager import get_background_manager

        tasks = get_background_manager().list()
        if self._on_open and 0 <= self.selected_index < len(tasks):
            self._on_open(tasks[self.selected_index].id)

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.select_prev()
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.select_next()
            return None
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            index = self._task_rows.get(mouse_event.position.y)
            if index is not None:
                self.selected_index = index
                self.open_selected()
                return None
        return NotImplemented

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        from infinidev.tools.shell.background_manager import get_background_manager

        usable = max(width - 2, 20)
        lines: list[list[tuple[str, str]]] = []
        self._task_rows.clear()

        tasks = get_background_manager().list()
        if not tasks:
            lines.append([(f"{TEXT_MUTED}", "  No background tasks have been started.")])
            lines.append([("", "")])
            lines.append([
                (f"{TEXT_MUTED}",
                 "  The agent starts them with run_in_background "
                 "(dev servers, watchers, builds).")
            ])
            self._line_count = len(lines)
            return self._content(lines)

        running = sum(1 for t in tasks if t.status == "running")
        header = f" Background Tasks — {len(tasks)} total, {running} running "
        lines.append([(f"bg:{PRIMARY} #ffffff bold", f"{header:<{usable + 1}}")])
        lines.append([("", "")])

        self.selected_index = min(self.selected_index, len(tasks) - 1)
        for index, t in enumerate(tasks):
            # Header line per task: running tasks in the accent colour so the
            # eye lands on what's still live; finished/failed ones muted.
            head_style = f"{ACCENT} bold" if t.status == "running" else f"{TEXT} bold"
            start = len(lines)
            selected = index == self.selected_index
            if selected:
                head_style = f"bg:{PRIMARY} #ffffff bold"
                self._scroll = start
            marker = "›" if selected else " "
            lines.append([(head_style, f"{marker} [{t.id}] {t.description}")])
            lines.append([(f"{TEXT_MUTED}", f"      {t.status_line()}")])

            tail = _output_tail(t, _TAIL_LINES)
            if tail:
                for raw in tail:
                    for wl in _wrap(f"      │ {raw}", usable):
                        lines.append([(f"{TEXT_MUTED}", wl)])
            lines.append([("", "")])
            self._task_rows.update({row: index for row in range(start, len(lines))})

        self._line_count = len(lines)
        return self._content(lines)

    def _content(self, lines: list[list[tuple[str, str]]]) -> UIContent:
        scroll = min(self._scroll, max(0, len(lines) - 1))

        def get_line(i: int):
            return lines[i] if 0 <= i < len(lines) else []

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=scroll),
        )


def _output_tail(task, n: int) -> list[str]:
    """Return the last ``n`` non-empty-ish lines of the task's combined output."""
    combined, _ = task.combined_output()
    combined = combined.strip()
    if not combined:
        return []
    rows = combined.splitlines()
    return [r.rstrip() for r in rows[-n:]]


def _wrap(text: str, width: int) -> list[str]:
    """Word-wrap a single string into lines that fit ``width``."""
    if len(text) <= width:
        return [text]
    result: list[str] = []
    while text:
        if len(text) <= width:
            result.append(text)
            break
        cut = text.rfind(" ", 0, width)
        if cut <= 0:
            cut = width
        result.append(text[:cut])
        text = text[cut:].lstrip(" ")
        if text:
            text = "      " + text  # indent continuation under the output gutter
    return result
