"""Background-tasks explorer — single-panel scrollable dialog.

Lists every command started with ``run_in_background`` and shows, per task,
its label, live status verdict, and a short tail of recent output. The
control reads the process-global background manager *live* on each render, so
the panel reflects current state every time the screen redraws (no frozen
snapshot) — a watcher that just exited or a server that just printed its
readiness line shows up the next frame.
"""

from __future__ import annotations

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIControl, UIContent

from infinidev.ui.theme import PRIMARY, ACCENT, TEXT, TEXT_MUTED

# How many trailing output lines to show per task. Enough to "explore" what a
# task is doing without turning the dialog into a full log viewer (the agent's
# background_status tool is the place for the full buffer).
_TAIL_LINES = 6


class BackgroundTasksControl(UIControl):
    """Scrollable view of the current background tasks and their output."""

    def __init__(self) -> None:
        self._scroll: int = 0
        self._line_count: int = 0

    def is_focusable(self) -> bool:
        return True

    def scroll_up(self) -> None:
        self._scroll = max(0, self._scroll - 1)

    def scroll_down(self) -> None:
        self._scroll = min(max(0, self._line_count - 1), self._scroll + 1)

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        from infinidev.tools.shell.background_manager import get_background_manager

        usable = max(width - 2, 20)
        lines: list[list[tuple[str, str]]] = []

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

        for t in tasks:
            # Header line per task: running tasks in the accent colour so the
            # eye lands on what's still live; finished/failed ones muted.
            head_style = f"{ACCENT} bold" if t.status == "running" else f"{TEXT} bold"
            lines.append([(head_style, f"  [{t.id}] {t.description}")])
            lines.append([(f"{TEXT_MUTED}", f"      {t.status_line()}")])

            tail = _output_tail(t, _TAIL_LINES)
            if tail:
                for raw in tail:
                    for wl in _wrap(f"      │ {raw}", usable):
                        lines.append([(f"{TEXT_MUTED}", wl)])
            lines.append([("", "")])

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
    out, err = task.output()
    combined = "\n".join(s for s in (out, err) if s).strip()
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
