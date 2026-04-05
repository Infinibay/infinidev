"""Notes browser — single-panel scrollable dialog for agent notes."""

from __future__ import annotations

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIControl, UIContent

from infinidev.ui.theme import PRIMARY, ACCENT, TEXT, TEXT_MUTED


class NotesBrowserControl(UIControl):
    """Scrollable view of session + task notes."""

    def __init__(self) -> None:
        self.session_notes: list[str] = []
        self.task_notes: list[str] = []
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
        lines: list[list[tuple[str, str]]] = []
        usable = max(width - 2, 20)

        # ── Session Notes ──
        lines.append([(f"bg:{PRIMARY} #ffffff bold", f" {'Session Notes':<{usable}} ")])
        if self.session_notes:
            for i, note in enumerate(self.session_notes, 1):
                for wl in _wrap(f"  {i}. {note}", usable):
                    lines.append([(f"{TEXT}", wl)])
        else:
            lines.append([(f"{TEXT_MUTED}", "  (no session notes)")])

        lines.append([("", "")])

        # ── Task Notes ──
        lines.append([(f"bg:{ACCENT} #ffffff bold", f" {'Task Notes':<{usable}} ")])
        if self.task_notes:
            for i, note in enumerate(self.task_notes, 1):
                for wl in _wrap(f"  {i}. {note}", usable):
                    lines.append([(f"{TEXT}", wl)])
        else:
            lines.append([(f"{TEXT_MUTED}", "  (no task notes)")])

        self._line_count = len(lines)
        scroll = min(self._scroll, max(0, self._line_count - 1))

        def get_line(i: int):
            return lines[i] if 0 <= i < len(lines) else []

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=scroll),
        )


def _wrap(text: str, width: int) -> list[str]:
    """Word-wrap a single string into lines that fit *width*."""
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
            text = "    " + text  # indent continuation
    return result
