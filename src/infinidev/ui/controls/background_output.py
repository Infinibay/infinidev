"""Live background output with bounded history and explicit tail following."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from infinidev.ui.controls.message_widgets import _wrap_fragments
from infinidev.ui.theme import TEXT_MUTED

if TYPE_CHECKING:
    from infinidev.tools.shell.background_manager import BackgroundTask


class BackgroundTaskOutputControl(UIControl):
    """Display captured stdout/stderr without blocking the UI on process exit."""

    def __init__(self, task: BackgroundTask) -> None:
        self.task = task
        self._follow_tail = True
        self._scroll_offset = 0
        self._line_count = 0
        self._height = 1
        self._cache_key: tuple[str, int, int, bool] | None = None
        self._lines: list[list[tuple[str, str]]] = []
        self._discarded = 0
        self._kb = KeyBindings()
        for key, action in (
            ("up", lambda: self.page_up(1)), ("down", lambda: self.page_down(1)),
            ("pageup", self.page_up), ("pagedown", self.page_down),
            ("home", self.scroll_home), ("end", self.scroll_end),
        ):
            self._kb.add(key)(lambda event, action=action: action())

    def is_focusable(self) -> bool:
        return True

    def get_key_bindings(self):
        return self._kb

    def status_fragments(self):
        mode = "Following output" if self._follow_tail else "Scrollback · End to follow"
        retained = " · older output discarded" if self._discarded else ""
        return [(TEXT_MUTED, f" {self.task.id} · {self.task.status_line()} · {mode}{retained}")]

    def page_up(self, lines: int | None = None) -> None:
        self._follow_tail = False
        self._scroll_offset = min(
            self._scroll_offset + (lines if lines is not None else self._height),
            max(0, self._line_count - self._height),
        )

    def page_down(self, lines: int | None = None) -> None:
        self._scroll_offset = max(
            0, self._scroll_offset - (lines if lines is not None else self._height),
        )
        self._follow_tail = self._scroll_offset == 0

    def scroll_home(self) -> None:
        self._follow_tail = False
        self._scroll_offset = max(0, self._line_count - self._height)

    def scroll_end(self) -> None:
        self._follow_tail = True
        self._scroll_offset = 0

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.page_up(3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.page_down(3)
            return None
        return NotImplemented

    def get_vertical_scroll(self, window) -> int:
        tail = max(0, self._line_count - self._height)
        return tail if self._follow_tail else max(0, tail - self._scroll_offset)

    def create_content(self, width: int, height: int | None) -> UIContent:
        text, self._discarded = self.task.combined_output()
        key = (text, width, self._discarded, self.task.is_running)
        self._height = max(1, height or 1)
        if key != self._cache_key:
            normalized = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
            fragments = to_formatted_text(ANSI(normalized))
            # Zero-width escape fragments may contain raw terminal commands;
            # a log viewer only needs the printable text and colour styles.
            fragments = [(style, part) for style, part in fragments
                         if "[ZeroWidthEscape]" not in style]
            lines = [row for line in split_lines(fragments)
                     for row in _wrap_fragments(line, max(1, width))]
            if not text:
                lines = [[(TEXT_MUTED, "Waiting for output…" if self.task.is_running
                           else "No output captured.")]]
            if not self._follow_tail and self._cache_key is not None:
                self._scroll_offset += len(lines) - self._line_count
            self._lines = lines
            self._line_count = len(lines)
            self._cache_key = key
        self._scroll_offset = min(max(0, self._scroll_offset),
                                  max(0, self._line_count - self._height))
        cursor = max(0, self._line_count - 1)
        if not self._follow_tail:
            cursor = max(0, cursor - self._scroll_offset)
        return UIContent(
            get_line=lambda i: self._lines[i] if 0 <= i < self._line_count else [],
            line_count=self._line_count,
            cursor_position=Point(x=0, y=cursor),
            show_cursor=False,
        )
