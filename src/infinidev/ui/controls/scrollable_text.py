"""Scrollable FormattedTextControl for sidebar sections.

Subclasses FormattedTextControl to enable mouse-wheel scrolling,
clickable-scrollbar interaction, and manual scroll-position tracking.

This mirrors the scroll-state pattern used by ChatHistoryControl:
``_follow_tail`` / ``_scroll_offset`` keep the ClickableScrollbar in
sync with the content window.
"""

from __future__ import annotations

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType


class ScrollableTextControl(FormattedTextControl):
    """A FormattedTextControl that supports mouse-wheel and scrollbar scrolling.

    Scroll state:
    - ``_follow_tail``: when True, the viewport sticks to the bottom.
    - ``_scroll_offset``: lines-from-bottom offset (used when not following).
    - ``_line_count``: total line count from the last ``create_content()`` call.

    These attributes are read/written by ClickableScrollbar._set_scroll()
    and by the Window's built-in scroll-wheel handler (via
    move_cursor_up / move_cursor_down).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._follow_tail: bool = True
        self._scroll_offset: int = 0
        self._line_count: int = 0

    # ── Focus / mouse ──────────────────────────────────────────────────

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event: MouseEvent):
        """Delegate everything to Window's built-in scroll handling."""
        return NotImplemented

    # ── Scroll helpers (called by Window on mouse wheel / by scrollbar) ─

    def move_cursor_down(self) -> None:
        """Called by Window._scroll_down() on mouse-wheel down."""
        if self._scroll_offset > 0:
            self._scroll_offset -= 1
        if self._scroll_offset == 0:
            self._follow_tail = True

    def move_cursor_up(self) -> None:
        """Called by Window._scroll_up() on mouse-wheel up."""
        self._follow_tail = False
        self._scroll_offset = min(
            self._scroll_offset + 1,
            max(0, self._line_count - 1),
        )

    # ── Content rendering ──────────────────────────────────────────────

    def create_content(self, width: int, height: int | None) -> UIContent:
        """Build UIContent with cursor positioned according to scroll state.

        This is the key difference from the base FormattedTextControl:
        we set ``cursor_position.y`` to the line we want visible, which
        prevents the Window from overriding our manual vertical_scroll.
        """
        content = super().create_content(width, height)
        line_count = content.line_count
        self._line_count = line_count

        if self._follow_tail:
            cursor_y = max(0, line_count - 1)
        else:
            cursor_y = max(0, line_count - 1 - self._scroll_offset)

        # Return a new UIContent with corrected cursor position.
        return UIContent(
            get_line=content.get_line,
            line_count=line_count,
            cursor_position=Point(x=0, y=cursor_y),
            show_cursor=False,
        )
