"""Clickable scrollbar — a 1-column UIControl that mirrors a target Window's
scroll state and responds to mouse clicks (jump-to-position), scroll wheel,
and arrow clicks.

Usage::

    from infinidev.ui.controls.clickable_scrollbar import scrollable_window
    win, container = scrollable_window(my_control, display_arrows=True)
    # Use 'container' in layout, 'win' for focus
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from infinidev.ui.theme import SCROLLBAR_BG, SCROLLBAR_FG

if TYPE_CHECKING:
    from prompt_toolkit.layout.containers import Window


# ── Styles ──────────────────────────────────────────────────────────────

_STYLE_BG = f"bg:{SCROLLBAR_BG} {SCROLLBAR_FG}"
_STYLE_THUMB = f"bg:{SCROLLBAR_FG} {SCROLLBAR_BG}"
_STYLE_ARROW = f"bg:{SCROLLBAR_BG} {SCROLLBAR_FG}"


class ClickableScrollbar(UIControl):
    """Renders a scrollbar track + thumb and handles mouse clicks.

    The control reads ``target_window.render_info`` each frame to stay
    in sync with the actual scroll position.  On click it sets
    ``target_window.vertical_scroll`` directly and, if the target content
    exposes ``_follow_tail`` / ``_scroll_offset`` (like ChatHistoryControl),
    updates those too.
    """

    def __init__(self, target_window: Window, display_arrows: bool = True) -> None:
        self._target = target_window
        self._display_arrows = display_arrows

    def is_focusable(self) -> bool:
        return False  # clicks still handled; focus stays on content

    def preferred_width(self, max_available_width: int) -> int | None:
        return 1

    def preferred_height(self, width: int, max_available_height: int,
                         wrap_lines: bool, get_line_prefix) -> int | None:
        return None

    # ── Rendering ───────────────────────────────────────────────────────

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        h = height or 1
        info = self._target.render_info

        track_height = h
        if self._display_arrows:
            track_height = max(1, h - 2)

        # Compute thumb position and size from render_info.
        if info and info.content_height > 0:
            content_h = info.content_height
            window_h = info.window_height or 1
            fraction_visible = min(1.0, window_h / content_h)
            fraction_above = info.vertical_scroll / content_h

            thumb_size = max(1, int(track_height * fraction_visible))
            thumb_top = int(track_height * fraction_above)
            # Clamp so thumb doesn't overflow track.
            thumb_top = min(thumb_top, track_height - thumb_size)
        else:
            thumb_size = track_height
            thumb_top = 0

        lines: list[list[tuple[str, str]]] = []

        if self._display_arrows:
            lines.append([(_STYLE_ARROW, "^")])

        for i in range(track_height):
            if thumb_top <= i < thumb_top + thumb_size:
                lines.append([(_STYLE_THUMB, " ")])
            else:
                lines.append([(_STYLE_BG, " ")])

        if self._display_arrows:
            lines.append([(_STYLE_ARROW, "v")])

        def get_line(i: int) -> list[tuple[str, str]]:
            if 0 <= i < len(lines):
                return lines[i]
            return []

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=0),
            show_cursor=False,
        )

    # ── Mouse handling ──────────────────────────────────────────────────

    def mouse_handler(self, mouse_event: MouseEvent):
        info = self._target.render_info
        if not info or info.content_height <= info.window_height:
            return NotImplemented

        y = mouse_event.position.y
        content_h = info.content_height
        window_h = info.window_height
        max_scroll = content_h - window_h

        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._set_scroll(max(0, self._target.vertical_scroll - 3),
                             content_h, window_h)
            return None

        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._set_scroll(min(max_scroll, self._target.vertical_scroll + 3),
                             content_h, window_h)
            return None

        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            h = window_h
            track_start = 0
            track_height = h

            if self._display_arrows:
                track_start = 1
                track_height = max(1, h - 2)

                # Arrow clicks: scroll by 3 lines.
                if y == 0:
                    self._set_scroll(max(0, self._target.vertical_scroll - 3),
                                     content_h, window_h)
                    return None
                if y >= h - 1:
                    self._set_scroll(min(max_scroll, self._target.vertical_scroll + 3),
                                     content_h, window_h)
                    return None

            # Track click: jump to proportional position.
            track_y = y - track_start
            fraction = track_y / max(1, track_height - 1)
            fraction = max(0.0, min(1.0, fraction))
            target_scroll = int(fraction * max_scroll)
            self._set_scroll(target_scroll, content_h, window_h)
            return None

        return NotImplemented

    def _set_scroll(self, target_scroll: int, content_h: int, window_h: int) -> None:
        """Set the target window's vertical scroll and sync content state."""
        self._target.vertical_scroll = target_scroll

        # Sync ChatHistoryControl's internal state if applicable.
        content = self._target.content
        max_scroll = content_h - window_h
        if hasattr(content, "_follow_tail") and hasattr(content, "_scroll_offset"):
            if target_scroll >= max_scroll:
                content._follow_tail = True
                content._scroll_offset = 0
            else:
                content._follow_tail = False
                content._scroll_offset = max_scroll - target_scroll


def scrollable_window(
    content,
    *,
    display_arrows: bool = True,
    **window_kwargs,
) -> tuple["Window", "VSplit"]:
    """Create a Window with a clickable scrollbar in a VSplit.

    Returns ``(window, container)`` — use *container* in the layout
    and *window* when you need a focus target.
    """
    from prompt_toolkit.layout.containers import VSplit, Window as _Window

    win = _Window(content=content, **window_kwargs)
    sb = ClickableScrollbar(win, display_arrows=display_arrows)
    sb_win = _Window(content=sb, width=1, style=f"bg:{SCROLLBAR_BG}")
    container = VSplit([win, sb_win])
    return win, container
