"""Scrollable FormattedTextControl for sidebar sections.

Subclasses FormattedTextControl to enable mouse-wheel scrolling
by making the control focusable and letting Window handle scroll events.
"""

from __future__ import annotations

from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent


class ScrollableTextControl(FormattedTextControl):
    """A FormattedTextControl that supports mouse-wheel scrolling.

    The only change from the base class: ``is_focusable()`` returns True
    and ``mouse_handler()`` returns NotImplemented, which tells the
    parent Window to handle scroll-wheel events via its built-in
    vertical scroll logic.
    """

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event: MouseEvent):
        """Let Window handle scroll wheel events."""
        return NotImplemented
