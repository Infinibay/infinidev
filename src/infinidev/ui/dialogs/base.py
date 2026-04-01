"""Base dialog pattern for modal Float overlays.

All dialogs follow this pattern:
1. A Float is registered in the FloatContainer
2. Visibility is controlled by app.active_dialog == dialog_name
3. Escape dismisses by setting active_dialog = None

Each dialog is a function that returns a Float containing the dialog layout,
plus a Condition for visibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import (
    Float, HSplit, VSplit, Window, ConditionalContainer,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.widgets import Box, Frame, Shadow

from infinidev.ui.theme import (
    SURFACE, SURFACE_DARK, SURFACE_LIGHT, PRIMARY, TEXT, TEXT_MUTED,
    ACCENT, WARNING, ERROR, SUCCESS,
    MODAL_OVERLAY_BG,
)

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp


def dialog_frame(title: str, body, width: int = 60, height: int = 20,
                 border_color: str = PRIMARY) -> HSplit:
    """Wrap a body container in a bordered dialog frame with title."""
    title_bar = Window(
        content=FormattedTextControl(lambda t=title: [
            (f"#ffffff bg:{border_color} bold", f" {t} "),
        ]),
        height=1,
    )

    hint_bar = Window(
        content=FormattedTextControl(lambda: [
            (f"{TEXT_MUTED}", " Esc = close"),
        ]),
        height=1,
        style=f"bg:{SURFACE_DARK}",
    )

    inner = HSplit([
        title_bar,
        body,
        hint_bar,
    ], style=f"bg:{SURFACE}")

    return Frame(
        body=inner,
        width=D(preferred=width),
        height=D(preferred=height),
        style=f"{border_color}",
    )


def make_dialog_float(name: str, app_state: InfinidevApp, content) -> Float:
    """Create a Float that's visible when app.active_dialog == name."""
    return Float(
        content=ConditionalContainer(
            content=content,
            filter=Condition(lambda n=name: app_state.active_dialog == n),
        ),
        transparent=False,
    )
