"""Base dialog pattern for modal Float overlays.

All dialogs follow this pattern:
1. A Float is registered in the FloatContainer
2. Visibility is controlled by app.active_dialog == dialog_name
3. Escape dismisses by setting active_dialog = None

Each dialog is a function that returns a Float containing the dialog layout,
plus a Condition for visibility.

The frame is drawn by hand rather than with ``prompt_toolkit.widgets.Frame``
for three reasons: ``Frame`` only draws square corners (``┌``) while the rest
of the UI is rounded, it cannot inline the title into the top border, and it
gives no interior padding — its content touches the border on every side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    Float,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Shadow

from infinidev.ui.theme import (
    PRIMARY,
    SURFACE,
    TEXT,
    TEXT_DIM,
    TEXT_MUTED,
)

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp

DEFAULT_HINTS = "esc close"


def dialog_frame(
    title: str,
    body,
    width: int = 60,
    height: int = 20,
    border_color: str = PRIMARY,
    hints: str = DEFAULT_HINTS,
) -> Shadow:
    """Wrap *body* in a rounded, titled, padded modal frame.

        ╭─ Title ─────────────────────────╮
        │                                 │
        │  body                           │
        │                                 │
        ╰──────────────────────── esc ────╯

    The title rides the top border and the key hints ride the bottom one,
    so neither costs an interior row — a 20-row dialog used to spend three
    of them on chrome.
    """
    border = f"{border_color} bg:{SURFACE}"

    def _title_fragments():
        return [
            (border, "─ "),
            (f"{TEXT} bold bg:{SURFACE}", title),
            (border, " "),
        ]

    title_width = get_cwidth(title) + 3

    top = VSplit(
        [
            Window(width=1, char="╭", style=border),
            Window(content=FormattedTextControl(_title_fragments), width=title_width),
            Window(char="─", style=border),
            Window(width=1, char="╮", style=border),
        ],
        height=1,
    )

    def _hint_fragments():
        return [(f"{TEXT_DIM} bg:{SURFACE}", f" {hints} ")]

    bottom = VSplit(
        [
            Window(width=1, char="╰", style=border),
            Window(char="─", style=border),
            Window(
                content=FormattedTextControl(_hint_fragments),
                width=get_cwidth(hints) + 2,
            ),
            Window(char="─", style=border, width=2),
            Window(width=1, char="╯", style=border),
        ],
        height=1,
    )

    middle = VSplit(
        [
            Window(width=1, char="│", style=border),
            Window(width=1, style=f"bg:{SURFACE}"),
            HSplit(
                [
                    Window(height=1, style=f"bg:{SURFACE}"),
                    body,
                    Window(height=1, style=f"bg:{SURFACE}"),
                ],
                style=f"bg:{SURFACE}",
            ),
            Window(width=1, style=f"bg:{SURFACE}"),
            Window(width=1, char="│", style=border),
        ]
    )

    return Shadow(
        body=HSplit(
            [top, middle, bottom],
            width=D(preferred=width),
            height=D(preferred=height),
            style=f"bg:{SURFACE}",
        )
    )


def section_title(text: str) -> Window:
    """A quiet section heading for use inside dialogs."""
    return Window(
        content=FormattedTextControl(
            lambda: [(f"{TEXT_MUTED} bold bg:{SURFACE}", text.upper())]
        ),
        height=1,
        style=f"bg:{SURFACE}",
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
