"""Modal confirmation for permission-gated tool calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    Float,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.widgets import Button

from infinidev.ui.dialogs.base import dialog_frame
from infinidev.ui.theme import SURFACE, SURFACE_DARK, TEXT, TEXT_MUTED, WARNING

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp

DIALOG_NAME = "permission_request"


def create_permission_detail(details: str):
    """Create a read-only permission detail viewer dialog."""
    buf = Buffer(
        document=Document(details),
        read_only=True,
        name="perm-detail",
    )
    control = BufferControl(buffer=buf, focusable=True)

    body = Window(content=control)
    return dialog_frame("Permission Detail", body, width=80, height=30, border_color=WARNING)


def create_permission_dialog(app_state: InfinidevApp) -> Float:
    """Create the centered Allow/Deny modal for a pending permission request."""

    def state() -> dict[str, str]:
        return getattr(app_state, "_permission_state", {}) or {}

    details_buffer = Buffer(read_only=True, name="permission-request-details")
    app_state._permission_details_buffer = details_buffer

    summary = Window(
        content=FormattedTextControl(
            lambda: [
                (f"{WARNING} bold", "Approval required\n"),
                (TEXT_MUTED, "An agent wants to perform an action that needs permission."),
            ]
        ),
        height=2,
        wrap_lines=True,
        style=f"bg:{SURFACE}",
    )
    tool_row = Window(
        content=FormattedTextControl(
            lambda: [
                (f"{TEXT_MUTED} bold", "Tool: "),
                (TEXT, state().get("tool_name", "")),
            ]
        ),
        height=1,
        style=f"bg:{SURFACE}",
    )
    description = Window(
        content=FormattedTextControl(
            lambda: [(TEXT, state().get("description", ""))]
        ),
        height=D(min=1, max=3, preferred=2),
        wrap_lines=True,
        style=f"bg:{SURFACE}",
    )
    detail_label = Window(
        content=FormattedTextControl(
            [(f"{TEXT_MUTED} bold", "REQUEST DETAILS")]
        ),
        height=1,
        style=f"bg:{SURFACE_DARK}",
    )
    detail_window = Window(
        content=BufferControl(buffer=details_buffer, focusable=True),
        height=D(min=3, max=10, preferred=7),
        wrap_lines=True,
        right_margins=[ScrollbarMargin(display_arrows=True)],
        style=f"bg:{SURFACE_DARK}",
    )

    allow_button = Button(
        "Allow",
        handler=lambda: app_state._resolve_permission(True),
        width=16,
        left_symbol="[",
        right_symbol="]",
    )
    deny_button = Button(
        "Deny",
        handler=lambda: app_state._resolve_permission(False),
        width=16,
        left_symbol="[",
        right_symbol="]",
    )
    app_state._permission_allow_button = allow_button
    app_state._permission_deny_button = deny_button

    buttons = VSplit(
        [
            Window(),
            allow_button,
            Window(width=3),
            deny_button,
            Window(),
        ],
        height=1,
        style=f"bg:{SURFACE}",
    )
    body = HSplit(
        [
            summary,
            Window(height=1, style=f"bg:{SURFACE}"),
            tool_row,
            description,
            Window(height=1, style=f"bg:{SURFACE}"),
            detail_label,
            detail_window,
            Window(height=1, style=f"bg:{SURFACE}"),
            buttons,
        ],
        style=f"bg:{SURFACE}",
    )
    frame = dialog_frame(
        "Permission required",
        body,
        width=82,
        height=21,
        border_color=WARNING,
        hints="y allow · n/esc deny · tab switch · enter select",
    )
    return Float(
        content=ConditionalContainer(
            content=frame,
            filter=Condition(lambda: app_state.active_dialog == DIALOG_NAME),
        ),
        transparent=False,
    )
