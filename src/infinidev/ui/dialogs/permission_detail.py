"""Permission detail viewer — read-only display of permission request code."""

from __future__ import annotations

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl

from infinidev.ui.theme import WARNING
from infinidev.ui.dialogs.base import dialog_frame

DIALOG_NAME = "permission_detail"


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
