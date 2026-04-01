"""Cancel task confirmation dialog."""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable

from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Button

from infinidev.ui.theme import ERROR, TEXT, TEXT_MUTED, SURFACE
from infinidev.ui.dialogs.base import dialog_frame

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp

DIALOG_NAME = "cancel_task"


def create_cancel_dialog(on_confirm: Callable, on_cancel: Callable):
    """Create the cancel task dialog content."""
    body = HSplit([
        Window(
            content=FormattedTextControl(lambda: [
                (f"{TEXT}", "\n  Cancel current task?\n"),
                (f"{TEXT_MUTED}", "  The task will stop after the current tool call finishes.\n\n"),
                (f"{ERROR} bold", "  Type 'y' to confirm, any other key to cancel."),
            ]),
            height=6,
        ),
    ])
    return dialog_frame("Cancel Task", body, width=50, height=8, border_color=ERROR)
