"""Unsaved changes confirmation dialog."""

from __future__ import annotations
from typing import Callable

from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from infinidev.ui.theme import WARNING, TEXT, TEXT_MUTED
from infinidev.ui.dialogs.base import dialog_frame

DIALOG_NAME = "unsaved_changes"


def create_unsaved_dialog(filename: str):
    """Create the unsaved changes dialog content."""
    body = HSplit([
        Window(
            content=FormattedTextControl(lambda f=filename: [
                (f"{TEXT}", f"\n  Unsaved changes in {f}\n\n"),
                (f"{TEXT_MUTED}", "  Do you want to save before closing?\n\n"),
                (f"{TEXT}", "  [s]ave  [d]iscard  [c]ancel"),
            ]),
            height=6,
        ),
    ])
    return dialog_frame(f"Unsaved Changes", body, width=55, height=8, border_color=WARNING)
