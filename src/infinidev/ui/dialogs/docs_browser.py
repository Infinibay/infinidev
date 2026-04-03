"""Documentation browser — three-panel dialog for cached library docs."""

from __future__ import annotations
from typing import Any

from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import UIControl, UIContent, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D

from infinidev.ui.theme import PRIMARY, TEXT, TEXT_MUTED
from infinidev.ui.dialogs.base import dialog_frame

DIALOG_NAME = "docs_browser"


from infinidev.ui.dialogs.docs_list_control import DocsListControl
from infinidev.ui.dialogs.docs_content_control import DocsContentControl

def create_docs_browser():
    """Create the three-panel docs browser dialog."""
    lib_list = DocsListControl()
    section_list = DocsListControl()
    content_view = DocsContentControl()

    body = VSplit([
        Window(content=lib_list, width=D(weight=30)),
        Window(width=1, char="│", style=f"{PRIMARY}"),
        Window(content=section_list, width=D(weight=25)),
        Window(width=1, char="│", style=f"{PRIMARY}"),
        Window(content=content_view, width=D(weight=45)),
    ])

    frame = dialog_frame("Documentation", body, width=90, height=30, border_color=PRIMARY)
    return frame, lib_list, section_list, content_view
