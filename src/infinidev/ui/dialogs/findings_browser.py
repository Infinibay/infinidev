"""Findings browser — two-panel dialog for browsing findings/knowledge."""

from __future__ import annotations
from typing import Any

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import UIControl, UIContent, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D

from infinidev.ui.theme import PRIMARY, TEXT, TEXT_MUTED, ACCENT
from infinidev.ui.dialogs.base import dialog_frame

DIALOG_NAME = "findings_browser"

_TYPE_ICONS = {
    "project": "P", "observation": "O", "conclusion": "C",
    "hypothesis": "?", "experiment": "E", "proof": "V",
}


from infinidev.ui.dialogs.findings_list_control import FindingsListControl
from infinidev.ui.dialogs.findings_detail_control import FindingsDetailControl

def create_findings_browser(title: str = "Findings"):
    """Create the findings browser dialog."""
    list_ctrl = FindingsListControl()
    detail_ctrl = FindingsDetailControl(list_ctrl)

    body = VSplit([
        Window(content=list_ctrl, width=D(weight=40)),
        Window(width=1, char="│", style=f"{PRIMARY}"),
        Window(content=detail_ctrl, width=D(weight=60)),
    ])

    frame = dialog_frame(title, body, width=90, height=30, border_color=PRIMARY)
    return frame, list_ctrl
