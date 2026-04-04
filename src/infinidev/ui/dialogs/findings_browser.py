"""Findings browser — two-panel dialog for browsing findings/knowledge."""

from __future__ import annotations

from prompt_toolkit.layout.containers import VSplit, Window
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.margins import ScrollbarMargin

from infinidev.ui.theme import PRIMARY
from infinidev.ui.dialogs.base import dialog_frame

DIALOG_NAME = "findings_browser"

from infinidev.ui.dialogs.findings_list_control import FindingsListControl
from infinidev.ui.dialogs.findings_detail_control import FindingsDetailControl

def create_findings_browser(title: str = "Findings"):
    """Create the findings browser dialog."""
    list_ctrl = FindingsListControl()
    detail_ctrl = FindingsDetailControl(list_ctrl)

    body = VSplit([
        Window(content=list_ctrl, width=D(weight=40),
               right_margins=[ScrollbarMargin()]),
        Window(width=1, char="│", style=f"{PRIMARY}"),
        Window(
            content=detail_ctrl,
            width=D(weight=60),
            wrap_lines=True,
            right_margins=[ScrollbarMargin()],
        ),
    ])

    frame = dialog_frame(title, body, width=90, height=30, border_color=PRIMARY)
    return frame, list_ctrl
