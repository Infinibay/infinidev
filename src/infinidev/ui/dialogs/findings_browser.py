"""Findings browser — two-panel dialog for browsing findings/knowledge."""

from __future__ import annotations

from prompt_toolkit.layout.containers import VSplit, Window, ScrollOffsets
from prompt_toolkit.layout.dimension import Dimension as D

from infinidev.ui.theme import PRIMARY
from infinidev.ui.dialogs.base import dialog_frame
from infinidev.ui.controls.clickable_scrollbar import scrollable_window

DIALOG_NAME = "findings_browser"

from infinidev.ui.dialogs.findings_list_control import FindingsListControl
from infinidev.ui.dialogs.findings_detail_control import FindingsDetailControl

def create_findings_browser(title: str = "Findings"):
    """Create the findings browser dialog."""
    list_ctrl = FindingsListControl()
    detail_ctrl = FindingsDetailControl(list_ctrl)

    _, list_container = scrollable_window(
        list_ctrl, display_arrows=False, width=D(weight=40),
    )
    _, detail_container = scrollable_window(
        detail_ctrl, display_arrows=True, width=D(weight=60),
        scroll_offsets=ScrollOffsets(top=1, bottom=1),
    )

    body = VSplit([
        list_container,
        Window(width=1, char="│", style=f"{PRIMARY}"),
        detail_container,
    ])

    frame = dialog_frame(title, body, width=90, height=30, border_color=PRIMARY)
    return frame, list_ctrl
