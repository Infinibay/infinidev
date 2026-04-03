"""Documentation browser — three-panel dialog for cached library docs."""

from __future__ import annotations
from typing import Any

from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import UIControl, UIContent, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D

from infinidev.ui.theme import PRIMARY, TEXT, TEXT_MUTED
from infinidev.ui.dialogs.base import dialog_frame

DIALOG_NAME = "docs_browser"


class DocsContentControl(UIControl):
    """Content viewer for documentation."""

    def __init__(self) -> None:
        self.content: str = ""

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        if not self.content:
            lines = [[(f"{TEXT_MUTED}", " Select a section")]]
        else:
            lines = []
            for line in self.content.split("\n"):
                lines.append([(f"{TEXT}", f" {line}")])

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []
        return UIContent(get_line=get_line, line_count=len(lines))


