"""Documentation browser — three-panel dialog for cached library docs."""

from __future__ import annotations
from typing import Any

from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import UIControl, UIContent, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D

from infinidev.ui.theme import PRIMARY, TEXT, TEXT_MUTED
from infinidev.ui.dialogs.base import dialog_frame

DIALOG_NAME = "docs_browser"


class DocsListControl(UIControl):
    """Generic selectable list for libraries or sections."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.cursor: int = 0

    def move_cursor(self, delta: int) -> None:
        if self.items:
            self.cursor = max(0, min(len(self.items) - 1, self.cursor + delta))

    def get_selected(self) -> dict | None:
        if 0 <= self.cursor < len(self.items):
            return self.items[self.cursor]
        return None

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        lines = []
        for i, item in enumerate(self.items):
            label = item.get("label", "?")[:width - 4]
            style = f"bg:{PRIMARY} #ffffff bold" if i == self.cursor else f"{TEXT}"
            lines.append([(style, f"  {label}")])

        if not lines:
            lines = [[(f"{TEXT_MUTED}", "  (empty)")]]

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []
        return UIContent(get_line=get_line, line_count=len(lines))


