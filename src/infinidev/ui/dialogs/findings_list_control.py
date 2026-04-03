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


class FindingsListControl(UIControl):
    """Selectable findings list."""

    def __init__(self) -> None:
        self.findings: list[dict[str, Any]] = []
        self.cursor: int = 0

    def move_cursor(self, delta: int) -> None:
        if self.findings:
            self.cursor = max(0, min(len(self.findings) - 1, self.cursor + delta))

    def get_selected(self) -> dict | None:
        if 0 <= self.cursor < len(self.findings):
            return self.findings[self.cursor]
        return None

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        lines = []
        for i, f in enumerate(self.findings):
            icon = _TYPE_ICONS.get(f.get("finding_type", ""), "*")
            topic = f.get("topic", "?")[:width - 6]
            style = f"bg:{PRIMARY} #ffffff bold" if i == self.cursor else f"{TEXT}"
            lines.append([(style, f" {icon} {topic}")])

        if not lines:
            lines = [[(f"{TEXT_MUTED}", " No findings")]]

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []
        return UIContent(get_line=get_line, line_count=len(lines))


