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


class FindingsDetailControl(UIControl):
    """Detail view for the selected finding."""

    def __init__(self, list_ctrl: FindingsListControl) -> None:
        self._list = list_ctrl

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        finding = self._list.get_selected()
        if not finding:
            lines = [[(f"{TEXT_MUTED}", " Select a finding")]]
        else:
            lines = []
            lines.append([(f"{TEXT} bold", f" {finding.get('topic', '?')}")])
            lines.append([(f"{TEXT_MUTED}",
                           f" Type: {finding.get('finding_type', '?')} | "
                           f"Confidence: {finding.get('confidence', '?')} | "
                           f"Status: {finding.get('status', '?')}")])
            lines.append([("", "")])
            content = finding.get("content", "")
            for line in content.split("\n"):
                lines.append([(f"{TEXT}", f" {line}")])

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []
        return UIContent(get_line=get_line, line_count=len(lines))


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
