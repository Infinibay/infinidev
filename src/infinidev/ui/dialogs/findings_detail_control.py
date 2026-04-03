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


