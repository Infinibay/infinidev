"""Findings detail control — right panel of the findings browser."""

from __future__ import annotations

from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent

from infinidev.ui.theme import TEXT, TEXT_MUTED
from infinidev.ui.dialogs.findings_list_control import FindingsListControl


class FindingsDetailControl(UIControl):
    """Detail view for the selected finding with scroll support."""

    def __init__(self, list_ctrl: FindingsListControl) -> None:
        self._list = list_ctrl

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event: MouseEvent):
        """Let Window handle scroll wheel."""
        return NotImplemented

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
