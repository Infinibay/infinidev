"""Findings detail control — right panel of the findings browser."""

from __future__ import annotations

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent

from infinidev.ui.theme import TEXT, TEXT_MUTED
from infinidev.ui.dialogs.findings_list_control import FindingsListControl


class FindingsDetailControl(UIControl):
    """Detail view for the selected finding with scroll and word wrap."""

    def __init__(self, list_ctrl: FindingsListControl) -> None:
        self._list = list_ctrl
        self._scroll_offset: int = 0
        self._line_count: int = 0
        self._prev_finding_id: int | None = None

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event: MouseEvent):
        return NotImplemented

    def move_cursor_down(self) -> None:
        if self._scroll_offset > 0:
            self._scroll_offset -= 1

    def move_cursor_up(self) -> None:
        self._scroll_offset = min(
            self._scroll_offset + 1,
            max(0, self._line_count - 1),
        )

    def preferred_width(self, max_available_width: int) -> int | None:
        # Don't request any width — let the VSplit decide via weights
        return None

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        finding = self._list.get_selected()

        # Reset scroll when switching findings
        fid = id(finding) if finding else None
        if fid != self._prev_finding_id:
            self._scroll_offset = 0
            self._prev_finding_id = fid

        if not finding:
            lines = [[(f"{TEXT_MUTED}", " Select a finding")]]
        else:
            lines = []
            lines.extend(_wrap_styled(f"{TEXT} bold", f" {finding.get('topic', '?')}", width))
            meta = (f" Type: {finding.get('finding_type', '?')} | "
                    f"Confidence: {finding.get('confidence', '?')} | "
                    f"Status: {finding.get('status', '?')}")
            lines.extend(_wrap_styled(f"{TEXT_MUTED}", meta, width))
            lines.append([("", "")])
            content = finding.get("content", "")
            for line in content.split("\n"):
                lines.extend(_wrap_styled(f"{TEXT}", f" {line}", width))

        self._line_count = len(lines)
        cursor_y = max(0, self._line_count - 1 - self._scroll_offset)

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=cursor_y),
        )


def _wrap_styled(style: str, text: str, width: int) -> list[list[tuple[str, str]]]:
    """Wrap text into multiple styled lines that fit within *width*."""
    if width <= 0:
        width = 80
    if len(text) <= width:
        return [[(style, text)]]
    result = []
    while text:
        if len(text) <= width:
            result.append([(style, text)])
            break
        cut = text.rfind(" ", 0, width)
        if cut <= 0:
            cut = width
        result.append([(style, text[:cut])])
        text = text[cut:].lstrip(" ")
        if text:
            text = "  " + text
    return result
