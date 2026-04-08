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

    # Scroll step for arrow keys in the detail panel. Three lines
    # per press feels noticeably faster than one without losing
    # control — page-down still uses native window paging.
    _SCROLL_STEP = 3

    def move_cursor_down(self) -> None:
        # "down" = scroll further into the content = bigger offset.
        self._scroll_offset = min(
            self._scroll_offset + self._SCROLL_STEP,
            max(0, self._line_count - 1),
        )

    def move_cursor_up(self) -> None:
        # "up" = scroll back toward the top = smaller offset.
        self._scroll_offset = max(0, self._scroll_offset - self._SCROLL_STEP)

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
    """Wrap *text* into styled lines that fit within *width*.

    Each continuation line is prefixed with two spaces of indent.

    Important invariant: every loop iteration MUST consume at least
    one character of *real* (non-leading-whitespace) text. The
    previous version of this function used ``text.rfind(" ", 0, width)``
    which would happily return a position INSIDE the continuation
    indent ("  "), producing a chunk of pure whitespace and an
    unchanged remainder — an infinite loop on any content that
    contained a word longer than *width* (file paths, URLs, long
    identifiers, hashes, ...). The findings browser hung the whole
    TUI because real findings always contain such words.
    """
    if width <= 0:
        width = 80
    if len(text) <= width:
        return [[(style, text)]]
    result: list[list[tuple[str, str]]] = []
    indent = "  "
    while text:
        if len(text) <= width:
            result.append([(style, text)])
            break
        # Skip past any leading whitespace when looking for a break
        # point — we never want to cut inside the continuation indent.
        leading_ws = len(text) - len(text.lstrip(" "))
        # Look for the rightmost space AFTER the leading whitespace
        # but within the line width.
        cut = text.rfind(" ", leading_ws + 1, width)
        if cut <= leading_ws:
            # No usable space in range — hard-cut at width. This
            # handles long unbreakable words (paths, URLs, hashes).
            cut = width
        # Safety net: cut MUST advance past leading whitespace,
        # otherwise we'd loop on the same string forever.
        if cut <= leading_ws:
            cut = leading_ws + 1
        result.append([(style, text[:cut])])
        text = text[cut:].lstrip(" ")
        if text:
            text = indent + text
    return result
