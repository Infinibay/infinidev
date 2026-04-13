"""Chat history control — renders all messages as FormattedText.

This replaces the Textual ChatHistory(VerticalScroll) that mounted Static
widgets. Here, messages are plain dicts in a list. The UIControl only
generates FormattedText for the visible viewport, giving us natural
viewport culling without the ±200px visibility hack.
"""

from __future__ import annotations

from typing import Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from infinidev.ui.theme import TEXT_MUTED, THINKING_FG, PRIMARY

import logging
import os

_copy_log = logging.getLogger("infinidev.copy_debug")
_copy_log_path = os.path.expanduser("~/.infinidev/copy_debug.log")
if not _copy_log.handlers:
    _fh = logging.FileHandler(_copy_log_path, mode="w")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _copy_log.addHandler(_fh)
    _copy_log.setLevel(logging.DEBUG)

# Style for the selected message in selection mode
_SELECT_HIGHLIGHT = f"#ffffff bg:{PRIMARY} bold"


class ChatHistoryControl(UIControl):
    """Custom UIControl that renders chat messages as formatted text lines.

    Messages are stored as a list of dicts:
        {"sender": str, "text": str, "type": str, "is_diff": bool, ...}
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages
        self._line_cache: list[list[tuple[str, str]]] | None = None
        self._cache_len = 0
        self._cache_width = 0
        self._show_thinking = False
        self._follow_tail: bool = True  # stick to bottom
        self._scroll_offset: int = 0    # lines from bottom (when not following)
        self._line_count: int = 0
        # Generalized click targets: line index → callback
        self._clickable_lines: dict[int, Any] = {}
        # Group collapse state: start_index of group → collapsed bool
        self._group_states: dict[int, bool] = {}
        # Message selection mode state
        self._select_mode: bool = False
        self._selected_msg_index: int = -1

    def invalidate_cache(self) -> None:
        """Force re-render on next frame and scroll to bottom."""
        self._line_cache = None
        self._follow_tail = True
        self._scroll_offset = 0

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event: MouseEvent):
        """Handle clicks on any registered clickable line; delegate rest to Window."""
        # Handle both MOUSE_UP and MOUSE_DOWN — some terminals (especially
        # via SSH, tmux, or screen) may only report one of the two.
        if mouse_event.event_type in (MouseEventType.MOUSE_UP, MouseEventType.MOUSE_DOWN):
            line_idx = mouse_event.position.y
            _copy_log.debug(
                "mouse %s at y=%d  clickable_keys=%s  match=%s",
                mouse_event.event_type, line_idx,
                sorted(self._clickable_lines.keys())[:20],
                line_idx in self._clickable_lines,
            )
            callback = self._clickable_lines.get(line_idx)
            if callback is not None:
                # Only fire on MOUSE_UP (preferred) or MOUSE_DOWN as fallback.
                # Track last handled click to avoid double-firing.
                import time
                now = time.monotonic()
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    self._last_click_time = now
                    _copy_log.debug("firing callback via MOUSE_UP for line %d", line_idx)
                    callback()
                    self._line_cache = None
                    return None
                elif mouse_event.event_type == MouseEventType.MOUSE_DOWN:
                    # Only use MOUSE_DOWN if no MOUSE_UP arrived recently
                    last = getattr(self, "_last_click_time", 0.0)
                    if now - last > 0.5:
                        self._last_click_time = now
                        _copy_log.debug("firing callback via MOUSE_DOWN for line %d", line_idx)
                        callback()
                        self._line_cache = None
                        return None
        return NotImplemented

    def move_cursor_down(self) -> None:
        """Called by Window._scroll_down() on mouse wheel down."""
        if self._scroll_offset > 0:
            self._scroll_offset -= 1
        if self._scroll_offset == 0:
            self._follow_tail = True

    def move_cursor_up(self) -> None:
        """Called by Window._scroll_up() on mouse wheel up."""
        self._follow_tail = False
        self._scroll_offset = min(
            self._scroll_offset + 1,
            max(0, self._line_count - 1),
        )

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("pageup")
        def _pgup(event):
            self._follow_tail = False
            self._scroll_offset = min(
                self._scroll_offset + 15,
                max(0, self._line_count - 1),
            )

        @kb.add("pagedown")
        def _pgdn(event):
            self._scroll_offset = max(0, self._scroll_offset - 15)
            if self._scroll_offset == 0:
                self._follow_tail = True

        @kb.add("home")
        def _home(event):
            self._follow_tail = False
            self._scroll_offset = max(0, self._line_count - 1)

        @kb.add("end")
        def _end(event):
            self._follow_tail = True
            self._scroll_offset = 0

        # Selection mode navigation
        @kb.add("up")
        def _select_up(event):
            if self._select_mode:
                self.select_prev_message()

        @kb.add("down")
        def _select_down(event):
            if self._select_mode:
                self.select_next_message()

        @kb.add("enter")
        def _select_confirm(event):
            if self._select_mode:
                self.copy_selected_message()

        return kb

    @property
    def show_thinking(self) -> bool:
        return self._show_thinking

    @show_thinking.setter
    def show_thinking(self, value: bool) -> None:
        if value != self._show_thinking:
            self._show_thinking = value

    def preferred_width(self, max_available_width: int) -> int | None:
        return None  # fill available

    def preferred_height(self, width: int, max_available_height: int,
                         wrap_lines: bool, get_line_prefix) -> int | None:
        return None  # fill available

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        """Build the displayable content from messages."""
        lines, line_count, custom_get_line = self._build_lines(width)
        if line_count == 0:
            lines = [[(f"{TEXT_MUTED}", " Welcome to Infinidev! Type your instruction or /help.")]]
            line_count = 1
            custom_get_line = None

        self._line_count = line_count

        if self._follow_tail:
            cursor_y = max(0, line_count - 1)
        else:
            cursor_y = max(0, line_count - 1 - self._scroll_offset)

        if custom_get_line:
            getter = custom_get_line
        else:
            def getter(i: int) -> list[tuple[str, str]]:
                if 0 <= i < len(lines):
                    return lines[i]
                return []

        return UIContent(
            get_line=getter,
            line_count=line_count,
            cursor_position=Point(x=0, y=cursor_y),
            show_cursor=False,
        )

    def _build_lines(self, width: int) -> list[list[tuple[str, str]]]:
        """Build flat line list with message grouping.

        Consecutive same-type messages are grouped. Groups of 2+ show a
        clickable header; when collapsed only the last message is visible.
        """
        from infinidev.ui.controls.message_groups import identify_groups
        from infinidev.ui.controls.message_widgets import get_widget

        msg_count = len(self._messages)
        # Invalidate cache in select mode (highlight changes without msg changes)
        if self._select_mode:
            self._line_cache = None
        if self._line_cache is not None and self._cache_len == msg_count and self._cache_width == width:
            lines = self._line_cache
        else:
            # Full rebuild
            lines = []
            self._clickable_lines = {}
            self._msg_line_ranges: dict[int, tuple[int, int]] = {}
            groups = identify_groups(self._messages)

            for group in groups:
                widget = get_widget(group.msg_type)
                if widget is None:
                    # Unknown type — render each message with fallback
                    for msg in group.messages:
                        lines.extend(self._render_fallback(msg, width))
                    continue

                if group.is_group:
                    # Multi-message group: render header + visible messages
                    collapsed = self._group_states.get(group.start_index, True)

                    header_result = widget.render_group_header(
                        len(group.messages), collapsed, width,
                    )
                    header_start = len(lines)
                    lines.extend(header_result.lines)

                    # Register group header click
                    def _toggle_group(idx=group.start_index):
                        self._group_states[idx] = not self._group_states.get(idx, True)
                    self._clickable_lines[header_start] = _toggle_group

                    if collapsed:
                        # Only render last message
                        visible_msgs = [group.messages[-1]]
                    else:
                        visible_msgs = group.messages

                    for msg in visible_msgs:
                        msg_idx = self._messages.index(msg)
                        result = widget.render(msg, width)
                        start = len(lines)
                        lines.extend(result.lines)
                        self._msg_line_ranges[msg_idx] = (start, len(lines))
                        # Register per-message clickable offsets
                        for offset, cb in result.clickable_offsets.items():
                            self._clickable_lines[start + offset] = cb
                else:
                    # Single message — no group header
                    msg = group.messages[0]
                    msg_idx = self._messages.index(msg)
                    result = widget.render(msg, width)
                    start = len(lines)
                    lines.extend(result.lines)
                    self._msg_line_ranges[msg_idx] = (start, len(lines))
                    for offset, cb in result.clickable_offsets.items():
                        self._clickable_lines[start + offset] = cb

            # Apply selection highlight
            if self._select_mode and self._selected_msg_index in self._msg_line_ranges:
                sel_start, sel_end = self._msg_line_ranges[self._selected_msg_index]
                for i in range(sel_start, sel_end):
                    if i < len(lines) and lines[i] and lines[i] != [("", "")]:
                        # Prepend reverse-video marker to highlight
                        lines[i] = [(_SELECT_HIGHLIGHT, frag_text)
                                     if frag_text.strip() else (style, frag_text)
                                     for style, frag_text in lines[i]]

            self._line_cache = lines
            self._cache_len = msg_count
            self._cache_width = width

        # Append thinking indicator if active
        if self._show_thinking:
            total = len(lines) + 2
            thinking_1 = [(f"{THINKING_FG}", "  Infinidev is thinking...")]
            thinking_2 = [(f"{THINKING_FG}", "  \u00b7\u00b7\u00b7")]

            def get_line_with_thinking(i: int) -> list[tuple[str, str]]:
                if i < len(lines):
                    return lines[i]
                if i == len(lines):
                    return thinking_1
                if i == len(lines) + 1:
                    return thinking_2
                return []

            return lines, total, get_line_with_thinking

        return lines, len(lines), None

    # ── Selection mode ──────────────────────────────────────────────

    def enter_select_mode(self) -> None:
        """Enter message selection mode, starting at the last message."""
        visible = [i for i, m in enumerate(self._messages)
                    if m.get("visible", True) and m.get("type") in ("user", "agent")]
        if not visible:
            return
        self._select_mode = True
        self._selected_msg_index = visible[-1]
        self._line_cache = None  # force redraw with highlight

    def exit_select_mode(self) -> None:
        """Leave message selection mode."""
        self._select_mode = False
        self._selected_msg_index = -1
        self._line_cache = None

    def select_prev_message(self) -> None:
        """Move selection to the previous visible message."""
        if not self._select_mode:
            return
        visible = [i for i, m in enumerate(self._messages)
                    if m.get("visible", True) and m.get("type") in ("user", "agent")]
        if not visible:
            return
        try:
            pos = visible.index(self._selected_msg_index)
        except ValueError:
            pos = len(visible) - 1
        if pos > 0:
            self._selected_msg_index = visible[pos - 1]
            self._line_cache = None

    def select_next_message(self) -> None:
        """Move selection to the next visible message."""
        if not self._select_mode:
            return
        visible = [i for i, m in enumerate(self._messages)
                    if m.get("visible", True) and m.get("type") in ("user", "agent")]
        if not visible:
            return
        try:
            pos = visible.index(self._selected_msg_index)
        except ValueError:
            pos = 0
        if pos < len(visible) - 1:
            self._selected_msg_index = visible[pos + 1]
            self._line_cache = None

    def copy_selected_message(self) -> bool:
        """Copy the currently selected message to clipboard. Returns True on success."""
        if not self._select_mode or self._selected_msg_index < 0:
            return False
        msg = self._messages[self._selected_msg_index]
        from infinidev.ui.clipboard import copy_to_clipboard
        from infinidev.ui.controls.message_widgets import _copy_feedback
        ok = copy_to_clipboard(msg.get("text", ""))
        if _copy_feedback:
            _copy_feedback(ok)
        self.exit_select_mode()
        return ok

    @property
    def select_mode(self) -> bool:
        return self._select_mode

    def _render_fallback(self, msg: dict, width: int) -> list[list[tuple[str, str]]]:
        """Minimal fallback for unknown message types."""
        if not msg.get("visible", True):
            return []
        text = msg.get("text", "")
        sender = msg.get("sender", "")
        lines = [
            [(f"{TEXT_MUTED}", f"  {sender}: {text[:width - 4]}")],
            [("", "")],
        ]
        return lines


def format_tool_chat_message(tool_name: str, detail: str,
                             error: str, output: str) -> str:
    """Format a tool call as a chat message string. Returns empty to skip."""
    if tool_name == "execute_command":
        cmd = detail or "..."
        msg = f"$ {cmd}"
        if error:
            msg += f"\n  x {error}"
        elif output:
            msg += f"\n  {output}"
        return msg

    if tool_name == "create_file":
        path = detail or "?"
        if error:
            return f"create {path}\n  x {error}"
        return f"+ created {path}"

    if tool_name in ("replace_lines", "edit_symbol", "add_symbol", "remove_symbol"):
        if error:
            path = detail or "?"
            label = {"replace_lines": "edit", "edit_symbol": "edit",
                     "add_symbol": "add", "remove_symbol": "remove"}.get(tool_name, "edit")
            return f"{label} {path}\n  x {error}"
        return ""

    if tool_name == "git_commit":
        return f"commit {detail}" if detail else "commit"

    if tool_name == "git_branch":
        return f"branch {detail}" if detail else "branch"

    return ""
