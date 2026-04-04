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

from infinidev.ui.theme import (
    STYLE_USER_MSG, STYLE_AGENT_MSG, STYLE_SYSTEM_MSG, STYLE_THINK_MSG,
    STYLE_PENDING_MSG, STYLE_QUEUED_MSG,
    STYLE_DIFF_TITLE,
    STYLE_USER_HEADER, STYLE_AGENT_HEADER, STYLE_SYSTEM_HEADER, STYLE_THINK_HEADER,
    MSG_USER_BORDER, MSG_AGENT_BORDER, MSG_SYSTEM_BORDER, MSG_THINK_BORDER,
    MSG_PENDING_BORDER,
    DIFF_TITLE_FG, DIFF_TITLE_BG,
    TEXT, TEXT_MUTED, THINKING_FG, ACCENT,
    SENDER_COLORS, NAME_COLORS,
)

# ── Message type → style mapping ────────────────────────────────────────

_MSG_STYLES = {
    "user": STYLE_USER_MSG,
    "agent": STYLE_AGENT_MSG,
    "system": STYLE_SYSTEM_MSG,
    "think": STYLE_THINK_MSG,
    "pending": STYLE_PENDING_MSG,
    "queued": STYLE_QUEUED_MSG,
}

_HEADER_STYLES = {
    "user": STYLE_USER_HEADER,
    "agent": STYLE_AGENT_HEADER,
    "system": STYLE_SYSTEM_HEADER,
    "think": STYLE_THINK_HEADER,
    "pending": STYLE_PENDING_MSG,
    "queued": STYLE_QUEUED_MSG,
}

_BORDER_CHARS = {
    "user": ("▌", MSG_USER_BORDER),
    "agent": ("▌", MSG_AGENT_BORDER),
    "system": ("▌", MSG_SYSTEM_BORDER),
    "think": ("▌", MSG_THINK_BORDER),
    "pending": ("┊", MSG_PENDING_BORDER),
    "queued": ("▌", ACCENT),
}

# Code block style for markdown rendering
_CODE_BLOCK_STYLE = f"{TEXT_MUTED}"


def _markdown_enabled() -> bool:
    """Check if markdown message rendering is enabled."""
    try:
        from infinidev.config.settings import settings
        return settings.MARKDOWN_MESSAGES
    except Exception:
        return False


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

    def invalidate_cache(self) -> None:
        """Force re-render on next frame and scroll to bottom."""
        self._line_cache = None
        self._follow_tail = True
        self._scroll_offset = 0

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event: MouseEvent):
        """Return NotImplemented so Window handles scroll wheel natively."""
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
        """Convert all messages to a flat list of styled line fragments.

        Incremental: only renders new messages since last build.
        """
        msg_count = len(self._messages)
        if self._line_cache is not None and self._cache_len <= msg_count and self._cache_width == width:
            # Append only new messages
            for msg in self._messages[self._cache_len:]:
                self._line_cache.extend(self._render_message(msg, width))
            self._cache_len = msg_count
            lines = self._line_cache
        else:
            # Full rebuild (width changed or cache invalid)
            lines = []
            for msg in self._messages:
                lines.extend(self._render_message(msg, width))
            self._line_cache = lines
            self._cache_len = msg_count
            self._cache_width = width

        # Append thinking indicator if active (without copying the list)
        if self._show_thinking:
            total = len(lines) + 2
            thinking_1 = [(f"{THINKING_FG}", "  Infinidev is thinking...")]
            thinking_2 = [(f"{THINKING_FG}", "  ···")]

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

    def _render_message(self, msg: dict, width: int) -> list[list[tuple[str, str]]]:
        """Render a single message to a list of line fragment lists."""
        msg_type = msg.get("type", "agent")

        # Diff messages get special rendering
        if msg_type == "diff":
            return self._render_diff_message(msg, width)

        sender = msg.get("sender", "")
        text = msg.get("text", "")

        border_char, border_color = _BORDER_CHARS.get(msg_type, ("▌", TEXT_MUTED))
        body_style = _MSG_STYLES.get(msg_type, f"{TEXT}")

        # Extract bg color from the body style for full-width fill
        bg_part = ""
        for part in body_style.split():
            if part.startswith("bg:"):
                bg_part = part
                break
        fill_style = bg_part  # just the bg for padding spaces

        # Determine header color: sender-specific override, or type-based
        header_style = NAME_COLORS.get(sender, "")
        if not header_style:
            header_style = _HEADER_STYLES.get(msg_type, f"{TEXT} bold")
        else:
            header_style = f"{header_style} bold"
        # Add bg to header too
        if bg_part and bg_part not in header_style:
            header_style = f"{header_style} {bg_part}"

        # Add bg to border style
        border_style = f"{border_color} {bg_part}" if bg_part else f"{border_color}"

        lines: list[list[tuple[str, str]]] = []

        # Header line
        suffix = " (pending):" if msg_type == "pending" else ":"
        header_text = f"{sender}{suffix}"
        header_used = 2 + len(header_text)  # border char + space + header
        lines.append([
            (border_style, f"{border_char} "),
            (header_style, header_text),
            (fill_style, " " * max(0, width - header_used)),
        ])

        # Body lines — word-wrap to width minus border indent
        content_width = max(width - 3, 20)  # 3 = border + space + margin

        use_markdown = (
            msg_type in ("agent", "think")
            and _markdown_enabled()
        )

        in_code_block = False
        for raw_line in text.split("\n"):
            if use_markdown:
                # Track code block state for plain rendering inside fences
                if raw_line.rstrip().startswith("```"):
                    in_code_block = not in_code_block
                    from infinidev.ui.controls.markdown_render import render_markdown_line
                    frags = render_markdown_line(raw_line, body_style, bg_part)
                    used = 2 + sum(len(t) for _, t in frags)
                    lines.append(
                        [(border_style, f"{border_char} ")]
                        + frags
                        + [(fill_style, " " * max(0, width - used))]
                    )
                    continue

                if in_code_block:
                    # Inside code block — render as plain code with code style
                    code_style = f"{_CODE_BLOCK_STYLE} {bg_part}" if bg_part else _CODE_BLOCK_STYLE
                    while len(raw_line) > content_width:
                        chunk = raw_line[:content_width]
                        used = 2 + len(chunk)
                        lines.append([
                            (border_style, f"{border_char} "),
                            (code_style, chunk),
                            (fill_style, " " * max(0, width - used)),
                        ])
                        raw_line = raw_line[content_width:]
                    used = 2 + len(raw_line)
                    lines.append([
                        (border_style, f"{border_char} "),
                        (code_style, raw_line),
                        (fill_style, " " * max(0, width - used)),
                    ])
                    continue

                # Markdown line — parse into styled fragments
                from infinidev.ui.controls.markdown_render import render_markdown_line
                frags = render_markdown_line(raw_line, body_style, bg_part)
                used = 2 + sum(len(t) for _, t in frags)
                # Simple overflow: if too wide, fall back to plain wrap
                if used > width:
                    # Fall back to plain text wrapping for very long lines
                    while len(raw_line) > content_width:
                        chunk = raw_line[:content_width]
                        u = 2 + len(chunk)
                        lines.append([
                            (border_style, f"{border_char} "),
                            (body_style, chunk),
                            (fill_style, " " * max(0, width - u)),
                        ])
                        raw_line = raw_line[content_width:]
                    used = 2 + len(raw_line)
                    lines.append([
                        (border_style, f"{border_char} "),
                        (body_style, raw_line),
                        (fill_style, " " * max(0, width - used)),
                    ])
                else:
                    lines.append(
                        [(border_style, f"{border_char} ")]
                        + frags
                        + [(fill_style, " " * max(0, width - used))]
                    )
            else:
                # Plain text rendering (original behavior)
                while len(raw_line) > content_width:
                    chunk = raw_line[:content_width]
                    used = 2 + len(chunk)
                    lines.append([
                        (border_style, f"{border_char} "),
                        (body_style, chunk),
                        (fill_style, " " * max(0, width - used)),
                    ])
                    raw_line = raw_line[content_width:]
                used = 2 + len(raw_line)
                lines.append([
                    (border_style, f"{border_char} "),
                    (body_style, raw_line),
                    (fill_style, " " * max(0, width - used)),
                ])

        # Blank line after message (no fill — separator)
        lines.append([("", "")])

        return lines

    def _render_diff_message(self, msg: dict, width: int) -> list[list[tuple[str, str]]]:
        """Render a file change diff message with colorized diff output."""
        from infinidev.ui.controls.file_diff import colorize_diff_fragments

        header_text = msg.get("text", "")
        diff_text = msg.get("diff_text", "")
        title_bg = f"bg:{DIFF_TITLE_BG}"

        lines: list[list[tuple[str, str]]] = []

        # Header: colored title bar showing filename and action
        pad = " " * max(0, width - len(header_text) - 2)
        lines.append([(f"{DIFF_TITLE_FG} {title_bg} bold", f" {header_text}{pad} ")])

        # Diff lines with syntax coloring
        if diff_text:
            diff_lines = colorize_diff_fragments(diff_text)
            for diff_line in diff_lines:
                lines.append(diff_line)

        # Blank separator
        lines.append([("", "")])

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
        if error:
            path = detail or "?"
            return f"create {path}\n  x {error}"
        return ""  # Diff widget handles successful creates

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
