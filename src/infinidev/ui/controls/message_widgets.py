"""Message widget system — modular rendering for chat message types.

Each message type (user, agent, system, think, diff, etc.) is rendered
by a registered widget.  Adding a new type = one class + one register()
call.  The ChatHistoryControl delegates to widgets via get_widget().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from infinidev.ui.theme import (
    STYLE_USER_MSG, STYLE_AGENT_MSG, STYLE_SYSTEM_MSG, STYLE_THINK_MSG,
    STYLE_PENDING_MSG, STYLE_QUEUED_MSG,
    STYLE_USER_HEADER, STYLE_AGENT_HEADER, STYLE_SYSTEM_HEADER, STYLE_THINK_HEADER,
    MSG_USER_BORDER, MSG_AGENT_BORDER, MSG_SYSTEM_BORDER, MSG_THINK_BORDER,
    MSG_PENDING_BORDER,
    DIFF_TITLE_FG, DIFF_TITLE_BG,
    TEXT, TEXT_MUTED, ACCENT,
    NAME_COLORS,
)

# Copy button labels shown on message headers
COPY_ICON = " [⧉] "
COPY_ICON_STYLE = f"{TEXT_MUTED}"
COPY_OK_ICON = " [✓ Copied] "
COPY_OK_STYLE = "#00ff00 bold"
COPY_FAIL_ICON = " [✗ Failed] "
COPY_FAIL_STYLE = "#ff4444 bold"
_COPY_FEEDBACK_DURATION = 2.0  # seconds the icon stays visible

# Module-level callback set by the app — receives (ok: bool) after a copy
_copy_feedback: Callable[[bool], None] | None = None

# Tracks recently-copied messages: id(msg) → (timestamp, ok)
import time as _time
_copy_highlight: dict[int, tuple[float, bool]] = {}


def set_copy_feedback(cb: Callable[[bool], None]) -> None:
    """Register a callback invoked after every copy attempt (ok=True/False)."""
    global _copy_feedback
    _copy_feedback = cb


# ── Render result ──────────────────────────────────────────────────────

@dataclass
class RenderResult:
    """Output from a widget's render() method."""
    lines: list[list[tuple[str, str]]]
    # Relative line offsets that are clickable → callback
    clickable_offsets: dict[int, Callable[[], None]] = field(default_factory=dict)


# ── Protocol ───────────────────────────────────────────────────────────

class MessageWidget(Protocol):
    msg_type: str
    group_label: str

    def render(self, msg: dict[str, Any], width: int) -> RenderResult: ...
    def render_group_header(self, count: int, collapsed: bool, width: int) -> RenderResult: ...


# ── Registry ───────────────────────────────────────────────────────────

_WIDGETS: dict[str, MessageWidget] = {}


def register(widget: MessageWidget) -> None:
    """Register a widget instance for a message type."""
    _WIDGETS[widget.msg_type] = widget


def get_widget(msg_type: str) -> MessageWidget | None:
    """Look up the widget for a message type."""
    return _WIDGETS.get(msg_type)


# ── Shared helpers ─────────────────────────────────────────────────────

_CODE_BLOCK_STYLE = f"{TEXT_MUTED}"


def _markdown_enabled() -> bool:
    try:
        from infinidev.config.settings import settings
        return settings.MARKDOWN_MESSAGES
    except Exception:
        return False


def _side_by_side_enabled() -> bool:
    try:
        from infinidev.config.settings import settings
        return settings.DIFF_DISPLAY_MODE == "side_by_side"
    except Exception:
        return False


def _render_bordered_body(
    text: str, width: int,
    border_char: str, border_style: str, body_style: str, fill_style: str,
    use_markdown: bool = False,
) -> list[list[tuple[str, str]]]:
    """Render message body lines with border, wrapping, and optional markdown."""
    content_width = max(width - 3, 20)
    lines: list[list[tuple[str, str]]] = []
    bg_part = ""
    for part in fill_style.split():
        if part.startswith("bg:"):
            bg_part = part
            break

    in_code_block = False
    for raw_line in text.split("\n"):
        if use_markdown:
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

            from infinidev.ui.controls.markdown_render import render_markdown_line
            frags = render_markdown_line(raw_line, body_style, bg_part)
            used = 2 + sum(len(t) for _, t in frags)
            if used > width:
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

    return lines


# ── BorderedWidget ─────────────────────────────────────────────────────

# Style tables (same data as the old _MSG_STYLES etc.)
_BORDERED_CONFIG: dict[str, dict[str, str]] = {
    "user":    {"body": STYLE_USER_MSG,    "header": STYLE_USER_HEADER,   "border_char": "\u258c", "border_color": MSG_USER_BORDER},
    "agent":   {"body": STYLE_AGENT_MSG,   "header": STYLE_AGENT_HEADER,  "border_char": "\u258c", "border_color": MSG_AGENT_BORDER},
    "system":  {"body": STYLE_SYSTEM_MSG,  "header": STYLE_SYSTEM_HEADER, "border_char": "\u258c", "border_color": MSG_SYSTEM_BORDER},
    "think":   {"body": STYLE_THINK_MSG,   "header": STYLE_THINK_HEADER,  "border_char": "\u258c", "border_color": MSG_THINK_BORDER},
    "pending": {"body": STYLE_PENDING_MSG, "header": STYLE_PENDING_MSG,   "border_char": "\u250a", "border_color": MSG_PENDING_BORDER},
    "queued":  {"body": STYLE_QUEUED_MSG,  "header": STYLE_QUEUED_MSG,    "border_char": "\u258c", "border_color": ACCENT},
}


class BorderedWidget:
    """Renders bordered messages: user, agent, system, think, pending, queued."""

    def __init__(self, msg_type: str, group_label: str) -> None:
        self.msg_type = msg_type
        self.group_label = group_label
        cfg = _BORDERED_CONFIG[msg_type]
        self._body_style = cfg["body"]
        self._header_style_default = cfg["header"]
        self._border_char = cfg["border_char"]
        self._border_color = cfg["border_color"]

    def render(self, msg: dict[str, Any], width: int) -> RenderResult:
        sender = msg.get("sender", "")
        text = msg.get("text", "")

        # Extract bg
        bg_part = ""
        for part in self._body_style.split():
            if part.startswith("bg:"):
                bg_part = part
                break
        fill_style = bg_part

        # Header style
        header_style = NAME_COLORS.get(sender, "")
        if not header_style:
            header_style = self._header_style_default
        else:
            header_style = f"{header_style} bold"
        if bg_part and bg_part not in header_style:
            header_style = f"{header_style} {bg_part}"

        border_style = f"{self._border_color} {bg_part}" if bg_part else f"{self._border_color}"

        lines: list[list[tuple[str, str]]] = []
        clickable: dict[int, Callable[[], None]] = {}

        # Header line (with copy button on the right)
        suffix = " (pending):" if self.msg_type == "pending" else ":"
        header_text = f"{sender}{suffix}"

        # Check if this message was recently copied → show feedback icon
        now = _time.monotonic()
        hl = _copy_highlight.get(id(msg))
        if hl and (now - hl[0]) < _COPY_FEEDBACK_DURATION:
            if hl[1]:
                copy_label, copy_style = COPY_OK_ICON, COPY_OK_STYLE
            else:
                copy_label, copy_style = COPY_FAIL_ICON, COPY_FAIL_STYLE
        else:
            copy_label, copy_style = COPY_ICON, COPY_ICON_STYLE

        header_used = 2 + len(header_text) + len(copy_label)
        gap = max(0, width - header_used)
        lines.append([
            (border_style, f"{self._border_char} "),
            (header_style, header_text),
            (fill_style, " " * gap),
            (copy_style, copy_label),
        ])

        # Register header line (offset 0) as clickable → copy message text
        def _copy_msg(m=msg):
            from infinidev.ui.clipboard import copy_to_clipboard
            ok = copy_to_clipboard(m.get("text", ""))
            # Store highlight so next render shows ✓/✗ icon
            _copy_highlight[id(m)] = (_time.monotonic(), ok)
            if _copy_feedback:
                _copy_feedback(ok)
        clickable[0] = _copy_msg

        # Body
        # Skip markdown while the message is mid-stream — unclosed
        # ``**bold`` / backticks / headers half-formed render as literal
        # text and look broken. Once the chat agent calls
        # notify_stream_end the flag flips and we re-render with full
        # markdown applied.
        use_markdown = (
            self.msg_type in ("agent", "think")
            and _markdown_enabled()
            and not msg.get("streaming", False)
        )
        lines.extend(_render_bordered_body(
            text, width, self._border_char, border_style, self._body_style,
            fill_style, use_markdown,
        ))

        # Blank separator
        lines.append([("", "")])
        return RenderResult(lines=lines, clickable_offsets=clickable)

    def render_group_header(self, count: int, collapsed: bool, width: int) -> RenderResult:
        arrow = "\u25b6" if collapsed else "\u25bc"
        label = f" {arrow} {self.group_label} ({count})"
        bg_part = ""
        for part in self._body_style.split():
            if part.startswith("bg:"):
                bg_part = part
                break
        style = f"{self._border_color} bold {bg_part}"
        pad = " " * max(0, width - len(label))
        lines = [[(style, f"{label}{pad}")]]
        return RenderResult(lines=lines)


# ── DiffWidget ─────────────────────────────────────────────────────────

class DiffWidget:
    """Renders file change diffs with per-message collapse."""

    msg_type = "diff"
    group_label = "Diffs"

    def render(self, msg: dict[str, Any], width: int) -> RenderResult:
        from infinidev.ui.controls.file_diff import (
            colorize_diff_fragments,
            colorize_diff_side_by_side,
        )

        header_text = msg.get("text", "")
        diff_text = msg.get("diff_text", "")
        collapsed = msg.get("collapsed", True)
        title_bg = f"bg:{DIFF_TITLE_BG}"
        arrow = "\u25b6" if collapsed else "\u25bc"

        lines: list[list[tuple[str, str]]] = []

        pad = " " * max(0, width - len(header_text) - 4)
        lines.append([(f"{DIFF_TITLE_FG} {title_bg} bold", f" {arrow} {header_text}{pad}")])

        if not collapsed and diff_text:
            col_width = max(20, (width - 3) // 2)  # subtract separator width
            if _side_by_side_enabled():
                for diff_line in colorize_diff_side_by_side(diff_text, column_width=col_width):
                    lines.append(diff_line)
            else:
                for diff_line in colorize_diff_fragments(diff_text):
                    lines.append(diff_line)

        lines.append([("", "")])

        # Line 0 (header) is clickable — toggles this diff's collapse
        def _toggle(m=msg):
            m["collapsed"] = not m.get("collapsed", True)

        return RenderResult(lines=lines, clickable_offsets={0: _toggle})

    def render_group_header(self, count: int, collapsed: bool, width: int) -> RenderResult:
        arrow = "\u25b6" if collapsed else "\u25bc"
        label = f" {arrow} {self.group_label} ({count})"
        title_bg = f"bg:{DIFF_TITLE_BG}"
        style = f"{DIFF_TITLE_FG} {title_bg} bold"
        pad = " " * max(0, width - len(label))
        lines = [[(style, f"{label}{pad}")]]
        return RenderResult(lines=lines)


# ── ErrorWidget ────────────────────────────────────────────────────────

# Styling for error messages — red border/fg so they stand out, and a
# dim body style for the traceback so it reads as "secondary content"
# once expanded.
_ERROR_TITLE_FG = "#ff4444"
_ERROR_BODY_STYLE = f"{TEXT_MUTED}"


class ErrorWidget:
    """Renders a chat-agent error with a collapsible traceback.

    The short message (``text``) is always visible and in red. The
    ``error_traceback`` body is hidden by default and togglable by
    clicking the header — same pattern as :class:`DiffWidget`.
    """

    msg_type = "error"
    group_label = "Errors"

    def render(self, msg: dict[str, Any], width: int) -> RenderResult:
        header_text = msg.get("text", "") or "Error"
        tb_text = msg.get("error_traceback", "") or ""
        collapsed = msg.get("collapsed", True)
        arrow = "\u25b6" if collapsed else "\u25bc"

        lines: list[list[tuple[str, str]]] = []

        label = f" {arrow} {header_text}"
        pad = " " * max(0, width - len(label))
        lines.append([(f"{_ERROR_TITLE_FG} bold", f"{label}{pad}")])

        if not collapsed and tb_text:
            # Traceback lines rendered one-per-visual-line so the user
            # can read the whole stack. No wrapping — long source lines
            # just get clipped by the panel.
            for tb_line in tb_text.rstrip().splitlines():
                lines.append([(_ERROR_BODY_STYLE, f"  {tb_line}")])

        lines.append([("", "")])

        def _toggle(m=msg):
            m["collapsed"] = not m.get("collapsed", True)

        return RenderResult(lines=lines, clickable_offsets={0: _toggle})

    def render_group_header(self, count: int, collapsed: bool, width: int) -> RenderResult:
        arrow = "\u25b6" if collapsed else "\u25bc"
        label = f" {arrow} {self.group_label} ({count})"
        pad = " " * max(0, width - len(label))
        lines = [[(f"{_ERROR_TITLE_FG} bold", f"{label}{pad}")]]
        return RenderResult(lines=lines)


# ── Register built-in widgets ──────────────────────────────────────────

register(BorderedWidget("user", "Messages"))
register(BorderedWidget("agent", "Responses"))
register(BorderedWidget("system", "System"))
register(BorderedWidget("think", "Thinking"))
register(BorderedWidget("pending", "Pending"))
register(BorderedWidget("queued", "Queued"))
register(DiffWidget())
register(ErrorWidget())
