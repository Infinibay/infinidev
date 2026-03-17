"""Collapsible widget that displays a unified diff for a changed file."""

from __future__ import annotations

import os
import re

from textual.widgets import Collapsible, Static


def colorize_diff(diff_text: str) -> str:
    """Apply Rich markup to unified diff lines for colored rendering."""
    lines = []
    for line in diff_text.splitlines():
        escaped = _escape_markup(line)
        if line.startswith("+++") or line.startswith("---"):
            lines.append(f"[bold #888888]{escaped}[/bold #888888]")
        elif line.startswith("+"):
            lines.append(f"[#00ee77]{escaped}[/#00ee77]")
        elif line.startswith("-"):
            lines.append(f"[#ff5577]{escaped}[/#ff5577]")
        elif line.startswith("@@"):
            lines.append(f"[#55aaff]{escaped}[/#55aaff]")
        else:
            lines.append(escaped)
    return "\n".join(lines)


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text."""
    return re.sub(r"(\[)", r"\\\1", text)


class FileChangeDiffWidget(Collapsible):
    """A collapsible showing the cumulative diff for a single file."""

    DEFAULT_CSS = """
    FileChangeDiffWidget {
        height: auto;
        margin: 0 0 1 0;
        border-left: tall #ff8800;
        background: #1a1200;
        padding: 0;
    }
    FileChangeDiffWidget .diff-content {
        height: auto;
        padding: 0 1;
        overflow-y: auto;
        max-height: 30;
    }
    """

    def __init__(self, file_path: str, diff_text: str, action: str = "modified") -> None:
        self._file_path = file_path
        self._action = action
        self._change_count = 1
        icon = "+" if action == "created" else "~"
        basename = os.path.basename(file_path)
        title = f" {icon} {basename}"
        super().__init__(title=title, collapsed=True)
        self._diff_static = Static(
            colorize_diff(diff_text),
            classes="diff-content",
            markup=True,
        )

    def compose(self):
        yield self._diff_static

    def update_diff(self, diff_text: str, change_count: int = 1, action: str = "modified") -> None:
        """Update the diff content and title."""
        self._change_count = change_count
        self._action = action
        self._diff_static.update(colorize_diff(diff_text))

        icon = "+" if action == "created" else "~"
        basename = os.path.basename(self._file_path)
        count_str = f" ({change_count} edits)" if change_count > 1 else ""
        self.title = f" {icon} {basename}{count_str}"
