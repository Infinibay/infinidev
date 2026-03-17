"""Collapsible widget that displays a unified diff for a changed file."""

from __future__ import annotations

import os
import re

from textual.widgets import Collapsible, Static


def colorize_diff(diff_text: str) -> str:
    """Apply Rich markup to unified diff lines with line numbers."""
    lines = []
    old_num = 0
    new_num = 0
    in_hunk = False

    for line in diff_text.splitlines():
        escaped = _escape_markup(line)

        if not in_hunk and (line.startswith("---") or line.startswith("+++")):
            # File headers (only before first hunk)
            lines.append(f"[bold #888888]     {escaped}[/bold #888888]")
        elif line.startswith("@@"):
            # Hunk header — parse line numbers: @@ -old,count +new,count @@
            in_hunk = True
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line)
            if match:
                old_num = int(match.group(1)) - 1
                new_num = int(match.group(2)) - 1
            lines.append(f"[bold #55aaff]     {escaped}[/bold #55aaff]")
        elif line.startswith("-"):
            old_num += 1
            lnum = f"{old_num:>4} "
            lines.append(f"[#ff5577]{lnum}{escaped}[/#ff5577]")
        elif line.startswith("+"):
            new_num += 1
            lnum = f"{new_num:>4} "
            lines.append(f"[#00ee77]{lnum}{escaped}[/#00ee77]")
        else:
            old_num += 1
            new_num += 1
            lnum = f"{new_num:>4} "
            lines.append(f"[dim]{lnum}[/dim]{escaped}")

    return "\n".join(lines)


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text."""
    return re.sub(r"(\[)", r"\\\1", text)


class FileChangeDiffWidget(Collapsible):
    """A collapsible showing the cumulative diff for a single file."""

    DEFAULT_CSS = """
    FileChangeDiffWidget {
        height: auto;
        max-height: 50%;
        margin: 0 0 1 0;
        padding: 0;
    }
    FileChangeDiffWidget > Contents {
        height: auto;
        max-height: 30;
        overflow-y: auto;
    }
    FileChangeDiffWidget .diff-content {
        height: auto;
        padding: 0 1;
    }
    FileChangeDiffWidget CollapsibleTitle {
        background: #1a1200;
        color: #ff8800;
        padding: 0 1;
    }
    FileChangeDiffWidget CollapsibleTitle:hover {
        background: #2a2000;
    }
    """

    def __init__(self, file_path: str, diff_text: str, action: str = "modified") -> None:
        self._file_path = file_path
        self._action = action
        self._change_count = 1
        self._diff_static = Static(
            colorize_diff(diff_text),
            classes="diff-content",
            markup=True,
        )
        title = self._build_title()
        # Pass the Static as a child to Collapsible (do NOT override compose)
        super().__init__(self._diff_static, title=title, collapsed=True)

    def _build_title(self) -> str:
        icon = "+" if self._action == "created" else "~"
        basename = os.path.basename(self._file_path)
        count_str = f" ({self._change_count} edits)" if self._change_count > 1 else ""
        return f" {icon} {basename}{count_str}"

    def update_diff(self, diff_text: str, change_count: int = 1, action: str = "modified") -> None:
        """Update the diff content and title."""
        self._change_count = change_count
        self._action = action
        self._diff_static.update(colorize_diff(diff_text))
        self.title = self._build_title()
