"""Collapsible file diff display as FormattedText.

Replaces the Textual FileChangeDiffWidget with a pure FormattedText renderer.
Each diff is stored as a dict in the chat messages list with type="diff".
"""

from __future__ import annotations

import os
import re

from infinidev.ui.theme import (
    STYLE_DIFF_REMOVED, STYLE_DIFF_ADDED, STYLE_DIFF_HUNK,
    STYLE_DIFF_HEADER, STYLE_DIFF_TITLE,
    TEXT, TEXT_MUTED, DIFF_REMOVED, DIFF_ADDED, DIFF_HUNK,
    DIFF_HEADER, DIFF_TITLE_FG, DIFF_TITLE_BG,
)


def colorize_diff_fragments(diff_text: str) -> list[list[tuple[str, str]]]:
    """Convert a unified diff to prompt_toolkit style fragment lines.

    Returns a list of lines, where each line is a list of (style, text) tuples.
    """
    lines: list[list[tuple[str, str]]] = []
    old_num = 0
    new_num = 0
    in_hunk = False

    for line in diff_text.splitlines():
        if not in_hunk and (line.startswith("---") or line.startswith("+++")):
            lines.append([(f"{DIFF_HEADER} bold", f"     {line}")])
        elif line.startswith("@@"):
            in_hunk = True
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line)
            if match:
                old_num = int(match.group(1)) - 1
                new_num = int(match.group(2)) - 1
            lines.append([(f"{DIFF_HUNK} bold", f"     {line}")])
        elif line.startswith("-"):
            old_num += 1
            lnum = f"{old_num:>4} "
            lines.append([(f"{DIFF_REMOVED}", f"{lnum}{line}")])
        elif line.startswith("+"):
            new_num += 1
            lnum = f"{new_num:>4} "
            lines.append([(f"{DIFF_ADDED}", f"{lnum}{line}")])
        else:
            old_num += 1
            new_num += 1
            lnum = f"{new_num:>4} "
            lines.append([
                (f"{TEXT_MUTED}", lnum),
                (f"{TEXT}", line),
            ])

    return lines


def diff_header_fragments(file_path: str, action: str = "modified",
                          change_count: int = 1) -> list[tuple[str, str]]:
    """Render a diff header line (collapsible title)."""
    icon = "+" if action == "created" else "~"
    basename = os.path.basename(file_path)
    count_str = f" ({change_count} edits)" if change_count > 1 else ""
    return [
        (f"{DIFF_TITLE_FG} bg:{DIFF_TITLE_BG}", f" {icon} {basename}{count_str} "),
    ]


class DiffState:
    """Tracks diff data for a file, including collapsed/expanded state."""

    def __init__(self, file_path: str, diff_text: str,
                 action: str = "modified", change_count: int = 1) -> None:
        self.file_path = file_path
        self.diff_text = diff_text
        self.action = action
        self.change_count = change_count
        self.collapsed = True
        self._diff_lines: list[list[tuple[str, str]]] | None = None

    def update(self, diff_text: str, action: str = "modified",
               change_count: int = 1) -> None:
        self.diff_text = diff_text
        self.action = action
        self.change_count = change_count
        self._diff_lines = None  # invalidate cache

    def get_lines(self) -> list[list[tuple[str, str]]]:
        """Get all display lines (header + diff if expanded)."""
        header = diff_header_fragments(self.file_path, self.action, self.change_count)
        lines = [header]

        if not self.collapsed:
            if self._diff_lines is None:
                self._diff_lines = colorize_diff_fragments(self.diff_text)
            lines.extend(self._diff_lines)

        return lines
