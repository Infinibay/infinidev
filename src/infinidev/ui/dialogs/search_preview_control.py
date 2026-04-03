"""Project-wide search dialog — Ctrl+Shift+F / Ctrl+G search."""

from __future__ import annotations

import os
import re
from typing import Any

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, UIControl, UIContent, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D

from infinidev.ui.theme import PRIMARY, TEXT, TEXT_MUTED, ACCENT, WARNING

DIALOG_NAME = "project_search"

# Directories/extensions to skip during search
_IGNORED_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".eggs", ".cache", ".next", ".nuxt", "coverage", "htmlcov",
    ".infinidev", "finetune",
}

_IGNORED_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".doc", ".docx",
}
from infinidev.ui.dialogs.search_results_control import SearchResultsControl


class SearchPreviewControl(UIControl):
    """Preview of the selected search result with context."""

    def __init__(self, results_ctrl: SearchResultsControl) -> None:
        self._results = results_ctrl

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        result = self._results.get_selected()
        if not result:
            lines = [[(f"{TEXT_MUTED}", " Select a result")]]
        else:
            lines = []
            # Read file context
            try:
                with open(result["file"], "r", errors="replace") as f:
                    file_lines = f.readlines()
                target = result["line"] - 1  # 0-based
                start = max(0, target - 3)
                end = min(len(file_lines), target + 4)
                for i in range(start, end):
                    num = f"{i + 1:>4} "
                    text = file_lines[i].rstrip()[:width - 6]
                    if i == target:
                        lines.append([
                            (f"{WARNING}", num),
                            (f"{TEXT} bold", f"{text}"),
                        ])
                    else:
                        lines.append([
                            (f"{TEXT_MUTED}", num),
                            (f"{TEXT}", f"{text}"),
                        ])
            except Exception:
                lines = [[(f"{TEXT_MUTED}", " Unable to read file")]]

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []
        return UIContent(get_line=get_line, line_count=len(lines))


