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


class SearchResultsControl(UIControl):
    """Selectable search results list."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []  # {file, line, text, match_start, match_end}
        self.cursor: int = 0
        self.status: str = ""

    def move_cursor(self, delta: int) -> None:
        if self.results:
            self.cursor = max(0, min(len(self.results) - 1, self.cursor + delta))

    def get_selected(self) -> dict | None:
        if 0 <= self.cursor < len(self.results):
            return self.results[self.cursor]
        return None

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        lines = []
        for i, r in enumerate(self.results):
            file_part = os.path.basename(r["file"])
            line_part = f":{r['line']}"
            text_part = r["text"].strip()[:width - len(file_part) - 10]

            if i == self.cursor:
                lines.append([
                    (f"bg:{PRIMARY} #ffffff bold", f" {file_part}{line_part} "),
                    (f"bg:{PRIMARY} #cccccc", f" {text_part}"),
                ])
            else:
                lines.append([
                    (f"{ACCENT}", f" {file_part}{line_part} "),
                    (f"{TEXT}", f" {text_part}"),
                ])

        if not lines:
            lines = [[(f"{TEXT_MUTED}", f" {self.status or 'Type to search...'}")]]

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []
        return UIContent(get_line=get_line, line_count=len(lines))


