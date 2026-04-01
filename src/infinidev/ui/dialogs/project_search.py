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


def run_search(query: str, root_dir: str, skip_junk: bool = True,
               max_results: int = 500) -> list[dict[str, Any]]:
    """Search for a query across all project files. Returns list of matches."""
    if not query.strip():
        return []

    results = []
    try:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    except re.error:
        return []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if skip_junk:
            dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]

        for fname in filenames:
            if skip_junk:
                ext = os.path.splitext(fname)[1].lower()
                if ext in _IGNORED_EXTENSIONS:
                    continue

            filepath = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(filepath)
                if size > 1_000_000:  # Skip files > 1MB
                    continue
                with open(filepath, "r", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern.search(line):
                            results.append({
                                "file": filepath,
                                "line": line_num,
                                "text": line.rstrip(),
                            })
                            if len(results) >= max_results:
                                return results
            except (PermissionError, OSError):
                continue

    return results


def create_project_search():
    """Create the project search dialog."""
    results_ctrl = SearchResultsControl()
    preview_ctrl = SearchPreviewControl(results_ctrl)

    search_buffer = Buffer(name="project-search")

    body = HSplit([
        # Search input row
        HSplit([
            Window(
                content=BufferControl(buffer=search_buffer, focusable=True),
                height=1,
            ),
        ]),
        # Status line
        Window(
            content=FormattedTextControl(lambda: [
                (f"{TEXT_MUTED}", f" {results_ctrl.status}"),
            ]),
            height=1,
        ),
        # Results + Preview
        VSplit([
            Window(content=results_ctrl, width=D(weight=50)),
            Window(width=1, char="│", style=f"{PRIMARY}"),
            Window(content=preview_ctrl, width=D(weight=50)),
        ]),
        Window(
            content=FormattedTextControl(lambda: [
                (f"{TEXT_MUTED}", " Enter = Open file | Esc = Close"),
            ]),
            height=1,
        ),
    ])

    from infinidev.ui.dialogs.base import dialog_frame
    frame = dialog_frame("Search Project", body, width=90, height=30, border_color=PRIMARY)
    return frame, search_buffer, results_ctrl
