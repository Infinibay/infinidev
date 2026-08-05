"""The single status line under the composer.

One line, two halves: what the session *is* on the left (model, project,
git branch, context budget) and what it can *do* on the right (the few
keybindings worth advertising). Everything is dropped in priority order
as the terminal narrows, so this never wraps and never pushes the
transcript around.

It replaces the old two-line status bar + footer: two full-width painted
bars for one line of information read as chrome, not as a status.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.utils import get_cwidth

from infinidev.ui.controls._widthutil import terminal_cols, truncate_cells
from infinidev.ui.theme import (
    ACCENT,
    ERROR,
    PRIMARY,
    PROGRESS_CRITICAL,
    PROGRESS_WARNING,
    SUCCESS,
    TEXT,
    TEXT_DIM,
    TEXT_MUTED,
)

SEPARATOR = "  ·  "

# Keybindings advertised on the right. Kept deliberately short — the full
# list lives behind `?`, which is the last hint to be dropped.
STATUS_HINTS: tuple[tuple[str, str], ...] = (
    ("?", "help"),
    ("^C", "stop"),
    ("^D", "exit"),
)

# git branch is read at most this often; a subprocess per repaint would
# turn a 60 fps scroll into a fork bomb.
_BRANCH_TTL = 5.0


class _BranchCache:
    """Throttled `git branch --show-current` with dirty-state detection."""

    def __init__(self) -> None:
        self._value = ""
        self._checked_at = 0.0
        self._cwd = ""

    def get(self, cwd: str) -> str:
        now = time.monotonic()
        if cwd == self._cwd and (now - self._checked_at) < _BRANCH_TTL:
            return self._value
        self._cwd = cwd
        self._checked_at = now
        self._value = self._read(cwd)
        return self._value

    @staticmethod
    def _read(cwd: str) -> str:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain=v1", "--branch"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=2,
            )
        except Exception:
            return ""
        if result.returncode != 0:
            return ""
        lines = result.stdout.splitlines()
        if not lines:
            return ""
        # "## main...origin/main [ahead 1]" → "main"
        header = lines[0].removeprefix("## ").split("...")[0].split(" ")[0]
        if header.startswith("No commits yet on "):
            header = header.removeprefix("No commits yet on ")
        dirty = any(not line.startswith("##") for line in lines)
        return f"{header}*" if dirty else header


_branch_cache = _BranchCache()


class StatusLineControl(FormattedTextControl):
    """Single-line session status with right-aligned key hints."""

    def __init__(self, app_state=None) -> None:
        self._app_state = app_state
        self._model = ""
        self._project = ""
        self._status = ""
        self._status_kind = "idle"
        super().__init__(self._get_text)

    # ── setters (mirror the old StatusBarControl API) ────────────────

    def set_model(self, model: str) -> None:
        self._model = model

    def set_project(self, project: str) -> None:
        self._project = project

    def set_status(self, status: str, kind: str = "info") -> None:
        self._status = status
        self._status_kind = kind

    # ── rendering ────────────────────────────────────────────────────

    def _context_fragment(self) -> tuple[str, str] | None:
        """Remaining context as a percentage, coloured by pressure."""
        calculator = getattr(self._app_state, "context_calculator", None)
        if calculator is None:
            return None
        try:
            # chat_usage_percentage is a 0..1 fraction of the *effective*
            # window, which is what the user actually has left to spend.
            used = float(calculator.chat_usage_percentage) * 100.0
        except Exception:
            return None
        if used <= 0:
            return None
        remaining = max(0.0, 100.0 - used)
        if remaining <= 10:
            colour = PROGRESS_CRITICAL
        elif remaining <= 25:
            colour = PROGRESS_WARNING
        else:
            colour = TEXT_MUTED
        return (colour, f"{remaining:.0f}% context left")

    def _status_fragment(self) -> tuple[str, str] | None:
        if not self._status:
            return None
        colour = {
            "error": ERROR,
            "warn": ACCENT,
            "success": SUCCESS,
        }.get(self._status_kind, PRIMARY)
        return (colour, self._status)

    def _left_segments(self) -> list[tuple]:
        """Segments in *drop-last-first* priority order."""
        segments: list[tuple] = []
        try:
            from infinidev.engine.council.observer import list_councils

            running = sum(
                member.get("status") == "running"
                for council in list_councils()
                if council.get("status") == "running"
                for member in council.get("members", {}).values()
            )
        except Exception:
            running = 0
        if running:
            def _open_agents(mouse_event):
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    self._app_state.show_agents()

            noun = "agent" if running == 1 else "agents"
            segments.append((ACCENT, f"Running {running} {noun}", _open_agents))
        status = self._status_fragment()
        if status:
            segments.append(status)
        if self._model:
            segments.append((TEXT, self._model))
        cwd = os.getcwd()
        branch = _branch_cache.get(cwd)
        if branch:
            segments.append((TEXT_MUTED, branch))
        project = self._project or Path(cwd).name
        if project:
            segments.append((TEXT_MUTED, project))
        context = self._context_fragment()
        if context:
            segments.append(context)
        return segments

    def _get_text(self) -> FormattedText:
        cols = terminal_cols()
        segments = self._left_segments()
        hints = list(STATUS_HINTS)

        if cols is None:
            return FormattedText(
                _join(segments) + [(TEXT_DIM, "   ")] + _render_hints(hints)
            )

        # Hints are cheap and fixed-width: reserve them first, dropping the
        # least important (leftmost, i.e. help) only when truly out of room.
        hint_fragments = _render_hints(hints)
        hint_width = _width(hint_fragments)
        while hints and hint_width > cols // 3:
            hints.pop(0)
            hint_fragments = _render_hints(hints)
            hint_width = _width(hint_fragments)

        budget = cols - hint_width - 2
        shown: list[tuple] = []
        used = 0
        for segment in segments:
            text = segment[1]
            width = get_cwidth(text) + (get_cwidth(SEPARATOR) if shown else 0)
            if used + width > budget:
                if not shown:  # never render an entirely empty status
                    shown.append((segment[0], truncate_cells(text, max(0, budget))))
                    used = budget
                break
            shown.append(segment)
            used += width

        left = _join(shown)
        pad = max(1, cols - used - hint_width)
        return FormattedText(left + [(TEXT_DIM, " " * pad)] + hint_fragments)


def _join(segments: list[tuple]) -> list[tuple]:
    fragments: list[tuple] = []
    for index, segment in enumerate(segments):
        if index:
            fragments.append((TEXT_DIM, SEPARATOR))
        fragments.append(segment)
    return fragments


def _render_hints(hints: list[tuple[str, str]]) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    for index, (key, label) in enumerate(hints):
        if index:
            fragments.append((TEXT_DIM, "  "))
        fragments.append((f"{TEXT_MUTED} bold", key))
        fragments.append((TEXT_DIM, f" {label}"))
    return fragments


def _width(fragments: list[tuple]) -> int:
    return sum(get_cwidth(fragment[1]) for fragment in fragments)
