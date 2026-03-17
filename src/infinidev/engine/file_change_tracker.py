"""Track file changes during a single task execution.

Stores the original content of each file before its first modification,
and the current content after each modification. Generates unified diffs
comparing original → current state, combining multiple edits into one diff.
"""

from __future__ import annotations

import difflib
import os


class FileChangeTracker:
    """Buffer that accumulates file changes within one task run."""

    def __init__(self) -> None:
        self._originals: dict[str, str] = {}   # path → content before first edit
        self._current: dict[str, str] = {}      # path → content after latest edit
        self._change_counts: dict[str, int] = {}
        self._active: bool = True

    @property
    def active(self) -> bool:
        return self._active

    def record(self, path: str, before: str | None, after: str) -> str | None:
        """Record a file change. Returns the cumulative unified diff, or None if inactive.

        Args:
            path: Absolute file path.
            before: Content before this specific edit (None for new files).
            after: Content after this edit.
        """
        if not self._active:
            return None

        path = os.path.abspath(path)

        # Store original only on first touch
        if path not in self._originals:
            self._originals[path] = before or ""

        self._current[path] = after
        self._change_counts[path] = self._change_counts.get(path, 0) + 1

        return self.get_diff(path)

    def get_diff(self, path: str) -> str | None:
        """Generate unified diff for a file (original → current)."""
        path = os.path.abspath(path)
        if path not in self._current:
            return None

        original = self._originals.get(path, "")
        current = self._current[path]

        if original == current:
            return None

        diff_lines = list(difflib.unified_diff(
            original.splitlines(),
            current.splitlines(),
            fromfile=f"a/{os.path.basename(path)}",
            tofile=f"b/{os.path.basename(path)}",
            lineterm="",
        ))

        if not diff_lines:
            return None

        # Truncate very long diffs
        max_lines = 500
        if len(diff_lines) > max_lines:
            diff_lines = diff_lines[:max_lines]
            diff_lines.append(f"\n... ({len(diff_lines)} more lines truncated)")

        return "\n".join(diff_lines)

    def get_change_count(self, path: str) -> int:
        return self._change_counts.get(os.path.abspath(path), 0)

    def get_action(self, path: str) -> str:
        """Return 'created' if original was empty, else 'modified'."""
        path = os.path.abspath(path)
        return "created" if not self._originals.get(path, "") else "modified"

    def get_all_paths(self) -> list[str]:
        return list(self._current.keys())

    def deactivate(self) -> None:
        self._active = False

    def reset(self) -> None:
        self._originals.clear()
        self._current.clear()
        self._change_counts.clear()
        self._active = True
