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
        self._reasons: dict[str, list[str]] = {}  # path → list of reasons for changes
        self._deleted_symbols: dict[str, set[str]] = {}  # path → set of removed symbol names
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
        total_lines = len(diff_lines)
        if total_lines > max_lines:
            truncated_count = total_lines - max_lines
            diff_lines = diff_lines[:max_lines]
            diff_lines.append(f"\n... ({truncated_count} more lines truncated)")

        return "\n".join(diff_lines)

    def get_change_count(self, path: str) -> int:
        return self._change_counts.get(os.path.abspath(path), 0)

    def get_action(self, path: str) -> str:
        """Return 'created' if original was empty, else 'modified'."""
        path = os.path.abspath(path)
        return "created" if not self._originals.get(path, "") else "modified"

    def record_reason(self, path: str, reason: str) -> None:
        """Record a reason/description for why a file was changed."""
        path = os.path.abspath(path)
        if reason and reason.strip():
            self._reasons.setdefault(path, []).append(reason.strip())

    def get_reasons(self, path: str) -> list[str]:
        """Return all recorded reasons for a file's changes."""
        return self._reasons.get(os.path.abspath(path), [])

    def get_all_paths(self) -> list[str]:
        return list(self._current.keys())

    def record_deleted_symbols(self, path: str, symbols: list[str]) -> None:
        """Record symbol names that were removed from a file.

        Called from `maybe_emit_file_change` when a file-write tool reports
        `removed_symbols` in its result. Consumed by VerificationEngine to
        run the orphaned-references check post-task.

        Args:
            path: Absolute file path that was changed.
            symbols: Simple or qualified symbol names removed from the file.
        """
        if not symbols:
            return
        path = os.path.abspath(path)
        self._deleted_symbols.setdefault(path, set()).update(symbols)

    def get_deleted_symbols(self) -> dict[str, set[str]]:
        """Return all removed symbols grouped by file path.

        Returns a dict mapping file path → set of removed symbol names.
        Used by VerificationEngine to check for orphaned references.
        """
        return dict(self._deleted_symbols)

    def deactivate(self) -> None:
        self._active = False

    def reset(self) -> None:
        self._originals.clear()
        self._current.clear()
        self._change_counts.clear()
        self._reasons.clear()
        self._deleted_symbols.clear()
        self._active = True
