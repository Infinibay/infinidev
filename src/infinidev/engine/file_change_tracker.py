"""Track file changes during a single task execution.

Stores the original content of each file before its first modification,
and the current content after each modification. Generates unified diffs
comparing original → current state, combining multiple edits into one diff.
"""

from __future__ import annotations

import difflib
import hashlib
import os

from infinidev.engine.workspace_baseline import WorkspaceBaseline


class FileChangeTracker:
    """Buffer that accumulates file changes within one task run."""

    def __init__(self, baseline: WorkspaceBaseline | None = None) -> None:
        self._originals: dict[str, str] = {}   # path → content before first edit
        self._current: dict[str, str] = {}      # path → content after latest edit
        self._change_counts: dict[str, int] = {}
        self._reasons: dict[str, list[str]] = {}  # path → list of reasons for changes
        self._deleted_symbols: dict[str, set[str]] = {}  # path → set of removed symbol names
        self._active: bool = True
        self._baseline = baseline

    @property
    def baseline(self) -> WorkspaceBaseline | None:
        return self._baseline

    def reconcile_workspace(self) -> None:
        """Merge final on-disk changes that bypassed known edit tools."""

        if self._baseline is None:
            return
        current = self._baseline.current_states()
        recorded = {
            os.path.relpath(path, self._baseline.root)
            for path in self._current
            if os.path.commonpath([self._baseline.root, os.path.realpath(path)])
            == self._baseline.root
        }
        for relative in sorted(set(self._baseline.files) | set(current) | recorded):
            before_state = self._baseline.files.get(relative)
            after_state = current.get(relative)
            path = os.path.join(self._baseline.root, relative)
            if before_state == after_state:
                # Disk truth wins over an earlier tool record: a later shell
                # command may have restored the original content, or removed
                # a file that was newly created during this task. Drop the
                # stale event entirely so rollback and review reflect final
                # workspace truth rather than historical tool activity.
                self._originals.pop(path, None)
                self._current.pop(path, None)
                self._change_counts.pop(path, None)
                self._reasons.pop(path, None)
                self._deleted_symbols.pop(path, None)
                continue
            before = before_state.text if before_state is not None else ""
            after = after_state.text if after_state is not None else ""
            # Oversized files are still detected, but their full text is not
            # retained in memory. Record an honest marker for review.
            if before is None:
                before = (
                    "[content omitted: file exceeded baseline capture limit; "
                    f"identity={before_state.digest}]\n"
                )
            if after is None:
                after = (
                    "[content omitted: file exceeded baseline capture limit; "
                    f"identity={after_state.digest}]\n"
                )
            if path not in self._originals:
                self._originals[path] = before
            self._current[path] = after
            self._change_counts[path] = max(1, self._change_counts.get(path, 0))
            reason = "Detected from task-start workspace baseline"
            if reason not in self._reasons.get(path, []):
                self._reasons.setdefault(path, []).append(reason)

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

        return self._render_diff(path)

    def get_diff(self, path: str) -> str | None:
        """Generate unified diff for a file (original → current)."""
        path = os.path.abspath(path)
        if self._active and path not in self._current:
            self.reconcile_workspace()
        return self._render_diff(path)

    def _render_diff(self, path: str) -> str | None:
        """Generate a diff from already-recorded state without scanning disk."""

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
        if self._active:
            self.reconcile_workspace()
        return [
            path
            for path, current in self._current.items()
            if self._originals.get(path, "") != current
        ]

    def change_fingerprint(
        self, *, reconcile: bool = False,
    ) -> tuple[tuple[str, str], ...]:
        """Return a stable identity for the current net task diff.

        Historical tool activity is intentionally absent. If an edit restores
        the prior content, its path disappears from this value. ``reconcile``
        is reserved for Step boundaries, where detecting shell-side writes is
        worth a bounded workspace scan; the per-tool progress path uses the
        already recorded in-memory state.
        """
        if reconcile and self._active:
            self.reconcile_workspace()
        rows: list[tuple[str, str]] = []
        for path, current in self._current.items():
            if self._originals.get(path, "") == current:
                continue
            digest = hashlib.sha256(
                current.encode("utf-8", errors="surrogatepass")
            ).hexdigest()
            rows.append((path, digest))
        return tuple(sorted(rows))

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
        if self._active:
            self.reconcile_workspace()
        self._active = False

    def reset(self) -> None:
        self._originals.clear()
        self._current.clear()
        self._change_counts.clear()
        self._reasons.clear()
        self._deleted_symbols.clear()
        self._active = True

    def merge_from(self, other: "FileChangeTracker") -> None:
        """Fold an earlier run's changes into this one.

        For the review's rework loop, which re-enters ``execute()`` on the
        same engine: each entry installs a fresh tracker, so after a rework
        pass the reviewer only saw the files that pass happened to touch.
        A file created in the first pass and merely edited in the second
        was reported as "modified", and a rework that wrote nothing at all
        reported no changes — which skips review entirely and silently
        drops the rejection that caused the rework.

        The OLDEST original wins, because that is what makes a diff read
        against the state before the turn began (and what keeps "created"
        from decaying into "modified"). Current content, being the newest,
        always comes from *self*.
        """
        for path, original in other._originals.items():
            # *other* ran first, so its "before" is the older one and wins
            # outright — this is what keeps a file created in the first
            # pass from reading as merely modified after the second.
            self._originals[path] = original
        for path, content in other._current.items():
            self._current.setdefault(path, content)
        for path, count in other._change_counts.items():
            self._change_counts[path] = self._change_counts.get(path, 0) + count
        for path, reasons in other._reasons.items():
            self._reasons[path] = list(reasons) + self._reasons.get(path, [])
        for path, symbols in other._deleted_symbols.items():
            self._deleted_symbols.setdefault(path, set()).update(symbols)
        if self._baseline is None:
            self._baseline = other._baseline
