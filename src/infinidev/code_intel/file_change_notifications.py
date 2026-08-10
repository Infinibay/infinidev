"""Thread-safe queue for "a file on disk is broken" notifications.

Bridge between three concurrent writers and one main-thread reader:

  * **Writers** (any thread, any call site): the indexer pushes a
    notification whenever ``index_file`` parses a source file and
    tree-sitter detects structural errors. Pushers include:
      - The file watcher's ``index_callback`` (background worker
        thread in both TUI and classic modes) when an external edit
        or shell redirect lands on disk.
      - The file tools' ``enqueue_or_sync`` calls (engine main thread)
        right after ``replace_lines`` / ``create_file`` / etc.
      - The ``/reindex`` slash command (main thread) when the user
        asks for a manual rebuild.

  * **Reader** (engine main thread): the prompt builder drains the
    queue at the start of each iteration and renders a
    ``<file_integrity_warning>`` block in the next LLM prompt. The
    model sees "file X is now broken — you probably just edited it
    via shell; read it and fix" without having to poll anything.

The queue is process-global, bounded (max 50 entries to survive a
pathological flood), and deduplicated by file path (a file that's
still broken after 5 edits produces ONE entry, not 5). A notification
is only emitted when a file's state flips from "valid → broken" OR
changes its error count by > 50 %, so a long session where the same
file stays broken doesn't spam the prompt.

The module is intentionally tiny — no dependencies on the indexer or
the loop state — so it's safe to import from any layer.
"""

from __future__ import annotations

from collections import Counter
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FileIntegrityNotification:
    """One entry on the broken-files queue."""

    file_path: str
    language: str
    issue_count: int
    first_issue_line: int
    first_issue_column: int
    first_issue_message: str

    def render(self) -> str:
        """Render the notification as a single human-readable line."""
        return (
            f"⚠ {self.file_path}: {self.issue_count} new syntax error(s) "
            f"at L{self.first_issue_line}:{self.first_issue_column} "
            f"({self.first_issue_message})"
        )


class _NotificationQueue:
    """Process-global, thread-safe queue with dedup and bounded size.

    Deduplication: the same ``file_path`` can only have ONE entry in
    the queue at a time. Re-pushing the same file replaces the
    existing entry (the newest state wins). When a file's syntax
    becomes valid again, ``clear_file`` removes it from the queue.

    Bounded size: max ``_MAX_ENTRIES`` entries in the queue. Beyond
    that, new pushes replace the oldest entry. Sheer flood protection
    for the edge case where a sweeping script breaks 100+ files.
    """

    _MAX_ENTRIES = 50

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, FileIntegrityNotification] = {}
        # Tracks files we've already warned about in the current task
        # so successive iterations don't re-warn until the file gets
        # fixed (then re-broken). Cleared explicitly on ``drain``.
        self._already_drained: set[str] = set()
        self._issue_states: dict[str, Counter[tuple[str, str]]] = {}

    def push(self, notification: FileIntegrityNotification) -> None:
        """Add or replace an entry for *notification.file_path*.

        If the queue is at capacity and a new path is being added,
        the oldest entry is evicted (LRU-style by insertion order
        since Python 3.7 dict).
        """
        with self._lock:
            existing = self._entries.get(notification.file_path)
            if existing and existing == notification:
                # Same exact state — skip to avoid churn.
                return
            self._entries[notification.file_path] = notification
            # Bounded eviction: drop oldest entries until at cap.
            while len(self._entries) > self._MAX_ENTRIES:
                # dict maintains insertion order in py3.7+, so the
                # first key is the oldest.
                oldest = next(iter(self._entries))
                self._entries.pop(oldest, None)
                self._already_drained.discard(oldest)

    @staticmethod
    def _issue_signature(issue: object) -> tuple[str, str]:
        message = str(getattr(issue, "message", "")).strip()
        snippet = " ".join(str(getattr(issue, "snippet", "")).split())
        return message, snippet

    def observe(
        self,
        file_path: str,
        language: str,
        issues: list[object],
        *,
        notify: bool,
    ) -> None:
        """Record parser state and optionally queue newly introduced issues."""
        current = Counter(self._issue_signature(issue) for issue in issues)
        with self._lock:
            previous = self._issue_states.get(file_path)
            self._issue_states[file_path] = current

            if not current:
                self._entries.pop(file_path, None)
                self._already_drained.discard(file_path)
                return
            if not notify:
                return

            novel = current if previous is None else current - previous
            novel_count = sum(novel.values())
            if novel_count <= 0:
                return

            remaining = novel.copy()
            first = issues[0]
            for issue in issues:
                signature = self._issue_signature(issue)
                if remaining[signature] > 0:
                    first = issue
                    break

            notification = FileIntegrityNotification(
                file_path=file_path,
                language=language or "",
                issue_count=novel_count,
                first_issue_line=int(getattr(first, "line", 0)),
                first_issue_column=int(getattr(first, "column", 0)),
                first_issue_message=str(getattr(first, "message", ""))[:120],
            )
            self._entries[file_path] = notification
            self._already_drained.discard(file_path)
            while len(self._entries) > self._MAX_ENTRIES:
                oldest = next(iter(self._entries))
                self._entries.pop(oldest, None)
                self._already_drained.discard(oldest)

    def has_baseline(self, file_path: str) -> bool:
        """Return whether parser state has been observed for file_path."""
        with self._lock:
            return file_path in self._issue_states


    def clear_file(self, file_path: str) -> None:
        """Remove *file_path* from the queue — its syntax is valid again.

        Called by the indexer when a previously-broken file is
        re-parsed successfully. Also clears it from the "already
        drained" set so a future breakage will warn again.
        """
        with self._lock:
            self._entries.pop(file_path, None)
            self._already_drained.discard(file_path)
            self._issue_states[file_path] = Counter()

    def drain(self) -> list[FileIntegrityNotification]:
        """Pop all entries NOT already delivered, mark them as delivered.

        Called by the engine main thread at the start of each
        iteration. The same file will NOT be re-drained on subsequent
        calls until it's either fixed (``clear_file``) or the state
        gets worse (new push with a different error count).

        Returns an empty list when nothing is pending, which is the
        fast path and costs a single lock acquire.
        """
        with self._lock:
            if not self._entries:
                return []
            fresh = [
                n for path, n in self._entries.items()
                if path not in self._already_drained
            ]
            for n in fresh:
                self._already_drained.add(n.file_path)
            return fresh

    def peek(self) -> list[FileIntegrityNotification]:
        """Return a snapshot of all current entries without draining.

        Used by tests and diagnostics. Thread-safe but does NOT
        update the drained set.
        """
        with self._lock:
            return list(self._entries.values())

    def reset(self) -> None:
        """Drop everything. Tests and ``/reindex --full`` use this."""
        with self._lock:
            self._entries.clear()
            self._already_drained.clear()

            self._issue_states.clear()

# Module-level singleton. All the push/drain helpers below delegate
# to this instance so call sites don't need to know it exists.
_queue = _NotificationQueue()


def push_notification(
    file_path: str,
    language: str,
    issues,
    *,
    notify: bool = True,
) -> None:
    """Record syntax state and optionally notify about newly introduced issues.

    With notify=False this establishes a baseline. With notify=True it
    compares against the last observation and queues only novel signatures.
    """
    _queue.observe(file_path, language, list(issues or []), notify=notify)


def record_text_integrity_baseline(file_path: str, text: str) -> None:
    """Parse text once and remember its pre-change syntax state."""
    if not text:
        return
    from infinidev.code_intel.syntax_check import check_syntax

    issues = check_syntax(text, file_path=file_path)
    push_notification(file_path, "", issues, notify=False)


def has_integrity_baseline(file_path: str) -> bool:
    """Return whether parser state for file_path is already known."""
    return _queue.has_baseline(file_path)


def clear_file_notification(file_path: str) -> None:
    """Public helper: remove a file from the queue (auto-heal path)."""
    _queue.clear_file(file_path)


def drain_pending_notifications() -> list[FileIntegrityNotification]:
    """Public helper: pop all undelivered notifications from the queue.

    Returns an empty list when nothing is pending — the common case,
    costs a single lock acquire. The engine main thread should call
    this once at the start of each iteration and render the result
    into the next LLM prompt if non-empty.
    """
    return _queue.drain()


def peek_pending_notifications() -> list[FileIntegrityNotification]:
    """Public helper: snapshot current entries without draining."""
    return _queue.peek()


def reset_notifications() -> None:
    """Public helper: drop all entries (tests and full rebuilds)."""
    _queue.reset()


__all__ = [
    "FileIntegrityNotification",
    "push_notification",
    "clear_file_notification",
    "drain_pending_notifications",
    "record_text_integrity_baseline",
    "has_integrity_baseline",
    "peek_pending_notifications",
    "reset_notifications",
]
