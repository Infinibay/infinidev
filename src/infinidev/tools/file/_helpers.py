"""Shared helpers for file tools — eliminates duplication across write/edit/insert/replace tools."""

from __future__ import annotations

import os
import sqlite3
import stat
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.tools.base.base_tool import InfinibayBaseTool


def guard_file_access(tool: InfinibayBaseTool, path: str, operation: str) -> str | None:
    """Validate sandbox boundaries and file permissions.

    Returns an error-JSON string if access is denied, or None if allowed.
    """
    sandbox_err = tool._validate_sandbox_path(path)
    if sandbox_err:
        return tool._error(sandbox_err)

    from infinidev.tools.base.permissions import check_file_permission
    perm_err = check_file_permission(operation, path)
    if perm_err:
        return tool._error(perm_err)

    return None


def detect_silent_deletions(
    file_path: str,
    old_content: str,
    new_content: str,
) -> list[str]:
    """Return a list of qualified symbol names present in old but missing
    in new. Empty list when nothing was deleted, when the language is
    unsupported, or when tree-sitter is unavailable.

    Used to surface "you accidentally removed a function" warnings on
    edit/replace tools. Soft signal — the caller decides whether to
    warn or block. Never raises.
    """
    from infinidev.engine.static_analysis_timer import measure
    with measure("silent_deletion"):
        try:
            from infinidev.code_intel.syntax_check import extract_top_level_symbols
            before = extract_top_level_symbols(old_content, file_path=file_path)
            after = extract_top_level_symbols(new_content, file_path=file_path)
        except Exception:
            return []
        if not before:
            return []  # nothing to compare against
        removed = sorted(before - after)
        return removed


def deletion_warning_text(removed: list[str], file_path: str) -> str:
    """Render a short, model-facing warning that lists deleted symbols.

    Returns "" when there's nothing to warn about, so callers can
    use truthiness directly: ``if (warn := deletion_warning_text(...)): ``.
    """
    if not removed:
        return ""
    if len(removed) == 1:
        return (
            f"You removed {removed[0]} from {file_path}. If this was "
            "intentional, ignore this notice. If not, restore the "
            "function — your edit may have collapsed too much."
        )
    head = ", ".join(removed[:5])
    extra = f" (+{len(removed) - 5} more)" if len(removed) > 5 else ""
    return (
        f"You removed {len(removed)} symbols from {file_path}: {head}{extra}. "
        "If this was intentional, ignore this notice. If not, restore them — "
        "your edit may have collapsed too much."
    )


def check_syntax_warning(
    tool: InfinibayBaseTool,
    file_path: str,
    new_content: str,
    *,
    operation: str = "write",
) -> str | None:
    """Pre-write syntax check via tree-sitter — returns an advisory warning string.

    Returns a human-readable warning body when tree-sitter flags issues in
    *new_content*, or ``None`` when the content parses cleanly, the language
    is unsupported, tree-sitter is unavailable, or the check is disabled via
    ``LOOP_VALIDATE_SYNTAX_BEFORE_WRITE``. The result is advisory only —
    callers include it in the success response; it never blocks the write.

    Previously this was ``validate_syntax_or_error`` and returned a JSON
    error that refused the write, but false positives (Jest mocks, TSX
    experimental syntax, TS project references) were rejecting legitimate
    edits — hence the demotion to warning.
    """
    try:
        from infinidev.config.settings import settings
        if not getattr(settings, "LOOP_VALIDATE_SYNTAX_BEFORE_WRITE", True):
            return None
    except Exception:
        pass

    from infinidev.engine.static_analysis_timer import measure
    with measure("syntax_check"):
        try:
            from infinidev.code_intel.syntax_check import check_syntax, format_issues
            issues = check_syntax(new_content, file_path=file_path)
        except Exception:
            return None
        if not issues:
            return None
        body = format_issues(issues)
        return (
            f"tree-sitter flagged {len(issues)} potential issue(s) in {file_path}. "
            "These may be false positives (Jest mocks, TS project config, "
            f"experimental syntax) — review before trusting:\n{body}"
        )


def atomic_write(file_path: str, content: str) -> None:
    """Write content to *file_path* atomically via tempfile + rename.

    Preserves the original file's permission bits if the file already exists.
    Raises on failure (caller should handle PermissionError, etc.).
    """
    from infinidev.engine.static_analysis_timer import measure as _sa_measure
    with _sa_measure("tool_io"):
        dir_name = os.path.dirname(file_path) or "."
        original_mode = os.stat(file_path).st_mode if os.path.exists(file_path) else None

        fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".infinibay_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            if original_mode is not None:
                os.chmod(tmp_path, stat.S_IMODE(original_mode))
            os.replace(tmp_path, file_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def record_artifact_change(
    tool: InfinibayBaseTool,
    file_path: str,
    action: str,
    before_hash: str | None,
    after_hash: str,
    size_bytes: int,
) -> None:
    """Record a file change in the artifact_changes audit table (best-effort).

    Silently ignores any DB errors — the caller's operation should not fail
    because audit logging is unavailable.
    """
    from infinidev.tools.base.db import execute_with_retry

    project_id = tool.project_id
    agent_run_id = tool.agent_run_id

    def _record(conn: sqlite3.Connection) -> None:
        conn.execute(
            """INSERT INTO artifact_changes
               (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes),
        )
        conn.commit()

    try:
        execute_with_retry(_record)
    except Exception:
        pass
