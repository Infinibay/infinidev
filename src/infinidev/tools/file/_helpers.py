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


def atomic_write(file_path: str, content: str) -> None:
    """Write content to *file_path* atomically via tempfile + rename.

    Preserves the original file's permission bits if the file already exists.
    Raises on failure (caller should handle PermissionError, etc.).
    """
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
