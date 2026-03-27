"""Tools for inserting content before or after a specific line."""

import hashlib
import os
import stat
import sqlite3
import tempfile
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


# ── Shared insert logic ──────────────────────────────────────────────────────


def _insert_at(tool: InfinibayBaseTool, path: str, content: str, insert_idx: int) -> str:
    """Insert content at a 0-based line index. Shared by both tools."""
    path = tool._resolve_path(os.path.expanduser(path))

    sandbox_err = tool._validate_sandbox_path(path)
    if sandbox_err:
        return tool._error(sandbox_err)

    from infinidev.tools.base.permissions import check_file_permission
    perm_err = check_file_permission("insert_lines", path)
    if perm_err:
        return tool._error(perm_err)

    if not os.path.exists(path):
        return tool._error(f"File not found: {path}")
    if not os.path.isfile(path):
        return tool._error(f"Not a file: {path}")

    file_size = os.path.getsize(path)
    if file_size > settings.MAX_FILE_SIZE_BYTES:
        return tool._error(
            f"File too large: {file_size} bytes (max {settings.MAX_FILE_SIZE_BYTES} bytes)"
        )

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except PermissionError:
        return tool._error(f"Permission denied: {path}")
    except Exception as e:
        return tool._error(f"Error reading file: {e}")

    total_lines = len(lines)

    # Clamp insert index
    insert_idx = max(0, min(insert_idx, total_lines))

    # Compute before-hash
    old_content = "".join(lines)
    before_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()[:16]

    # Build new lines
    new_lines = content.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    result_lines = lines[:insert_idx] + new_lines + lines[insert_idx:]
    new_content = "".join(result_lines)

    new_size = len(new_content.encode("utf-8"))
    if new_size > settings.MAX_FILE_SIZE_BYTES:
        return tool._error(
            f"Resulting file too large: {new_size} bytes (max {settings.MAX_FILE_SIZE_BYTES} bytes)"
        )

    # Atomic write
    try:
        dir_name = os.path.dirname(path)
        original_mode = os.stat(path).st_mode
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".infinibay_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            os.chmod(tmp_path, stat.S_IMODE(original_mode))
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise
    except PermissionError:
        return tool._error(f"Permission denied: {path}")
    except Exception as e:
        return tool._error(f"Error writing file: {e}")

    after_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()[:16]

    # Audit
    project_id = tool.project_id
    agent_run_id = tool.agent_run_id

    def _record(conn: sqlite3.Connection):
        conn.execute(
            """INSERT INTO artifact_changes
               (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (project_id, agent_run_id, path, "modified", before_hash, after_hash, new_size),
        )
        conn.commit()

    try:
        execute_with_retry(_record)
    except Exception:
        pass

    return tool._success({
        "path": path,
        "action": "modified",
        "inserted_at": insert_idx + 1,
        "lines_added": len(new_lines),
        "total_lines": len(result_lines),
        "size_bytes": new_size,
    })


# ── add_content_after_line ────────────────────────────────────────────────────


class AddContentAfterLineInput(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    line_number: int = Field(..., description="Line number to insert AFTER (1-based)")
    content: str = Field(..., description="Content to insert")


class AddContentAfterLineTool(InfinibayBaseTool):
    name: str = "add_content_after_line"
    description: str = "Insert content after a specific line number."
    args_schema: Type[BaseModel] = AddContentAfterLineInput

    def _run(self, file_path: str, line_number: int, content: str) -> str:
        if line_number < 0:
            return self._error(f"line_number must be >= 0, got {line_number}")
        self._log_tool_usage(f"add_content_after_line: line {line_number} in {file_path}")
        return _insert_at(self, file_path, content, insert_idx=line_number)


# ── add_content_before_line ───────────────────────────────────────────────────


class AddContentBeforeLineInput(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    line_number: int = Field(..., description="Line number to insert BEFORE (1-based)")
    content: str = Field(..., description="Content to insert")


class AddContentBeforeLineTool(InfinibayBaseTool):
    name: str = "add_content_before_line"
    description: str = "Insert content before a specific line number."
    args_schema: Type[BaseModel] = AddContentBeforeLineInput

    def _run(self, file_path: str, line_number: int, content: str) -> str:
        if line_number < 1:
            return self._error(f"line_number must be >= 1, got {line_number}")
        self._log_tool_usage(f"add_content_before_line: line {line_number} in {file_path}")
        return _insert_at(self, file_path, content, insert_idx=line_number - 1)
