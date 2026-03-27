"""Tool for replacing a range of lines in a file."""

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


class ReplaceLinesInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to edit")
    content: str = Field(
        ...,
        description="New content to insert (replaces the specified line range)",
    )
    start_line: int = Field(
        ..., description="First line to replace (1-based, inclusive)"
    )
    end_line: int = Field(
        ..., description="Last line to replace (1-based, inclusive)"
    )


class ReplaceLinesTool(InfinibayBaseTool):
    name: str = "replace_lines"
    description: str = "Replace a range of lines in a file with new content."
    args_schema: Type[BaseModel] = ReplaceLinesInput

    def _run(
        self,
        file_path: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> str:
        path = self._resolve_path(os.path.expanduser(file_path))

        # Sandbox check
        sandbox_err = self._validate_sandbox_path(path)
        if sandbox_err:
            return self._error(sandbox_err)

        # Permission check
        from infinidev.tools.base.permissions import check_file_permission
        perm_err = check_file_permission("replace_lines", path)
        if perm_err:
            return self._error(perm_err)

        if not os.path.exists(path):
            return self._error(f"File not found: {path}")
        if not os.path.isfile(path):
            return self._error(f"Not a file: {path}")

        file_size = os.path.getsize(path)
        if file_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"File too large: {file_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Read existing content
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except PermissionError:
            return self._error(f"Permission denied: {path}")
        except Exception as e:
            return self._error(f"Error reading file: {e}")

        total_lines = len(lines)

        # Validate line range
        if start_line < 1:
            return self._error(f"start_line must be >= 1, got {start_line}")
        if end_line < start_line - 1:
            return self._error(
                f"end_line ({end_line}) must be >= start_line - 1 ({start_line - 1})"
            )
        if start_line > total_lines + 1:
            return self._error(
                f"start_line ({start_line}) is beyond end of file ({total_lines} lines)"
            )

        # Compute before-hash
        old_content = "".join(lines)
        before_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()[:16]

        # Build new content lines (ensure each line ends with \n)
        if content == "":
            new_lines = []
        else:
            new_lines = content.splitlines(keepends=True)
            # Ensure last line has newline
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

        # Splice: replace lines[start_line-1 : end_line] with new_lines
        # When start_line == end_line + 1, this is a pure insert (no lines removed)
        start_idx = start_line - 1
        end_idx = min(end_line, total_lines)
        result_lines = lines[:start_idx] + new_lines + lines[end_idx:]

        new_content = "".join(result_lines)

        # Check resulting size
        new_size = len(new_content.encode("utf-8"))
        if new_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"Resulting file too large: {new_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Atomic write (preserve permissions)
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
            return self._error(f"Permission denied: {path}")
        except Exception as e:
            return self._error(f"Error writing file: {e}")

        # Compute after-hash
        after_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()[:16]

        # Audit trail
        project_id = self.project_id
        agent_run_id = self.agent_run_id
        lines_removed = end_idx - start_idx
        lines_added = len(new_lines)

        def _record_change(conn: sqlite3.Connection):
            conn.execute(
                """INSERT INTO artifact_changes
                   (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (project_id, agent_run_id, path, "modified", before_hash, after_hash, new_size),
            )
            conn.commit()

        try:
            execute_with_retry(_record_change)
        except Exception:
            pass

        self._log_tool_usage(
            f"Replaced lines {start_line}-{end_line} in {path} "
            f"(-{lines_removed} +{lines_added} lines, {new_size} bytes)"
        )
        return self._success({
            "path": path,
            "action": "modified",
            "lines_removed": lines_removed,
            "lines_added": lines_added,
            "total_lines": len(result_lines),
            "size_bytes": new_size,
        })
