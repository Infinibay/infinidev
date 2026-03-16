"""Tool for surgical file edits using search-and-replace."""

import hashlib
import json
import os
import stat
import sqlite3
import tempfile
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class EditFileInput(BaseModel):
    path: str = Field(..., description="Path to the file to edit")
    old_string: str = Field(
        ...,
        description=(
            "The exact text to find in the file. Must match exactly "
            "(including indentation and whitespace). Must be unique in the "
            "file — if it appears more than once, provide more surrounding "
            "context to make it unique, or use replace_all=true."
        ),
    )
    new_string: str = Field(
        ...,
        description="The text to replace old_string with. Must differ from old_string.",
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "If true, replace ALL occurrences of old_string in the file. "
            "Useful for renaming variables or updating repeated patterns."
        ),
    )


class EditFileTool(InfinibayBaseTool):
    name: str = "edit_file"
    description: str = (
        "Make a surgical edit to an existing file by replacing a specific "
        "text snippet. This is far more efficient than rewriting the entire "
        "file with write_file — use edit_file for all modifications to "
        "existing files. The old_string must match exactly and be unique in "
        "the file (unless replace_all is true). Workflow: use code_search "
        "to find the location, read_file with offset/limit to see context, "
        "then edit_file to make the change."
    )
    args_schema: Type[BaseModel] = EditFileInput

    def _run(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        if old_string == new_string:
            return self._error("old_string and new_string are identical — nothing to change.")

        path = self._resolve_path(os.path.expanduser(path))

        if self._is_pod_mode():
            return self._run_in_pod(path, old_string, new_string, replace_all)

        # Sandbox check
        sandbox_err = self._validate_sandbox_path(path)
        if sandbox_err:
            return self._error(sandbox_err)

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
                content = f.read()
        except PermissionError:
            return self._error(f"Permission denied: {path}")
        except Exception as e:
            return self._error(f"Error reading file: {e}")

        # Validate old_string exists
        count = content.count(old_string)
        if count == 0:
            return self._error(
                f"old_string not found in {path}. "
                "Ensure the text matches exactly, including indentation and whitespace. "
                "Use code_search or read_file to verify the exact content."
            )

        if count > 1 and not replace_all:
            return self._error(
                f"old_string appears {count} times in {path}. "
                "Provide more surrounding context to make it unique, "
                "or set replace_all=true to replace all occurrences."
            )

        # Compute before-hash
        before_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        # Perform replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        # Check resulting size
        new_size = len(new_content.encode("utf-8"))
        if new_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"Resulting file too large: {new_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Atomic write (preserve original permissions)
        try:
            dir_name = os.path.dirname(path)
            original_mode = os.stat(path).st_mode if os.path.exists(path) else None
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".infinibay_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(new_content)
                if original_mode is not None:
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

        # Record in artifact_changes for audit
        project_id = self.project_id
        agent_run_id = self.agent_run_id

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
            pass  # Don't fail the edit if audit logging fails

        self._log_tool_usage(
            f"Edited {path} ({replacements} replacement{'s' if replacements > 1 else ''}, {new_size} bytes)"
        )
        return self._success({
            "path": path,
            "action": "modified",
            "replacements": replacements,
            "size_bytes": new_size,
        })

    def _run_in_pod(
        self, path: str, old_string: str, new_string: str, replace_all: bool,
    ) -> str:
        """Edit file via infinibay-file-helper inside the pod."""
        req = {
            "op": "edit",
            "path": path,
            "old_string": old_string,
            "new_string": new_string,
            "replace_all": replace_all,
        }

        try:
            result = self._exec_in_pod(
                ["infinibay-file-helper"],
                stdin_data=json.dumps(req),
            )
        except RuntimeError as e:
            return self._error(f"Pod execution failed: {e}")

        if result.exit_code != 0:
            return self._error(f"File helper error: {result.stderr.strip()}")

        try:
            resp = json.loads(result.stdout)
        except json.JSONDecodeError:
            return self._error(f"Invalid response from file helper: {result.stdout[:200]}")

        if not resp.get("ok"):
            return self._error(resp.get("error", "Unknown error"))

        data = resp["data"]

        # Record in artifact_changes for audit
        project_id = self.project_id
        agent_run_id = self.agent_run_id

        def _record_change(conn: sqlite3.Connection):
            conn.execute(
                """INSERT INTO artifact_changes
                   (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (project_id, agent_run_id, path, "modified",
                 data.get("before_hash"), data["after_hash"], data["size_bytes"]),
            )
            conn.commit()

        try:
            execute_with_retry(_record_change)
        except Exception:
            pass

        replacements = data["replacements"]
        self._log_tool_usage(
            f"Edited {path} (pod, {replacements} replacement{'s' if replacements > 1 else ''}, {data['size_bytes']} bytes)"
        )
        return self._success({
            "path": path,
            "action": "modified",
            "replacements": replacements,
            "size_bytes": data["size_bytes"],
        })
