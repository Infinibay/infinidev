"""Tool for surgical file edits using search-and-replace."""

import difflib
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
    model_config = {"populate_by_name": True}

    file_path: str = Field(..., description="Path to the file to edit")
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
    reason: str = Field(
        default="",
        alias="description",
        description=(
            "Brief explanation of WHY this edit is being made. "
            "Used by the code reviewer to understand intent."
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
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        reason: str = "",
    ) -> str:
        if old_string == new_string:
            return self._error("old_string and new_string are identical — nothing to change.")

        file_path = self._resolve_path(os.path.expanduser(file_path))

        if self._is_pod_mode():
            return self._run_in_pod(file_path, old_string, new_string, replace_all)

        # Sandbox check
        sandbox_err = self._validate_sandbox_path(file_path)
        if sandbox_err:
            return self._error(sandbox_err)

        # Permission check
        from infinidev.tools.base.permissions import check_file_permission
        perm_err = check_file_permission("edit_file", file_path)
        if perm_err:
            return self._error(perm_err)

        if not os.path.exists(file_path):
            return self._error(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            return self._error(f"Not a file: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"File too large: {file_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Read existing content
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except PermissionError:
            return self._error(f"Permission denied: {file_path}")
        except Exception as e:
            return self._error(f"Error reading file: {e}")

        # Validate old_string exists
        count = content.count(old_string)
        if count == 0:
            # Find the most similar line(s) to help the LLM correct its attempt
            hint = ""
            old_lines = old_string.strip().splitlines()
            if old_lines:
                file_lines = content.splitlines()
                first_old = old_lines[0].strip()
                if first_old:
                    close = difflib.get_close_matches(
                        first_old, [l.strip() for l in file_lines], n=1, cutoff=0.5
                    )
                    if close:
                        # Find the actual line (with original indentation)
                        for fl in file_lines:
                            if fl.strip() == close[0]:
                                hint = f' Closest match found: "{fl}"'
                                break
            return self._error(
                f"old_string not found in {file_path}. "
                "Ensure the text matches exactly, including indentation and whitespace."
                f"{hint}"
            )

        if count > 1 and not replace_all:
            return self._error(
                f"old_string appears {count} times in {file_path}. "
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
            dir_name = os.path.dirname(file_path)
            original_mode = os.stat(file_path).st_mode if os.path.exists(file_path) else None
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".infinibay_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(new_content)
                if original_mode is not None:
                    os.chmod(tmp_path, stat.S_IMODE(original_mode))
                os.replace(tmp_path, file_path)
            except Exception:
                os.unlink(tmp_path)
                raise
        except PermissionError:
            return self._error(f"Permission denied: {file_path}")
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
                (project_id, agent_run_id, file_path, "modified", before_hash, after_hash, new_size),
            )
            conn.commit()

        try:
            execute_with_retry(_record_change)
        except Exception:
            pass  # Don't fail the edit if audit logging fails

        self._log_tool_usage(
            f"Edited {file_path} ({replacements} replacement{'s' if replacements > 1 else ''}, {new_size} bytes)"
        )
        result = {
            "file_path": file_path,
            "action": "modified",
            "replacements": replacements,
            "size_bytes": new_size,
        }
        if reason:
            result["reason"] = reason
        return self._success(result)

    def _run_in_pod(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool,
    ) -> str:
        """Edit file via infinibay-file-helper inside the pod."""
        req = {
            "op": "edit",
            "file_path": file_path,
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
                (project_id, agent_run_id, file_path, "modified",
                 data.get("before_hash"), data["after_hash"], data["size_bytes"]),
            )
            conn.commit()

        try:
            execute_with_retry(_record_change)
        except Exception:
            pass

        replacements = data["replacements"]
        self._log_tool_usage(
            f"Edited {file_path} (pod, {replacements} replacement{'s' if replacements > 1 else ''}, {data['size_bytes']} bytes)"
        )
        return self._success({
            "file_path": file_path,
            "action": "modified",
            "replacements": replacements,
            "size_bytes": data["size_bytes"],
        })
