"""Tool for creating new files with audit trail."""

import hashlib
import os
import sqlite3
import tempfile
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class CreateFileInput(BaseModel):
    file_path: str = Field(..., description="Path for the new file")
    content: str = Field(..., description="Content to write")


class CreateFileTool(InfinibayBaseTool):
    name: str = "create_file"
    description: str = "Create a new file. Fails if the file already exists."
    args_schema: Type[BaseModel] = CreateFileInput

    def _run(self, file_path: str, content: str) -> str:
        file_path = self._resolve_path(os.path.expanduser(file_path))

        if self._is_pod_mode():
            return self._run_in_pod(file_path, content)

        # Sandbox check
        sandbox_err = self._validate_sandbox_path(file_path)
        if sandbox_err:
            return self._error(sandbox_err)

        # Permission check
        from infinidev.tools.base.permissions import check_file_permission
        perm_err = check_file_permission("create_file", file_path)
        if perm_err:
            return self._error(perm_err)

        # Fail if file already exists
        if os.path.exists(file_path):
            return self._error(
                f"File already exists: {file_path}. "
                "Use replace_lines or edit_symbol to modify existing files."
            )

        # Check content size
        content_size = len(content.encode("utf-8"))
        if content_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"Content too large: {content_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Create parent directories
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Atomic write: write to temp file then rename
        try:
            dir_name = os.path.dirname(file_path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".infinibay_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp_path, file_path)
            except Exception:
                os.unlink(tmp_path)
                raise
        except PermissionError:
            return self._error(f"Permission denied: {file_path}")
        except Exception as e:
            return self._error(f"Error creating file: {e}")

        # Compute hash and size
        try:
            with open(file_path, "rb") as f:
                after_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            size_bytes = os.path.getsize(file_path)
        except Exception:
            after_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            size_bytes = content_size

        # Record in artifact_changes for audit
        project_id = self.project_id
        agent_run_id = self.agent_run_id

        def _record_change(conn: sqlite3.Connection):
            conn.execute(
                """INSERT INTO artifact_changes
                   (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (project_id, agent_run_id, file_path, "created", None, after_hash, size_bytes),
            )
            conn.commit()

        try:
            execute_with_retry(_record_change)
        except Exception:
            pass

        self._log_tool_usage(f"Created {file_path} ({size_bytes} bytes)")
        return self._success({
            "file_path": file_path,
            "action": "created",
            "size_bytes": size_bytes,
        })

    def _run_in_pod(self, file_path: str, content: str) -> str:
        """Create file via infinibay-file-helper inside the pod."""
        import json
        req = {"op": "write", "file_path": file_path, "content": content, "mode": "x"}

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

        project_id = self.project_id
        agent_run_id = self.agent_run_id

        def _record_change(conn: sqlite3.Connection):
            conn.execute(
                """INSERT INTO artifact_changes
                   (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (project_id, agent_run_id, file_path, "created",
                 None, data["after_hash"], data["size_bytes"]),
            )
            conn.commit()

        try:
            execute_with_retry(_record_change)
        except Exception:
            pass

        self._log_tool_usage(f"Created {file_path} (pod, {data['size_bytes']} bytes)")
        return self._success({
            "file_path": file_path,
            "action": "created",
            "size_bytes": data["size_bytes"],
        })
