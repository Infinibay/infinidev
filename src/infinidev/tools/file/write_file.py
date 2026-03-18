"""Tool for writing file contents with audit trail."""

import hashlib
import json
import os
import stat
import sqlite3
import tempfile
from typing import Literal, Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class WriteFileInput(BaseModel):
    path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")
    mode: Literal["w", "a"] = Field(
        default="w", description="Write mode: 'w' to overwrite, 'a' to append"
    )


class WriteFileTool(InfinibayBaseTool):
    name: str = "write_file"
    description: str = (
        "Write content to a file. Creates parent directories if needed. "
        "Use mode='w' to overwrite or mode='a' to append."
    )
    args_schema: Type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str, mode: str = "w") -> str:
        path = self._resolve_path(os.path.expanduser(path))

        if self._is_pod_mode():
            return self._run_in_pod(path, content, mode)

        # Sandbox check (resolves symlinks, enforces directory boundaries)
        sandbox_err = self._validate_sandbox_path(path)
        if sandbox_err:
            return self._error(sandbox_err)

        # Permission check
        from infinidev.tools.base.permissions import check_file_permission
        perm_err = check_file_permission("write_file", path)
        if perm_err:
            return self._error(perm_err)

        # Check content size before writing
        content_size = len(content.encode("utf-8"))
        existing_size = os.path.getsize(path) if os.path.exists(path) else 0
        total_size = (existing_size + content_size) if mode == "a" else content_size
        if total_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"Content too large: {total_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Compute before-hash if file exists
        before_hash = None
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    before_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                pass

        # Create parent directories
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Atomic write: write to temp file then rename
        try:
            dir_name = os.path.dirname(path)
            if mode == "w":
                original_mode = os.stat(path).st_mode if os.path.exists(path) else None
                fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".infinibay_")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(content)
                    if original_mode is not None:
                        os.chmod(tmp_path, stat.S_IMODE(original_mode))
                    os.replace(tmp_path, path)
                except Exception:
                    os.unlink(tmp_path)
                    raise
            else:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)
        except PermissionError:
            return self._error(f"Permission denied: {path}")
        except Exception as e:
            return self._error(f"Error writing file: {e}")

        # Compute after-hash from the actual file content
        try:
            with open(path, "rb") as f:
                after_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            size_bytes = os.path.getsize(path)
        except Exception:
            after_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            size_bytes = len(content.encode("utf-8"))

        # Record in artifact_changes for audit
        project_id = self.project_id
        agent_run_id = self.agent_run_id
        action = "modified" if before_hash else "created"

        def _record_change(conn: sqlite3.Connection):
            conn.execute(
                """INSERT INTO artifact_changes
                   (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (project_id, agent_run_id, path, action, before_hash, after_hash, size_bytes),
            )
            conn.commit()

        try:
            execute_with_retry(_record_change)
        except Exception:
            pass  # Don't fail the write if audit logging fails

        self._log_tool_usage(f"Wrote {path} ({size_bytes} bytes, {action})")
        return self._success({
            "path": path,
            "action": action,
            "size_bytes": size_bytes,
        })

    def _run_in_pod(self, path: str, content: str, mode: str) -> str:
        """Write file via infinibay-file-helper inside the pod."""
        req = {"op": "write", "path": path, "content": content, "mode": mode}

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
                (project_id, agent_run_id, path, data["action"],
                 data.get("before_hash"), data["after_hash"], data["size_bytes"]),
            )
            conn.commit()

        try:
            execute_with_retry(_record_change)
        except Exception:
            pass

        self._log_tool_usage(f"Wrote {path} (pod, {data['size_bytes']} bytes, {data['action']})")
        return self._success({
            "path": path,
            "action": data["action"],
            "size_bytes": data["size_bytes"],
        })
