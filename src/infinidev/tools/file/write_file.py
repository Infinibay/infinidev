"""Tool for writing file contents with audit trail."""

import hashlib
import json
import os
from typing import Literal, Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


class WriteFileInput(BaseModel):
    model_config = {"populate_by_name": True}

    file_path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")
    mode: Literal["w", "a"] = Field(
        default="w", description="Write mode: 'w' to overwrite, 'a' to append"
    )
    reason: str = Field(
        default="",
        alias="description",
        description=(
            "Brief explanation of WHY this file is being created/written. "
            "Used by the code reviewer to understand intent."
        ),
    )


class WriteFileTool(InfinibayBaseTool):
    name: str = "write_file"
    description: str = (
        "Write content to a file. Creates parent directories if needed. "
        "Use mode='w' to overwrite or mode='a' to append."
    )
    args_schema: Type[BaseModel] = WriteFileInput

    def _run(self, file_path: str, content: str, mode: str = "w", reason: str = "") -> str:
        file_path = self._resolve_path(os.path.expanduser(file_path))

        if self._is_pod_mode():
            return self._run_in_pod(file_path, content, mode)

        access_err = guard_file_access(self, file_path, "write_file")
        if access_err:
            return access_err

        # Check content size before writing
        content_size = len(content.encode("utf-8"))
        existing_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        total_size = (existing_size + content_size) if mode == "a" else content_size
        if total_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"Content too large: {total_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Compute before-hash if file exists
        before_hash = None
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    before_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                pass

        # Create parent directories
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Atomic write: write to temp file then rename
        try:
            if mode == "w":
                atomic_write(file_path, content)
            else:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)
        except PermissionError:
            return self._error(f"Permission denied: {file_path}")
        except Exception as e:
            return self._error(f"Error writing file: {e}")

        # Compute after-hash from the actual file content
        try:
            with open(file_path, "rb") as f:
                after_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            size_bytes = os.path.getsize(file_path)
        except Exception:
            after_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            size_bytes = len(content.encode("utf-8"))

        # Record in artifact_changes for audit
        action = "modified" if before_hash else "created"
        record_artifact_change(self, file_path, action, before_hash, after_hash, size_bytes)

        self._log_tool_usage(f"Wrote {file_path} ({size_bytes} bytes, {action})")
        result = {
            "file_path": file_path,
            "action": action,
            "size_bytes": size_bytes,
        }
        if reason:
            result["reason"] = reason
        return self._success(result)

    def _run_in_pod(self, file_path: str, content: str, mode: str) -> str:
        """Write file via infinibay-file-helper inside the pod."""
        req = {"op": "write", "file_path": file_path, "content": content, "mode": mode}

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
        record_artifact_change(
            self, file_path, data["action"],
            data.get("before_hash"), data["after_hash"], data["size_bytes"],
        )

        self._log_tool_usage(f"Wrote {file_path} (pod, {data['size_bytes']} bytes, {data['action']})")
        return self._success({
            "file_path": file_path,
            "action": data["action"],
            "size_bytes": data["size_bytes"],
        })
