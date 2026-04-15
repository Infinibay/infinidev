"""Tool for creating new files with audit trail."""

import hashlib
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.file._helpers import (
    guard_file_access,
    atomic_write,
    record_artifact_change,
    check_syntax_warning,
)
from infinidev.tools.file.create_file_input import CreateFileInput


class CreateFileTool(InfinibayBaseTool):
    name: str = "create_file"
    description: str = "Create a new file. Fails if the file already exists."
    args_schema: Type[BaseModel] = CreateFileInput

    def _run(self, file_path: str, content: str) -> str:
        file_path = self._resolve_path(os.path.expanduser(file_path))

        if self._is_pod_mode():
            return self._run_in_pod(file_path, content)

        access_err = guard_file_access(self, file_path, "create_file")
        if access_err:
            return access_err

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

        # Pre-write syntax check (tree-sitter) — advisory only, never blocks
        syntax_warn = check_syntax_warning(self, file_path, content, operation="create_file")

        # Create parent directories
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Atomic write: write to temp file then rename
        try:
            atomic_write(file_path, content)
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
        record_artifact_change(self, file_path, "created", None, after_hash, size_bytes)

        self._log_tool_usage(f"Created {file_path} ({size_bytes} bytes)")
        result = {
            "file_path": file_path,
            "action": "created",
            "size_bytes": size_bytes,
        }
        if syntax_warn:
            result["syntax_warning"] = syntax_warn
        return self._success(result)

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

        record_artifact_change(
            self, file_path, "created",
            None, data["after_hash"], data["size_bytes"],
        )

        self._log_tool_usage(f"Created {file_path} (pod, {data['size_bytes']} bytes)")
        return self._success({
            "file_path": file_path,
            "action": "created",
            "size_bytes": data["size_bytes"],
        })

