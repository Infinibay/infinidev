"""Tool for reading file contents with optional line-range selection."""

import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


class ReadFileInput(BaseModel):
    path: str = Field(..., description="Path to the file to read")
    offset: int | None = Field(
        default=None,
        description=(
            "Line number to start reading from (1-based). "
            "If omitted, reads from the beginning of the file."
        ),
    )
    limit: int | None = Field(
        default=None,
        description=(
            "Maximum number of lines to read. "
            "If omitted, reads until the end of the file."
        ),
    )


class ReadFileTool(InfinibayBaseTool):
    name: str = "read_file"
    description: str = (
        "Read the contents of a file. Returns the file content as numbered "
        "lines. Use `offset` and `limit` to read a specific range of lines "
        "instead of the entire file — this is strongly recommended for large "
        "files. Combine with code_search to find the relevant line numbers "
        "first, then read only the region you need."
    )
    args_schema: Type[BaseModel] = ReadFileInput

    def _run(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        path = self._resolve_path(os.path.expanduser(path))

        if self._is_pod_mode():
            return self._run_in_pod(path, offset, limit)

        # Sandbox check (resolves symlinks, enforces directory boundaries)
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

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except PermissionError:
            return self._error(f"Permission denied: {path}")
        except Exception as e:
            return self._error(f"Error reading file: {e}")

        total_lines = len(all_lines)

        # Apply offset/limit for partial reads
        if offset is not None or limit is not None:
            start = max((offset or 1) - 1, 0)  # convert 1-based to 0-based
            end = start + limit if limit is not None else total_lines
            selected = all_lines[start:end]
            # Format with line numbers for easy reference
            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")
            content = "\n".join(numbered)
            desc = f"lines {start + 1}-{min(end, total_lines)} of {total_lines}"
        else:
            # Full read — still add line numbers for consistency
            numbered = []
            for i, line in enumerate(all_lines, start=1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")
            content = "\n".join(numbered)
            desc = f"{total_lines} lines"

        self._log_tool_usage(f"Read {path} ({desc})")
        return content

    def _run_in_pod(
        self, path: str, offset: int | None, limit: int | None,
    ) -> str:
        """Read file via infinibay-file-helper inside the pod."""
        req = {"op": "read", "path": path}
        if offset is not None:
            req["offset"] = offset
        if limit is not None:
            req["limit"] = limit

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
        self._log_tool_usage(f"Read {path} (pod, {data.get('total_lines', '?')} lines)")
        return data["content"]
