"""Tool for listing directory contents."""

import fnmatch
import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file.list_directory_input import ListDirectoryInput


class ListDirectoryTool(InfinibayBaseTool):
    name: str = "list_directory"
    description: str = "List files and directories at a path."
    args_schema: Type[BaseModel] = ListDirectoryInput

    def _run(
        self, file_path: str = ".", recursive: bool = False, pattern: str | None = None
    ) -> str:
        # Normalise — LLMs sometimes pass "None" as a string instead of null
        if pattern is not None and pattern.lower() in ("none", "null", ""):
            pattern = None

        file_path = self._resolve_path(os.path.expanduser(file_path))

        if self._is_pod_mode():
            return self._run_in_pod(file_path, recursive, pattern)

        # Sandbox check (resolves symlinks, enforces directory boundaries)
        sandbox_err = self._validate_sandbox_path(file_path)
        if sandbox_err:
            return self._error(sandbox_err)

        if not os.path.exists(file_path):
            return self._error(f"Directory not found: {file_path}")
        if not os.path.isdir(file_path):
            return self._error(f"Not a directory: {file_path}")

        entries = []
        count = 0
        max_entries = settings.MAX_DIR_LISTING

        # Hidden dirs and .git excluded by default
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", ".tox"}

        try:
            if recursive:
                for root, dirs, files in os.walk(file_path):
                    # Prune hidden/skip dirs
                    dirs[:] = [
                        d for d in dirs
                        if d not in skip_dirs and not d.startswith(".")
                    ]
                    for name in files:
                        if count >= max_entries:
                            break
                        full = os.path.join(root, name)
                        rel = os.path.relpath(full, file_path)
                        if pattern and not fnmatch.fnmatch(name, pattern):
                            continue
                        try:
                            stat = os.stat(full)
                            entries.append({
                                "file_path": rel,
                                "size": stat.st_size,
                                "mtime": stat.st_mtime,
                                "type": "file",
                            })
                        except OSError:
                            entries.append({"file_path": rel, "type": "file"})
                        count += 1
                    if count >= max_entries:
                        break
            else:
                for name in sorted(os.listdir(file_path)):
                    if count >= max_entries:
                        break
                    if name in skip_dirs or name.startswith("."):
                        continue
                    full = os.path.join(file_path, name)
                    if pattern and not fnmatch.fnmatch(name, pattern):
                        continue
                    is_dir = os.path.isdir(full)
                    try:
                        stat = os.stat(full)
                        entries.append({
                            "file_path": name,
                            "size": stat.st_size if not is_dir else None,
                            "mtime": stat.st_mtime,
                            "type": "directory" if is_dir else "file",
                        })
                    except OSError:
                        entries.append({
                            "file_path": name,
                            "type": "directory" if is_dir else "file",
                        })
                    count += 1
        except PermissionError:
            return self._error(f"Permission denied: {file_path}")

        truncated = count >= max_entries
        return self._success({
            "file_path": file_path,
            "entries": entries,
            "total": len(entries),
            "truncated": truncated,
        })

    def _run_in_pod(
        self, file_path: str, recursive: bool, pattern: str | None,
    ) -> str:
        """List directory via infinibay-file-helper inside the pod."""
        req = {"op": "list", "file_path": file_path, "recursive": recursive}
        if pattern:
            req["pattern"] = pattern

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
        return self._success({
            "file_path": file_path,
            "entries": data["entries"],
            "total": data["count"],
            "truncated": data["truncated"],
        })

