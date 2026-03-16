"""Tool for listing directory contents."""

import fnmatch
import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="Directory path to list")
    recursive: bool = Field(default=False, description="Whether to list recursively")
    pattern: str | None = Field(
        default=None, description="Glob pattern to filter files (e.g. '*.py')"
    )


class ListDirectoryTool(InfinibayBaseTool):
    name: str = "list_directory"
    description: str = (
        "List files and directories in a given path. "
        "Supports recursive listing and glob pattern filtering. "
        "Automatically excludes .git, __pycache__, node_modules, .venv, "
        ".tox, and hidden files/directories."
    )
    args_schema: Type[BaseModel] = ListDirectoryInput

    def _run(
        self, path: str = ".", recursive: bool = False, pattern: str | None = None
    ) -> str:
        # Normalise — LLMs sometimes pass "None" as a string instead of null
        if pattern is not None and pattern.lower() in ("none", "null", ""):
            pattern = None

        path = self._resolve_path(os.path.expanduser(path))

        if self._is_pod_mode():
            return self._run_in_pod(path, recursive, pattern)

        # Sandbox check (resolves symlinks, enforces directory boundaries)
        sandbox_err = self._validate_sandbox_path(path)
        if sandbox_err:
            return self._error(sandbox_err)

        if not os.path.exists(path):
            return self._error(f"Directory not found: {path}")
        if not os.path.isdir(path):
            return self._error(f"Not a directory: {path}")

        entries = []
        count = 0
        max_entries = settings.MAX_DIR_LISTING

        # Hidden dirs and .git excluded by default
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", ".tox"}

        try:
            if recursive:
                for root, dirs, files in os.walk(path):
                    # Prune hidden/skip dirs
                    dirs[:] = [
                        d for d in dirs
                        if d not in skip_dirs and not d.startswith(".")
                    ]
                    for name in files:
                        if count >= max_entries:
                            break
                        full = os.path.join(root, name)
                        rel = os.path.relpath(full, path)
                        if pattern and not fnmatch.fnmatch(name, pattern):
                            continue
                        try:
                            stat = os.stat(full)
                            entries.append({
                                "path": rel,
                                "size": stat.st_size,
                                "mtime": stat.st_mtime,
                                "type": "file",
                            })
                        except OSError:
                            entries.append({"path": rel, "type": "file"})
                        count += 1
                    if count >= max_entries:
                        break
            else:
                for name in sorted(os.listdir(path)):
                    if count >= max_entries:
                        break
                    if name in skip_dirs or name.startswith("."):
                        continue
                    full = os.path.join(path, name)
                    if pattern and not fnmatch.fnmatch(name, pattern):
                        continue
                    is_dir = os.path.isdir(full)
                    try:
                        stat = os.stat(full)
                        entries.append({
                            "path": name,
                            "size": stat.st_size if not is_dir else None,
                            "mtime": stat.st_mtime,
                            "type": "directory" if is_dir else "file",
                        })
                    except OSError:
                        entries.append({
                            "path": name,
                            "type": "directory" if is_dir else "file",
                        })
                    count += 1
        except PermissionError:
            return self._error(f"Permission denied: {path}")

        truncated = count >= max_entries
        return self._success({
            "path": path,
            "entries": entries,
            "total": len(entries),
            "truncated": truncated,
        })

    def _run_in_pod(
        self, path: str, recursive: bool, pattern: str | None,
    ) -> str:
        """List directory via infinibay-file-helper inside the pod."""
        req = {"op": "list", "path": path, "recursive": recursive}
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
            "path": path,
            "entries": data["entries"],
            "total": data["count"],
            "truncated": data["truncated"],
        })
