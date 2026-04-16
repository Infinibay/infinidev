"""Tool for listing directory contents."""

import fnmatch
import json
import os
import subprocess
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file.list_directory_input import ListDirectoryInput

# Directories that are almost never interesting to browse.
# Used as fallback when git-based ignore detection is unavailable.
_FALLBACK_SKIP_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
    "dist", "build", ".eggs", ".nox", ".hg", ".svn",
    "coverage", "htmlcov", ".coverage", ".hypothesis",
    ".next", ".nuxt", ".output", ".turbo",
    "target",          # Rust / Java
    "vendor",          # Go (when not explicit)
    "bower_components",
}
_FALLBACK_SKIP_SUFFIXES: set[str] = {".egg-info"}


class ListDirectoryTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "list_directory"
    description: str = "List files and directories at a path."
    args_schema: Type[BaseModel] = ListDirectoryInput

    # ── git-based ignore helpers ──────────────────────────────────────

    @staticmethod
    def _find_git_root(path: str) -> str | None:
        """Return the repo root if *path* lives inside a git repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, cwd=path, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _git_ignored_set(dir_path: str, names: list[str]) -> set[str]:
        """Return the subset of *names* that git would ignore in *dir_path*.

        Uses ``git check-ignore`` which respects .gitignore at every level.
        Falls back to an empty set on any error.
        """
        if not names:
            return set()
        try:
            result = subprocess.run(
                ["git", "check-ignore", "--stdin", "-z"],
                input="\0".join(
                    os.path.join(dir_path, n) for n in names
                ),
                capture_output=True, text=True, cwd=dir_path, timeout=5,
            )
            if result.returncode <= 1:  # 0 = some ignored, 1 = none ignored
                ignored_paths = {
                    p for p in result.stdout.split("\0") if p
                }
                return {os.path.basename(p) for p in ignored_paths}
        except Exception:
            pass
        return set()

    def _should_skip_fallback(self, name: str) -> bool:
        """Fallback skip check when git is not available."""
        if name.startswith("."):
            return True
        if name in _FALLBACK_SKIP_DIRS:
            return True
        for suffix in _FALLBACK_SKIP_SUFFIXES:
            if name.endswith(suffix):
                return True
        return False

    # ── main entry ────────────────────────────────────────────────────

    def _run(
        self,
        file_path: str = ".",
        recursive: bool = False,
        pattern: str | None = None,
        include_ignored: bool = False,
        max_depth: int | None = None,
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

        use_git = not include_ignored and self._find_git_root(file_path) is not None

        entries = []
        count = 0
        max_entries = settings.MAX_DIR_LISTING

        try:
            if recursive:
                for root, dirs, files in os.walk(file_path):
                    # Depth limiting
                    if max_depth is not None:
                        rel_root = os.path.relpath(root, file_path)
                        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
                        if depth >= max_depth:
                            dirs.clear()  # don't descend further
                            continue

                    if not include_ignored:
                        if use_git:
                            ignored = self._git_ignored_set(root, dirs + files)
                            dirs[:] = [
                                d for d in dirs
                                if d not in ignored and d != ".git"
                            ]
                            files = [f for f in files if f not in ignored]
                        else:
                            dirs[:] = [
                                d for d in dirs
                                if not self._should_skip_fallback(d)
                            ]
                            files = [
                                f for f in files
                                if not f.startswith(".")
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
                all_names = sorted(os.listdir(file_path))
                if not include_ignored:
                    if use_git:
                        ignored = self._git_ignored_set(file_path, all_names)
                        all_names = [
                            n for n in all_names
                            if n not in ignored and n != ".git"
                        ]
                    else:
                        all_names = [
                            n for n in all_names
                            if not self._should_skip_fallback(n)
                        ]

                for name in all_names:
                    if count >= max_entries:
                        break
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

