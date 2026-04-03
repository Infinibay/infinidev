"""Tool for finding files by name pattern with optional content filtering."""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.file.glob_input import GlobInput

# Shared with list_directory — dirs that are almost never interesting to browse.
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


class GlobTool(InfinibayBaseTool):
    name: str = "glob"
    description: str = "Find files by glob pattern with optional content filtering."
    args_schema: Type[BaseModel] = GlobInput

    # ── git-based ignore helpers ──────────────────────────────────────

    @staticmethod
    def _find_git_root(path: str) -> str | None:
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
    def _git_is_ignored(file_path: str, cwd: str) -> bool:
        """Check if a single path is git-ignored."""
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", file_path],
                capture_output=True, cwd=cwd, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _should_skip_fallback(name: str) -> bool:
        if name.startswith("."):
            return True
        if name in _FALLBACK_SKIP_DIRS:
            return True
        for suffix in _FALLBACK_SKIP_SUFFIXES:
            if name.endswith(suffix):
                return True
        return False

    def _is_ignored_path(
        self, parts: tuple[str, ...], base_str: str, use_git: bool,
    ) -> bool:
        """Check if any directory component of the path should be skipped."""
        for p in parts[:-1]:  # check directory components, not the file itself
            if use_git:
                if p == ".git":
                    return True
            else:
                if self._should_skip_fallback(p):
                    return True
        # For git mode, check the full path at once
        if use_git and len(parts) > 1:
            full_rel = os.path.join(*parts)
            return self._git_is_ignored(full_rel, base_str)
        return False

    # ── main entry ────────────────────────────────────────────────────

    def _run(
        self,
        pattern: str,
        file_path: str = ".",
        content_pattern: str | None = None,
        case_sensitive: bool = True,
        max_results: int = 100,
        include_ignored: bool = False,
        max_depth: int | None = None,
    ) -> str:
        file_path = self._resolve_path(os.path.expanduser(file_path))

        if self._is_pod_mode():
            return self._run_in_pod(
                pattern, file_path, content_pattern, case_sensitive, max_results,
            )

        # Sandbox check
        sandbox_err = self._validate_sandbox_path(file_path)
        if sandbox_err:
            return self._error(sandbox_err)

        if not os.path.isdir(file_path):
            return self._error(f"Directory not found: {file_path}")

        # Compile content regex if provided
        content_re = None
        if content_pattern:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                content_re = re.compile(content_pattern, flags)
            except re.error as e:
                return self._error(f"Invalid content_pattern regex: {e}")

        use_git = (
            not include_ignored and self._find_git_root(file_path) is not None
        )

        # Batch git check-ignore for performance: collect all glob results
        # first, then filter in one pass.
        base = Path(file_path)
        raw_paths: list[Path] = []
        for p in base.glob(pattern):
            if p.is_dir():
                continue
            # Depth filter: count path components relative to base
            if max_depth is not None:
                parts = p.relative_to(base).parts
                if len(parts) > max_depth:
                    continue
            raw_paths.append(p)

        # Build ignore set via batch git check-ignore
        ignored_rels: set[str] = set()
        if not include_ignored and use_git and raw_paths:
            rel_strs = []
            for p in raw_paths:
                rel_strs.append(str(p.relative_to(base)))
            try:
                result = subprocess.run(
                    ["git", "check-ignore", "--stdin", "-z"],
                    input="\0".join(rel_strs),
                    capture_output=True, text=True,
                    cwd=file_path, timeout=10,
                )
                if result.returncode <= 1:
                    ignored_rels = {
                        s for s in result.stdout.split("\0") if s
                    }
            except Exception:
                pass

        matches = []
        scanned = 0

        try:
            for fpath in raw_paths:
                parts = fpath.relative_to(base).parts
                rel = str(fpath.relative_to(base))

                if not include_ignored:
                    if use_git:
                        # Always skip .git itself; use batch result for the rest
                        if any(p == ".git" for p in parts[:-1]):
                            continue
                        if rel in ignored_rels:
                            continue
                    else:
                        # Fallback: skip known junk dirs and dotfiles
                        if any(self._should_skip_fallback(p) for p in parts[:-1]):
                            continue
                        if (
                            fpath.name.startswith(".")
                            and not pattern.startswith(".")
                            and "/." not in pattern
                        ):
                            continue

                scanned += 1

                # Content filter
                if content_re is not None:
                    try:
                        text = fpath.read_text(encoding="utf-8", errors="replace")
                        if not content_re.search(text):
                            continue
                    except (PermissionError, OSError):
                        continue

                try:
                    stat = fpath.stat()
                    matches.append({
                        "file_path": rel,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                except OSError:
                    matches.append({"file_path": rel})

                if len(matches) >= max_results:
                    break
        except PermissionError:
            return self._error(f"Permission denied: {file_path}")

        truncated = len(matches) >= max_results

        content_desc = f", content ~/{content_pattern}/" if content_pattern else ""
        self._log_tool_usage(
            f"Glob '{pattern}' in {file_path}{content_desc} — "
            f"{len(matches)} matches ({scanned} scanned)"
        )

        return json.dumps({
            "pattern": pattern,
            "file_path": file_path,
            "content_pattern": content_pattern,
            "match_count": len(matches),
            "truncated": truncated,
            "matches": matches,
        })

    def _run_in_pod(
        self,
        pattern: str,
        file_path: str,
        content_pattern: str | None,
        case_sensitive: bool,
        max_results: int,
    ) -> str:
        """Glob via infinibay-file-helper inside the pod."""
        req = {
            "op": "glob",
            "pattern": pattern,
            "file_path": file_path,
            "case_sensitive": case_sensitive,
            "max_results": max_results,
        }
        if content_pattern:
            req["content_pattern"] = content_pattern

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
        content_desc = f", content ~/{content_pattern}/" if content_pattern else ""
        self._log_tool_usage(
            f"Glob '{pattern}' in {file_path}{content_desc} (pod) — "
            f"{data['match_count']} matches"
        )

        return json.dumps({
            "pattern": pattern,
            "file_path": file_path,
            "content_pattern": content_pattern,
            "match_count": data["match_count"],
            "truncated": data["truncated"],
            "matches": data["matches"],
        })
