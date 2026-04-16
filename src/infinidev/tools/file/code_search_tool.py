"""Tool for searching code patterns across a codebase."""

import json
import os
import shlex
import subprocess
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file.code_search_input import CodeSearchInput


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell usage."""
    return shlex.quote(s)


class CodeSearchTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "code_search"
    description: str = "Search code files for a text pattern or regex."
    args_schema: Type[BaseModel] = CodeSearchInput

    @staticmethod
    def _has_git(path: str) -> bool:
        """Return True if *path* is inside a git repository."""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, cwd=path, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _run(
        self,
        pattern: str = "",
        file_path: str = ".",
        file_extensions: list[str] | None = None,
        case_sensitive: bool = True,
        max_results: int = 50,
        context_lines: int = 0,
        max_depth: int | None = None,
    ) -> str:
        file_path = self._resolve_path(os.path.expanduser(file_path))

        if self._is_pod_mode():
            return self._run_in_pod(
                pattern, file_path, file_extensions, case_sensitive,
                max_results, context_lines,
            )

        # Sandbox check (resolves symlinks, enforces directory boundaries)
        sandbox_err = self._validate_sandbox_path(file_path)
        if sandbox_err:
            return self._error(sandbox_err)

        if not os.path.isdir(file_path):
            return self._error(f"Directory not found: {file_path}")

        # Prefer git grep when inside a git repo — it respects .gitignore
        # and skips untracked build artifacts automatically.
        use_git_grep = self._has_git(file_path)

        if use_git_grep:
            cmd = ["git", "grep", "-n", "-E"]
            if not case_sensitive:
                cmd.append("-i")
            if context_lines > 0:
                cmd.extend(["-C", str(context_lines)])
            if max_depth is not None:
                cmd.extend(["--max-depth", str(max_depth)])
            cmd.extend(["--", pattern])
            if file_extensions:
                for ext in file_extensions:
                    ext = ext if ext.startswith(".") else f".{ext}"
                    cmd.append(f"*{ext}")
        else:
            # Fallback to plain grep
            cmd = ["grep", "-rn", "-E"]
            if not case_sensitive:
                cmd.append("-i")
            if context_lines > 0:
                cmd.extend(["-C", str(context_lines)])
            if file_extensions:
                for ext in file_extensions:
                    ext = ext if ext.startswith(".") else f".{ext}"
                    cmd.extend(["--include", f"*{ext}"])
            for exclude_dir in [
                ".git", "node_modules", "__pycache__", ".venv", "venv",
                ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
                "dist", "build", ".eggs", ".nox", ".tox",
            ]:
                cmd.extend(["--exclude-dir", exclude_dir])

            if max_depth is not None:
                # Use find with -maxdepth piped to xargs grep
                find_cmd = ["find", ".", "-maxdepth", str(max_depth), "-type", "f"]
                if file_extensions:
                    find_expr = []
                    for ext in file_extensions:
                        ext = ext if ext.startswith(".") else f".{ext}"
                        if find_expr:
                            find_expr.append("-o")
                        find_expr.extend(["-name", f"*{ext}"])
                    if find_expr:
                        find_cmd.extend(["("] + find_expr + [")"])
                # Strip -r from grep for file-list mode
                cmd = [g for g in cmd if g != "-r"]
                cmd.extend(["--", pattern])
                shell_cmd = (
                    " ".join(_shell_quote(c) for c in find_cmd)
                    + " -print0 | xargs -0 "
                    + " ".join(_shell_quote(c) for c in cmd)
                )
                try:
                    result = subprocess.run(
                        shell_cmd, shell=True,
                        capture_output=True, text=True, timeout=30,
                        cwd=file_path,
                    )
                except subprocess.TimeoutExpired:
                    return self._error("Search timed out (pattern may be too broad)")
                except FileNotFoundError:
                    return self._error("grep is not installed or not in PATH")

                if result.returncode in (1, 123):
                    return json.dumps({
                        "pattern": pattern, "file_path": file_path,
                        "match_count": 0, "truncated": False, "matches": [],
                    })
                if result.returncode not in (0, 1, 123):
                    return self._error(f"Search failed: {result.stderr.strip()}")

                # Skip the normal command execution below
                cmd = None
            else:
                cmd.extend(["--", pattern, file_path])

        if cmd is not None:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30,
                    cwd=file_path,
                )
            except subprocess.TimeoutExpired:
                return self._error("Search timed out (pattern may be too broad)")
            except FileNotFoundError:
                return self._error(
                    "git/grep is not installed or not in PATH"
                )

        # grep/git-grep returns 1 when no matches found
        if result.returncode in (1, 123):
            return json.dumps({
                "pattern": pattern,
                "file_path": file_path,
                "match_count": 0,
                "truncated": False,
                "matches": [],
            })
        if result.returncode not in (0, 1):
            return self._error(f"Search failed: {result.stderr.strip()}")

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

        # Parse matches (skip context separator lines "--")
        matches = []
        for line in lines:
            if line == "--":
                continue
            # grep output format: file:line:content (or file-line-content for context)
            # Use partition to handle colons in content
            first_sep = line.find(":")
            if first_sep == -1:
                continue
            file_part = line[:first_sep]
            rest = line[first_sep + 1:]
            second_sep = rest.find(":")
            if second_sep == -1:
                # Context line (uses - separator)
                second_sep = rest.find("-")
                if second_sep == -1:
                    continue
            line_num_str = rest[:second_sep]
            content = rest[second_sep + 1:]

            try:
                line_num = int(line_num_str)
            except ValueError:
                continue

            matches.append({
                "file": file_part,
                "line": line_num,
                "content": content,
            })

            if len(matches) >= max_results:
                break

        truncated = len(lines) > len(matches) or (
            len(matches) == max_results and len(lines) > max_results
        )

        self._log_tool_usage(
            f"Searched '{pattern}' in {file_path} — {len(matches)} matches"
        )

        return json.dumps({
            "pattern": pattern,
            "file_path": file_path,
            "match_count": len(matches),
            "truncated": truncated,
            "matches": matches,
        })

    def _run_in_pod(
        self,
        pattern: str,
        file_path: str,
        file_extensions: list[str] | None,
        case_sensitive: bool,
        max_results: int,
        context_lines: int,
    ) -> str:
        """Search code via infinibay-file-helper inside the pod."""
        req = {
            "op": "search",
            "pattern": pattern,
            "file_path": file_path,
            "case_sensitive": case_sensitive,
            "max_results": max_results,
            "context_lines": context_lines,
        }
        if file_extensions:
            req["file_extensions"] = file_extensions

        try:
            result = self._exec_in_pod(
                ["infinibay-file-helper"],
                stdin_data=json.dumps(req),
                timeout=30,
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
        self._log_tool_usage(
            f"Searched '{pattern}' in {file_path} (pod) — {data['match_count']} matches"
        )
        return json.dumps({
            "pattern": pattern,
            "file_path": file_path,
            "match_count": data["match_count"],
            "truncated": False,
            "matches": data["matches"],
        })

