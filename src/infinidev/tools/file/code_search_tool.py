"""Tool for searching code patterns across a codebase."""

import json
import os
import subprocess
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file.code_search_input import CodeSearchInput


class CodeSearchTool(InfinibayBaseTool):
    name: str = "code_search"
    description: str = "Search code files for a text pattern or regex."
    args_schema: Type[BaseModel] = CodeSearchInput

    def _run(
        self,
        pattern: str = "",
        file_path: str = ".",
        file_extensions: list[str] | None = None,
        case_sensitive: bool = True,
        max_results: int = 50,
        context_lines: int = 0,
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

        cmd = ["grep", "-rn", "-E"]

        if not case_sensitive:
            cmd.append("-i")

        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])

        if file_extensions:
            for ext in file_extensions:
                ext = ext if ext.startswith(".") else f".{ext}"
                cmd.extend(["--include", f"*{ext}"])

        # Exclude common non-source directories
        for exclude_dir in [".git", "node_modules", "__pycache__", ".venv", "venv"]:
            cmd.extend(["--exclude-dir", exclude_dir])

        cmd.extend(["--", pattern, file_path])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return self._error("Search timed out (pattern may be too broad)")
        except FileNotFoundError:
            return self._error("grep is not installed or not in PATH")

        # grep returns 1 when no matches found (not an error)
        if result.returncode == 1:
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

