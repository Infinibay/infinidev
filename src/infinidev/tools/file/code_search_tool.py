"""Tool for searching code patterns across a codebase."""

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.engine.subprocess_runner import run_captured
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file.code_search_input import CodeSearchInput


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell usage."""
    return shlex.quote(s)


def _limited(metadata: dict[str, Any], reason: str, *, unknown: bool = False) -> None:
    if reason not in metadata["reasons"]:
        metadata["reasons"].append(reason)
    if unknown or metadata["coverage"] == "unknown":
        metadata["coverage"] = "unknown"
    else:
        metadata["coverage"] = "limited"


def _record_process(metadata: dict[str, Any], result: Any, *, aggregate: bool = False) -> None:
    """Record mechanical backend facts without interpreting diagnostic prose."""
    metadata["backend_exit_code"] = result.exit_code
    if result.stderr:
        metadata["backend_stderr"] = result.stderr[:2000]
        metadata["backend_stderr_truncated"] = len(result.stderr) > 2000
        _limited(metadata, "backend_stderr")
    if result.exit_code == 123:
        _limited(metadata, "aggregate_exit" if aggregate else "backend_exit_uncertain",
                 unknown=True)
    if result.exit_code in (1, 123) and result.stdout:
        _limited(metadata, "omitted_output", unknown=True)
    if getattr(result, "timed_out", False):
        _limited(metadata, "backend_timeout", unknown=True)


def _directory_record(line: str, *, git_backend: bool) -> tuple[str, int, str, bool] | None:
    """Decode filename-NUL records without guessing separators inside paths."""
    filename, separator, payload = line.partition("\0")
    if not separator or not filename:
        return None
    if git_backend:
        fields = payload.split("\0")
        if len(fields) == 3:
            number, column, content = fields
            if not re.fullmatch(r"[1-9][0-9]*", column):
                return None
            matched = True
        elif len(fields) == 2:
            number, content = fields
            matched = False
        else:
            return None
    else:
        parsed = re.fullmatch(r"([1-9][0-9]*)([:-])(.*)", payload)
        if parsed is None:
            return None
        number, marker, content = parsed.groups()
        matched = marker == ":"
    if not re.fullmatch(r"[1-9][0-9]*", number):
        return None
    return filename, int(number), content, matched


class CodeSearchTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "code_search"
    description: str = "Search code files for a text pattern or regex."
    args_schema: Type[BaseModel] = CodeSearchInput

    @staticmethod
    def _has_git(path: str) -> bool:
        """Return True when Git can search the requested directory faithfully."""
        try:
            result = run_captured(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path,
                timeout=5,
            )
            if result.timed_out or result.exit_code != 0:
                return False
            ignored = run_captured(
                ["git", "check-ignore", "-q", "."],
                cwd=path,
                timeout=5,
            )
            return ignored.timed_out or ignored.exit_code != 0
        except OSError:
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
        metadata: dict[str, Any] = {
            "schema_version": 1, "outcome": "error", "mode": "regex",
            "backend": "unknown", "dialect": "unknown", "scope": "unknown",
            "root": None, "coverage": "unknown", "filters_applied": {}, "reasons": [],
        }
        body = self._run_search(pattern, file_path, file_extensions, case_sensitive,
                                max_results, context_lines, max_depth, _metadata=metadata)
        payload = json.loads(body)
        if "error" in payload:
            if metadata["outcome"] != "unavailable":
                metadata["outcome"] = "error"
            metadata["coverage"] = "unknown"
        else:
            if payload["truncated"]:
                _limited(metadata, "result_limit")
            metadata["outcome"] = (
                "partial" if metadata["coverage"] != "completed_scope"
                else "matches" if payload["match_count"] else "empty")
        payload["search_metadata"] = metadata
        return json.dumps(payload)

    def _run_search(
        self,
        pattern: str = "",
        file_path: str = ".",
        file_extensions: list[str] | None = None,
        case_sensitive: bool = True,
        max_results: int = 50,
        context_lines: int = 0,
        max_depth: int | None = None,
        *,
        _metadata: dict[str, Any],
    ) -> str:
        file_path = self._resolve_path(os.path.expanduser(file_path))
        _metadata["root"] = file_path

        if self._is_pod_mode():
            _metadata.update(backend="pod_helper", mode="unknown", scope="remote_unknown",
                             filters_applied=None)
            _metadata["reasons"].append("remote_metadata_unknown")
            return self._run_in_pod(
                pattern, file_path, file_extensions, case_sensitive,
                max_results, context_lines, _metadata=_metadata,
            )

        # Sandbox check (resolves symlinks, enforces directory boundaries)
        sandbox_err = self._validate_sandbox_path(file_path)
        if sandbox_err:
            return self._error(sandbox_err)

        if os.path.isfile(file_path):
            _metadata.update(backend="python_re", dialect="python", scope="file",
                             coverage="completed_scope", filters_applied={
                                 "case_sensitive": case_sensitive,
                                 "max_results": max_results, "context_lines": context_lines})
            return self._search_single_file(
                pattern,
                file_path,
                case_sensitive=case_sensitive,
                max_results=max_results,
                context_lines=context_lines,
            )
        if not os.path.isdir(file_path):
            return self._error(f"Directory not found: {file_path}")

        # Prefer git grep when inside a git repo — it respects .gitignore
        # and skips untracked build artifacts automatically.
        use_git_grep = self._has_git(file_path)
        _metadata.update(backend="git_grep" if use_git_grep else "grep", dialect="ere",
                         scope="directory", coverage="completed_scope", filters_applied={
                             "file_extensions": [ext if ext.startswith(".") else f".{ext}"
                                                 for ext in file_extensions] if file_extensions else None,
                             "max_depth": max_depth, "case_sensitive": case_sensitive,
                             "max_results": max_results, "context_lines": context_lines,
                             "excludes": "git_defaults" if use_git_grep else "grep_defaults"})
        if use_git_grep and max_depth is not None and file_extensions:
            # Git documents wildcard pathspecs as an exception to --max-depth;
            # effectiveness differs from merely supplying the flag.
            _metadata["filters_applied"]["max_depth"] = None
            _limited(_metadata, "git_depth_with_pathspec_uncertain", unknown=True)
        elif not use_git_grep and max_depth is not None:
            # find supplies files directly, so grep's recursive directory
            # exclusions do not prune their parent directories.
            _metadata["filters_applied"]["excludes"] = None
            _limited(_metadata, "directory_exclusions_not_applied", unknown=True)

        if use_git_grep:
            # Include non-ignored untracked files: an editor or coding agent
            # must be able to find a file before its first git add.
            cmd = ["git", "grep", "--untracked", "-n", "--column", "-z", "-E"]
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
            # Fallback to plain grep. Keep -r and -n as separate tokens so the
            # max_depth file-list branch below can strip the recursion flag.
            cmd = ["grep", "-r", "-n", "--null", "-E"]
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
                cmd.append("-H")
                cmd.extend(["--", pattern])
                shell_cmd = (
                    " ".join(_shell_quote(c) for c in find_cmd)
                    + " -print0 | xargs -0 "
                    + " ".join(_shell_quote(c) for c in cmd)
                )
                try:
                    result = run_captured(
                        shell_cmd,
                        shell=True,
                        cwd=file_path,
                        timeout=30,
                    )
                except OSError:
                    _metadata["outcome"] = "unavailable"
                    return self._error("grep is not installed or not in PATH")
                _record_process(_metadata, result, aggregate=True)
                if result.timed_out:
                    return self._error("Search timed out (pattern may be too broad)")

                if result.exit_code in (1, 123):
                    return json.dumps({
                        "pattern": pattern, "file_path": file_path,
                        "match_count": 0, "truncated": False, "matches": [],
                    })
                if result.exit_code not in (0, 1, 123):
                    return self._error(f"Search failed: {result.stderr.strip()}")

                # Skip the normal command execution below
                cmd = None
            else:
                cmd.extend(["--", pattern, file_path])

        if cmd is not None:
            try:
                result = run_captured(
                    cmd,
                    cwd=file_path,
                    timeout=30,
                )
            except OSError:
                _metadata["outcome"] = "unavailable"
                return self._error(
                    "git/grep is not installed or not in PATH"
                )
            _record_process(_metadata, result)
            if result.timed_out:
                return self._error("Search timed out (pattern may be too broad)")

        # grep/git-grep returns 1 when no matches found
        if result.exit_code in (1, 123):
            return json.dumps({
                "pattern": pattern,
                "file_path": file_path,
                "match_count": 0,
                "truncated": False,
                "matches": [],
            })
        if result.exit_code not in (0, 1):
            return self._error(f"Search failed: {result.stderr.strip()}")

        # Git grep and find emit paths relative to their search cwd.
        # File tools interpret relative paths from the agent's workspace.
        match_base = Path(file_path)
        workspace = Path(self._resolve_path("."))
        if match_base.is_relative_to(workspace):
            match_base = match_base.relative_to(workspace)

        matches = []
        context_by_file: dict[str, dict[int, str]] = {}
        hit_count = 0
        unparsed = 0
        lines = result.stdout.split("\n")
        if lines[-1] == "":
            lines.pop()
        for line in lines:
            if line == "--":
                continue
            record = _directory_record(line, git_backend=use_git_grep)
            if record is None:
                unparsed += 1
                continue
            filename, number, content, matched = record
            filename = str(match_base / filename)
            if context_lines:
                context_by_file.setdefault(filename, {})[number] = content
            if matched:
                hit_count += 1
                if len(matches) < max_results:
                    matches.append({"file": filename, "line": number, "content": content})
        # Match caps count hits, never surrounding context rows.
        hit_cap = hit_count >= max_results
        if context_lines:
            for match in matches:
                nearby = context_by_file[match["file"]]
                match["context"] = [
                    {"line": number, "content": nearby[number]}
                    for number in range(max(1, match["line"] - context_lines),
                                        match["line"] + context_lines + 1)
                    if number in nearby
                ]

        # Only flag truncation when the match cap was actually reached.
        # The old heuristic compared len(lines) to len(matches), but `lines`
        # also contains context lines (context_lines>0) and "--" group
        # separators, so it reported truncated=True for complete results.
        truncated = hit_cap
        if unparsed:
            _metadata["unparsed_output_lines"] = unparsed
            _limited(_metadata, "unparsed_output", unknown=True)

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

    def _search_single_file(
        self,
        pattern: str,
        file_path: str,
        *,
        case_sensitive: bool,
        max_results: int,
        context_lines: int,
    ) -> str:
        """Search one file directly when the model supplies a file path."""
        try:
            matcher = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            return self._error(f"Invalid search regex: {exc}")
        try:
            with open(file_path, encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError as exc:
            return self._error(f"Could not read {file_path}: {exc}")

        matches: list[dict[str, object]] = []
        hit_cap = False
        for index, line in enumerate(lines):
            if not matcher.search(line):
                continue
            if len(matches) >= max_results:
                hit_cap = True
                break
            start = max(0, index - context_lines)
            end = min(len(lines), index + context_lines + 1)
            match: dict[str, object] = {
                "file": file_path,
                "line": index + 1,
                "content": line,
            }
            if context_lines:
                match["context"] = [
                    {"line": line_index + 1, "content": lines[line_index]}
                    for line_index in range(start, end)
                ]
            matches.append(match)

        return json.dumps({
            "pattern": pattern,
            "file_path": file_path,
            "match_count": len(matches),
            "truncated": hit_cap,
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
        *,
        _metadata: dict[str, Any] | None = None,
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

        if _metadata is not None:
            _record_process(_metadata, result)
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
