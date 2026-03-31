"""Tool for finding files by name pattern with optional content filtering."""

import json
import os
import re
from pathlib import Path
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


class GlobInput(BaseModel):
    pattern: str = Field(
        ...,
        description=(
            "Glob pattern to match file paths. Supports ** for recursive "
            "matching. Examples: '**/*.py' (all Python files), "
            "'src/**/*.test.ts' (all test files under src), "
            "'**/migrations/*.sql' (all SQL migrations), "
            "'*.md' (markdown files in current dir only)."
        ),
    )
    file_path: str = Field(
        default=".",
        description="Base directory to search from (default: current directory).",
    )
    content_pattern: str | None = Field(
        default=None,
        description=(
            "Optional regex pattern to filter files by content. Only files "
            "whose content matches this pattern will be returned. "
            "Example: 'class.*Tool', 'def test_', 'TODO|FIXME'."
        ),
    )
    case_sensitive: bool = Field(
        default=True,
        description="Whether the content_pattern match is case sensitive (default: true).",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of matching files to return (default: 100).",
    )


class GlobTool(InfinibayBaseTool):
    name: str = "glob"
    description: str = "Find files by glob pattern with optional content filtering."
    args_schema: Type[BaseModel] = GlobInput

    # Directories to always skip
    _SKIP_DIRS: set[str] = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache"}

    def _run(
        self,
        pattern: str,
        file_path: str = ".",
        content_pattern: str | None = None,
        case_sensitive: bool = True,
        max_results: int = 100,
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

        base = Path(file_path)
        matches = []
        scanned = 0

        try:
            for file_path in base.glob(pattern):
                # Skip directories in results
                if file_path.is_dir():
                    continue

                # Skip hidden/excluded directories anywhere in the file_path
                parts = file_path.relative_to(base).parts
                if any(p in self._SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
                    continue
                # Skip hidden files unless the pattern explicitly targets them
                if file_path.name.startswith(".") and not pattern.startswith(".") and "/." not in pattern:
                    continue

                scanned += 1

                # Content filter
                if content_re is not None:
                    try:
                        text = file_path.read_text(encoding="utf-8", errors="replace")
                        if not content_re.search(text):
                            continue
                    except (PermissionError, OSError):
                        continue

                rel = str(file_path.relative_to(base))
                try:
                    stat = file_path.stat()
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
