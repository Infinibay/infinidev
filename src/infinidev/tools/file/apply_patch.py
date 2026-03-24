"""Tool for applying unified diff patches to files."""

import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import tempfile
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class ApplyPatchInput(BaseModel):
    patch: str = Field(
        ...,
        description=(
            "A unified diff string (like output of `git diff`). "
            "Can contain changes to one or multiple files. "
            "Must include diff headers (--- a/file, +++ b/file) and hunks (@@ ... @@)."
        ),
    )
    strip: int = Field(
        default=1,
        description="Number of leading path components to strip (like `patch -pN`). Default 1.",
    )


class ApplyPatchTool(InfinibayBaseTool):
    name: str = "apply_patch"
    description: str = (
        "Apply a unified diff patch to one or more files. More efficient than "
        "multiple edit_file calls for multi-file changes. The patch must be in "
        "unified diff format (like `git diff` output). Files are modified atomically."
    )
    args_schema: Type[BaseModel] = ApplyPatchInput

    def _run(self, patch: str, strip: int = 1) -> str:
        if not patch.strip():
            return self._error("Empty patch.")

        workspace = self.workspace_path or os.getcwd()

        # Try system `patch` command first (most robust)
        patch_bin = _find_patch_binary()
        if patch_bin:
            return self._apply_with_patch_binary(patch, strip, workspace, patch_bin)

        # Fallback: Python-based patch application
        return self._apply_python(patch, strip, workspace)

    def _apply_with_patch_binary(
        self, patch: str, strip: int, workspace: str, patch_bin: str,
    ) -> str:
        """Apply patch using the system `patch` command."""
        # Dry run first
        try:
            dry_run = subprocess.run(
                [patch_bin, f"-p{strip}", "--dry-run", "--forward", "--no-backup-if-mismatch"],
                input=patch,
                capture_output=True,
                text=True,
                cwd=workspace,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return self._error("Patch dry-run timed out.")
        except FileNotFoundError:
            return self._error("patch binary not found.")

        if dry_run.returncode != 0:
            stderr = dry_run.stderr.strip() or dry_run.stdout.strip()
            return self._error(
                f"Patch would not apply cleanly:\n{stderr[:500]}"
            )

        # Collect files that will be modified (for before-hashes)
        files = _extract_files_from_patch(patch, strip)
        before_hashes = {}
        for f in files:
            fpath = os.path.join(workspace, f)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "rb") as fh:
                        before_hashes[f] = hashlib.sha256(fh.read()).hexdigest()[:16]
                except Exception:
                    pass

        # Apply for real
        try:
            result = subprocess.run(
                [patch_bin, f"-p{strip}", "--forward", "--no-backup-if-mismatch"],
                input=patch,
                capture_output=True,
                text=True,
                cwd=workspace,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return self._error("Patch application timed out.")

        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            return self._error(f"Patch failed:\n{stderr[:500]}")

        # Record audit for each modified file
        self._record_changes(files, before_hashes, workspace)

        output = result.stdout.strip()
        self._log_tool_usage(
            f"Applied patch to {len(files)} file(s): {', '.join(files[:5])}"
        )
        return self._success({
            "files_modified": files,
            "count": len(files),
            "output": output[:300] if output else "Patch applied successfully.",
        })

    def _apply_python(self, patch: str, strip: int, workspace: str) -> str:
        """Fallback Python-based patch applicator for simple unified diffs."""
        hunks_by_file = _parse_unified_diff(patch, strip)
        if not hunks_by_file:
            return self._error(
                "Could not parse patch. Ensure it is in unified diff format "
                "with proper --- a/file and +++ b/file headers."
            )

        # Validate all files exist (except new files)
        for filepath, hunks in hunks_by_file.items():
            fpath = os.path.join(workspace, filepath)
            is_new = all(h["old_start"] == 0 and h["old_count"] == 0 for h in hunks)
            if not is_new and not os.path.isfile(fpath):
                return self._error(f"File not found: {fpath}")

        before_hashes = {}
        modified_files = []

        for filepath, hunks in hunks_by_file.items():
            fpath = os.path.join(workspace, filepath)

            # Sandbox check
            sandbox_err = self._validate_sandbox_path(fpath)
            if sandbox_err:
                return self._error(sandbox_err)

            is_new = all(h["old_start"] == 0 and h["old_count"] == 0 for h in hunks)

            if is_new:
                # New file — collect all added lines
                new_lines = []
                for h in hunks:
                    new_lines.extend(h["add_lines"])
                content = "\n".join(new_lines)
                if new_lines and not content.endswith("\n"):
                    content += "\n"
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
                modified_files.append(filepath)
                continue

            # Existing file
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    original = f.read()
            except Exception as e:
                return self._error(f"Error reading {fpath}: {e}")

            before_hashes[filepath] = hashlib.sha256(
                original.encode("utf-8")
            ).hexdigest()[:16]

            lines = original.splitlines(keepends=True)
            # Apply hunks in reverse order to preserve line numbers
            for hunk in reversed(hunks):
                old_start = hunk["old_start"] - 1  # 0-based
                old_count = hunk["old_count"]
                new_lines = [l + "\n" for l in hunk["add_lines"]]
                lines[old_start:old_start + old_count] = new_lines

            new_content = "".join(lines)

            # Atomic write
            try:
                dir_name = os.path.dirname(fpath)
                original_mode = os.stat(fpath).st_mode
                fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".infinibay_")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    os.chmod(tmp_path, stat.S_IMODE(original_mode))
                    os.replace(tmp_path, fpath)
                except Exception:
                    os.unlink(tmp_path)
                    raise
            except Exception as e:
                return self._error(f"Error writing {fpath}: {e}")

            modified_files.append(filepath)

        self._record_changes(modified_files, before_hashes, workspace)

        self._log_tool_usage(
            f"Applied patch (Python) to {len(modified_files)} file(s)"
        )
        return self._success({
            "files_modified": modified_files,
            "count": len(modified_files),
            "output": "Patch applied successfully.",
        })

    def _record_changes(
        self,
        files: list[str],
        before_hashes: dict[str, str],
        workspace: str,
    ) -> None:
        """Record audit entries for modified files."""
        project_id = self.project_id
        agent_run_id = self.agent_run_id

        def _record(conn: sqlite3.Connection):
            for f in files:
                fpath = os.path.join(workspace, f)
                after_hash = ""
                size = 0
                try:
                    with open(fpath, "rb") as fh:
                        data = fh.read()
                        after_hash = hashlib.sha256(data).hexdigest()[:16]
                        size = len(data)
                except Exception:
                    pass
                conn.execute(
                    """INSERT INTO artifact_changes
                       (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (project_id, agent_run_id, fpath, "modified",
                     before_hashes.get(f, ""), after_hash, size),
                )
            conn.commit()

        try:
            execute_with_retry(_record)
        except Exception:
            pass


def _find_patch_binary() -> str | None:
    """Find the system patch binary."""
    import shutil
    return shutil.which("patch")


def _extract_files_from_patch(patch: str, strip: int) -> list[str]:
    """Extract file paths from a unified diff."""
    files = []
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            # Remove timestamp if present
            path = path.split("\t")[0]
            # Strip leading components
            parts = path.split("/")
            if strip > 0 and len(parts) > strip:
                path = "/".join(parts[strip:])
            if path and path != "/dev/null":
                files.append(path)
    return files


def _parse_unified_diff(patch: str, strip: int) -> dict[str, list[dict]]:
    """Parse a unified diff into a dict of {filepath: [hunks]}.

    Each hunk is a dict with: old_start, old_count, add_lines.
    """
    result: dict[str, list[dict]] = {}
    current_file = None
    current_hunk = None
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip().split("\t")[0]
            parts = path.split("/")
            if strip > 0 and len(parts) > strip:
                path = "/".join(parts[strip:])
            if path and path != "/dev/null":
                current_file = path
                if current_file not in result:
                    result[current_file] = []
            continue

        if line.startswith("--- "):
            continue

        m = hunk_re.match(line)
        if m and current_file is not None:
            if current_hunk is not None:
                result[current_file].append(current_hunk)
            current_hunk = {
                "old_start": int(m.group(1)),
                "old_count": int(m.group(2) or "1"),
                "add_lines": [],
            }
            continue

        if current_hunk is not None:
            if line.startswith("+"):
                current_hunk["add_lines"].append(line[1:])
            elif line.startswith(" "):
                current_hunk["add_lines"].append(line[1:])
            # Lines starting with "-" are removed (not added to new content)

    # Flush last hunk
    if current_hunk is not None and current_file is not None:
        result[current_file].append(current_hunk)

    return result
