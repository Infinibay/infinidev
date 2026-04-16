"""Tool for replacing a range of lines in a file."""

import hashlib
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import (
    guard_file_access,
    atomic_write,
    record_artifact_change,
    check_syntax_warning,
    detect_silent_deletions,
    deletion_warning_text,
    find_external_usages,
)
from infinidev.tools.file.replace_lines_input import ReplaceLinesInput


class ReplaceLinesTool(InfinibayBaseTool):
    name: str = "replace_lines"
    description: str = "Replace a range of lines in a file with new content."
    args_schema: Type[BaseModel] = ReplaceLinesInput

    def _run(
        self,
        file_path: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> str:
        path = self._resolve_path(os.path.expanduser(file_path))

        access_err = guard_file_access(self, path, "replace_lines")
        if access_err:
            return access_err

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

        # Read existing content
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except PermissionError:
            return self._error(f"Permission denied: {path}")
        except Exception as e:
            return self._error(f"Error reading file: {e}")

        total_lines = len(lines)

        # Validate line range
        if start_line < 1:
            return self._error(f"start_line must be >= 1, got {start_line}")
        if end_line < start_line - 1:
            return self._error(
                f"end_line ({end_line}) must be >= start_line - 1 ({start_line - 1})"
            )
        if start_line > total_lines + 1:
            return self._error(
                f"start_line ({start_line}) is beyond end of file ({total_lines} lines)"
            )

        # Compute before-hash
        old_content = "".join(lines)
        before_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()[:16]

        # Build new content lines (ensure each line ends with \n)
        if content == "":
            new_lines = []
        else:
            new_lines = content.splitlines(keepends=True)
            # Ensure last line has newline
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

        # Splice: replace lines[start_line-1 : end_line] with new_lines
        # When start_line == end_line + 1, this is a pure insert (no lines removed)
        start_idx = start_line - 1
        end_idx = min(end_line, total_lines)
        result_lines = lines[:start_idx] + new_lines + lines[end_idx:]

        new_content = "".join(result_lines)

        # Check resulting size
        new_size = len(new_content.encode("utf-8"))
        if new_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"Resulting file too large: {new_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Pre-write syntax check on the SPLICED result (advisory only) so
        # the model can notice seam-level errors between old and new lines
        # without blocking the write on tree-sitter false positives.
        syntax_warn = check_syntax_warning(self, path, new_content, operation="replace_lines")

        # Detect symbols (functions/classes/methods) that disappeared
        # between the old and the new content. Soft signal — we still
        # write the file, but the success response carries a warning so
        # the model can notice and restore if it was an accident.
        deleted_symbols = detect_silent_deletions(path, old_content, new_content)
        symbol_usages = find_external_usages(
            deleted_symbols, path, getattr(self, "workspace_path", None)
        )

        # Atomic write (preserve permissions)
        try:
            atomic_write(path, new_content)
        except PermissionError:
            return self._error(f"Permission denied: {path}")
        except Exception as e:
            return self._error(f"Error writing file: {e}")

        # Compute after-hash
        after_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()[:16]

        # Audit trail
        lines_removed = end_idx - start_idx
        lines_added = len(new_lines)
        record_artifact_change(self, path, "modified", before_hash, after_hash, new_size)

        self._log_tool_usage(
            f"Replaced lines {start_line}-{end_line} in {path} "
            f"(-{lines_removed} +{lines_added} lines, {new_size} bytes)"
        )
        result: dict = {
            "path": path,
            "action": "modified",
            "lines_removed": lines_removed,
            "lines_added": lines_added,
            "total_lines": len(result_lines),
            "size_bytes": new_size,
        }
        if (warn := deletion_warning_text(deleted_symbols, path, symbol_usages)):
            result["warning"] = warn
            result["removed_symbols"] = deleted_symbols
            if symbol_usages:
                result["removed_symbol_usages"] = symbol_usages
        if syntax_warn:
            result["syntax_warning"] = syntax_warn
        return self._success(result)

