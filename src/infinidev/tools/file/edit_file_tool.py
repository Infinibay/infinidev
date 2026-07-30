"""Edit a file by replacing exact text.

The one way to change an existing file. It replaced five tools that each
addressed the same operation differently — a line range, a symbol name, an
insertion point before or after a line — because addressing is not a
capability, and five ways to say "change this" is a decision the model has
to get right before it can start working.

Text matching is not just fewer tools, it is a better failure mode. Line
numbers shift the moment an earlier edit in the same step lands, so an
off-by-one writes real content into the wrong place and returns success. An
exact match either finds its text or refuses, and when the text appears more
than once it says so instead of picking one.
"""

from __future__ import annotations

import hashlib
import os
from typing import Type

from pydantic import BaseModel

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import (
    atomic_write,
    check_syntax_warning,
    deletion_warning_text,
    detect_silent_deletions,
    guard_file_access,
    record_artifact_change,
)
from infinidev.tools.file.edit_file_input import EditFileInput


class EditFileTool(InfinibayBaseTool):
    name: str = "edit_file"
    description: str = (
        "Replace exact text in an existing file. old_string must match the "
        "file byte for byte, indentation included, and must be unique unless "
        "replace_all is set — add surrounding lines until it is. An empty "
        "new_string deletes the text. Use create_file for a new file."
    )
    args_schema: Type[BaseModel] = EditFileInput

    def _run(
        self,
        file_path: str,
        old_string: str,
        new_string: str = "",
        replace_all: bool = False,
        rationale: str = "",
    ) -> str:
        del rationale  # read by the critic off the tool call, not needed here

        path = self._resolve_path(os.path.expanduser(file_path))

        access_err = guard_file_access(self, path, "edit_file")
        if access_err:
            return access_err

        if not os.path.exists(path):
            return self._error(
                f"File not found: {path}. Use create_file to make a new one."
            )
        if not os.path.isfile(path):
            return self._error(f"Not a file: {path}")

        size = os.path.getsize(path)
        if size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"File too large: {size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Read strictly: the whole file is rewritten below, so errors="replace"
        # would quietly turn untouched bytes into U+FFFD.
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except UnicodeDecodeError:
            return self._error(
                f"File is not valid UTF-8; refusing to edit to avoid "
                f"corrupting binary content: {path}"
            )
        except PermissionError:
            return self._error(f"Permission denied: {path}")
        except Exception as exc:
            return self._error(f"Error reading file: {exc}")

        if not old_string:
            return self._error(
                "old_string is empty. Pass the exact text to replace, or use "
                "create_file to write a whole file."
            )
        if old_string == new_string:
            return self._error("old_string and new_string are identical; nothing to do.")

        occurrences = content.count(old_string)
        if occurrences == 0:
            return self._error(
                f"old_string not found in {path}. Nothing was changed. Copy the "
                "text exactly as it appears in the file, including indentation "
                "— read the file again if it may have changed."
            )
        if occurrences > 1 and not replace_all:
            return self._error(
                f"old_string appears {occurrences} times in {path}. Nothing was "
                "changed. Add surrounding lines until the match is unique, or "
                "pass replace_all=true to change every occurrence."
            )

        before_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )

        new_size = len(new_content.encode("utf-8"))
        if new_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"Resulting file too large: {new_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        syntax_warn = check_syntax_warning(
            self, path, new_content, operation="edit_file",
        )
        deleted_symbols = detect_silent_deletions(path, content, new_content)

        try:
            atomic_write(path, new_content)
        except PermissionError:
            return self._error(f"Permission denied: {path}")
        except Exception as exc:
            return self._error(f"Error writing file: {exc}")

        after_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()[:16]
        record_artifact_change(self, path, "modified", before_hash, after_hash, new_size)

        applied = occurrences if replace_all else 1
        self._log_tool_usage(
            f"Edited {path} ({applied} replacement"
            f"{'s' if applied > 1 else ''}, {new_size} bytes)"
        )
        result: dict = {
            "file_path": path,
            "action": "modified",
            "replacements": applied,
            "size_bytes": new_size,
        }
        if (warn := deletion_warning_text(deleted_symbols, path)):
            result["warning"] = warn
            result["removed_symbols"] = deleted_symbols
        if syntax_warn:
            result["syntax_warning"] = syntax_warn
        return self._success(result)
