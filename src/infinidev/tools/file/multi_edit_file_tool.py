"""Tool for applying multiple search-and-replace edits to a single file atomically."""

import hashlib
import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import (
    guard_file_access,
    atomic_write,
    record_artifact_change,
    validate_syntax_or_error,
)
from infinidev.tools.file.edit_operation import EditOperation
from infinidev.tools.file.multi_edit_file_input import MultiEditFileInput


class MultiEditFileTool(InfinibayBaseTool):
    name: str = "multi_edit_file"
    description: str = (
        "Apply multiple search-and-replace edits to a single file atomically. "
        "All edits are validated before any are applied — if any old_string is "
        "not found, none of the edits take effect. Use this instead of multiple "
        "edit_file calls when you need 2+ changes in the same file."
    )
    args_schema: Type[BaseModel] = MultiEditFileInput

    def _run(
        self,
        file_path: str,
        edits: list[dict] | list[EditOperation],
        reason: str = "",
    ) -> str:
        # Normalize edits to dicts
        edit_list = []
        for e in edits:
            if isinstance(e, dict):
                edit_list.append(e)
            else:
                edit_list.append({"old_string": e.old_string, "new_string": e.new_string})

        if not edit_list:
            return self._error("No edits provided.")

        file_path = self._resolve_path(os.path.expanduser(file_path))

        access_err = guard_file_access(self, file_path, "edit_file")
        if access_err:
            return access_err

        if not os.path.exists(file_path):
            return self._error(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            return self._error(f"Not a file: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"File too large: {file_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Read existing content
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except PermissionError:
            return self._error(f"Permission denied: {file_path}")
        except Exception as e:
            return self._error(f"Error reading file: {e}")

        # Phase 1: Validate ALL old_strings exist in original content
        for i, edit in enumerate(edit_list):
            old_s = edit.get("old_string", "")
            new_s = edit.get("new_string", "")
            if not old_s:
                return self._error(f"Edit {i}: old_string is empty.")
            if old_s == new_s:
                return self._error(f"Edit {i}: old_string and new_string are identical.")
            count = content.count(old_s)
            if count == 0:
                return self._error(
                    f"Edit {i}: old_string not found in {file_path}. "
                    "None of the edits were applied. "
                    "Ensure text matches exactly including indentation."
                )
            if count > 1:
                return self._error(
                    f"Edit {i}: old_string appears {count} times in {file_path}. "
                    "Provide more surrounding context to make it unique. "
                    "None of the edits were applied."
                )

        # Phase 2: Collect positions for all edits (on original content)
        # to detect overlapping edits
        positions = []
        for i, edit in enumerate(edit_list):
            old_s = edit["old_string"]
            start = content.find(old_s)
            end = start + len(old_s)
            positions.append((start, end, i))

        # Sort by position and check for overlaps
        positions.sort()
        for j in range(1, len(positions)):
            if positions[j][0] < positions[j - 1][1]:
                idx_a = positions[j - 1][2]
                idx_b = positions[j][2]
                return self._error(
                    f"Edits {idx_a} and {idx_b} overlap in the file. "
                    "None of the edits were applied."
                )

        # Phase 3: Apply all edits (from end to start to preserve positions)
        before_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        new_content = content

        # Apply in reverse position order so earlier positions stay valid
        for start, end, i in reversed(positions):
            new_s = edit_list[i]["new_string"]
            new_content = new_content[:start] + new_s + new_content[end:]

        # Check resulting size
        new_size = len(new_content.encode("utf-8"))
        if new_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"Resulting file too large: {new_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        # Pre-write syntax check on the post-merge result
        syntax_err = validate_syntax_or_error(self, file_path, new_content, operation="multi_edit_file")
        if syntax_err:
            return syntax_err

        # Atomic write
        try:
            atomic_write(file_path, new_content)
        except PermissionError:
            return self._error(f"Permission denied: {file_path}")
        except Exception as e:
            return self._error(f"Error writing file: {e}")

        after_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()[:16]

        # Record audit
        record_artifact_change(self, file_path, "modified", before_hash, after_hash, new_size)

        num_edits = len(edit_list)
        self._log_tool_usage(
            f"Multi-edited {file_path} ({num_edits} edit{'s' if num_edits > 1 else ''}, {new_size} bytes)"
        )
        result = {
            "file_path": file_path,
            "action": "modified",
            "edits_applied": num_edits,
            "size_bytes": new_size,
        }
        if reason:
            result["reason"] = reason
        return self._success(result)

