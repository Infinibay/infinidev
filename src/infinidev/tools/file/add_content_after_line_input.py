"""Tools for inserting content before or after a specific line."""

import hashlib
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


# ── Shared insert logic ──────────────────────────────────────────────────────


def _insert_at(tool: InfinibayBaseTool, path: str, content: str, insert_idx: int) -> str:
    """Insert content at a 0-based line index. Shared by both tools."""
    path = tool._resolve_path(os.path.expanduser(path))

    access_err = guard_file_access(tool, path, "insert_lines")
    if access_err:
        return access_err

    if not os.path.exists(path):
        return tool._error(f"File not found: {path}")
    if not os.path.isfile(path):
        return tool._error(f"Not a file: {path}")

    file_size = os.path.getsize(path)
    if file_size > settings.MAX_FILE_SIZE_BYTES:
        return tool._error(
            f"File too large: {file_size} bytes (max {settings.MAX_FILE_SIZE_BYTES} bytes)"
        )

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except PermissionError:
        return tool._error(f"Permission denied: {path}")
    except Exception as e:
        return tool._error(f"Error reading file: {e}")

    total_lines = len(lines)

    # Clamp insert index
    insert_idx = max(0, min(insert_idx, total_lines))

    # Compute before-hash
    old_content = "".join(lines)
    before_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()[:16]

    # Build new lines
    new_lines = content.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    result_lines = lines[:insert_idx] + new_lines + lines[insert_idx:]
    new_content = "".join(result_lines)

    new_size = len(new_content.encode("utf-8"))
    if new_size > settings.MAX_FILE_SIZE_BYTES:
        return tool._error(
            f"Resulting file too large: {new_size} bytes (max {settings.MAX_FILE_SIZE_BYTES} bytes)"
        )

    # Atomic write
    try:
        atomic_write(path, new_content)
    except PermissionError:
        return tool._error(f"Permission denied: {path}")
    except Exception as e:
        return tool._error(f"Error writing file: {e}")

    after_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()[:16]

    # Audit
    record_artifact_change(tool, path, "modified", before_hash, after_hash, new_size)

    return tool._success({
        "path": path,
        "action": "modified",
        "inserted_at": insert_idx + 1,
        "lines_added": len(new_lines),
        "total_lines": len(result_lines),
        "size_bytes": new_size,
    })


# ── add_content_after_line ────────────────────────────────────────────────────


class AddContentAfterLineInput(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    line_number: int = Field(..., description="Line number to insert AFTER (1-based)")
    content: str = Field(..., description="Content to insert")


