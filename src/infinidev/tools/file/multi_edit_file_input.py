"""Tool for applying multiple search-and-replace edits to a single file atomically."""

import hashlib
import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change
from infinidev.tools.file.edit_operation import EditOperation


class MultiEditFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to edit")
    edits: list[EditOperation] = Field(
        ...,
        description=(
            "Ordered list of edits to apply. Each edit has old_string and new_string. "
            "All old_strings are validated against the ORIGINAL content before any "
            "changes are made — edits are atomic (all succeed or none apply)."
        ),
    )
    reason: str = Field(
        default="",
        description="Brief explanation of why these edits are being made.",
    )


