"""Tool for surgical file edits using search-and-replace."""

import difflib
import hashlib
import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


class EditFileInput(BaseModel):
    model_config = {"populate_by_name": True}

    file_path: str = Field(..., description="Path to the file to edit")
    old_string: str = Field(
        ...,
        description=(
            "The exact text to find in the file. Must match exactly "
            "(including indentation and whitespace). Must be unique in the "
            "file — if it appears more than once, provide more surrounding "
            "context to make it unique, or use replace_all=true."
        ),
    )
    new_string: str = Field(
        ...,
        description="The text to replace old_string with. Must differ from old_string.",
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "If true, replace ALL occurrences of old_string in the file. "
            "Useful for renaming variables or updating repeated patterns."
        ),
    )
    reason: str = Field(
        default="",
        alias="description",
        description=(
            "Brief explanation of WHY this edit is being made. "
            "Used by the code reviewer to understand intent."
        ),
    )


