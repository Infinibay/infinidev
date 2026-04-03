"""Tool for replacing a range of lines in a file."""

import hashlib
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


class ReplaceLinesInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to edit")
    content: str = Field(
        ...,
        description="New content to insert (replaces the specified line range)",
    )
    start_line: int = Field(
        ..., description="First line to replace (1-based, inclusive)"
    )
    end_line: int = Field(
        ..., description="Last line to replace (1-based, inclusive)"
    )


