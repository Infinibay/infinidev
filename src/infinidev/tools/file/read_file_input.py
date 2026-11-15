"""Tool for reading file contents with optional line-range selection."""

import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to read")
    offset: int | None = Field(
        default=None,
        description=(
            "Line number to start reading from (1-based). "
            "If omitted, reads from the beginning of the file."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum number of lines to read; non-negative, zero returns an empty result. "
            "If omitted, reads until the end of the file."
        ),
    )


