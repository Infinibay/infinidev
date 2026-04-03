"""Tool for listing directory contents."""

import fnmatch
import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


class ListDirectoryInput(BaseModel):
    file_path: str = Field(default=".", description="Directory file_path to list")
    recursive: bool = Field(default=False, description="Whether to list recursively")
    pattern: str | None = Field(
        default=None, description="Glob pattern to filter files (e.g. '*.py')"
    )


