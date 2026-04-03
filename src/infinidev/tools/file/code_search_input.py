"""Tool for searching code patterns across a codebase."""

import json
import os
import subprocess
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


class CodeSearchInput(BaseModel):
    pattern: str = Field(
        ..., description="Search pattern (text or regex) to find in source files"
    )
    file_path: str = Field(
        default=".", description="Directory to search in (default: current directory)"
    )
    file_extensions: list[str] | None = Field(
        default=None,
        description="Filter by file extensions, e.g. ['.py', '.js']. None searches all files.",
    )
    case_sensitive: bool = Field(
        default=True, description="Whether the search is case sensitive"
    )
    max_results: int = Field(
        default=50, ge=1, le=200, description="Maximum number of matches to return"
    )
    context_lines: int = Field(
        default=0, ge=0, le=5, description="Lines of context before and after each match"
    )


