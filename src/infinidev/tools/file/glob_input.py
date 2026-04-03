"""Tool for finding files by name pattern with optional content filtering."""

import json
import os
import re
from pathlib import Path
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


class GlobInput(BaseModel):
    pattern: str = Field(
        ...,
        description=(
            "Glob pattern to match file paths. Supports ** for recursive "
            "matching. Examples: '**/*.py' (all Python files), "
            "'src/**/*.test.ts' (all test files under src), "
            "'**/migrations/*.sql' (all SQL migrations), "
            "'*.md' (markdown files in current dir only)."
        ),
    )
    file_path: str = Field(
        default=".",
        description="Base directory to search from (default: current directory).",
    )
    content_pattern: str | None = Field(
        default=None,
        description=(
            "Optional regex pattern to filter files by content. Only files "
            "whose content matches this pattern will be returned. "
            "Example: 'class.*Tool', 'def test_', 'TODO|FIXME'."
        ),
    )
    case_sensitive: bool = Field(
        default=True,
        description="Whether the content_pattern match is case sensitive (default: true).",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of matching files to return (default: 100).",
    )
    include_ignored: bool = Field(
        default=False,
        description="Include files normally ignored (caches, build artifacts, node_modules, etc.). By default these are hidden.",
    )
    max_depth: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum directory depth for results. 1 = files in base dir only. None = unlimited.",
    )


