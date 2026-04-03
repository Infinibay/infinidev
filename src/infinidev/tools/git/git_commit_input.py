"""Tool for Git commit operations."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.git._helpers import run_git, GitToolError


class GitCommitInput(BaseModel):
    message: str = Field(..., description="Commit message")
    files: list[str] | None = Field(
        default=None,
        description="Specific files to stage. If None, stages all changes (git add -A).",
    )


