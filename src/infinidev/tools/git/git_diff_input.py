"""Tool for viewing Git diffs."""

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.git._helpers import run_git, GitToolError


class GitDiffInput(BaseModel):
    branch: str | None = Field(
        default=None, description="Branch to diff against (e.g. 'main')"
    )
    file: str | None = Field(
        default=None, description="Specific file to diff"
    )
    staged: bool = Field(
        default=False, description="Show only staged changes"
    )


