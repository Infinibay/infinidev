"""Tool for Git push operations (async-capable)."""

import asyncio
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.git._helpers import run_git, GitToolError


class GitPushInput(BaseModel):
    branch: str | None = Field(
        default=None, description="Branch to push. If None, pushes current branch."
    )
    force: bool = Field(default=False, description="Force push (use with caution)")


