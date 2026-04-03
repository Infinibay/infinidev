"""Tool for viewing Git status."""

from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.git._helpers import run_git, GitToolError


class GitStatusInput(BaseModel):
    pass


