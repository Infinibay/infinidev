"""Tool for Git branch operations."""

import logging
import re
import sqlite3
from typing import Type

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.git._helpers import run_git, GitToolError


class GitBranchInput(BaseModel):
    branch_name: str = Field(..., description="Name of the branch")
    create: bool = Field(default=True, description="Create the branch if True, checkout if False")
    base_branch: str = Field(default="main", description="Base branch to create from")


