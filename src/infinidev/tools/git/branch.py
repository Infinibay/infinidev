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


from infinidev.tools.git.git_branch_input import GitBranchInput
from infinidev.tools.git.git_branch_tool import GitBranchTool
