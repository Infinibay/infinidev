"""Tool for viewing Git diffs."""

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.git._helpers import run_git, GitToolError


from infinidev.tools.git.git_diff_input import GitDiffInput
from infinidev.tools.git.git_diff_tool import GitDiffTool
