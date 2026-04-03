"""Tool for executing shell commands in Infinidev CLI."""

import logging
import os
import shlex
import subprocess
from typing import Type
from pydantic import BaseModel, Field
from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


class ExecuteCommandInput(BaseModel):
    command: str = Field(..., description="Command to execute")
    timeout: int = Field(
        default=300, description="Max execution time in seconds. 0 or negative = no timeout."
    )
    cwd: str | None = Field(
        default=None, description="Working directory for the command"
    )
    env: dict[str, str] | None = Field(
        default=None, description="Additional environment variables"
    )

