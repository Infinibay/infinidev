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
    rationale: str = Field(
        ...,
        min_length=30,
        description=(
            "REQUIRED. Explain WHAT you expect this command to do and "
            "WHY you need to run it (≥30 chars). The assistant critic "
            "reads this before the command runs. Do NOT use vague "
            "phrases like 'running tests' — say which tests, what "
            "outcome you expect, and why the outcome matters."
        ),
    )
    timeout: int = Field(
        default=300, description="Max execution time in seconds. 0 or negative = no timeout."
    )
    cwd: str | None = Field(
        default=None, description="Working directory for the command"
    )
    env: dict[str, str] | None = Field(
        default=None, description="Additional environment variables"
    )

