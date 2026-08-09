"""Tool for executing shell commands in Infinidev CLI."""

import logging
import os
import shlex
import subprocess
from typing import Type
from pydantic import BaseModel, Field, model_validator
from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


class ExecuteCommandInput(BaseModel):
    command: str = Field(..., description="Command to execute")
    rationale: str = Field(
        default="",
        description=(
            "Explain WHAT you expect this command to do and "
            "WHY you need to run it. The assistant critic "
            "reads this before the command runs. Do NOT use vague "
            "phrases like 'running tests'. If omitted, the controller derives "
            "a bounded rationale from the command instead of rejecting it."
        ),
    )
    timeout: int = Field(
        default=120,
        description=(
            "Requested execution time in seconds. The configured COMMAND_TIMEOUT "
            "is a hard ceiling; zero or negative uses that ceiling."
        ),
    )
    cwd: str | None = Field(
        default=None,
        description=(
            "Working directory for this command. Every execute_command call starts "
            "independently; a prior shell `cd` does not persist. Pass cwd explicitly "
            "when the command must run in a repository or subdirectory."
        ),
    )
    env: dict[str, str] | None = Field(
        default=None, description="Additional environment variables"
    )

    @model_validator(mode="after")
    def _derive_missing_rationale(self) -> "ExecuteCommandInput":
        if not self.rationale.strip():
            shown = " ".join(self.command.split())[:100]
            self.rationale = (
                f"Run `{shown}` and inspect its observable result for the active task."
            )
        return self
