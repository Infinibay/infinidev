"""Tool for executing Python code in a sandboxed interpreter."""

import logging
import os
import subprocess
import tempfile
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


class CodeInterpreterInput(BaseModel):
    code: str = Field(..., description="Python code to execute")
    libraries_used: list[str] | None = Field(
        default=None,
        description="List of libraries the code uses (informational only)",
    )
    timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Max execution time in seconds",
    )


