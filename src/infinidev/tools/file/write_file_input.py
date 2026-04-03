"""Tool for writing file contents with audit trail."""

import hashlib
import json
import os
from typing import Literal, Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


class WriteFileInput(BaseModel):
    model_config = {"populate_by_name": True}

    file_path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")
    mode: Literal["w", "a"] = Field(
        default="w", description="Write mode: 'w' to overwrite, 'a' to append"
    )
    reason: str = Field(
        default="",
        alias="description",
        description=(
            "Brief explanation of WHY this file is being created/written. "
            "Used by the code reviewer to understand intent."
        ),
    )


