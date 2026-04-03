"""Tool for creating new files with audit trail."""

import hashlib
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


class CreateFileInput(BaseModel):
    file_path: str = Field(..., description="Path for the new file")
    content: str = Field(..., description="Content to write")


