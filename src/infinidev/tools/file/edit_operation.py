"""Tool for applying multiple search-and-replace edits to a single file atomically."""

import hashlib
import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


class EditOperation(BaseModel):
    old_string: str = Field(
        ...,
        description="The exact text to find. Must match exactly including indentation.",
    )
    new_string: str = Field(
        ...,
        description="The replacement text.",
    )


