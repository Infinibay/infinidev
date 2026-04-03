"""Tool: add a method or function to a file or class.

Inserts code at the end of a class (if class_name given) or end of file.
Auto-detects indentation.
"""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class AddMethodInput(BaseModel):
    file_path: str = Field(..., description="Path to the target file")
    code: str = Field(
        ...,
        description="Complete method/function source code (including def line)",
    )
    class_name: str = Field(
        default="",
        description="Class to add the method to. If empty, appends to end of file.",
    )


