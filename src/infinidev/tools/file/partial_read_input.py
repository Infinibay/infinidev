"""Tool for reading a specific range of lines from a file."""

import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file.read_file import ReadFileTool


class PartialReadInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to read")
    start_line: int = Field(..., description="First line to read (1-based, inclusive)")
    end_line: int = Field(..., description="Last line to read (1-based, inclusive)")


