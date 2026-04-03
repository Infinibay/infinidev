"""Tool for reading a specific range of lines from a file."""

import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file.read_file import ReadFileTool
from infinidev.tools.file.partial_read_input import PartialReadInput


class PartialReadTool(InfinibayBaseTool):
    name: str = "partial_read"
    description: str = "Read a specific range of lines from a file."
    args_schema: Type[BaseModel] = PartialReadInput

    def _run(self, file_path: str, start_line: int, end_line: int) -> str:
        if start_line < 1:
            return self._error(f"start_line must be >= 1, got {start_line}")
        if end_line < start_line:
            return self._error(
                f"end_line ({end_line}) must be >= start_line ({start_line})"
            )

        # Delegate to ReadFileTool with offset/limit
        reader = ReadFileTool()
        self._bind_delegate(reader)
        offset = start_line
        limit = end_line - start_line + 1
        return reader._run(file_path=file_path, offset=offset, limit=limit)

