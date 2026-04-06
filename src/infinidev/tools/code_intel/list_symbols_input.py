"""Tool: list all symbols in a file."""

from typing import Type
from pydantic import BaseModel, Field, field_validator

from infinidev.tools.base.base_tool import InfinibayBaseTool


class ListSymbolsInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to list symbols from (cannot be empty — pass a real file like 'src/main.py')", min_length=1)
    kind: str = Field(
        default="",
        description="Optional filter: 'function', 'method', 'class', 'variable'",
    )

    @field_validator("file_path")
    @classmethod
    def _path_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "file_path must be a real file path (e.g. 'src/main.py'). "
                "To list files in a directory use list_directory instead."
            )
        return v


