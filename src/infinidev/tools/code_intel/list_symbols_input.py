"""Tool: list all symbols in a file."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class ListSymbolsInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to list symbols from")
    kind: str = Field(
        default="",
        description="Optional filter: 'function', 'method', 'class', 'variable'",
    )


