"""Tool: fuzzy search across all symbol names."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class SearchSymbolsInput(BaseModel):
    query: str = Field(..., description="Search text (supports partial matches)")
    kind: str = Field(
        default="",
        description="Optional filter: 'function', 'method', 'class', 'variable'",
    )


