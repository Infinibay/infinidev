"""Tool: find where a symbol is defined."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class FindDefinitionInput(BaseModel):
    name: str = Field(..., description="Symbol name to find (function, class, variable)")
    kind: str = Field(
        default="",
        description="Optional filter: 'function', 'method', 'class', 'variable', 'constant'",
    )


