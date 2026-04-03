"""Tool: get the full source code of a function, method, or class."""

import os
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class GetSymbolCodeInput(BaseModel):
    name: str = Field(..., description="Symbol name (function, method, or class)")
    kind: str = Field(
        default="",
        description="Optional: 'function', 'method', 'class' to narrow results",
    )
    file_path: str = Field(
        default="",
        description="Optional: file path to narrow search and ensure indexing",
    )


