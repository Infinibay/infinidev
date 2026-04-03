"""Tool: rename a symbol and update all references across the project."""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class RenameSymbolInput(BaseModel):
    symbol: str = Field(
        ...,
        description="Qualified symbol name: 'ClassName.method_name' or 'function_name'",
    )
    new_name: str = Field(
        ...,
        description="New name for the symbol (just the name, not qualified)",
    )
    file_path: str = Field(
        default="",
        description="File path hint if symbol is ambiguous",
    )


