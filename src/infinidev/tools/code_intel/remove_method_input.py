"""Tool: remove a method or function by symbol name.

Uses tree-sitter index to find and delete the symbol's source lines.
"""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class RemoveMethodInput(BaseModel):
    symbol: str = Field(
        ...,
        description=(
            "Qualified symbol name: 'ClassName.method_name' for methods, "
            "'function_name' for top-level functions"
        ),
    )
    file_path: str = Field(
        default="",
        description="File path hint if symbol is ambiguous (optional)",
    )


