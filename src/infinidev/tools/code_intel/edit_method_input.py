"""Tool: edit (replace) a method or function by symbol name.

Uses tree-sitter index to find the exact location of the symbol,
then replaces it with new code. No old_string matching needed.
"""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class EditMethodInput(BaseModel):
    symbol: str = Field(
        ...,
        description=(
            "Qualified symbol name: 'ClassName.method_name' for methods, "
            "'function_name' for top-level functions"
        ),
    )
    new_code: str = Field(
        ...,
        description="Complete new source code for the method/function (including def line)",
    )
    file_path: str = Field(
        default="",
        description="File path hint if symbol is ambiguous (optional)",
    )


