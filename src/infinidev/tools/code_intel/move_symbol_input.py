"""Tool: move a symbol (function/method/class) to another file or class."""

import os
import tempfile
import stat
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class MoveSymbolInput(BaseModel):
    symbol: str = Field(
        ...,
        description="Qualified symbol name: 'ClassName.method_name', 'function_name', or 'ClassName'",
    )
    target_file: str = Field(
        ...,
        description="Destination file path",
    )
    target_class: str = Field(
        default="",
        description="Class to move into. Empty = top-level in file.",
    )
    after_line: int = Field(
        default=0,
        description="Insert after this line number (1-based). 0 = end of file/class.",
    )


