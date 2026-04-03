"""Tool: run heuristic code analysis on indexed data."""

import json
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class AnalyzeCodeInput(BaseModel):
    file_path: str = Field(
        default="",
        description="File to analyze. Empty = analyze whole project.",
    )
    checks: str = Field(
        default="",
        description="Comma-separated checks: broken_imports,undefined_symbols,unused_imports,unused_definitions. Empty = all.",
    )


