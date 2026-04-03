"""Tool: show project structure with semantic descriptions from the code index."""

import os
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class ProjectStructureInput(BaseModel):
    file_path: str = Field(
        default=".",
        description="Directory to show structure of. Defaults to project root.",
    )
    depth: int = Field(
        default=2,
        description="How many levels deep to show (1-5). Default 2.",
    )


