"""Tool: find all usages of a symbol."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class FindReferencesInput(BaseModel):
    name: str = Field(..., description="Symbol name to find usages of")
    ref_kind: str = Field(
        default="",
        description="Optional filter: 'call', 'import', 'type_ref', 'assignment', 'usage'",
    )


