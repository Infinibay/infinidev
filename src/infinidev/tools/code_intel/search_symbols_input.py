"""Tool: fuzzy search across all symbol names."""

from typing import Type
from pydantic import BaseModel, Field, field_validator

from infinidev.tools.base.base_tool import InfinibayBaseTool


class SearchSymbolsInput(BaseModel):
    query: str = Field(..., description="Search text (1+ chars, supports partial matches)", min_length=1)
    kind: str = Field(
        default="",
        description="Optional filter: 'function', 'method', 'class', 'variable'",
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty — give a name fragment to search for")
        return v


