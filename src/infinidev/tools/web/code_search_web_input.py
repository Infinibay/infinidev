"""Input schema for code-focused web search."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CodeSearchWebInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Natural language query about code, API usage, or library documentation. "
            "Example: 'python asyncio gather exception handling', "
            "'django queryset annotate with subquery'."
        ),
    )
    language: str = Field(
        default="",
        description="Optional programming language filter (e.g. 'python', 'rust', 'typescript').",
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Number of results to return (1-15).",
    )
