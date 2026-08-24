"""Input schema for general web search."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    num_results: int = Field(
        default=10, ge=1, le=20, description="Number of results to return"
    )
