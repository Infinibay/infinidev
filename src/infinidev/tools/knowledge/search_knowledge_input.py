"""Tool for unified cross-source knowledge search using FTS5."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, sanitize_fts5_query


class SearchKnowledgeInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Full-text search query. Supports operators: "
            "'term1 | term2' for OR, 'term1 & term2' for AND, "
            "'arch*' for prefix matching, '\"exact phrase\"' for phrases. "
            "Examples: 'react | vue | angular', 'auth & security', 'micros*'"
        ),
    )
    sources: list[str] = Field(
        default=["findings", "reports"],
        description="Sources to search: findings, reports",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max results per source")
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum confidence filter (findings only)",
    )


