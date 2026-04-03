"""Tool for searching findings by semantic similarity."""

import sqlite3
from typing import Type

import numpy as np
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, parse_query_or_terms


class SearchFindingsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Text to search for among findings. "
            "Matches by semantic similarity against topic and content. "
            "Supports | for OR: 'security | auth' matches findings similar to either term."
        ),
    )
    threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity (0-1). Lower = broader matches.",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Filter to a specific session. Omit to use current session context. "
            "Pass '0' to search all project findings."
        ),
    )
    include_content: bool = Field(
        default=False,
        description="Include full finding content (default: topics only for speed).",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max results")


