"""Tool for reading/searching research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, sanitize_fts5_query


class ReadFindingsInput(BaseModel):
    query: str | None = Field(
        default=None,
        description=(
            "Full-text search query across findings. "
            "Supports: | for OR, & for AND, * for prefix, \"quotes\" for exact phrases. "
            "Example: 'security | auth', 'API & design*'"
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Filter findings by session ID. When omitted, defaults to the "
            "current session from agent context (if available). Pass '0' to "
            "explicitly disable session filtering and see all project findings."
        ),
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence filter"
    )
    finding_type: str | None = Field(
        default=None, description="Filter by finding type"
    )
    limit: int = Field(default=50, ge=1, le=200, description="Max results")


