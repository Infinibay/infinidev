"""Tool for searching code examples and documentation on the web."""

import json
import logging
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.web.backends import search_ddg

logger = logging.getLogger(__name__)

# In-memory cache for code search results
_cache: dict[str, str] = {}
_CACHE_MAX = 50


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


