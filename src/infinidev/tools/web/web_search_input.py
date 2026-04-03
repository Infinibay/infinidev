"""Web search tool using DuckDuckGo."""

import collections
import hashlib
import time
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool

# Simple in-memory LRU cache (bounded to 256 entries)
_MAX_CACHE_SIZE = 256
_search_cache: collections.OrderedDict[str, tuple[float, list]] = collections.OrderedDict()


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    num_results: int = Field(
        default=10, ge=1, le=20, description="Number of results to return"
    )


