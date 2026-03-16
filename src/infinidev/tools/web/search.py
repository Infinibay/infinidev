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


class WebSearchTool(InfinibayBaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web using DuckDuckGo. "
        "Returns a list of results with title, URL, and snippet."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str, num_results: int = 10) -> str:
        # Check cache
        cache_key = hashlib.sha256(f"{query}:{num_results}".encode()).hexdigest()
        if cache_key in _search_cache:
            cached_time, cached_results = _search_cache[cache_key]
            if time.time() - cached_time < settings.WEB_CACHE_TTL_SECONDS:
                return self._success({"results": cached_results, "cached": True})

        from infinidev.tools.web.backends import search_ddg

        try:
            results = search_ddg(query, num_results)
        except Exception as e:
            return self._error(f"Search failed: {e}")

        # Cache results (evict oldest if at capacity)
        if len(_search_cache) >= _MAX_CACHE_SIZE:
            _search_cache.popitem(last=False)
        _search_cache[cache_key] = (time.time(), results)

        return self._success({"results": results, "count": len(results)})
