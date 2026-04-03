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
from infinidev.tools.web.code_search_web_input import CodeSearchWebInput


_CACHE_MAX = 50


class CodeSearchWebTool(InfinibayBaseTool):
    name: str = "code_search_web"
    description: str = (
        "Search the web for code examples, API documentation, and programming "
        "solutions. Searches Stack Overflow, GitHub, and official docs. "
        "Use when you need to find correct API usage, library patterns, "
        "or solutions to specific programming problems."
    )
    args_schema: Type[BaseModel] = CodeSearchWebInput

    def _run(
        self,
        query: str,
        language: str = "",
        num_results: int = 5,
    ) -> str:
        if not query.strip():
            return self._error("Empty query.")

        # Build search query with site filters for code-relevant sources
        sites = "site:stackoverflow.com OR site:github.com OR site:docs.python.org OR site:developer.mozilla.org"
        search_query = f"{query}"
        if language:
            search_query = f"{language} {search_query}"
        search_query = f"{search_query} ({sites})"

        # Check cache
        cache_key = f"{search_query}:{num_results}"
        if cache_key in _cache:
            return _cache[cache_key]

        results = search_ddg(search_query, num_results=num_results)

        if not results:
            return self._error(
                f"No results found for: {query}. "
                "Try rephrasing the query or being more specific."
            )

        # Format results
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. **{r['title']}**\n"
                f"   URL: {r['url']}\n"
                f"   {r['snippet']}"
            )

        output = f"Code search results for: {query}\n\n" + "\n\n".join(formatted)

        self._log_tool_usage(f"Code search: {query} ({len(results)} results)")

        # Cache result
        if len(_cache) >= _CACHE_MAX:
            # Evict oldest entry
            oldest = next(iter(_cache))
            del _cache[oldest]
        _cache[cache_key] = output

        return output

