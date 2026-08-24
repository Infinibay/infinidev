"""Tool for searching code examples and documentation on the web."""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Type

from pydantic import BaseModel

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.web.backends import search_ddg
from infinidev.tools.web.code_search_web_input import CodeSearchWebInput

logger = logging.getLogger(__name__)

# Bounded, expiring cache for code search results.
_cache: collections.OrderedDict[tuple[str, int], tuple[float, str]] = collections.OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX = 50


class CodeSearchWebTool(InfinibayBaseTool):
    # Searches the web for code examples; no workspace mutation, so it is
    # read-only for role filtering.
    is_read_only: bool = True
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
        query = query.strip()
        language = language.strip()
        if not query:
            return self._error("Empty query.")

        sites = (
            "site:stackoverflow.com OR site:github.com OR "
            "site:docs.python.org OR site:developer.mozilla.org"
        )
        search_query = query
        if language:
            search_query = f"{language} {search_query}"
        search_query = f"{search_query} ({sites})"

        cache_key = (search_query, num_results)
        now = time.monotonic()
        with _cache_lock:
            cached = _cache.get(cache_key)
            if cached is not None:
                cached_at, cached_output = cached
                if now - cached_at < settings.WEB_CACHE_TTL_SECONDS:
                    _cache.move_to_end(cache_key)
                    return cached_output
                del _cache[cache_key]

        try:
            results = search_ddg(search_query, num_results=num_results)
        except Exception as exc:
            logger.warning("Code search failed for %r: %s", query, exc)
            return self._error(f"Code search failed: {exc}")

        if not results:
            return self._error(
                f"No results found for: {query}. "
                "Try rephrasing the query or being more specific."
            )

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. **{result['title']}**\n"
                f"   URL: {result['url']}\n"
                f"   {result['snippet']}"
            )

        output = f"Code search results for: {query}\n\n" + "\n\n".join(formatted)
        self._log_tool_usage(f"Code search: {query} ({len(results)} results)")

        with _cache_lock:
            if _CACHE_MAX > 0:
                while len(_cache) >= _CACHE_MAX:
                    _cache.popitem(last=False)
                _cache[cache_key] = (time.monotonic(), output)

        return output
