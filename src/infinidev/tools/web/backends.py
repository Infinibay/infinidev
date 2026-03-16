"""Web search and fetch backends for Infinidev web tools.

Provides pure functions (no BaseTool dependency) for search and content fetching.
"""

from __future__ import annotations

import logging

from infinidev.config.settings import settings
from infinidev.tools.web.rate_limiter import web_rate_limiter

logger = logging.getLogger(__name__)


def search_ddg(query: str, num_results: int = 10) -> list[dict]:
    """Search via DuckDuckGo.

    Returns list of ``{title, url, snippet}`` dicts.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("Neither ddgs nor duckduckgo_search installed")
            return []

    web_rate_limiter.acquire()

    try:
        ua_headers = {"User-Agent": "Mozilla/5.0 (compatible; InfinidevBot/1.0)"}
        try:
            ddgs = DDGS(headers=ua_headers)
        except TypeError:
            ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=num_results))
    except Exception as exc:
        logger.warning("DDG search failed: %s", exc)
        return []

    results = []
    for r in raw_results:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("href", r.get("link", "")),
            "snippet": r.get("body", r.get("snippet", "")),
        })
    return results


def fetch_with_trafilatura(url: str, timeout: int | None = None) -> str | None:
    """Fetch URL and extract content with trafilatura.

    Returns extracted text or ``None`` on failure.
    """
    try:
        import httpx
    except ImportError:
        return None

    try:
        import trafilatura
    except ImportError:
        return None

    if timeout is None:
        timeout = settings.WEB_TIMEOUT

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; InfinidevBot/1.0)"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text
    except Exception:
        return None

    try:
        content = trafilatura.extract(html, output_format="txt", favor_recall=True)
    except Exception:
        return None

    return content
