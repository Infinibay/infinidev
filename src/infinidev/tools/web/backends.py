"""Web search and fetch backends for Infinidev web tools.

Provides pure functions (no BaseTool dependency) for search and content fetching.
"""

from __future__ import annotations

import logging
import threading

from infinidev.config.settings import settings
from infinidev.tools.web.rate_limiter import web_rate_limiter

logger = logging.getLogger(__name__)

_DDG_MAX_IN_FLIGHT = 4
_ddg_worker_slots = threading.BoundedSemaphore(_DDG_MAX_IN_FLIGHT)


def search_ddg(query: str, num_results: int = 10) -> list[dict]:
    """Search via DuckDuckGo.

    Returns list of ``{title, url, snippet}`` dicts.
    Uses a hard timeout to prevent hanging the engine when DDG is slow
    or unreachable (backend='auto' tries multiple backends sequentially).
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

    timeout = settings.WEB_TIMEOUT

    def _do_search() -> list[dict]:
        ua_headers = {"User-Agent": "Mozilla/5.0 (compatible; InfinidevBot/1.0)"}
        try:
            ddgs = DDGS(headers=ua_headers, timeout=timeout)
        except TypeError:
            ddgs = DDGS()
        return list(ddgs.text(query, max_results=num_results))

    # ThreadPoolExecutor's context manager waits for running workers during
    # shutdown, which defeats Future.result(timeout=...). A daemon worker
    # lets the caller honor the wall-clock deadline while DDGS finishes its
    # own bounded network attempt in the background.
    worker_slots = _ddg_worker_slots
    if not worker_slots.acquire(blocking=False):
        logger.warning(
            "DDG search skipped because %d workers are still in flight",
            _DDG_MAX_IN_FLIGHT,
        )
        return []

    completed = threading.Event()
    raw_results: list[dict] = []
    error: list[Exception] = []

    def _worker() -> None:
        try:
            raw_results.extend(_do_search())
        except Exception as exc:
            error.append(exc)
        finally:
            worker_slots.release()
            completed.set()

    worker = threading.Thread(target=_worker, name="infinidev-ddg-search", daemon=True)
    try:
        worker.start()
    except RuntimeError as exc:
        worker_slots.release()
        logger.warning("Failed to start DDG search worker: %s", exc)
        return []
    if not completed.wait(timeout=max(float(timeout), 0.0)):
        logger.warning("DDG search timed out after %ss for query: %s", timeout, query[:80])
        return []
    if error:
        logger.warning("DDG search failed: %s", error[0])
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
