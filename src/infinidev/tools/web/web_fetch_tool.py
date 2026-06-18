"""Web fetch tool for extracting readable content from URLs."""

import ipaddress
import socket
import sqlite3
from typing import Literal, Type
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.db.service import execute_with_retry
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.web.rate_limiter import web_rate_limiter
from infinidev.tools.web.robots_checker import robots_checker
from infinidev.tools.web.web_fetch_input import WebFetchInput


def _validate_fetch_url(url: str) -> str | None:
    """Return an error string if *url* is unsafe to fetch (SSRF guard), else None.

    The LLM controls the url and this tool is exposed to the read-only tiers,
    so a prompt-injected page could otherwise point it at localhost, RFC-1918
    intranet hosts, link-local, or the cloud metadata endpoint
    (169.254.169.254). Restrict to http/https and reject any host that resolves
    to a private/loopback/link-local/reserved address.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "Only http/https URLs are supported"
    host = parsed.hostname
    if not host:
        return "URL has no host"
    try:
        infos = socket.getaddrinfo(host, parsed.port or None)
    except socket.gaierror:
        return "Could not resolve host"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return "Could not parse resolved address"
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return "Refusing to fetch private/internal address"
    return None


class WebFetchTool(InfinibayBaseTool):
    # Read-only with respect to the workspace — retrieves a URL, never
    # writes. Exposed to the read-only exploration tiers.
    is_read_only: bool = True
    name: str = "web_fetch"
    description: str = (
        "Fetch and extract readable content from a URL. "
        "Returns clean text or markdown, stripping ads and navigation."
    )
    args_schema: Type[BaseModel] = WebFetchInput

    def _check_cache(self, url: str, format: str) -> str | None:
        """Return cached content if fresh enough, else None."""
        def _query(conn: sqlite3.Connection) -> str | None:
            row = conn.execute(
                """\
                SELECT content, fetched_at
                FROM web_cache
                WHERE url = ? AND format = ?
                  AND fetched_at > datetime('now', ?)
                """,
                (url, format, f"-{settings.WEB_CACHE_TTL_SECONDS} seconds"),
            ).fetchone()
            return row["content"] if row else None
        try:
            return execute_with_retry(_query)
        except Exception:
            return None

    def _store_cache(self, url: str, format: str, content: str) -> None:
        """Store fetched content in the cache."""
        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                """\
                INSERT INTO web_cache (url, format, content)
                VALUES (?, ?, ?)
                ON CONFLICT(url, format) DO UPDATE SET
                    content    = excluded.content,
                    fetched_at = CURRENT_TIMESTAMP
                """,
                (url, format, content),
            )
            conn.commit()
        try:
            execute_with_retry(_insert)
        except Exception:
            pass  # Cache write failure is non-fatal

    def _run(self, url: str, format: str = "markdown", bypass_cache: bool = False) -> str:
        # SSRF guard FIRST — before cache/robots/rate-limit, since the robots
        # check itself reaches the target host.
        err = _validate_fetch_url(url)
        if err is not None:
            return self._error(err)

        # Check cache first
        if not bypass_cache:
            cached = self._check_cache(url, format)
            if cached is not None:
                return cached

        try:
            import httpx
        except ImportError:
            return self._error("httpx not installed. Run: pip install httpx")

        try:
            import trafilatura
        except ImportError:
            return self._error(
                "trafilatura not installed. Run: pip install trafilatura"
            )

        # Check robots.txt
        if not robots_checker.is_allowed(url, "InfinidevBot/1.0"):
            return self._error("robots.txt disallows fetching this URL")

        # Rate limit
        web_rate_limiter.acquire()

        # Fetch URL. Follow redirects MANUALLY (httpx auto-follow disabled) so
        # each hop's target is re-validated by the SSRF guard — otherwise a
        # permitted public host could 30x-redirect us into the internal
        # network. Common http→https / canonicalization redirects still work.
        try:
            with httpx.Client(
                timeout=settings.WEB_TIMEOUT,
                follow_redirects=False,
                headers={"User-Agent": "Mozilla/5.0 (compatible; InfinidevBot/1.0)"},
            ) as client:
                current = url
                response = None
                for _ in range(5):  # cap redirect hops
                    response = client.get(current)
                    location = response.headers.get("location")
                    if response.is_redirect and location:
                        nxt = str(response.url.join(location))
                        verr = _validate_fetch_url(nxt)
                        if verr is not None:
                            return self._error(f"Refusing redirect to unsafe URL: {verr}")
                        current = nxt
                        continue
                    break
                response.raise_for_status()
                html = response.text
        except Exception as e:
            return self._error(f"Fetch failed: {e}")

        # Extract content
        output_format = "markdown" if format == "markdown" else "txt"
        try:
            content = trafilatura.extract(
                html,
                output_format=output_format,
                include_links=(format == "markdown"),
                include_tables=True,
                favor_recall=True,
            )
        except Exception as e:
            return self._error(f"Content extraction failed: {e}")

        if not content:
            return self._error(f"No readable content extracted from {url}")

        self._store_cache(url, format, content)
        return content

