"""Thread-safe robots.txt checker with caching for web tools."""

import threading
import time
import urllib.parse
import urllib.robotparser

from infinidev.config.settings import settings


class RobotsChecker:
    """Checks robots.txt rules with a TTL-based cache per domain."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], tuple[float, urllib.robotparser.RobotFileParser]] = {}
        self._ttl: int = settings.WEB_ROBOTS_CACHE_TTL

    def _get_parser(
        self, domain: str, scheme: str
    ) -> urllib.robotparser.RobotFileParser | None:
        key = (scheme, domain)
        with self._lock:
            if key in self._cache:
                cached_time, parser = self._cache[key]
                if time.time() - cached_time < self._ttl:
                    return parser

        robots_url = f"{scheme}://{domain}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
        except Exception:
            return None

        with self._lock:
            self._cache[key] = (time.time(), parser)
        return parser

    def is_allowed(self, url: str, user_agent: str) -> bool:
        """Return True if *user_agent* may fetch *url* per robots.txt rules.

        Permissive policy: if robots.txt cannot be fetched, access is allowed.
        """
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme or "https"
        # Use hostname (drops any user:pass@ userinfo so credentials never leak
        # into the robots.txt request) plus the port, and key the cache on
        # (scheme, domain) so http/https for the same host don't collide.
        host = parsed.hostname or ""
        domain = host if parsed.port is None else f"{host}:{parsed.port}"
        parser = self._get_parser(domain, scheme)
        if parser is None:
            return True
        return parser.can_fetch(user_agent, url)


robots_checker = RobotsChecker()
