"""Thread-safe rate limiter for web tools (RPM-based sliding window)."""

import collections
import threading
import time

from infinidev.config.settings import settings


class WebRateLimiter:
    """Sliding-window rate limiter that caps requests per minute."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timestamps: collections.deque[float] = collections.deque()
        self._rpm_limit: int = settings.WEB_RPM_LIMIT
        self._window: float = 60.0

    def acquire(self) -> None:
        """Block until a request slot is available within the RPM window."""
        with self._lock:
            while True:
                now = time.time()
                cutoff = now - self._window
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self._rpm_limit:
                    break

                sleep_time = self._timestamps[0] + self._window - time.time()
                if sleep_time > 0:
                    self._lock.release()
                    try:
                        time.sleep(sleep_time)
                    finally:
                        self._lock.acquire()

            self._timestamps.append(time.time())


web_rate_limiter = WebRateLimiter()
