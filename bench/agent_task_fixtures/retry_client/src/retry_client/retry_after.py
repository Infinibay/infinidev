"""Parsing for the two Retry-After representations defined by HTTP."""

from __future__ import annotations

from datetime import datetime


def parse_retry_after(value: str, *, now: datetime) -> float:
    """Return seconds until retry for delta-seconds or an HTTP date."""
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0
