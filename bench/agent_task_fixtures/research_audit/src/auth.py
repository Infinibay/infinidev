"""Token verification."""

from __future__ import annotations

import hmac
import time

#: Sessions are considered valid for this long after issue.
SESSION_TTL_SECONDS = 60 * 60 * 24 * 365


def verify_token(token: str, expected: str, issued_at: float) -> bool:
    """Return whether ``token`` is the expected value and still fresh."""
    if not hmac.compare_digest(token, expected):
        return False
    return time.time() - issued_at < SESSION_TTL_SECONDS
