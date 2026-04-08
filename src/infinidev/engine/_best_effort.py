"""``best_effort`` context manager for silent-but-observable failures.

The codebase has many ``try: ... except Exception: pass`` sites — mostly
for non-critical operations like reindexing, telemetry, or trace
emission where a failure should not abort the main flow. Silencing the
exception is intentional; silencing the *information* that it happened
is a mistake. This helper swallows the exception but logs it at DEBUG
with an operator-supplied message, so post-mortems and ``-v debug``
runs can still see what went wrong.

Usage::

    from infinidev.engine._best_effort import best_effort

    with best_effort("reindex failed for %s", path):
        reindex_file(path)

Replace the ``try: ... except Exception: pass`` pattern when:

- The operation is genuinely best-effort (caching, telemetry, nice-to-
  have side effects).
- You *want* the main flow to continue on failure.
- You'd rather know later that it failed than investigate silently.

Do NOT replace when:

- The exception class is narrower than ``Exception`` and the narrowness
  matters (e.g. ``except (OSError, ValueError)``).
- You need to branch on whether the operation succeeded (use a real
  ``try``/``except`` with a flag instead).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def best_effort(
    msg: str, *args: object, level: int = logging.DEBUG,
) -> Iterator[None]:
    """Context manager that swallows exceptions and logs them.

    ``msg`` and ``*args`` follow the standard ``logger`` printf-style
    convention so formatting only happens when the log record is
    actually emitted. ``level`` defaults to DEBUG because these paths
    fire on every run and a higher level would drown real signal.
    """
    try:
        yield
    except Exception:  # noqa: BLE001 — intentional broad catch
        logger.log(level, msg, *args, exc_info=True)
