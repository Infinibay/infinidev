"""Process-wide registry for the background file-indexing queue.

The :class:`infinidev.cli.index_queue.IndexQueue` is a worker thread
that calls ``ensure_indexed`` for paths it pulls off a thread-safe
queue. The CLI starts one at boot, but tools like ``read_file`` need
to *push* paths to it without knowing about the CLI module — and they
also need a sensible fallback when no queue exists (tests, scripts,
isolated tool invocations).

This module is the bridge:

  * :func:`set_global_queue` — replaces the registration explicitly.
  * :func:`get_global_queue` — for introspection / tests.
  * :func:`acquire_global_queue` — atomically registers a queue only when
    no other instance owns the process-wide slot.
  * :func:`release_global_queue` — atomically clears a queue only while
    it remains the registered instance.
  * :func:`enqueue_or_sync` — the public entry point for tools.
    Pushes the path to the global queue if one is running; otherwise
    falls back to a synchronous ``ensure_indexed`` call so callers
    don't have to special-case "no queue available".

Both happy paths are non-blocking on the caller's hot path:
  * with a queue: a single ``queue.put`` (~µs)
  * without a queue: the same sync ``ensure_indexed`` we used to do

Cleanly thread-safe via a single module-level lock.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.cli.index_queue import IndexQueue

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_global_queue: "IndexQueue | None" = None


def set_global_queue(queue: "IndexQueue | None") -> None:
    """Register the process-wide IndexQueue instance.

    Pass ``None`` to clear the registration (useful in test teardown
    or when stopping the CLI). Idempotent.
    """
    global _global_queue
    with _lock:
        _global_queue = queue


def get_global_queue() -> "IndexQueue | None":
    """Return the registered IndexQueue, or None if none is set."""
    with _lock:
        return _global_queue


def acquire_global_queue(queue: "IndexQueue") -> bool:
    """Register *queue* unless another running queue owns the global slot."""
    global _global_queue
    with _lock:
        if _global_queue is not None and _global_queue.is_running():
            return False
        _global_queue = queue
        return True


def release_global_queue(queue: "IndexQueue") -> bool:
    """Clear *queue* if it is still the registered instance.

    Returns ``True`` when the queue was released. A different queue may be
    registered while the caller stops its worker; in that case this leaves the
    replacement untouched and returns ``False``.
    """
    global _global_queue
    with _lock:
        if _global_queue is not queue:
            return False
        _global_queue = None
        return True


def enqueue_or_sync(
    project_id: int,
    file_path: str,
    *,
    notify_integrity: bool = True,
) -> None:
    """Enqueue *file_path* for background indexing, or do it synchronously.

    Behaviour:
      1. If a global IndexQueue is registered AND running, push the
         path to it and return immediately. The worker thread will
         pick it up.
      2. Otherwise (no queue, queue stopped, or queue is for a
         different project), fall back to a synchronous
         ``ensure_indexed`` call so the file still gets indexed.

    Never raises — failures inside indexing are logged at debug level
    and swallowed, just like the previous in-line behaviour.
    """
    queue = get_global_queue()
    if queue is not None and queue.project_id == project_id:
        try:
            if queue.enqueue_if_running(
                file_path,
                notify_integrity=notify_integrity,
            ):
                return
        except Exception:
            logger.debug(
                "background_indexer: enqueue failed for %s",
                file_path,
                exc_info=True,
            )
            # Fall through to sync fallback below.

    # Sync fallback — same code path as before this module existed.
    try:
        from infinidev.code_intel.smart_index import ensure_indexed
        ensure_indexed(
            project_id, file_path, notify_integrity=notify_integrity,
        )
    except Exception:
        logger.debug(
            "background_indexer: sync index failed for %s",
            file_path,
            exc_info=True,
        )


__all__ = [
    "set_global_queue",
    "get_global_queue",
    "acquire_global_queue",
    "release_global_queue",
    "enqueue_or_sync",
]
