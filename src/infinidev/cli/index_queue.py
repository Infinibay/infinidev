"""Background indexing queue for auto-indexing files on change.

Runs a worker daemon thread that processes file paths from a queue,
calling ensure_indexed() for each. Hash-based skip in ensure_indexed()
prevents redundant re-parsing.
"""

import logging
import queue
from threading import Thread, Event, Lock
from typing import Callable

logger = logging.getLogger(__name__)

_STOP = object()


class IndexQueue:
    """Thread-safe queue that processes file indexing in background."""

    def __init__(
        self,
        project_id: int,
        post_index_callback: Callable[[str], None] | None = None,
    ):
        self._project_id = project_id
        self._post_index = post_index_callback
        self._queue: queue.Queue[tuple[str, bool] | object] = queue.Queue()
        self._stop = Event()
        self._worker: Thread | None = None
        self._stop_lock = Lock()
        self._stopped = False

    def enqueue(self, file_path: str, *, notify_integrity: bool = True) -> None:
        """Add a file while preserving whether this is a baseline observation."""
        self._queue.put((file_path, notify_integrity))

    def _process(self) -> None:
        """Worker loop: pull from queue, call ensure_indexed()."""
        from infinidev.code_intel.smart_index import ensure_indexed

        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            if self._stop.is_set():
                return

            path, notify_integrity = item
            try:
                reindexed = ensure_indexed(
                    self._project_id, path, notify_integrity=notify_integrity,
                )
                if reindexed and self._post_index:
                    self._post_index(path)
            except Exception as exc:
                logger.debug("IndexQueue: failed to index %s: %s", path, exc)

    def start(self) -> None:
        """Start the background worker thread.

        Resets ``_stopped`` under the lock so a stop→start→stop cycle
        works correctly: without the reset, the second ``stop()`` would
        early-return as a no-op and never join the new worker, leaving
        a thread alive past ``os._exit``.
        """
        if self._worker and self._worker.is_alive():
            return
        with self._stop_lock:
            self._stopped = False
        self._stop.clear()
        # A previous stop enqueued a wake-up sentinel. It is normally consumed
        # by that worker; drain any leftover one before a stop→start cycle so
        # the replacement worker cannot exit immediately on stale control data.
        pending: list[tuple[str, bool]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not _STOP:
                pending.append(item)
        for item in pending:
            self._queue.put(item)
        self._worker = Thread(target=self._process, daemon=True, name="index-queue")
        self._worker.start()
        logger.info("IndexQueue started (project_id=%s)", self._project_id)

    def stop(self) -> None:
        """Stop the worker thread gracefully. Idempotent and blocking.

        Called from the shutdown path right before ``os._exit(0)``. If a
        second caller races in, it must wait until the first completes —
        otherwise the worker thread can still be walking Python objects
        when ``_exit`` tears the interpreter down, producing a SIGSEGV.
        """
        with self._stop_lock:
            if self._stopped:
                return
            self._stop.set()
            worker = self._worker
            if worker and worker.is_alive():
                self._queue.put(_STOP)
                worker.join(timeout=3.0)
            self._worker = None
            self._stopped = True
            logger.info("IndexQueue stopped")

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()
