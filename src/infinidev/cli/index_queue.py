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


class IndexQueue:
    """Thread-safe queue that processes file indexing in background."""

    def __init__(
        self,
        project_id: int,
        post_index_callback: Callable[[str], None] | None = None,
    ):
        self._project_id = project_id
        self._post_index = post_index_callback
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = Event()
        self._worker: Thread | None = None
        self._stop_lock = Lock()
        self._stopped = False

    def enqueue(self, file_path: str) -> None:
        """Add a file to be re-indexed. Safe to call from any thread."""
        self._queue.put(file_path)

    def _process(self) -> None:
        """Worker loop: pull from queue, call ensure_indexed()."""
        from infinidev.code_intel.smart_index import ensure_indexed

        while not self._stop.is_set():
            try:
                path = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                reindexed = ensure_indexed(self._project_id, path)
                if reindexed and self._post_index:
                    self._post_index(path)
            except Exception as exc:
                logger.debug("IndexQueue: failed to index %s: %s", path, exc)

    def start(self) -> None:
        """Start the background worker thread."""
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
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
                worker.join(timeout=3.0)
            self._worker = None
            self._stopped = True
            logger.info("IndexQueue stopped")

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()
