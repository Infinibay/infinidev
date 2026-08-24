"""Background indexing queue for auto-indexing files on change.

Runs a worker daemon thread that processes file paths from a queue,
calling ensure_indexed() for each. Hash-based skip in ensure_indexed()
prevents redundant re-parsing.
"""

import logging
import queue
from threading import Event, Lock, Thread, current_thread
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
        self._lifecycle_lock = Lock()
        self._state_lock = Lock()
        self._stopped = False

    @property
    def project_id(self) -> int:
        """Return the project whose files this queue indexes."""
        return self._project_id

    def enqueue(self, file_path: str, *, notify_integrity: bool = True) -> None:
        """Add a file while preserving whether this is a baseline observation.

        Raise when no worker can consume the item instead of silently leaving it
        queued across shutdown. Race-aware callers that provide their own fallback
        should use :meth:`enqueue_if_running`.
        """
        if not self.enqueue_if_running(file_path, notify_integrity=notify_integrity):
            raise RuntimeError("IndexQueue is not running")

    def enqueue_if_running(
        self,
        file_path: str,
        *,
        notify_integrity: bool = True,
    ) -> bool:
        """Enqueue a file only while this queue owns a live worker.

        The state lock makes the liveness check and enqueue atomic with the
        state updates in ``start()`` and ``stop()``. A caller that loses a race
        with shutdown can therefore use a synchronous fallback instead of
        stranding work.
        """
        with self._state_lock:
            if (
                self._worker is None
                or not self._worker.is_alive()
                or self._stop.is_set()
                or self._stopped
            ):
                return False
            self._queue.put((file_path, notify_integrity))
            return True

    def _process(self) -> None:
        """Worker loop: pull from queue, call ensure_indexed()."""
        from infinidev.code_intel.smart_index import ensure_indexed

        worker = current_thread()
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return

                    # Shutdown appends the sentinel while holding the same state lock used by
                    # enqueue_if_running(). FIFO ordering therefore guarantees that
                    # every item accepted before shutdown is processed before this worker exits.
                    path, notify_integrity = item
                    try:
                        reindexed = ensure_indexed(
                            self._project_id, path, notify_integrity=notify_integrity,
                        )
                    except Exception:
                        logger.debug("IndexQueue: failed to index %s", path, exc_info=True)
                        continue

                    if reindexed and self._post_index:
                        try:
                            self._post_index(path)
                        except Exception:
                            logger.debug(
                                "IndexQueue: post-index callback failed for %s",
                                path,
                                exc_info=True,
                            )
                finally:
                    self._queue.task_done()
        finally:
            with self._state_lock:
                if self._worker is worker:
                    self._worker = None
                    self._stop.set()
                    self._stopped = True

    def start(self) -> None:
        """Start the background worker thread.

        The lifecycle lock covers the complete transition so concurrent starts
        cannot create duplicate workers and ``stop()`` cannot return while a
        worker is still being published.
        """
        with self._lifecycle_lock:
            with self._state_lock:
                if self._worker and self._worker.is_alive():
                    return
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
                    self._queue.task_done()
                for item in pending:
                    self._queue.put(item)
                worker = Thread(target=self._process, daemon=True, name="index-queue")
                self._worker = worker
                try:
                    worker.start()
                except RuntimeError:
                    # ``Thread.start`` normally fails before launching. Roll that
                    # clean failure back so callers do not observe a published,
                    # unusable worker. If a custom thread did launch before
                    # raising, retain it so ``stop()`` can still join it.
                    if not worker.is_alive():
                        self._worker = None
                        self._stop.set()
                        self._stopped = True
                    raise
                logger.info("IndexQueue started (project_id=%s)", self._project_id)

    def stop(self) -> None:
        """Stop the worker thread gracefully. Idempotent and blocking.

        Called from the shutdown path right before ``os._exit(0)``. If a
        second caller races in, it must wait until the first completes —
        otherwise the worker thread can still be walking Python objects
        when ``_exit`` tears the interpreter down, producing a SIGSEGV.
        """
        with self._lifecycle_lock:
            with self._state_lock:
                if self._stopped:
                    return
                shutdown_started = self._stop.is_set()
                self._stop.set()
                worker = self._worker
                if worker and worker.is_alive() and not shutdown_started:
                    self._queue.put(_STOP)

            # Do not hold the state lock while joining. A post-index callback may
            # inspect this queue as its final item completes during shutdown.
            if worker and worker.is_alive():
                if worker is current_thread():
                    return
                worker.join(timeout=3.0)
                if worker.is_alive():
                    raise TimeoutError("IndexQueue worker did not stop within 3 seconds")

            with self._state_lock:
                self._worker = None
                self._stopped = True
                logger.info("IndexQueue stopped")

    def is_running(self) -> bool:
        """Return whether the worker is alive after any lifecycle transition completes."""
        with self._state_lock:
            return self._worker is not None and self._worker.is_alive()
