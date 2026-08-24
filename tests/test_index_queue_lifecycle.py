"""Lifecycle tests for ``IndexQueue`` — idempotent stop + start cycles.

Targets the B1 fix from commit ``6964f5c`` and the review finding from
commit ``c951eec``. The existing ``test_index_queue.py`` covers the
happy-path enqueue/process flow; these tests cover the concurrency
edge cases that were historically silent crashes on shutdown.
"""

from __future__ import annotations

import threading

import pytest

import infinidev.cli.index_queue as index_queue_module
from infinidev.cli.index_queue import IndexQueue
from infinidev.code_intel import background_indexer


# ── Idempotent stop ──────────────────────────────────────────────────────


class TestIdempotentStop:
    """``stop()`` must be safe to call multiple times."""

    def test_stop_without_start_is_noop(self):
        """Calling stop() before start() must not raise."""
        q = IndexQueue(project_id=1)
        q.stop()  # no worker; should just set _stopped and return

    def test_stop_twice_does_not_raise(self):
        """A second stop() after the worker joined must not re-join."""
        q = IndexQueue(project_id=1)
        q.start()
        q.stop()
        q.stop()  # idempotent — second call is a no-op

    def test_stop_from_two_threads_serializes(self):
        """Concurrent stop() calls must not double-join the worker."""
        q = IndexQueue(project_id=1)
        q.start()
        errors: list[BaseException] = []

        def _stop():
            try:
                q.stop()
            except BaseException as exc:  # noqa: BLE001 — record any error
                errors.append(exc)

        threads = [threading.Thread(target=_stop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert not errors, f"concurrent stop raised: {errors}"
        # Worker should be gone.
        assert not q.is_running()

    def test_enqueue_rejects_work_without_live_worker(self):
        """The convenience API must not strand work before start or after stop."""
        q = IndexQueue(project_id=1)

        with pytest.raises(RuntimeError, match="IndexQueue is not running"):
            q.enqueue("before-start.py")

        q.start()
        q.stop()

        with pytest.raises(RuntimeError, match="IndexQueue is not running"):
            q.enqueue("after-stop.py")

    def test_stop_timeout_preserves_live_worker_ownership(self):
        """A worker alive after join must remain discoverable for a later retry."""

        class TimedOutWorker:
            def __init__(self):
                self.join_timeouts = []

            def is_alive(self):
                return True

            def join(self, timeout=None):
                self.join_timeouts.append(timeout)

        q = IndexQueue(project_id=1)
        worker = TimedOutWorker()
        q._worker = worker

        with pytest.raises(TimeoutError, match="did not stop within 3 seconds"):
            q.stop()

        assert worker.join_timeouts == [3.0]
        assert q._worker is worker
        assert q._stopped is False
        assert q.is_running()
        assert not q.enqueue_if_running("late.py")

    def test_stop_retry_does_not_append_duplicate_sentinels(self):
        """A timed-out worker needs one wake-up sentinel across stop retries."""

        class TimedOutWorker:
            def is_alive(self):
                return True

            def join(self, timeout=None):
                return None

        q = IndexQueue(project_id=1)
        q._worker = TimedOutWorker()

        for _attempt in range(2):
            with pytest.raises(TimeoutError, match="did not stop within 3 seconds"):
                q.stop()

        assert q._queue.qsize() == 1
        assert q._queue.unfinished_tasks == 1


# ── Serialized lifecycle transitions ───────────────────────────────────────


class ControlledWorker:
    """Thread stand-in whose publication can be paused deterministically."""

    instances: list[ControlledWorker] = []
    start_entered = threading.Event()
    allow_start = threading.Event()

    def __init__(self, **_kwargs):
        self.alive = False
        self.join_timeouts = []
        self.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.start_entered = threading.Event()
        cls.allow_start = threading.Event()

    def start(self) -> None:
        self.start_entered.set()
        assert self.allow_start.wait(5.0), "test did not release worker start"
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout=None) -> None:
        self.join_timeouts.append(timeout)
        self.alive = False


class TestSerializedLifecycle:
    """Complete start and stop transitions must not overlap."""

    def test_start_failure_rolls_back_unstarted_worker(self, monkeypatch):
        """A clean thread-start failure must leave the queue retryable."""

        class FailingWorker:
            def __init__(self, **_kwargs):
                self.start_calls = 0

            def start(self) -> None:
                self.start_calls += 1
                raise RuntimeError("cannot start worker")

            def is_alive(self) -> bool:
                return False

        failing_worker = FailingWorker()
        monkeypatch.setattr(
            index_queue_module,
            "Thread",
            lambda **_kwargs: failing_worker,
        )
        queue = IndexQueue(project_id=1)

        with pytest.raises(RuntimeError, match="cannot start worker"):
            queue.start()

        assert queue._worker is None
        assert queue._stopped is True
        assert queue._stop.is_set()
        assert not queue.is_running()
        assert failing_worker.start_calls == 1

    def test_stop_waits_for_start_to_publish_and_then_stops_worker(self, monkeypatch):
        ControlledWorker.reset()
        monkeypatch.setattr(index_queue_module, "Thread", ControlledWorker)
        q = IndexQueue(project_id=1)
        stop_finished = threading.Event()

        start_thread = threading.Thread(target=q.start)
        start_thread.start()
        assert ControlledWorker.start_entered.wait(5.0), "start did not reach worker startup"

        stop_thread = threading.Thread(target=lambda: (q.stop(), stop_finished.set()))
        stop_thread.start()
        assert not stop_finished.wait(0.1), "stop returned before start published its worker"

        ControlledWorker.allow_start.set()
        start_thread.join(timeout=5.0)
        stop_thread.join(timeout=5.0)

        assert not start_thread.is_alive()
        assert not stop_thread.is_alive()
        assert stop_finished.is_set()
        assert len(ControlledWorker.instances) == 1
        assert not ControlledWorker.instances[0].is_alive()
        assert not q.is_running()

    def test_concurrent_starts_create_one_worker(self, monkeypatch):
        ControlledWorker.reset()
        monkeypatch.setattr(index_queue_module, "Thread", ControlledWorker)
        q = IndexQueue(project_id=1)

        starts = [threading.Thread(target=q.start) for _ in range(2)]
        for thread in starts:
            thread.start()
        assert ControlledWorker.start_entered.wait(5.0), "start did not reach worker startup"

        ControlledWorker.allow_start.set()
        for thread in starts:
            thread.join(timeout=5.0)

        assert all(not thread.is_alive() for thread in starts)
        assert len(ControlledWorker.instances) == 1
        assert q.is_running()
        q.stop()

    def test_global_acquisition_waits_for_registered_queue_start(self, monkeypatch):
        ControlledWorker.reset()
        monkeypatch.setattr(index_queue_module, "Thread", ControlledWorker)
        queue = IndexQueue(project_id=1)
        replacement = IndexQueue(project_id=2)
        acquisition_entered = threading.Event()
        acquisition_result = []
        background_indexer.set_global_queue(queue)

        def acquire_replacement() -> None:
            acquisition_entered.set()
            acquisition_result.append(background_indexer.acquire_global_queue(replacement))

        start_thread = threading.Thread(target=queue.start)
        acquire_thread = threading.Thread(target=acquire_replacement)
        try:
            start_thread.start()
            assert ControlledWorker.start_entered.wait(5.0), "start did not reach worker startup"

            acquire_thread.start()
            assert acquisition_entered.wait(5.0), "acquisition did not begin"
            acquire_thread.join(timeout=0.1)
            assert acquire_thread.is_alive(), "acquisition observed a partial start"
            assert acquisition_result == []

            ControlledWorker.allow_start.set()
            start_thread.join(timeout=5.0)
            acquire_thread.join(timeout=5.0)

            assert not start_thread.is_alive()
            assert not acquire_thread.is_alive()
            assert acquisition_result == [False]
            assert background_indexer.get_global_queue() is queue
        finally:
            ControlledWorker.allow_start.set()
            start_thread.join(timeout=5.0)
            acquire_thread.join(timeout=5.0)
            if queue.is_running():
                queue.stop()
            background_indexer.set_global_queue(None)

    def test_stop_drains_work_accepted_before_shutdown(self, monkeypatch):
        """The stop sentinel must remain behind every lifecycle-accepted item."""
        queue = IndexQueue(project_id=1)
        first_started = threading.Event()
        release_first = threading.Event()
        processed: list[str] = []

        def ensure_indexed(
            _project_id: int,
            file_path: str,
            *,
            notify_integrity: bool = True,
        ) -> bool:
            if file_path == "first.py":
                first_started.set()
                assert release_first.wait(5.0), "test did not release first index"
            processed.append(file_path)
            return True

        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            ensure_indexed,
        )
        queue.start()
        stop_thread = threading.Thread(target=queue.stop)
        try:
            assert queue.enqueue_if_running("first.py")
            assert first_started.wait(5.0), "worker did not begin first item"
            assert queue.enqueue_if_running("accepted.py")

            stop_thread.start()
            release_first.set()
            stop_thread.join(timeout=5.0)

            assert not stop_thread.is_alive()
            assert processed == ["first.py", "accepted.py"]
            assert not queue.is_running()
        finally:
            release_first.set()
            stop_thread.join(timeout=5.0)
            if queue.is_running():
                queue.stop()

    def test_unexpected_worker_exit_publishes_state_and_allows_restart(
        self,
        monkeypatch,
    ):
        """A fatal worker error must release ownership before a replacement starts."""

        class FatalWorkerError(BaseException):
            pass

        queue = IndexQueue(project_id=1)
        queue._worker = threading.current_thread()
        queue._queue.put(("fatal.py", True))

        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(FatalWorkerError()),
        )

        with pytest.raises(FatalWorkerError):
            queue._process()

        assert queue._worker is None
        assert queue._stop.is_set()
        assert queue._stopped is True
        assert not queue.is_running()
        assert queue._queue.unfinished_tasks == 0

        processed: list[str] = []
        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            lambda _project_id, path, *, notify_integrity=True: (
                processed.append(path) or True
            ),
        )
        queue.start()
        assert queue.enqueue_if_running("recovered.py")
        queue.stop()

        assert processed == ["recovered.py"]
        assert not queue.is_running()

    def test_indexing_failure_does_not_strand_later_work(
        self,
        monkeypatch,
        caplog,
    ):
        """One indexing failure must not terminate the shared queue worker."""
        processed: list[str] = []
        second_finished = threading.Event()

        def ensure_indexed(
            _project_id: int,
            file_path: str,
            *,
            notify_integrity: bool = True,
        ) -> bool:
            processed.append(file_path)
            if file_path == "first.py":
                raise RuntimeError("indexing failed")
            second_finished.set()
            return True

        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            ensure_indexed,
        )
        queue = IndexQueue(project_id=1)
        caplog.set_level("DEBUG", logger=index_queue_module.__name__)
        queue.start()
        try:
            assert queue.enqueue_if_running("first.py")
            assert queue.enqueue_if_running("second.py")
            assert second_finished.wait(5.0), "later queued work was stranded"

            assert processed == ["first.py", "second.py"]
            assert queue.is_running()
            assert "failed to index first.py" in caplog.text
            assert "RuntimeError: indexing failed" in caplog.text
        finally:
            queue.stop()

    def test_restart_preserves_fifo_order_across_worker_cycles(self, monkeypatch):
        """Stopping and restarting must not lose or reorder accepted paths."""
        processed: list[str] = []

        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            lambda _project_id, file_path, *, notify_integrity=True: (
                processed.append(file_path) or True
            ),
        )
        queue = IndexQueue(project_id=1)

        queue.start()
        assert queue.enqueue_if_running("first.py")
        assert queue.enqueue_if_running("second.py")
        queue.stop()

        queue.start()
        assert queue.enqueue_if_running("third.py")
        assert queue.enqueue_if_running("fourth.py")
        queue.stop()

        assert processed == ["first.py", "second.py", "third.py", "fourth.py"]
        assert not queue.is_running()

    def test_completed_work_balances_queue_accounting(self, monkeypatch):
        """Processed items and shutdown sentinels must all call ``task_done()``."""
        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            lambda *_args, **_kwargs: True,
        )
        queue = IndexQueue(project_id=1)

        queue.start()
        assert queue.enqueue_if_running("first.py")
        assert queue.enqueue_if_running("second.py")
        queue.stop()

        assert queue._queue.unfinished_tasks == 0
        queue._queue.join()

    def test_failed_work_balances_queue_accounting(self, monkeypatch):
        """Indexing failures must acknowledge both work items and the sentinel."""

        def fail_index(*_args, **_kwargs):
            raise RuntimeError("indexing failed")

        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            fail_index,
        )
        queue = IndexQueue(project_id=1)

        queue.start()
        assert queue.enqueue_if_running("failed.py")
        queue.stop()

        assert queue._queue.unfinished_tasks == 0
        queue._queue.join()

    def test_callback_failure_does_not_strand_later_work(
        self,
        monkeypatch,
        caplog,
    ):
        """One callback failure must not terminate the shared queue worker."""
        processed: list[str] = []
        callback_paths: list[str] = []
        second_callback_finished = threading.Event()

        def ensure_indexed(
            _project_id: int,
            file_path: str,
            *,
            notify_integrity: bool = True,
        ) -> bool:
            processed.append(file_path)
            return True

        def post_index(file_path: str) -> None:
            callback_paths.append(file_path)
            if file_path == "first.py":
                raise RuntimeError("callback failed")
            second_callback_finished.set()

        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            ensure_indexed,
        )
        queue = IndexQueue(project_id=1, post_index_callback=post_index)
        caplog.set_level("DEBUG", logger=index_queue_module.__name__)
        queue.start()
        try:
            assert queue.enqueue_if_running("first.py")
            assert queue.enqueue_if_running("second.py")
            assert second_callback_finished.wait(5.0), "later queued work was stranded"

            assert processed == ["first.py", "second.py"]
            assert callback_paths == ["first.py", "second.py"]
            assert queue.is_running()
            assert "post-index callback failed for first.py" in caplog.text
            assert "RuntimeError: callback failed" in caplog.text
        finally:
            queue.stop()

    def test_post_index_callback_can_stop_and_restart_queue(self, monkeypatch):
        """A worker callback may stop its own queue without joining itself."""
        processed: list[str] = []
        callbacks_finished = [threading.Event(), threading.Event()]
        callback_count = 0
        queue = IndexQueue(project_id=1)

        def post_index(_path: str) -> None:
            nonlocal callback_count
            callback_index = callback_count
            callback_count += 1
            queue.stop()
            callbacks_finished[callback_index].set()

        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            lambda _project_id, path, *, notify_integrity=True: (
                processed.append(path) or True
            ),
        )
        queue._post_index = post_index

        for cycle, path in enumerate(("first.py", "restarted.py")):
            queue.start()
            worker = queue._worker
            assert worker is not None
            assert queue.enqueue_if_running(path)
            assert callbacks_finished[cycle].wait(5.0), "self-stop did not return"
            worker.join(timeout=5.0)

            assert not worker.is_alive()
            assert not queue.is_running()
            assert queue._worker is None
            assert queue._stopped is True

        assert processed == ["first.py", "restarted.py"]
        assert callback_count == 2
        assert queue._queue.unfinished_tasks == 0

    def test_stop_allows_final_callback_to_inspect_queue(self, monkeypatch):
        """Shutdown must not deadlock a callback that reads lifecycle state."""
        callback_started = threading.Event()
        release_callback = threading.Event()
        callback_finished = threading.Event()
        observed_running: list[bool] = []
        queue = IndexQueue(project_id=1)

        def post_index(_path: str) -> None:
            callback_started.set()
            assert release_callback.wait(5.0), "test did not release callback"
            observed_running.append(queue.is_running())
            callback_finished.set()

        queue._post_index = post_index
        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            lambda *_args, **_kwargs: True,
        )
        queue.start()
        stop_thread = threading.Thread(target=queue.stop)
        try:
            assert queue.enqueue_if_running("final.py")
            assert callback_started.wait(5.0), "worker did not reach callback"
            stop_thread.start()
            assert not callback_finished.wait(0.1)

            release_callback.set()
            assert callback_finished.wait(1.0), "callback blocked on queue state lock"
            stop_thread.join(timeout=5.0)

            assert not stop_thread.is_alive()
            assert observed_running == [True]
            assert not queue.is_running()
        finally:
            release_callback.set()
            stop_thread.join(timeout=5.0)
            if queue.is_running():
                queue.stop()

    def test_enqueue_or_sync_uses_public_queue_identity(self, monkeypatch):
        """The bridge must route without depending on private queue state."""

        class PublicQueue:
            project_id = 1

            def enqueue_if_running(
                self,
                file_path: str,
                *,
                notify_integrity: bool = True,
            ) -> bool:
                accepted.append((file_path, notify_integrity))
                return True

        accepted: list[tuple[str, bool]] = []
        queue = PublicQueue()
        background_indexer.set_global_queue(queue)
        try:
            background_indexer.enqueue_or_sync(
                1,
                "changed.py",
                notify_integrity=False,
            )
        finally:
            background_indexer.set_global_queue(None)

        assert accepted == [("changed.py", False)]

    def test_enqueue_or_sync_falls_back_when_shutdown_wins(self, monkeypatch):
        """Work observed before shutdown must not be stranded on a stopped queue."""
        queue = IndexQueue(project_id=1)
        queue.start()
        background_indexer.set_global_queue(queue)
        check_entered = threading.Event()
        allow_check = threading.Event()
        accepted: list[str] = []
        indexed: list[tuple[int, str, bool]] = []
        errors: list[BaseException] = []
        enqueue_if_running = queue.enqueue_if_running

        def paused_enqueue(file_path: str, *, notify_integrity: bool = True) -> bool:
            check_entered.set()
            assert allow_check.wait(5.0), "test did not release enqueue check"
            return enqueue_if_running(file_path, notify_integrity=notify_integrity)

        def call_enqueue_or_sync() -> None:
            try:
                background_indexer.enqueue_or_sync(1, "changed.py")
            except BaseException as exc:  # pragma: no cover - surfaced by assertion
                errors.append(exc)

        monkeypatch.setattr(queue, "enqueue_if_running", paused_enqueue)
        monkeypatch.setattr(
            queue,
            "enqueue",
            lambda file_path, **_kwargs: accepted.append(file_path),
        )
        monkeypatch.setattr(
            "infinidev.code_intel.smart_index.ensure_indexed",
            lambda project_id, file_path, *, notify_integrity=True: indexed.append(
                (project_id, file_path, notify_integrity)
            ),
        )
        caller = threading.Thread(target=call_enqueue_or_sync)

        try:
            caller.start()
            assert check_entered.wait(5.0), "enqueue_or_sync did not select the queue"
            queue.stop()
            allow_check.set()
            caller.join(timeout=5.0)

            assert not caller.is_alive()
            assert not errors
            assert accepted == []
            assert indexed == [(1, "changed.py", True)]
        finally:
            allow_check.set()
            caller.join(timeout=5.0)
            if queue.is_running():
                queue.stop()
            background_indexer.set_global_queue(None)


# ── Start after stop resets the lifecycle ──────────────────────────────────


class TestStartAfterStop:
    """A stop→start→stop cycle must correctly join the new worker."""

    def test_start_resets_stopped_flag(self):
        """start() resets _stopped so the next stop() actually joins."""
        q = IndexQueue(project_id=1)
        q.start()
        q.stop()
        assert q._stopped is True

        q.start()  # this MUST reset _stopped so the next stop works
        assert q._stopped is False

        # The worker from the second start should also stop cleanly.
        q.stop()
        assert q._stopped is True
        assert not q.is_running()

    def test_stopped_then_started_worker_is_distinct(self):
        """After restart, there's a live worker again."""
        q = IndexQueue(project_id=1)
        q.start()
        first_worker = q._worker
        q.stop()
        q.start()
        second_worker = q._worker
        assert second_worker is not None
        assert second_worker is not first_worker
        assert second_worker.is_alive()
        q.stop()


# ── is_running semantics ──────────────────────────────────────────────────


class TestIsRunning:
    """is_running() reflects the current state."""

    def test_not_running_before_start(self):
        q = IndexQueue(project_id=1)
        assert not q.is_running()

    def test_running_after_start(self):
        q = IndexQueue(project_id=1)
        q.start()
        try:
            assert q.is_running()
        finally:
            q.stop()

    def test_not_running_after_stop(self):
        q = IndexQueue(project_id=1)
        q.start()
        q.stop()
        assert not q.is_running()
