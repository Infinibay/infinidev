"""Lifecycle tests for ``IndexQueue`` — idempotent stop + start cycles.

Targets the B1 fix from commit ``6964f5c`` and the review finding from
commit ``c951eec``. The existing ``test_index_queue.py`` covers the
happy-path enqueue/process flow; these tests cover the concurrency
edge cases that were historically silent crashes on shutdown.
"""

from __future__ import annotations

import threading
import time

import pytest

from infinidev.cli.index_queue import IndexQueue


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
