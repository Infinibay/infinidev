"""Tests for the taskqueue module."""
import pytest
import time
import threading
from taskqueue import TaskQueue, TaskStatus, Priority


class TestTaskExecution:
    """Test basic task execution."""

    def test_simple_task(self):
        """Test that a simple task executes successfully."""
        queue = TaskQueue(num_workers=2)

        def simple_task():
            return "Hello, World!"

        task_id = queue.submit(simple_task)
        queue.wait_all(timeout=5)

        status = queue.status(task_id)
        assert status is not None
        assert status['status'] == 'completed'
        assert status['result'] == "Hello, World!"

        queue.shutdown()

    def test_task_with_kwargs(self):
        """Test task execution with keyword arguments."""
        queue = TaskQueue(num_workers=2)

        def add(a, b=10):
            return a + b

        task_id = queue.submit(add, kwargs={'a': 5, 'b': 15})
        queue.wait_all(timeout=5)

        status = queue.status(task_id)
        assert status['status'] == 'completed'
        assert status['result'] == 20

        queue.shutdown()

    def test_task_failure(self):
        """Test that failing tasks are marked as failed."""
        queue = TaskQueue(num_workers=2)

        def fail_task():
            raise ValueError("Intentional failure")

        task_id = queue.submit(fail_task, max_retries=0)
        queue.wait_all(timeout=5)

        status = queue.status(task_id)
        assert status['status'] == 'failed'
        assert "Intentional failure" in status['error']

        queue.shutdown()


class TestPriorityOrdering:
    """Test priority-based task ordering."""

    def test_priority_ordering(self):
        """Test that high priority tasks are processed first."""
        execution_order = []
        lock = threading.Lock()

        def record_order(name):
            time.sleep(0.05)  # Small delay to ensure ordering
            with lock:
                execution_order.append(name)
            return name

        queue = TaskQueue(num_workers=1)

        # Submit tasks in reverse priority order
        queue.submit(record_order, args=("low",), priority=Priority.LOW)
        queue.submit(record_order, args=("medium",), priority=Priority.MEDIUM)
        queue.submit(record_order, args=("high",), priority=Priority.HIGH)

        # Small delay to ensure all tasks are in the priority heap
        time.sleep(0.1)

        queue.start_processing()
        queue.wait_all(timeout=10)

        # High priority should execute first
        assert execution_order[0] == "high"
        assert execution_order[1] == "medium"
        assert execution_order[2] == "low"

        queue.shutdown()


class TestRetryBehavior:
    """Test retry with exponential backoff."""

    def test_retry_on_failure(self):
        """Test that tasks are retried on failure."""
        attempt_count = 0
        lock = threading.Lock()

        def flaky_task():
            nonlocal attempt_count
            with lock:
                attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "Success after retries"

        queue = TaskQueue(num_workers=2, retry_delay=0.1)
        task_id = queue.submit(flaky_task, max_retries=3)
        queue.wait_all(timeout=10)

        status = queue.status(task_id)
        assert status['status'] == 'completed'
        assert status['retries'] == 2

        queue.shutdown()

    def test_max_retries_exceeded(self):
        """Test that tasks fail after max retries."""
        def always_fails():
            raise ValueError("Always fails")

        queue = TaskQueue(num_workers=2, retry_delay=0.1)
        task_id = queue.submit(always_fails, max_retries=3)
        queue.wait_all(timeout=10)

        status = queue.status(task_id)
        assert status['status'] == 'failed'
        assert status['retries'] == 3

        queue.shutdown()


class TestTimeoutHandling:
    """Test task timeout handling."""

    def test_task_timeout(self):
        """Test that slow tasks are timed out."""
        def slow_task():
            time.sleep(10)
            return "Completed"

        queue = TaskQueue(num_workers=2)
        task_id = queue.submit(slow_task, timeout=0.5, max_retries=0)
        queue.wait_all(timeout=5)

        status = queue.status(task_id)
        assert status['status'] == 'timeout'
        assert "timed out" in status['error'].lower()

        queue.shutdown()

    def test_task_within_timeout(self):
        """Test that fast tasks complete within timeout."""
        def fast_task():
            time.sleep(0.1)
            return "Done"

        queue = TaskQueue(num_workers=2)
        task_id = queue.submit(fast_task, timeout=2.0)
        queue.wait_all(timeout=5)

        status = queue.status(task_id)
        assert status['status'] == 'completed'
        assert status['result'] == "Done"

        queue.shutdown()


class TestProgressTracking:
    """Test progress tracking and status methods."""

    def test_get_progress(self):
        """Test getting queue progress."""
        queue = TaskQueue(num_workers=2)

        def quick_task(n):
            time.sleep(0.1)
            return n

        # Submit 3 tasks
        for i in range(3):
            queue.submit(quick_task, args=(i,))

        time.sleep(0.5)  # Wait for tasks to complete

        progress = queue.get_progress()
        assert progress['total'] == 3
        assert progress['completed'] == 3
        assert progress['failed'] == 0
        assert progress['progress'] == 100.0

        queue.shutdown()

    def test_results(self):
        """Test getting results."""
        queue = TaskQueue(num_workers=2)

        def multiply(a, b):
            return a * b

        task_id = queue.submit(multiply, args=(3, 4))
        queue.wait_all(timeout=5)

        results = queue.results()
        assert task_id in results
        assert results[task_id]['result'] == 12

        queue.shutdown()

    def test_status_task_not_found(self):
        """Test status for non-existent task."""
        queue = TaskQueue(num_workers=2)

        status = queue.status("nonexistent")
        assert status is None

        queue.shutdown()


class TestWaitAll:
    """Test wait_all functionality."""

    def test_wait_all_timeout(self):
        """Test that wait_all respects timeout."""
        queue = TaskQueue(num_workers=2)

        def long_task():
            time.sleep(10)
            return "Done"

        queue.submit(long_task)
        result = queue.wait_all(timeout=0.5)

        assert result is False  # Should timeout

        queue.shutdown()


class TestContextManager:
    """Test context manager support."""

    def test_context_manager(self):
        """Test using TaskQueue as a context manager."""
        def simple_task():
            return "Done"

        with TaskQueue(num_workers=2) as queue:
            task_id = queue.submit(simple_task)
            queue.wait_all(timeout=5)

            status = queue.status(task_id)
            assert status['status'] == 'completed'

        # Queue should be shut down after context
        assert not queue.is_running()
