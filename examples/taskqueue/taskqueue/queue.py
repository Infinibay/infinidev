"""Task queue with thread pool worker management.

This module provides a TaskQueue class that accepts callables with arguments,
distributes work to a configurable pool of worker threads, and supports
priority levels, retry with exponential backoff, timeout, and progress tracking.
"""
import threading
import time
import logging
from typing import Any, Dict, Optional, Callable
from heapq import heappush, heappop

from .task import Task, TaskStatus, Priority
from .worker import WorkerPool

logger = logging.getLogger(__name__)


class TaskQueue:
    """Thread-safe task queue with priority support and worker pool.

    Features:
        - Priority levels (HIGH, MEDIUM, LOW)
        - Configurable worker pool size
        - Task retry with exponential backoff
        - Task timeout support
        - Progress tracking and status monitoring

    Example:
        >>> queue = TaskQueue(num_workers=4)
        >>> task_id = queue.submit(my_function, args=(1, 2), priority=Priority.HIGH)
        >>> status = queue.status(task_id)
        >>> results = queue.results()
        >>> queue.wait_all()
    """

    def __init__(
        self,
        num_workers: int = 4,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        start_processing: bool = True
    ) -> None:
        """Initialize the task queue.

        Args:
            num_workers: Number of worker threads in the pool
            retry_delay: Initial delay in seconds before first retry
            backoff_factor: Multiplier for retry delay (exponential backoff)
            start_processing: If True, start processing tasks immediately
        """
        self._lock = threading.RLock()
        self._worker_pool = WorkerPool(num_workers)
        self._worker_pool.set_task_callback(self._on_task_complete)

        self._tasks: Dict[str, Task] = {}
        self._pending_tasks: list = []  # heap of (priority_value, timestamp, task_id)
        self._retry_delay = retry_delay
        self._backoff_factor = backoff_factor
        self._running = True
        self._results_ready = threading.Condition(self._lock)
        self._new_task_event = threading.Event()

        # Start dispatcher thread
        self._dispatcher = threading.Thread(
            target=self._dispatcher_loop, daemon=True
        )
        self._dispatcher.start()

        # Statistics
        self._stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'timeout': 0,
        }

        self._processing_started = start_processing
        self._start_processing_event = threading.Event()
        if start_processing:
            self._start_processing_event.set()
        logger.info(f"TaskQueue initialized with {num_workers} workers")

    def start_processing(self) -> None:
        """Start processing tasks from the queue.

        This is useful when you want to batch submit tasks before processing begins.
        """
        if not self._processing_started:
            self._processing_started = True
            self._start_processing_event.set()
            self._new_task_event.set()
            logger.info("Task processing started")

    def submit(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: Priority = Priority.MEDIUM,
        timeout: Optional[float] = None,
        max_retries: int = 3
    ) -> str:
        """Submit a task to the queue.

        Args:
            func: Callable to execute
            args: Positional arguments for the callable
            kwargs: Keyword arguments for the callable
            priority: Task priority level (HIGH, MEDIUM, LOW)
            timeout: Maximum execution time in seconds
            max_retries: Maximum number of retry attempts

        Returns:
            Task ID for tracking the submitted task
        """
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )

        with self._lock:
            self._tasks[task.id] = task
            heappush(self._pending_tasks, (task.priority.value, task.created_at, task.id))
            self._stats['submitted'] += 1
            logger.debug(f"Task {task.id} submitted with priority {priority.name}")
            self._new_task_event.set()

        return task.id

    def status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a task.

        Args:
            task_id: The task identifier

        Returns:
            Dictionary with task status information, or None if task not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            elapsed = task.get_elapsed_time() if task.started_at else 0.0
            return {
                'id': task.id,
                'status': task.status.value,
                'priority': task.priority.name,
                'result': task.result,
                'error': task.error,
                'retries': task.retries,
                'max_retries': task.max_retries,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'elapsed_time': elapsed,
            }

    def results(self) -> Dict[str, Dict[str, Any]]:
        """Get results of all completed tasks.

        Returns:
            Dictionary mapping task IDs to their results/status
        """
        with self._lock:
            return {
                task_id: self._format_task_result(task)
                for task_id, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT)
            }

    def _format_task_result(self, task: Task) -> Dict[str, Any]:
        """Format a task result for return."""
        return {
            'status': task.status.value,
            'result': task.result,
            'error': task.error,
            'retries': task.retries,
        }

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """Wait for all tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()

        while True:
            with self._lock:
                pending = sum(
                    1 for t in self._tasks.values()
                    if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                )

                if pending == 0:
                    return True

                remaining = timeout - (time.time() - start_time) if timeout else None
                if remaining is not None and remaining <= 0:
                    return False

                self._results_ready.wait(timeout=min(remaining or 0.1, 0.1))

    def get_progress(self) -> Dict[str, Any]:
        """Get overall queue progress statistics.

        Returns:
            Dictionary with progress information
        """
        with self._lock:
            total = len(self._tasks)
            completed = sum(
                1 for t in self._tasks.values()
                if t.status == TaskStatus.COMPLETED
            )
            failed = sum(
                1 for t in self._tasks.values()
                if t.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT)
            )
            pending = sum(
                1 for t in self._tasks.values()
                if t.status == TaskStatus.PENDING
            )
            running = sum(
                1 for t in self._tasks.values()
                if t.status == TaskStatus.RUNNING
            )

            return {
                'total': total,
                'completed': completed,
                'failed': failed,
                'pending': pending,
                'running': running,
                'progress': (completed / total * 100) if total > 0 else 0.0,
            }

    def _get_next_task(self) -> Optional[Task]:
        """Get the next task from the queue (highest priority first).

        Returns:
            Next Task to execute, or None if queue is empty
        """
        while self._pending_tasks:
            _, _, task_id = heappop(self._pending_tasks)
            task = self._tasks.get(task_id)

            if task and task.status == TaskStatus.PENDING:
                task.mark_running()
                return task

        return None

    def _dispatcher_loop(self) -> None:
        """Dispatcher thread that assigns tasks from priority heap to workers.

        This ensures tasks are executed in priority order.
        """
        while self._running:
            self._new_task_event.wait(timeout=0.05)
            self._new_task_event.clear()

            while self._running:
                next_task = self._get_next_task_with_lock()
                if next_task is None:
                    break

                self._worker_pool.task_queue.put(next_task)

    def _get_next_task_with_lock(self) -> Optional[Task]:
        """Get next task with lock held."""
        with self._lock:
            while self._pending_tasks:
                _, _, task_id = heappop(self._pending_tasks)
                task = self._tasks.get(task_id)

                if task and task.status == TaskStatus.PENDING:
                    task.mark_running()
                    return task

        return None

    def _on_task_complete(self, task: Task) -> None:
        """Callback when a worker completes a task.

        Args:
            task: The completed task
        """
        with self._lock:
            if task.status == TaskStatus.COMPLETED:
                self._stats['completed'] += 1
                logger.debug(f"Task {task.id} completed")

            elif task.status == TaskStatus.FAILED:
                if task.should_retry():
                    # Schedule retry with exponential backoff
                    task.increment_retries()
                    delay = self._retry_delay * (self._backoff_factor ** task.retries)
                    logger.warning(
                        f"Task {task.id} failed, retry {task.retries}/{task.max_retries} in {delay:.1f}s"
                    )
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                    task.error = None
                    heappush(
                        self._pending_tasks,
                        (task.priority.value, time.time(), task.id)
                    )
                    self._new_task_event.set()
                else:
                    # Final failure
                    self._stats['failed'] += 1
                    logger.error(f"Task {task.id} failed permanently: {task.error}")

            elif task.status == TaskStatus.TIMEOUT:
                if task.should_retry():
                    task.increment_retries()
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                    task.error = None
                    heappush(
                        self._pending_tasks,
                        (task.priority.value, time.time(), task.id)
                    )
                    logger.warning(
                        f"Task {task.id} timed out, retry {task.retries}/{task.max_retries}"
                    )
                    self._new_task_event.set()
                else:
                    # Already marked as TIMEOUT by worker, just update stats
                    self._stats['timeout'] += 1
                    logger.error(f"Task {task.id} timed out permanently")

            # Notify waiters
            self._results_ready.notify_all()

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the task queue and worker pool.

        Args:
            wait: If True, wait for running tasks to complete
        """
        logger.info("Shutting down TaskQueue...")
        self._running = False
        self._new_task_event.set()
        self._worker_pool.shutdown(wait=wait)
        logger.info("TaskQueue shutdown complete")

    def is_running(self) -> bool:
        """Check if the queue is still running."""
        return self._running

    @property
    def stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self._lock:
            return self._stats.copy()

    def __len__(self) -> int:
        """Return total number of submitted tasks."""
        with self._lock:
            return len(self._tasks)

    def __enter__(self) -> 'TaskQueue':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - ensures cleanup."""
        self.shutdown(wait=True)
        return False
