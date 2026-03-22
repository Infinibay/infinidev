"""Worker pool implementation with thread management and task execution.

This module provides a worker pool that executes tasks from a shared queue,
handling timeouts, retries, and graceful shutdown.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from queue import Queue, Empty
from typing import Callable, Optional, Any, List

from .task import Task, TaskStatus

logger = logging.getLogger(__name__)


class WorkerPool:
    """Pool of worker threads that process tasks from a queue.

    Attributes:
        num_workers: Number of worker threads in the pool
        task_queue: Shared queue of tasks to process
    """

    def __init__(self, num_workers: int = 4) -> None:
        """Initialize the worker pool.

        Args:
            num_workers: Number of worker threads to create
        """
        self.num_workers = num_workers
        self.task_queue: Queue = Queue()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._task_callback: Optional[Callable[[Task], None]] = None
        self._delayed_tasks: List[tuple] = []  # (delay_until, task_id)
        self._delayed_lock = threading.Lock()
        self._start_workers()

    def set_task_callback(self, callback: Callable[[Task], None]) -> None:
        """Set the callback to be called when a task completes.

        Args:
            callback: Function to call with completed Task objects
        """
        self._task_callback = callback

    def _start_workers(self) -> None:
        """Start all worker threads."""
        self._running = True
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self._workers.append(worker)
        logger.info(f"Worker pool started with {self.num_workers} workers")

    def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop - fetch and execute tasks.

        Args:
            worker_id: Unique identifier for this worker
        """
        while self._running:
            task = None
            try:
                task = self.task_queue.get(timeout=0.1)
            except Empty:
                continue

            if task is not None:
                try:
                    self._execute_task(worker_id, task)
                finally:
                    self.task_queue.task_done()

    def _execute_task(self, worker_id: int, task: Task) -> None:
        """Execute a single task with timeout handling.

        Args:
            worker_id: Worker identifier
            task: Task to execute
        """
        logger.debug(f"Worker {worker_id} executing task {task.id}")
        task.mark_running()

        try:
            if task.timeout:
                result = self._execute_with_timeout(
                    task.func, task.args, task.kwargs, task.timeout
                )
            else:
                result = task.func(*task.args, **task.kwargs)

            task.mark_completed(result)
            logger.debug(f"Worker {worker_id} completed task {task.id}")

        except FuturesTimeoutError:
            task.mark_timeout()
            logger.warning(f"Worker {worker_id} task {task.id} timed out")

        except Exception as e:
            task.mark_failed(str(e))
            logger.warning(f"Worker {worker_id} task {task.id} failed: {e}")

        finally:
            if self._task_callback:
                self._task_callback(task)

    def _execute_with_timeout(
        self, func: Callable, args: tuple, kwargs: dict, timeout: float
    ) -> Any:
        """Execute a function with a timeout.

        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Maximum execution time in seconds

        Returns:
            Function result

        Raises:
            FuturesTimeoutError: If execution exceeds timeout
        """
        result_container = {}
        error_container = {}
        completed = threading.Event()

        def wrapper():
            try:
                result_container['result'] = func(*args, **kwargs)
            except Exception as e:
                error_container['error'] = e
            finally:
                completed.set()

        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()

        if not completed.wait(timeout=timeout):
            # Timeout - thread keeps running in background (daemon)
            raise FuturesTimeoutError(f"Task exceeded {timeout}s timeout")

        if 'error' in error_container:
            raise error_container['error']

        return result_container.get('result')

    def get_task(self) -> Optional[Task]:
        """Get the next task from the queue (for testing)."""
        try:
            return self.task_queue.get_nowait()
        except Empty:
            return None

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool.

        Args:
            wait: If True, wait for workers to finish current tasks
        """
        logger.info("Shutting down worker pool")
        self._running = False

        if wait:
            for worker in self._workers:
                worker.join(timeout=1.0)

        self._workers.clear()
        logger.info("Worker pool shutdown complete")

    def queue_size(self) -> int:
        """Return the number of tasks in the queue."""
        return self.task_queue.qsize()

    def active_workers(self) -> int:
        """Return the number of workers actively processing tasks."""
        return self.queue_size()

    def add_delayed_task(self, task_id: str, delay: float) -> None:
        """Add a task to be re-queued after a delay.

        Args:
            task_id: Task identifier
            delay: Delay in seconds before re-queuing
        """
        with self._delayed_lock:
            self._delayed_tasks.append((time.time() + delay, task_id))

    def process_delayed_tasks(self, tasks_dict: dict) -> None:
        """Process delayed tasks and re-queue them.

        Args:
            tasks_dict: Dictionary mapping task IDs to Task objects
        """
        current_time = time.time()
        tasks_to_requeue = []

        with self._delayed_lock:
            remaining = []
            for delay_until, task_id in self._delayed_tasks:
                if delay_until <= current_time:
                    tasks_to_requeue.append(task_id)
                else:
                    remaining.append((delay_until, task_id))
            self._delayed_tasks = remaining

        for task_id in tasks_to_requeue:
            task = tasks_dict.get(task_id)
            if task:
                self.task_queue.put(task)
                logger.debug(f"Re-queued delayed task {task_id}")

    def start_delayed_task_processor(
        self, tasks_dict: dict, stop_event: threading.Event
    ) -> threading.Thread:
        """Start a background thread to process delayed tasks.

        Args:
            tasks_dict: Dictionary mapping task IDs to Task objects
            stop_event: Event to signal the thread to stop

        Returns:
            The started thread
        """
        thread = threading.Thread(
            target=self._delayed_task_loop, args=(tasks_dict, stop_event), daemon=True
        )
        thread.start()
        return thread

    def _delayed_task_loop(
        self, tasks_dict: dict, stop_event: threading.Event
    ) -> None:
        """Background loop to process delayed tasks.

        Args:
            tasks_dict: Dictionary mapping task IDs to Task objects
            stop_event: Event to signal the thread to stop
        """
        while not stop_event.is_set():
            self.process_delayed_tasks(tasks_dict)
            stop_event.wait(timeout=0.1)
