"""Task representation with status tracking and metadata."""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
import uuid


class TaskStatus(Enum):
    """Status states for a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Priority(Enum):
    """Task priority levels."""
    HIGH = 0
    MEDIUM = 1
    LOW = 2


@dataclass
class Task:
    """Represents a unit of work to be executed.
    
    Attributes:
        id: Unique task identifier
        func: Callable to execute
        args: Positional arguments for the callable
        kwargs: Keyword arguments for the callable
        priority: Task priority level (HIGH, MEDIUM, LOW)
        status: Current status of the task
        result: Result of task execution (if completed)
        error: Error message if task failed
        retries: Number of retry attempts made
        max_retries: Maximum retry attempts allowed
        timeout: Maximum execution time in seconds
        created_at: Timestamp when task was created
        started_at: Timestamp when task started executing
        completed_at: Timestamp when task finished
    """
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    
    # Task metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def mark_running(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
    
    def mark_completed(self, result: Any) -> None:
        """Mark task as successfully completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
    
    def mark_failed(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()
    
    def mark_timeout(self) -> None:
        """Mark task as timed out."""
        self.status = TaskStatus.TIMEOUT
        self.error = f"Task timed out after {self.timeout}s"
        self.completed_at = time.time()
    
    def should_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retries < self.max_retries
    
    def increment_retries(self) -> None:
        """Increment retry counter."""
        self.retries += 1
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since task started."""
        if self.started_at:
            return time.time() - self.started_at
        return 0.0
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __lt__(self, other: 'Task') -> bool:
        """Compare tasks by priority for queue ordering."""
        return self.priority.value < other.priority.value
