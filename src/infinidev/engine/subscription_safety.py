"""Host-wide single-flight coordination for subscription-backed evaluations."""

from __future__ import annotations

import fcntl
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator


SUBSCRIPTION_LOCK_PATH = Path(tempfile.gettempdir()) / "infinidev-subscription-global.lock"
_request_interval: ContextVar[float] = ContextVar("infinidev_request_interval", default=0.0)
_last_request_started: ContextVar[float | None] = ContextVar(
    "infinidev_last_request_started", default=None
)


@contextmanager
def subscription_single_flight(path: Path = SUBSCRIPTION_LOCK_PATH) -> Iterator[None]:
    """Reject concurrent subscription campaigns across every Infinidev runner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another subscription-backed evaluation owns {path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def paced_llm_requests(min_interval_seconds: float) -> Iterator[None]:
    """Apply a task-local minimum interval at the global LiteLLM boundary."""
    if min_interval_seconds < 0:
        raise ValueError("minimum LLM request interval cannot be negative")
    interval_token = _request_interval.set(float(min_interval_seconds))
    started_token = _last_request_started.set(None)
    try:
        yield
    finally:
        _last_request_started.reset(started_token)
        _request_interval.reset(interval_token)


def pace_llm_request() -> None:
    """Wait before an LLM request only when an evaluation pacing scope is active."""
    interval = _request_interval.get()
    if interval <= 0:
        return
    now = time.monotonic()
    previous = _last_request_started.get()
    if previous is not None:
        remaining = interval - (now - previous)
        if remaining > 0:
            time.sleep(remaining)
    _last_request_started.set(time.monotonic())
