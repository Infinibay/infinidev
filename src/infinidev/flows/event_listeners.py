"""Centralized event bus for engine → UI communication.

All engines (loop, tree, analysis, review, gather) emit events here.
Consumers (TUI, WebSocket, etc.) subscribe once and receive all events.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, int, str, dict[str, Any]], None]


class EventBus:
    """Thread-safe publish/subscribe event bus.

    Signature for all callbacks:
        callback(event_type: str, project_id: int, agent_id: str, data: dict)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[EventCallback] = []

    def subscribe(self, callback: EventCallback) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    def emit(
        self,
        event_type: str,
        project_id: int,
        agent_id: str,
        data: dict[str, Any],
    ) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(event_type, project_id, agent_id, data)
            except Exception:
                logger.debug("EventBus: subscriber %r failed on %s", cb, event_type, exc_info=True)

    @property
    def has_subscribers(self) -> bool:
        with self._lock:
            return len(self._subscribers) > 0


event_bus = EventBus()
