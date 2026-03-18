"""Permission request mechanism for tools that need user approval.

Tools call `request_permission()` which blocks until the UI responds.
The UI registers a handler via `set_permission_handler()`.
"""

from __future__ import annotations

import threading
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Handler signature: (tool_name, description, details) -> bool
# Must be thread-safe — called from the loop engine's worker thread.
_permission_handler: Callable[[str, str, str], bool] | None = None
_handler_lock = threading.Lock()


def set_permission_handler(handler: Callable[[str, str, str], bool] | None) -> None:
    """Register a UI handler for permission requests.

    Args:
        handler: Callable(tool_name, description, details) -> bool.
                 Called from a background thread. Must block until
                 the user responds and return True (allow) or False (deny).
    """
    global _permission_handler
    with _handler_lock:
        _permission_handler = handler


def request_permission(tool_name: str, description: str, details: str = "") -> bool:
    """Request user permission to execute an action.

    Blocks until the user responds. Returns True if approved, False if denied.
    If no handler is registered, defaults to True (auto-approve).
    """
    with _handler_lock:
        handler = _permission_handler

    if handler is None:
        # No UI registered — auto-approve (classic mode fallback)
        return True

    try:
        return handler(tool_name, description, details)
    except Exception as e:
        logger.error("Permission handler failed: %s — denying for safety", e)
        return False
