"""Stdin-prompt request mechanism for execute_command.

Parallel to ``tools/permission.py`` but for the case where a running
subprocess pauses waiting for input (password, passphrase, username).
Tools call ``request_stdin_input()`` which blocks until the UI
responds. The UI registers a handler via ``set_stdin_input_handler()``.

Return contract:
    - str  → write that string + newline to the subprocess's stdin
    - None → kill the subprocess
"""

from __future__ import annotations

import threading
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Handler signature: (command, prompt_text, stdout_so_far, stderr_so_far) -> str | None
# Must be thread-safe — called from a worker thread. Must block
# until the user responds. Return a string to forward to stdin, or
# None to instruct the caller to kill the process.
_handler: Callable[[str, str, str, str], "str | None"] | None = None
_handler_lock = threading.Lock()


def set_stdin_input_handler(
    handler: Callable[[str, str, str, str], "str | None"] | None,
) -> None:
    """Register a UI handler for stdin-prompt requests.

    Args:
        handler: Callable(command, prompt_text, stdout_so_far,
                 stderr_so_far) → str | None.
                 Called from a background thread; must block until the
                 user responds. Return a string to feed stdin, or
                 None to kill the process. The stdout/stderr snapshots
                 let the UI show what the process has emitted so far
                 so the user can decide whether to reply or kill.
    """
    global _handler
    with _handler_lock:
        _handler = handler


def has_stdin_input_handler() -> bool:
    """Return True iff a UI handler is currently registered."""
    with _handler_lock:
        return _handler is not None


def request_stdin_input(
    command: str,
    prompt_text: str,
    stdout_so_far: str = "",
    stderr_so_far: str = "",
) -> "str | None":
    """Request a stdin value from the user in response to a detected
    subprocess prompt. Blocks until the user responds.

    Returns the reply string (to be piped into the subprocess's stdin
    followed by a newline), or None to signal "kill the process".

    If no handler is registered, returns None immediately so the caller
    kills the process rather than hanging — the classic-mode fallback
    uses ``stdin=DEVNULL`` and never reaches this path in practice.
    """
    with _handler_lock:
        handler = _handler
    if handler is None:
        return None
    try:
        return handler(command, prompt_text, stdout_so_far, stderr_so_far)
    except Exception as exc:
        logger.error("stdin handler failed: %s — killing for safety", exc)
        return None
