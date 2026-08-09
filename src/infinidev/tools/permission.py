"""Permission request mechanism for tools that need user approval.

Tools call `request_permission()` which blocks until the UI responds.
The UI registers a handler via `set_permission_handler()`.
"""

from __future__ import annotations

import re
import threading
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Handler signature: (tool_name, description, details) -> bool
# Must be thread-safe — called from the loop engine's worker thread.
_permission_handler: Callable[[str, str, str], bool] | None = None
_handler_lock = threading.Lock()

_TEST_AUTHORIZATION_RE = re.compile(
    r"\b(?:run|execute|verify|validate|test|ejecut(?:a|ar|e)|corr(?:e|er)|"
    r"prob(?:a|ar|á)|verific(?:a|ar)|valid(?:a|ar))\b[^.\n]{0,100}"
    r"\b(?:tests?|pytest|test\s+suite|pruebas?|suite\s+de\s+pruebas)\b|"
    r"\b(?:tests?|pytest|pruebas?)\b[^.\n]{0,100}"
    r"\b(?:run|execute|verify|validate|ejecut(?:a|ar|e)|corr(?:e|er)|"
    r"prob(?:a|ar|á)|verific(?:a|ar)|valid(?:a|ar))\b",
    re.IGNORECASE,
)
_TEST_DENIAL_RE = re.compile(
    r"\b(?:do\s+not|don't|never)\s+(?:run|execute)?\s*(?:the\s+)?tests?\b|"
    r"\b(?:no|nunca)\s+(?:ejecut(?:es|ar)|corr(?:as|er)|"
    r"prueb(?:es|ar))\s+(?:las?\s+)?pruebas?\b",
    re.IGNORECASE,
)
_VERIFICATION_AUTHORIZATION_RE = re.compile(
    r"\b(?:verify|validate|verific(?:a|ar|á)|valid(?:a|ar))\b|"
    r"\bverification\b[^.\n]{0,60}\b(?:complete|passing|required)\b|"
    r"\bverificaci[oó]n\b[^.\n]{0,60}\b(?:completa|requerida)\b",
    re.IGNORECASE,
)
_VERIFICATION_DENIAL_RE = re.compile(
    r"\b(?:do\s+not|don't|never)\s+(?:verify|validate)\b|"
    r"\b(?:no|nunca)\s+(?:verific(?:ar|a|á)|valid(?:ar|a|á))\b",
    re.IGNORECASE,
)
_EXPLICIT_PYTHON_SCRIPT_RE = re.compile(
    r"\bpython(?:3(?:\.\d+)?)?\s+(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py"
    r"(?:\s+[A-Za-z0-9_./:=+-]+)*",
    re.IGNORECASE,
)


def _explicit_python_command_is_authorized(command: str, request: str) -> bool:
    """Recognize a Python script command the user literally asked to run."""
    command_without_redirects = re.sub(r"\s+\d*[<>].*$", "", command).strip()
    command_match = _EXPLICIT_PYTHON_SCRIPT_RE.search(command_without_redirects)
    if command_match is None:
        return False
    normalized = " ".join(command_match.group(0).casefold().split())
    for match in _EXPLICIT_PYTHON_SCRIPT_RE.finditer(request):
        candidate = " ".join(match.group(0).casefold().split())
        if candidate != normalized:
            continue
        prefix = request[max(0, match.start() - 160):match.start()]
        if re.search(
            r"\b(?:do\s+not|don't|never|no|nunca)\b[^.\n]{0,80}$",
            prefix,
            re.IGNORECASE,
        ):
            return False
        return bool(re.search(
            r"\b(?:begin|start|run|execute|use|empieza|comienza|ejecuta|usa)\b"
            r"[^.\n]{0,140}$",
            prefix,
            re.IGNORECASE,
        ))
    return False


def make_noninteractive_permission_handler(
    user_request: str,
) -> Callable[[str, str, str], bool]:
    """Authorize only explicitly requested test verification.

    A one-shot CLI has no UI capable of answering an approval prompt. The
    user's literal request can still authorize a bounded verification action:
    a recognized test command may run when tests were requested. Provably
    read-only shell and Python commands bypass this handler in ``auto`` mode.
    Pipelines are accepted only when every other segment is also proven
    read-only, so test authority cannot be widened into a mutation.
    """
    request = (user_request or "").strip()
    tests_authorized = bool(request) and bool(
        _TEST_AUTHORIZATION_RE.search(request)
        or _VERIFICATION_AUTHORIZATION_RE.search(request)
    ) and not bool(
        _TEST_DENIAL_RE.search(request)
        or _VERIFICATION_DENIAL_RE.search(request)
    )

    def decide(tool_name: str, _description: str, details: str) -> bool:
        if tool_name != "execute_command":
            return False
        command = (details or "").strip()
        if not command:
            return False
        from infinidev.engine.guidance.test_runners import is_test_command
        from infinidev.tools.base.command_risk import (
            classify_command,
            split_shell_segments,
        )

        # Merging stderr into stdout does not create a file and is commonly
        # appended to diagnostic commands. Remove only that exact shell form;
        # ordinary output redirections remain visible to the risk parser and
        # fail closed.
        risk_command = re.sub(r"(?<!\S)2>&1(?=\s|$|;|\|)", "", command)
        segments, _ = split_shell_segments(risk_command)
        if segments is None:
            return False
        saw_test = False
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            if is_test_command(segment) and tests_authorized:
                saw_test = True
                continue
            if _explicit_python_command_is_authorized(segment, request):
                saw_test = True
                continue
            if not classify_command(segment)[0]:
                return False
        return saw_test

    return decide


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


def is_permission_handler_registered() -> bool:
    """Whether an interactive UI handler is available to approve prompts.

    Used by ``auto`` mode to fail CLOSED in non-interactive contexts (headless
    ``--prompt``, server): without a handler ``request_permission`` would
    auto-approve, so a risky-but-escalated operation would run silently. The
    ``auto`` branches deny instead when this returns False.
    """
    with _handler_lock:
        return _permission_handler is not None


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
