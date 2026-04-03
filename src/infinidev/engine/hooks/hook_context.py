"""Hook system for intercepting and modifying engine behavior.

Unlike EventBus (fire-and-forget observation for UI), hooks run inline
in the execution pipeline and can modify arguments, results, or skip
execution entirely.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)
from infinidev.engine.hooks.hook_event import HookEvent


@dataclass
class HookContext:
    """Mutable bag of data passed through the hook chain.

    Each hook event populates relevant fields.  Hooks read and modify
    these freely.  Setting ``skip = True`` in a pre-hook prevents the
    default action (e.g. tool execution).

    A single ``HookContext`` instance is reused for the PRE and POST
    phases of the same operation so that hooks can stash data in
    ``metadata`` during the PRE phase and read it back in POST.
    """

    event: HookEvent
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    skip: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    project_id: int = 0
    agent_id: str = ""


