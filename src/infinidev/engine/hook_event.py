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


class HookEvent(str, Enum):
    """All hookable events in the engine lifecycle."""

    # Tool hooks
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"

    # Step lifecycle
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"
    STEP_TRANSITION = "step_transition"

    # Loop lifecycle
    LOOP_START = "loop_start"
    LOOP_END = "loop_end"

    # LLM call
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"


