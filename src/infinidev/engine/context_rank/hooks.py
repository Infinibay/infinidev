"""ContextRank engine hooks — thin integration layer.

Wired into LoopEngine to emit interaction events and context messages.
All methods are no-ops when logging is disabled.  Failures are caught
and logged — never propagated to the engine.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ContextRankHooks:
    """Stateful helper attached to a single LoopEngine.execute() invocation."""

    def __init__(self) -> None:
        self._enabled = False
        self._session_id: str = ""
        self._task_id: str = ""
        self._task_context_id: int | None = None
        self._active_step_context_id: int | None = None

    def start(self, session_id: str, task_id: str, task_description: str) -> None:
        """Called once at the start of execute(). Logs the task input context."""
        from infinidev.config.settings import settings as _s
        self._enabled = _s.CONTEXT_RANK_LOGGING_ENABLED
        if not self._enabled:
            return
        self._session_id = session_id
        self._task_id = task_id

        from infinidev.engine.context_rank.logger import log_context, compute_context_embedding
        self._task_context_id = log_context(
            session_id, task_id, "task_input", task_description,
        )
        self._active_step_context_id = self._task_context_id
        # Compute embedding eagerly in a background thread so it's ready
        # for predictive scoring on iteration 0.
        if self._task_context_id is not None:
            threading.Thread(
                target=compute_context_embedding,
                args=(self._task_context_id,),
                daemon=True,
            ).start()

    def on_step_activated(self, title: str, explanation: str, iteration: int, step_index: int) -> None:
        """Called when a new plan step becomes active."""
        if not self._enabled:
            return
        from infinidev.engine.context_rank.logger import log_context, compute_context_embedding

        # Level 2: step title
        title_ctx_id = log_context(
            self._session_id, self._task_id, "step_title",
            title, iteration, step_index,
        )
        # Level 3: step description (if non-empty)
        desc_ctx_id = None
        if explanation and explanation.strip():
            desc_ctx_id = log_context(
                self._session_id, self._task_id, "step_description",
                explanation, iteration, step_index,
            )

        # Use the most specific context_id for linking subsequent tool calls
        self._active_step_context_id = desc_ctx_id or title_ctx_id or self._task_context_id

        # Background embed
        for cid in (title_ctx_id, desc_ctx_id):
            if cid is not None:
                threading.Thread(
                    target=compute_context_embedding,
                    args=(cid,),
                    daemon=True,
                ).start()

    def on_tool_call(self, tool_name: str, arguments: str | dict, iteration: int) -> None:
        """Called after each regular tool call completes."""
        if not self._enabled:
            return
        # Parse arguments if they come as a JSON string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        from infinidev.engine.context_rank.logger import log_tool_call
        log_tool_call(
            self._session_id, self._task_id,
            self._active_step_context_id, iteration,
            tool_name, arguments,
        )

    def finish(self) -> None:
        """Called once when the task ends. Snapshots session scores."""
        if not self._enabled:
            return
        from infinidev.engine.context_rank.logger import snapshot_session_scores
        try:
            snapshot_session_scores(self._session_id, self._task_id)
        except Exception:
            logger.debug("ContextRank snapshot failed", exc_info=True)
