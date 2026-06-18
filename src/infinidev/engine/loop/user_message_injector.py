"""User-message injection for the loop engine.

A thread-safe queue plus the logic that injects live user messages into
the running loop at the points where the engine drains them: at step
start, mid-step before an LLM call, and on a late ``step_complete``.
Extracted verbatim from ``LoopEngine`` so the engine module stays
focused on the loop; behavior is unchanged.
"""

from __future__ import annotations

import queue
from typing import Any, TYPE_CHECKING

from infinidev.engine.engine_logging import emit_log as _emit_log

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext


class UserMessageInjector:
    """Owns the user-message queue and the inject/drain/reject logic."""

    def __init__(self) -> None:
        # Thread-safe queue for user messages injected mid-task.
        self._queue: queue.Queue[str] = queue.Queue()

    def inject(self, message: str) -> None:
        """Inject a user message into the running loop (thread-safe).

        The message will be included in the next iteration's prompt as
        a ``<user-message>`` block, giving the LLM live guidance without
        interrupting the current step.
        """
        self._queue.put(message)

    def drain(self) -> list[str]:
        """Drain all pending user messages from the queue."""
        messages = []
        while not self._queue.empty():
            try:
                messages.append(self._queue.get_nowait())
            except Exception:
                break
        return messages

    def inject_mid_step(
        self, ctx: "ExecutionContext", messages: list[dict[str, Any]],
    ) -> None:
        """Drain any pending user messages and inject them as urgent
        ``user``-role turns before the next LLM call.

        No-op if the queue is empty. Used at the top of the inner loop
        so the model always sees the freshest user input even when the
        user speaks while an LLM call is in flight.
        """
        drained = self.drain()
        if not drained:
            return
        _emit_log(
            "info",
            f"⚡ mid-step user message drained ({len(drained)} msg(s)) "
            f"— injecting before next LLM call",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        for m in drained:
            messages.append({
                "role": "user",
                "content": (
                    "URGENT — I just sent this while you were working. "
                    "Acknowledge it with `send_message` as your VERY NEXT "
                    f"tool call before continuing your current step:\n\n{m}"
                ),
            })

    def reject_step_complete_on_late_message(
        self,
        ctx: "ExecutionContext",
        messages: list[dict[str, Any]],
        step_complete_id: str,
    ) -> bool:
        """If the user spoke AFTER the model called ``step_complete`` but
        BEFORE we processed the completion, reject the step and force
        one more LLM call so the user can be acknowledged.

        Writes a ``tool``-role message on the ``step_complete`` tool id
        — providers treat that as "your previous close was overridden
        by this feedback", which is exactly the framing we want.
        Returns ``True`` if the rejection fired (caller should
        ``continue`` the loop), ``False`` if the queue was empty.
        """
        drained = self.drain()
        if not drained:
            return False

        _emit_log(
            "info",
            f"⚡ late mid-step user message drained ({len(drained)} msg(s)) "
            f"— overriding step_complete, forcing one more LLM call",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        rejection_body = (
            "step_complete REJECTED — the user just spoke while "
            "you were finishing your last action. You MUST "
            "acknowledge them BEFORE completing this step. Call "
            "`send_message` with a brief (1-2 sentence) reply "
            "that addresses what they said, then call "
            "step_complete again. The user's message(s) were:\n\n"
            + "\n\n---\n\n".join(drained)
        )

        self._overwrite_step_complete_tool_result(
            messages, step_complete_id, rejection_body,
        )
        return True

    @staticmethod
    def _overwrite_step_complete_tool_result(
        messages: list[dict[str, Any]],
        step_complete_id: str,
        new_body: str,
    ) -> None:
        """Override the ``acknowledged`` stub on a step_complete tool id.

        Anthropic requires exactly one tool_result per tool_use_id, so
        we locate the existing tool message (the "acknowledged" stub
        appended by ``_execute_regular_tools`` /
        ``_build_pseudo_only_messages``) and rewrite its content in
        place rather than appending a second one. On OpenAI both
        approaches work; on Anthropic appending duplicates raises.
        Falls back to a fresh append if no prior result is found —
        that path keeps the loop well-formed even if the assumption
        breaks.
        """
        for msg in reversed(messages):
            if (
                msg.get("role") == "tool"
                and msg.get("tool_call_id") == step_complete_id
            ):
                msg["content"] = new_body
                return
        messages.append({
            "role": "tool",
            "tool_call_id": step_complete_id,
            "content": new_body,
        })
