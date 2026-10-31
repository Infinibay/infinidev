"""User-message injection for the loop engine.

A thread-safe queue projects messages at Step start, between model calls,
and on a late ``step_complete``. Literal guidance and image references enter
Task state before delivery; image payloads remain ephemeral.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING
from uuid import uuid4

from infinidev.engine.engine_logging import emit_log as _emit_log
from infinidev.engine.loop.user_guidance import retain_user_guidance

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext
    from infinidev.engine.loop.loop_state import LoopState
    from infinidev.engine.multimodal import ImageAttachment


@dataclass(frozen=True)
class InjectedUserMessage:
    """User guidance and images submitted while a task is running."""

    text: str
    attachments: tuple["ImageAttachment", ...] = ()
    message_id: str = field(default_factory=lambda: uuid4().hex)


class UserMessageInjector:
    """Owns the user-message queue and the inject/drain/reject logic."""

    def __init__(
        self, on_context_change: Callable[[list[str]], None] | None = None
    ) -> None:
        # Thread-safe queue for user messages injected mid-task.
        self._queue: deque[InjectedUserMessage] = deque()
        self._lock = threading.Lock()
        self._on_context_change = on_context_change

    def inject(
        self,
        message: str,
        attachments: list["ImageAttachment"] | None = None,
    ) -> None:
        """Inject user guidance and optional images into the running loop."""
        with self._lock:
            self._queue.append(InjectedUserMessage(message, tuple(attachments or ())))

    @property
    def has_pending(self) -> bool:
        """Whether a fresh user message is still awaiting model delivery."""
        with self._lock:
            return bool(self._queue)

    def retain_pending(self, state: LoopState | None) -> bool:
        """Checkpoint pending input without consuming its urgent delivery."""
        with self._lock:
            pending = list(self._queue)
        retain_user_guidance(state, pending)
        return bool(pending)

    def drain(self) -> list[str]:
        """Drain pending text in FIFO order (backward-compatible API)."""
        return [item.text for item in self._drain_items()]

    def drain_for_state(self, state: LoopState | None) -> list[InjectedUserMessage]:
        """Drain complete messages and preserve their literal Task context."""
        messages = self._drain_items()
        retain_user_guidance(state, messages)
        return messages

    @staticmethod
    def _message_content(item: InjectedUserMessage, prefix: str) -> Any:
        text = prefix + item.text
        if not item.attachments:
            return text
        from infinidev.engine.multimodal import build_user_content, mention_paths_as_text

        try:
            from infinidev.config.model_capabilities import get_capability_snapshot

            supports_vision = get_capability_snapshot().supports_vision
        except Exception:
            supports_vision = False
        if supports_vision:
            return build_user_content(text, item.attachments)
        return mention_paths_as_text(text, item.attachments)

    def _drain_items(self) -> list[InjectedUserMessage]:
        """Drain structured pending messages for internal projection paths."""
        with self._lock:
            messages = list(self._queue)
            self._queue.clear()
        return messages

    def inject_mid_step(
        self, ctx: "ExecutionContext", messages: list[dict[str, Any]],
    ) -> list[str]:
        """Drain any pending user messages and inject them as urgent
        ``user``-role turns before the next LLM call.

        No-op if the queue is empty. Used at the top of the inner loop
        so the model always sees the freshest user input even when the
        user speaks while an LLM call is in flight.
        """
        drained = self.drain_for_state(getattr(ctx, "state", None))
        if not drained:
            return []
        _emit_log(
            "info",
            f"⚡ mid-step user message drained ({len(drained)} msg(s)) "
            f"— injecting before next LLM call",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        for item in drained:
            prefix = (
                "URGENT — I just sent this while you were working. "
                "Acknowledge it with `send_message` as your VERY NEXT "
                "tool call before continuing your current step:\n\n"
            )
            messages.append({"role": "user", "content": self._message_content(item, prefix)})
        texts = [item.text for item in drained]
        if self._on_context_change is not None:
            self._on_context_change(texts)
        return texts

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
        drained = self.drain_for_state(getattr(ctx, "state", None))
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
            + "\n\n---\n\n".join(item.text for item in drained)
        )
        if self._on_context_change is not None:
            self._on_context_change([item.text for item in drained])

        self._overwrite_step_complete_tool_result(
            messages, step_complete_id, rejection_body,
        )
        for item in drained:
            if item.attachments:
                messages.append({
                    "role": "user",
                    "content": self._message_content(
                        item, "Images from the user's latest guidance:\n\n",
                    ),
                })
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

        This is the single place four of the five step_complete gates
        deliver their feedback, so it is also the single place that has
        to know what shape the conversation is in. In manual mode there
        is no tool channel at all — the assistant turn is prose, acks
        come back as ``user`` — and appending a ``role: "tool"`` message
        answering a call no assistant ever announced makes the next
        request invalid. The check is on the transcript rather than on
        ``ctx.manual_tc`` because this function is reached from four
        callers and one static wrapper, and a fact already visible in
        the messages does not need to be threaded through all of them.
        """
        for msg in reversed(messages):
            if (
                msg.get("role") == "tool"
                and msg.get("tool_call_id") == step_complete_id
            ):
                msg["content"] = new_body
                return

        speaks_tool_protocol = any(
            msg.get("role") == "tool" or msg.get("tool_calls") for msg in messages
        )
        if not speaks_tool_protocol:
            messages.append({"role": "user", "content": new_body})
            return

        messages.append({
            "role": "tool",
            "tool_call_id": step_complete_id,
            "content": new_body,
        })
