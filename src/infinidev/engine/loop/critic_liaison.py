"""Everything the loop needs to know about the pair-programming critic.

The critic itself lives in ``critic.py``. What lived in the engine was the
*relationship* with it, and it was scattered across three places in one
357-line method: a lazy constructor, a ThreadPoolExecutor that raced the
critic against tool execution, and a hundred-line block on the
``step_complete`` path that could veto a step from closing. Read in
sequence they looked like three unrelated concerns.

They are one concern with one rule: **the critic is advisory except on
``step_complete``**. While tools run it can only annotate; when the
principal tries to close a step it can send it back for one more turn.
Making that asymmetry a single class with two entry points is the point of
this module.

Placement matters as much as content. A verdict is appended to the *last
tool message* rather than added as a fresh user turn, because models weight
nearby tokens more heavily and the critique belongs inside the same
attentional block as the action that provoked it.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any, Callable

from infinidev.config.settings import settings
from infinidev.engine.engine_logging import emit_loop_event, emit_log
from infinidev.engine.loop.critic import AssistantCritic, CriticVerdict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StepCompleteReview:
    """What the critic decided about a step that wants to close.

    ``blocked`` is the only outcome that changes control flow; everything
    else is carried forward as advice for the next step's preamble.
    """

    blocked: bool = False
    followup: dict[str, Any] | None = None


class CriticLiaison:
    """Owns the engine's side of the critic relationship."""

    def __init__(self) -> None:
        self._critic: AssistantCritic | None = None
        # A bad critic config should fail loudly once and then be absent —
        # retrying per step would turn one misconfiguration into a warning
        # on every iteration of every run.
        self._init_failed: bool = False
        # Advisory verdicts survive the inner loop's ``break`` by sitting
        # here until the next step's preamble drains them onto a fresh
        # messages list. The list they were produced against is gone.
        self.pending_messages: list[dict[str, Any]] = []

    # ── lifecycle ────────────────────────────────────────────────────

    def get(self, ctx: Any) -> AssistantCritic | None:
        """The critic for this run, built on first use. ``None`` if off."""
        if not settings.ASSISTANT_LLM_ENABLED:
            return None
        if self._critic is not None:
            return self._critic
        if self._init_failed:
            return None
        try:
            descriptions = {
                name: (getattr(tool, "description", None) or "")
                for name, tool in (ctx.tool_dispatch or {}).items()
            }
            self._critic = AssistantCritic(descriptions)
            try:
                from infinidev.engine.loop.critic import set_active_critic

                # Registering the instance is what lets ``ConsultAssistantTool``
                # — the principal's own way of asking for a second opinion —
                # reach this critic without threading it through the tool
                # factory.
                set_active_critic(self._critic)
            except Exception:
                logger.debug("set_active_critic failed", exc_info=True)
            return self._critic
        except Exception as exc:
            logger.warning(
                "assistant critic init failed (%s); disabling for this run", exc,
            )
            self._init_failed = True
            return None

    def drain_pending(self) -> list[dict[str, Any]]:
        """Take the advisory verdicts owed to the next step, if any."""
        pending, self.pending_messages = self.pending_messages, []
        return pending

    # ── while tools run: advisory only ───────────────────────────────

    def review_alongside(
        self,
        ctx: Any,
        messages: list[dict[str, Any]],
        tool_calls: list[Any],
        reasoning: str | None,
        run_tools: Callable[[], int],
    ) -> int:
        """Run *run_tools* and the critic at the same wall clock.

        The critic's LLM call and the principal's tool I/O are independent
        and usually land on different endpoints, so overlapping them costs
        roughly nothing in the steady state. Returns whatever *run_tools*
        returns; the verdict, if any, is appended to *messages*.

        The snapshots are taken *before* the tools run: the critic should
        judge what the principal saw when it decided, not the state its
        actions produced. Reasoning is passed separately because
        ``ContextManager.expire_thinking`` strips it from the history, so
        the critic would otherwise see the actions without the thinking
        that led to them.
        """
        critic = self.get(ctx)
        if critic is None:
            return run_tools()

        messages_snapshot = list(messages)
        calls_snapshot = list(tool_calls)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tools_future = pool.submit(run_tools)
            critic_future = pool.submit(
                critic.review, messages_snapshot, calls_snapshot, reasoning,
            )
            action_tool_calls = tools_future.result()
            try:
                verdict = critic_future.result()
            except Exception as exc:
                logger.warning("assistant critic raised: %s", exc)
                verdict = None

        if verdict is not None and not verdict.is_silent:
            self.attach(ctx, messages, verdict, source="tools")
        return action_tool_calls

    def attach(
        self,
        ctx: Any,
        messages: list[dict[str, Any]],
        verdict: CriticVerdict,
        *,
        source: str,
    ) -> None:
        """Anchor a verdict next to the tool call that provoked it.

        Appending to the last ``role: "tool"`` message keeps the critique
        inside the same attentional block as the principal's own output,
        which models weight more heavily than a separate trailing turn. The
        user-role fallback only fires when there is no tool message at all.
        """
        if verdict.is_silent:
            return
        model_tag = self._critic.model_short_name if self._critic else "assistant"
        note = (
            f"\n\n--- critic note ---\n"
            f"[ASSISTANT ({model_tag}) - {verdict.action}]: {verdict.message}"
        )

        last_tool_index = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if messages[i].get("role") == "tool"
            ),
            -1,
        )
        if last_tool_index >= 0:
            existing = messages[last_tool_index].get("content") or ""
            if not isinstance(existing, str):
                existing = str(existing)
            messages[last_tool_index]["content"] = existing + note
        else:
            messages.append({"role": "user", "content": note.lstrip("\n")})

        try:
            emit_loop_event(
                "loop_assistant_message",
                ctx.project_id,
                ctx.agent_id,
                {
                    "action": verdict.action,
                    "message": verdict.message,
                    "model": model_tag,
                    "source": source,
                },
            )
        except Exception:
            pass

    # ── on step_complete: may veto ───────────────────────────────────

    def review_step_complete(
        self,
        ctx: Any,
        messages: list[dict[str, Any]],
        step_complete_call: Any,
        reasoning: str | None,
        overwrite_result: Callable[[list[dict[str, Any]], str, str], None],
    ) -> StepCompleteReview:
        """Judge a step that is trying to close.

        Synchronous: there is no tool execution left to overlap with, the
        principal is one statement away from leaving the inner loop.

        A ``reject`` verdict actually blocks. The objection replaces the
        ``step_complete`` tool result — the same mechanism a late user
        message uses — so the model reads it as "your close was overridden"
        and gets one more turn. Following a tool result is the model's
        natural mode after a tool call, which is why this framing lands far
        more often than a bare user-role message would.
        """
        if not settings.ASSISTANT_LLM_INCLUDE_STEP_COMPLETE:
            return StepCompleteReview()
        critic = self.get(ctx)
        if critic is None:
            return StepCompleteReview()

        try:
            verdict = critic.review(messages, [step_complete_call], reasoning)
        except Exception as exc:
            logger.warning("assistant critic (step_complete) raised: %s", exc)
            return StepCompleteReview()
        if verdict is None or verdict.is_silent:
            return StepCompleteReview()

        model_tag = critic.model_short_name
        if verdict.action == "reject":
            overwrite_result(
                messages,
                step_complete_call.id,
                (
                    f"step_complete REJECTED by the assistant critic "
                    f"({model_tag}). Address the objection below before "
                    f"closing this step — either fix the issue and call "
                    f"step_complete again, or push back with a brief "
                    f"explanation if you disagree.\n\n"
                    f"Critic objection:\n{verdict.message}"
                ),
            )
            self._emit(ctx, verdict, model_tag, blocked=True)
            emit_log(
                "info",
                f"⚠ step_complete blocked by assistant critic "
                f"({model_tag}): {verdict.message[:120]}",
                project_id=ctx.project_id,
                agent_id=ctx.agent_id,
            )
            return StepCompleteReview(blocked=True)

        # Advisory: the current messages list dies with the break, so the
        # note is queued for the next step's preamble instead.
        followup = {
            "role": "user",
            "content": (
                f"[ASSISTANT ({model_tag}) - {verdict.action}] "
                f"(re: step_complete): {verdict.message}"
            ),
        }
        self.pending_messages.append(followup)
        self._emit(ctx, verdict, model_tag, blocked=False)
        return StepCompleteReview(followup=followup)

    @staticmethod
    def _emit(ctx: Any, verdict: CriticVerdict, model_tag: str, *, blocked: bool) -> None:
        try:
            payload: dict[str, Any] = {
                "action": verdict.action,
                "message": verdict.message,
                "model": model_tag,
                "source": "step_complete",
            }
            if blocked:
                payload["blocked"] = True
            emit_loop_event(
                "loop_assistant_message", ctx.project_id, ctx.agent_id, payload,
            )
        except Exception:
            pass
