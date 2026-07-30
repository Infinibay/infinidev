"""The conditions under which a step is allowed to close.

``step_complete`` is the model's own claim that a step is finished, and four
separate things can override it. In the engine they had grown into 165 lines
of nested branches inside the inner loop, each with its own idea of how to
say no, and the shape of the decision was invisible: you could not tell from
reading it how many gates there were or in what order they fired.

They are a chain. Each gate either lets the step close or sends the model
back for one more turn, and the order is deliberate — cheapest and most
mechanical first, LLM-backed last, so a step that fails the note-discipline
check never pays for a critic call:

1. **Notes** — a small model that never called ``add_note`` is about to
   throw away everything it learned, since raw tool output does not survive
   the step boundary. Fires at most once per step: a second attempt is
   always honoured, or the model would deadlock between "add a note" and
   "that note was not good enough".
2. **Late user message** — the user typed while the model was generating.
   They are owed an acknowledgement before a step boundary, not after it.
3. **Critic** — a ``reject`` verdict from the pair-programming critic.
4. **Objective** — the step's own deterministic verification. The model does
   not get to decide this one; a check does.

Every gate refuses the same way: by overwriting the ``step_complete`` tool
result. The model reads that as "your close was overridden by this
feedback", and following a tool result is its natural mode right after a
tool call — far more reliable than a bare user-role message.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.engine._best_effort import best_effort
from infinidev.engine.engine_logging import emit_log, emit_loop_event
from infinidev.engine.loop.step_manager import _get_settings

logger = logging.getLogger(__name__)

_NOTE_NUDGE = (
    "Hold on — before you complete this step, save the key facts you "
    "discovered with add_note (file paths, function names, line numbers, "
    "decisions). Anything not saved is discarded when this step ends. "
    "Example: add_note(note='auth.py:42 verify_token() uses JWT, no exp "
    "check'). Then call step_complete again — the second call will be "
    "honored."
)

# Below this many tool calls the model has not learned enough for the
# missing note to be a real loss, and nagging it reads as noise.
_MIN_CALLS_BEFORE_NOTE_GATE = 2


def step_complete_status(step_complete_call: Any) -> str:
    """Best-effort read of the ``status`` argument, defaulting to continue."""
    try:
        raw = step_complete_call.function.arguments
        args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        if isinstance(args, dict):
            return str(args.get("status", "continue"))
    except Exception:
        pass
    return "continue"


class StepCompleteGate:
    """Decides whether a ``step_complete`` call is honoured."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        # One note nudge per step, tracked here rather than on the engine so
        # the "fires once" rule lives next to the rule it bounds.
        self._note_gate_fired: bool = False
        # Per-step count of forced objective-verification correction turns,
        # so one stuck objective cannot starve the whole budget.
        self._verify_attempts: dict[int, int] = {}

    def reset_run(self) -> None:
        """Forget per-run state at the start of an execution."""
        self._verify_attempts = {}
        self._note_gate_fired = False

    # ── the chain ────────────────────────────────────────────────────

    def blocks(
        self,
        ctx: Any,
        step_complete_call: Any,
        messages: list[dict[str, Any]],
        *,
        action_tool_calls: int,
        reasoning: str | None,
    ) -> bool:
        """``True`` when the step must stay open for one more turn."""
        if self._notes_missing(ctx, step_complete_call, messages, action_tool_calls):
            return True
        self._note_gate_fired = False

        if self._engine._reject_step_complete_on_late_message(
            ctx, messages, step_complete_call.id,
        ):
            return True

        review = self._engine._critic.review_step_complete(
            ctx, messages, step_complete_call, reasoning,
            self._engine._overwrite_step_complete_tool_result,
        )
        if review.blocked:
            return True

        return self.objective_unmet(ctx, step_complete_call, messages)

    # ── gate 1: notes ────────────────────────────────────────────────

    def _notes_missing(
        self,
        ctx: Any,
        step_complete_call: Any,
        messages: list[dict[str, Any]],
        action_tool_calls: int,
    ) -> bool:
        """Small models lose everything they learned without ``add_note``."""
        if not getattr(_get_settings(), "LOOP_REQUIRE_NOTE_BEFORE_COMPLETE", True):
            return False
        if not ctx.is_small or ctx.state.notes:
            return False
        if action_tool_calls < _MIN_CALLS_BEFORE_NOTE_GATE or self._note_gate_fired:
            return False

        self._note_gate_fired = True
        if ctx.manual_tc:
            messages.append({"role": "user", "content": _NOTE_NUDGE})
            return True

        # Anthropic requires exactly one tool_result per tool_use id, so the
        # existing stub is replaced rather than a second one appended.
        for message in reversed(messages):
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") == step_complete_call.id
            ):
                message["content"] = _NOTE_NUDGE
                return True
        messages.append({
            "role": "tool",
            "tool_call_id": step_complete_call.id,
            "content": _NOTE_NUDGE,
        })
        return True

    # ── gate 4: the step's own verification ──────────────────────────

    def objective_unmet(
        self,
        ctx: Any,
        step_complete_call: Any,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Run the active step's deterministic check before letting it close.

        A no-op when the feature is off, when the step carries no executable
        check, or when the model said ``status='blocked'`` — that is a
        legitimate give-up, and forcing verification on it would only
        produce a failure the model already knows about.

        Only *deterministic* checks gate per step. An ``llm_judge`` check is
        deferred to the post-loop verifier so no step_complete ever costs an
        LLM call here.
        """
        settings_ = _get_settings()
        if not getattr(settings_, "LOOP_OBJECTIVE_VERIFY_ENABLED", True):
            return False

        active = ctx.state.plan.active_step if ctx.state.plan else None
        check = getattr(active, "verify", None) if active is not None else None
        if check is None or not check.is_deterministic:
            return False
        if step_complete_status(step_complete_call) == "blocked":
            return False

        result = self._run_check(ctx, check, active)
        if result is None:
            return False  # fail open — the gate must never crash the loop
        if result.passed:
            emit_log(
                "info",
                f"✓ step {active.index} objective verified "
                f"({check.kind}: {check.spec[:80]})",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            return False

        return self._block_for_correction(ctx, messages, step_complete_call,
                                          active, check, result, settings_)

    @staticmethod
    def _run_check(ctx: Any, check: Any, active: Any) -> Any | None:
        """Execute a verification against the project, not the process cwd."""
        from infinidev.engine.analysis.objective_verifier import ObjectiveVerifier

        workspace = getattr(ctx, "workspace_path", None)
        if not workspace:
            with best_effort("objective-gate workspace resolve failed"):
                from infinidev.tools.base.context import get_current_workspace_path

                workspace = get_current_workspace_path()
        try:
            return ObjectiveVerifier(workspace).verify(check)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "objective verification raised for step %s: %s",
                getattr(active, "index", "?"), exc,
            )
            return None

    def _block_for_correction(
        self, ctx: Any, messages: list[dict[str, Any]],
        step_complete_call: Any, active: Any, check: Any,
        result: Any, settings_: Any,
    ) -> bool:
        """Send the model back to fix the check, up to a bounded number of tries."""
        attempts = self._verify_attempts.get(active.index, 0) + 1
        self._verify_attempts[active.index] = attempts
        cap = getattr(settings_, "LOOP_OBJECTIVE_VERIFY_MAX_ATTEMPTS", 3)

        if attempts > cap:
            # Stop blocking to protect the global budget — but an objective
            # that quietly went unmet is worse than one that failed loudly,
            # so it is recorded in the notes the summary will read.
            note = (
                f"⚠ Step {active.index} objective NOT verified after {cap} "
                f"attempts — {check.kind}: {check.spec}"
            )
            emit_log("error", note, project_id=ctx.project_id, agent_id=ctx.agent_id)
            with best_effort("objective-unmet note append failed"):
                ctx.state.notes.append(note)
            return False

        self._engine._overwrite_step_complete_tool_result(
            messages,
            step_complete_call.id,
            (
                f"step_complete BLOCKED — the step's verification check did "
                f"not pass. You do not decide this verdict; an automated "
                f"check does.\n\n"
                f"Verification ({check.kind}): {check.spec}\n"
                f"{result.format_for_developer() or result.summary}\n\n"
                f"Fix the issue so this check passes, then call "
                f"step_complete again (attempt {attempts}/{cap})."
            ),
        )
        with best_effort("objective-gate loop event failed"):
            emit_loop_event(
                "loop_objective_verify", ctx.project_id, ctx.agent_id,
                {
                    "step_index": active.index,
                    "kind": check.kind,
                    "spec": check.spec,
                    "passed": False,
                    "attempt": attempts,
                    "blocked": True,
                },
            )
        emit_log(
            "info",
            f"⚠ step_complete blocked — step {active.index} verification "
            f"failed ({check.kind}: {check.spec[:80]}), "
            f"attempt {attempts}/{cap}",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        return True
