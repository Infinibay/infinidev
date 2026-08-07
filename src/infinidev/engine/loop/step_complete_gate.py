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

0. **Task completion** — ``status="done"`` while any planned Step remains
   open. A rolling horizon is a commitment until its Steps are completed,
   blocked, or removed with evidence. A list comprehension over the plan makes
   an attempt to end the run early cheap to refuse.
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
5. **User hook** — a ``step_end_instruction`` command from the user's
   ``.infinidev/hooks.json``, holding the step for one more pass of work.

Every gate refuses the same way: by overwriting the ``step_complete`` tool
result. The model reads that as "your close was overridden by this
feedback", and following a tool result is its natural mode right after a
tool call — far more reliable than a bare user-role message.

The hook gate is deliberately **last**. It fires only once the engine's own
four have agreed the step is finished, which buys two things: the user's
command is never asked to comment on a step that failed its own
verification, and it does not run again for each of the correction turns
that a failing check can cost. Like the note gate it fires at most once per
step — a hook that re-fired on the retry would hold the step forever, since
the second ``step_complete`` is indistinguishable from the first.
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

# How many times a run may be told that ending here would abandon plan steps.
# Bounded for the same reason the objective gate is: a model that has decided
# it is finished twice will not be talked out of it a third time, and the run
# is worth more finished-with-the-gap-recorded than spinning. Two, not three —
# the first refusal is the useful one, the second catches a misread.
_SCOPE_GATE_MAX_ATTEMPTS = 2


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
        # Step indices whose note nudge has already fired. Keyed the same
        # way as ``_hook_fired`` and for the same reason: a single flag was
        # cleared as soon as the gate *passed*, which is not when the step
        # ends, so any later gate that held the step re-armed it and the
        # model got "save a note" alternating with "fix the verification".
        self._note_fired: set[int] = set()
        # Per-step count of forced objective-verification correction turns,
        # so one stuck objective cannot starve the whole budget.
        self._verify_attempts: dict[int, int] = {}
        # Step indices whose end-of-step user hook has already fired. Keyed by
        # index rather than a flag because steps can be added mid-run, and a
        # single flag would let step 4 inherit step 3's "already fired".
        self._hook_fired: set[int] = set()
        # Run-level, unlike the other three: the scope gate is about the plan
        # as a whole, not about the step being closed. Keying it per step would
        # hand the model a fresh pair of attempts at every step it tries to end
        # the run from.
        self._scope_attempts = 0

    @staticmethod
    def _step_key(ctx: Any) -> int:
        """Index of the step a once-per-step guarantee is scoped to.

        -1 stands in for "no plan yet" — the bootstrap step, before the
        model has called add_step. It is a real step and still deserves
        the guarantee. Keyed by index rather than a flag because steps can
        be added mid-run, and a flag would let step 4 inherit step 3's
        "already fired".
        """
        state = getattr(ctx, "state", None)
        plan = getattr(state, "plan", None) if state is not None else None
        active = getattr(plan, "active_step", None) if plan is not None else None
        return getattr(active, "index", -1) if active is not None else -1

    def reset_run(self) -> None:
        """Forget per-run state at the start of an execution."""
        self._verify_attempts = {}
        self._note_fired = set()
        self._hook_fired = set()
        self._scope_attempts = 0

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
        if self._scope_open(ctx, step_complete_call, messages):
            return True

        if self._notes_missing(ctx, step_complete_call, messages, action_tool_calls):
            return True

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

        if self.objective_unmet(ctx, step_complete_call, messages):
            return True

        return self._user_hook_holds(ctx, step_complete_call, messages)

    # ── gate 0: the user's scope ─────────────────────────────────────

    def _scope_open(
        self,
        ctx: Any,
        step_complete_call: Any,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Refuse a ``done`` that would leave planned work unstarted.

        User-approved Steps are scope records; developer-authored Steps are
        the current execution commitment. Either must be explicitly completed,
        blocked, or removed before the Task may close. Otherwise a model can
        create a rolling horizon, perform its first item, and silently skip the
        rest by declaring the whole Task done.

        Ordered first in the chain because it is a comprehension over at most a
        handful of steps: an attempt to finish early is refused before it can
        cost a critic call or a verification subprocess.

        ``blocked`` is deliberately not caught. A step closed as blocked was
        attempted and reported, which is the honest outcome this gate exists to
        distinguish from a silent drop.
        """
        if step_complete_status(step_complete_call) != "done":
            return False

        plan = getattr(ctx.state, "plan", None)
        if plan is None or not plan.steps:
            return False

        undischarged = plan.undischarged(
            exclude_index=self._step_key(ctx),
            approved_only=False,
        )
        if not undischarged:
            return False

        self._scope_attempts += 1
        titles = [f"  {s.index}. {s.title}" for s in undischarged]

        if self._scope_attempts > _SCOPE_GATE_MAX_ATTEMPTS:
            # Honour the close rather than spin, but a plan step that quietly
            # went undone is exactly what the reviewer and the next turn need
            # to be told about. ``notes`` is the channel that survives the
            # prompt rebuild and reaches both.
            note = (
                f"⚠ Run ended with {len(undischarged)} plan step(s) not "
                "executed from the approved plan:\n"
                + "\n".join(titles)
            )
            emit_log("error", note, project_id=ctx.project_id, agent_id=ctx.agent_id)
            with best_effort("scope-gate note append failed"):
                ctx.state.notes.append(note)
            return False

        exits = (
            "Finish these steps, remove work you now know is unnecessary, or "
            "close each one you cannot do with status=\"blocked\" and say why. "
            "Rewording a step does not discharge it."
        )
        self._engine._overwrite_step_complete_tool_result(
            messages,
            step_complete_call.id,
            (
                f"step_complete BLOCKED — you set status=\"done\", but "
                f"{len(undischarged)} open step(s) remain in the plan:\n"
                + "\n".join(titles) + "\n\n" + exits + " Then set "
                f"status=\"done\" (attempt {self._scope_attempts}/"
                f"{_SCOPE_GATE_MAX_ATTEMPTS})."
            ),
        )
        with best_effort("scope-gate loop event failed"):
            emit_loop_event(
                "loop_scope_gate", ctx.project_id, ctx.agent_id,
                {
                    "undischarged": [s.index for s in undischarged],
                    "attempt": self._scope_attempts,
                    "blocked": True,
                },
            )
        emit_log(
            "info",
            f"⚠ step_complete blocked — {len(undischarged)} open step(s) "
            f"still pending, attempt {self._scope_attempts}/"
            f"{_SCOPE_GATE_MAX_ATTEMPTS}",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        return True

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
        step_key = self._step_key(ctx)
        if action_tool_calls < _MIN_CALLS_BEFORE_NOTE_GATE or step_key in self._note_fired:
            return False

        self._note_fired.add(step_key)
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

    # ── gate 5: the user's end-of-step hook ──────────────────────────

    def _user_hook_holds(
        self,
        ctx: Any,
        step_complete_call: Any,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Hold the step while a configured ``step_end_instruction`` runs.

        Returns ``False`` — let the step close — for all the ordinary
        reasons: no hook configured, the hook already fired for this step,
        the hook printed nothing, or it failed. Only text coming back turns
        into another turn of work.

        The output is injected but never summarised: whatever the model
        *does* in response is captured by the step summary, while the
        instruction itself dies with the step's messages. That asymmetry is
        the point — the instruction was scaffolding, the work is the record.
        """
        from infinidev.engine.user_hooks import (
            UserHookEvent, run_hooks, step_instruction, step_payload,
        )

        step_key = self._step_key(ctx)
        if step_key in self._hook_fired:
            return False

        output = None
        with best_effort("step_end_instruction hook failed"):
            output = run_hooks(
                UserHookEvent.STEP_END_INSTRUCTION,
                step_payload(ctx, status=step_complete_status(step_complete_call)),
                workspace_path=getattr(ctx, "workspace_path", None),
            )
        # Marked fired even when the hook produced nothing, so a hook that
        # prints only sometimes cannot fire twice inside one step.
        self._hook_fired.add(step_key)
        if not output:
            return False

        self._engine._overwrite_step_complete_tool_result(
            messages, step_complete_call.id, step_instruction(output.text),
        )
        emit_log(
            "info",
            f"⚠ step_complete held — end-of-step hook added work "
            f"(step {step_key})",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        return True
