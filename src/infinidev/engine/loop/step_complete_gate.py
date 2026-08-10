"""The conditions under which a step is allowed to close.

``step_complete`` is the model's own claim that a step is finished, and seven
separate things can override it. In the engine they had grown into 165 lines
of nested branches inside the inner loop, each with its own idea of how to
say no, and the shape of the decision was invisible: you could not tell from
reading it how many gates there were or in what order they fired.

They are a chain. Each gate either lets the step close or sends the model
back for one more turn, and the order is deliberate — cheapest and most
mechanical first, LLM-backed last, so a step that fails the note-discipline
check never pays for a critic call:

0. **Recoverable tool error** — ``status="blocked"`` immediately after an
   unknown-tool result that names available alternatives. The model gets one
   correction turn instead of converting a naming miss into a blocked Task.
1. **Latest test outcome** — ``status="done"`` cannot override a recognised
   test command whose latest exit code is nonzero. A later green run clears
   this veto; ``blocked`` remains available for a genuine environment issue.
2. **Notes** — a small model that never called ``add_note`` is about to
   throw away everything it learned, since raw tool output does not survive
   the step boundary. Fires at most once per step: a second attempt is
   always honoured, or the model would deadlock between "add a note" and
   "that note was not good enough".
3. **Late user message** — the user typed while the model was generating.
   They are owed an acknowledgement before a step boundary, not after it.
4. **Critic** — a ``reject`` verdict from the pair-programming critic.
5. **Objective** — the step's own deterministic verification. The model does
   not get to decide this one; a check does.
6. **User hook** — a ``step_end_instruction`` command from the user's
   ``.infinidev/hooks.json``, holding the step for one more pass of work.

Every gate refuses the same way: by overwriting the ``step_complete`` tool
result. The model reads that as "your close was overridden by this
feedback", and following a tool result is its natural mode right after a
tool call — far more reliable than a bare user-role message.

The hook gate is deliberately **last**. It fires only once the engine's own
gates have agreed the step is finished, which buys two things: the user's
command is never asked to comment on a step that failed its own
verification, and it does not run again for each of the correction turns
that a failing check can cost. Like the note gate it fires at most once per
step — a hook that re-fired on the retry would hold the step forever, since
the second ``step_complete`` is indistinguishable from the first.
"""

from __future__ import annotations

import json
import logging
import re
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


def _step_complete_text(step_complete_call: Any) -> str:
    """Return the model's summary/final answer for recovery classification."""
    try:
        raw = step_complete_call.function.arguments
        args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        if isinstance(args, dict):
            return " ".join(
                str(args.get(key) or "") for key in ("summary", "final_answer")
            ).strip()
    except Exception:
        pass
    return ""


_RECOVERABLE_TOOL_BLOCK_RE = re.compile(
    r"\btool\b.{0,120}\b(?:not callable|not available|unavailable|unknown|"
    r"not exposed|not advertised|missing)\b|"
    r"\b(?:not callable|not available|unavailable|unknown|missing)\b.{0,120}\btool\b",
    re.IGNORECASE,
)

_RECOVERY_INTERNAL_BLOCK_RE = re.compile(
    r"\b(?:discovery suppression|recovery (?:mode|restriction|allowance))\b|"
    r"\b(?:need|needs|needed|require|requires|required)\b.{0,100}"
    r"\b(?:more local context|source (?:code|lines?|files?)|read_file|"
    r"read|inspect|inspection|discovery)\b|"
    r"\b(?:insufficient|missing|not enough)\b.{0,80}"
    r"\b(?:local context|source context|information|source lines?)\b|"
    r"\b(?:read_file|reads?|inspection)\b.{0,80}"
    r"\b(?:suppress(?:ed|ion)?|unavailable|hidden|not available)\b",
    re.IGNORECASE,
)

_RECOVERY_EXTERNAL_BLOCK_RE = re.compile(
    r"\b(?:permission denied|read[- ]only file system|authentication|"
    r"credentials?|api key|network unavailable|user action|outside (?:the )?"
    r"process|hardware unavailable|no space left)\b",
    re.IGNORECASE,
)


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
        # Step indices that have already received the recoverable-error
        # correction turn. The second blocked claim is honoured so a broken
        # suggestion cannot deadlock the Task.
        self._recoverable_error_fired: set[int] = set()
        self._recovery_escape_fired: set[int] = set()

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
        self._recoverable_error_fired = set()
        self._recovery_escape_fired = set()

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
        if self._recoverable_tool_error(ctx, step_complete_call, messages):
            return True

        if self._workspace_recovery_escape(ctx, step_complete_call, messages):
            return True

        if self._latest_test_failed(ctx, step_complete_call, messages):
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

    def _latest_test_failed(
        self,
        ctx: Any,
        step_complete_call: Any,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Refuse a successful close while the latest recognised test is red."""
        if step_complete_status(step_complete_call) != "done":
            return False
        state = getattr(ctx, "state", None)
        exit_code = getattr(state, "last_test_exit_code", None)
        command = str(getattr(state, "last_test_command", "") or "")
        if not command or exit_code in (None, 0):
            return False

        feedback = (
            "step_complete BLOCKED — the latest recognised test command "
            f"exited {exit_code}: {command[:220]}. Correct the command or the "
            "implementation and run a passing test before reporting "
            "status=\"done\". If the environment genuinely cannot be repaired, "
            "report status=\"blocked\" with the concrete requirement."
        )
        self._engine._overwrite_step_complete_tool_result(
            messages, step_complete_call.id, feedback,
        )
        emit_log(
            "warning",
            f"⚠ step_complete blocked — latest test exited {exit_code}",
            project_id=ctx.project_id,
            agent_id=ctx.agent_id,
        )
        return True

    # ── gate 0: recoverable tool errors ─────────────────────────────

    def _recoverable_tool_error(
        self,
        ctx: Any,
        step_complete_call: Any,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Refuse one premature ``blocked`` after a correctable tool-name miss."""
        if step_complete_status(step_complete_call) != "blocked":
            return False

        step_key = self._step_key(ctx)
        if step_key in self._recoverable_error_fired:
            return False

        last_result = ""
        for message in reversed(messages):
            if message.get("role") != "tool":
                continue
            if message.get("tool_call_id") == step_complete_call.id:
                continue
            last_result = str(message.get("content") or "")
            break

        unknown_with_suggestion = (
            "Unknown tool:" in last_result
            and "Did you mean one of:" in last_result
        )
        model_reported_tool_surface_miss = bool(
            _RECOVERABLE_TOOL_BLOCK_RE.search(_step_complete_text(step_complete_call))
        )
        if not (unknown_with_suggestion or model_reported_tool_surface_miss):
            return False

        self._recoverable_error_fired.add(step_key)
        feedback = (
            "step_complete BLOCKED — this is a recoverable tool-surface error, "
            "not a user-action blocker. Retry the intended operation once using "
            "a concrete tool schema advertised in this turn (for shell commands, "
            "use execute_command). "
            "Only report status=\"blocked\" if that corrected call also cannot "
            "proceed or requires user action."
        )
        self._engine._overwrite_step_complete_tool_result(
            messages, step_complete_call.id, feedback,
        )
        emit_log(
            "info",
            "⚠ step_complete blocked — unknown tool has a suggested correction",
            project_id=ctx.project_id,
            agent_id=ctx.agent_id,
        )
        with best_effort("recoverable-tool-error loop event failed"):
            emit_loop_event(
                "loop_recoverable_tool_error_gate",
                ctx.project_id,
                ctx.agent_id,
                {"step_index": step_key, "blocked": True},
            )
        return True

    def _workspace_recovery_escape(
        self,
        ctx: Any,
        step_complete_call: Any,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Refuse a blocked escape caused only by local missing context."""
        if step_complete_status(step_complete_call) != "blocked":
            return False
        if not (
            getattr(ctx, "suppress_discovery_this_step", False)
            and getattr(ctx, "recovery_requires_workspace_change", False)
        ):
            return False

        step_key = self._step_key(ctx)
        summary = _step_complete_text(step_complete_call)
        local_context_block = bool(_RECOVERY_INTERNAL_BLOCK_RE.search(summary))
        concrete_external_block = bool(_RECOVERY_EXTERNAL_BLOCK_RE.search(summary))
        repeated = step_key in self._recovery_escape_fired
        if repeated and (not local_context_block or concrete_external_block):
            return False

        state = getattr(ctx, "state", None)
        active = getattr(getattr(state, "plan", None), "active_step", None)
        tracker = getattr(ctx, "file_tracker", None)
        if tracker is not None and hasattr(tracker, "change_fingerprint"):
            current = tracker.change_fingerprint(reconcile=True)
            entry = getattr(state, "step_entry_change_fingerprints", {}).get(
                getattr(active, "index", step_key)
            )
            if entry is not None and current != entry:
                return False

        self._recovery_escape_fired.add(step_key)
        unlimited_reads = bool(getattr(ctx, "unlimited_recovery_reads", False))
        if unlimited_reads:
            ctx.semantic_recovery_context_calls = 0
        else:
            ctx.semantic_recovery_context_calls = max(
                2, int(getattr(ctx, "semantic_recovery_context_calls", 0) or 0)
            )
        if repeated and local_context_block:
            feedback = (
                "step_complete BLOCKED — needing more local source or context is "
                "still not an external blocker when repeated. Do not use blocked "
                "to request more discovery. Direct read_file remains available "
                "without a call-count allowance; use it on the most plausible "
                "target and make the smallest "
                "reversible edit, or close a completed discovery/verification Step "
                "with status=\"continue\" and transition to one concrete change "
                "Step. The Step has no call budget."
            )
        else:
            feedback = (
                "step_complete BLOCKED — the engine's discovery recovery is not an "
                "external blocker. The Step remains active and has no call budget. "
                "Direct read_file remains available on the already grounded source "
                "target without a call-count allowance. Read the exact missing lines, "
                "then make the smallest implementation change and run its focused "
                "test. Repeat status=\"blocked\" only with concrete evidence "
                "of a requirement outside this process."
            )
        self._engine._overwrite_step_complete_tool_result(
            messages, step_complete_call.id, feedback,
        )
        emit_log(
            "info",
            "⚠ step_complete blocked — recovery mode is not an external blocker",
            project_id=ctx.project_id,
            agent_id=ctx.agent_id,
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

    # ── gate 4: the step's own verification ─────────────────────────

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
            ctx.state.objectively_verified_step_indices.add(active.index)
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
