"""LoopEngine — the plan-execute-summarize loop itself.

What remains here is the *shape* of a run: seed a plan, iterate, and inside
each iteration ask the model what to do until the step closes or a budget
runs out. Everything a step needs on the way is a collaborator:

    ContextBuilder      what a run needs, and what each iteration says
    LLMCaller           one model call, with retries and manual-mode parsing
    ToolProcessor       classifying calls into real tools and pseudo-tools
    ToolRunner          executing them and writing the result into messages
    CriticLiaison       the pair-programming critic's advice and its one veto
    StepCompleteGate    the four reasons a step may not be allowed to close
    LoopGuard           repetition, error circuits, text-only stalls
    StepManager         advancing the plan, summarising, finishing
    run_report          what the finished run tells the reviewer

The loop reads top to bottom because none of those live in it. That
separation is also what the loop *is*: unlike a ReAct agent it rebuilds its
prompt from scratch every iteration out of compact summaries, so the only
state carried forward is the plan and the summaries — never the transcript.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from infinidev.engine.base import AgentEngine
from infinidev.engine.hooks.hooks import (
    hook_manager as _hook_manager,
    HookContext as _HookContext,
    HookEvent as _HookEvent,
)
from infinidev.engine.loop import run_report
from infinidev.engine.loop.behavior_tracker import BehaviorTracker
from infinidev.engine.loop.context_builder import (
    build_execution_context,
    build_iteration_messages,
)
from infinidev.engine.loop.context_manager import ContextManager
from infinidev.engine.loop.critic_liaison import CriticLiaison
from infinidev.engine.loop.execution_context import ExecutionContext
from infinidev.engine.loop.guardrail_runner import apply_guardrail
from infinidev.engine.loop.guidance_handler import GuidanceHandler
from infinidev.engine.loop.llm_caller import (
    COMPLETION_PLAN_TOOL_NAMES,
    ClassifiedCalls,
    LLMCaller,
    LLMCallResult,
)
from infinidev.engine.loop.loop_guard import LoopGuard
from infinidev.engine.loop.loop_plan import (
    _MAX_CONSECUTIVE_DECOMPOSITIONS,
    _step_concepts,
    _step_phase,
    _step_targets,
)
from infinidev.engine.loop.models import ActionRecord, LoopState, StepResult
from infinidev.engine.loop.step_complete_gate import (
    StepCompleteGate,
    step_complete_status,
)
from infinidev.engine.loop.step_manager import StepManager
from infinidev.engine.loop.tool_processor import ToolProcessor
from infinidev.engine.loop.tool_runner import ToolRunner
from infinidev.engine.loop.user_message_injector import UserMessageInjector
from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.formats.tool_call_parser import (
    parse_step_complete_args as _parse_step_complete_args,
)
from infinidev.engine.engine_logging import (
    emit_loop_event as _emit_loop_event,
    emit_log as _emit_log,
    log_step_start as _log_step_start,
    log_step_done as _log_step_done,
    log_plan as _log_plan,
    YELLOW as _YELLOW,
    RED as _RED,
    RESET as _RESET,
)
from infinidev.tools.base.context import (
    bind_tools_to_agent,
    set_capability_requester,
    set_file_tracker,
    set_loop_state,
)
from infinidev.engine._best_effort import best_effort
from infinidev.engine.trace_log import (
    trace_run_start as _trace_run_start,
    trace_iteration_prompt as _trace_iter_prompt,
    trace_llm_response as _trace_llm_response,
    trace_plan as _trace_plan,
    trace_step_done as _trace_step_done,
)

logger = logging.getLogger(__name__)

# Explorations one step may request before it is made to get on with the
# work. The step no longer advances on ``explore``, so without this the
# model could ask for the same decomposition indefinitely.
_MAX_EXPLORE_PER_STEP = 2

# Once execution tools reach their Step fuse, the model sees only notes and
# ``step_complete``.  Some models correctly preserve a session note first and
# need one following turn to emit the terminal call.  Both turns are bounded
# and cannot execute workspace tools.
_MAX_COMPLETION_ONLY_TURNS = 2

# One productive Step may consume a second local window before context is
# compacted. Further progress is preserved in the next outer iteration rather
# than allowing a sequence of edits to grow one conversation without bound.
_MAX_STEP_PROGRESS_RENEWALS = 1


def _seed_state_from_plan(state, plan) -> None:
    """Populate ``state.plan`` from an analyst-emitted Plan.

    Each step preserves the authority declared at the handoff boundary.
    Planner-authored steps default to ``model_inferred`` and remain editable;
    only direct or confirmed user instructions receive ``user_approved``
    protections. The first step is set active so the developer has somewhere
    to start.

    Plans with an overview but no steps (analyst fallback path) seed
    only the overview so the developer still has context, and then
    drop into the LoopEngine bootstrap branch where the model is
    instructed to call ``add_step`` to build its own decomposition.
    """
    from infinidev.engine.authority import is_user_authorized
    from infinidev.engine.loop.plan_step import PlanStep

    state.plan.overview = plan.overview or ""
    state.plan.rolling_horizon_limit = max(
        0, int(getattr(plan, "rolling_horizon_limit", 0) or 0)
    )
    state.plan.steps = [
        PlanStep(
            index=idx + 1,
            title=spec.title,
            detail=spec.detail,
            expected_output=spec.expected_output,
            verify=getattr(spec, "verify", None),
            authority=(authority := getattr(spec, "authority", "model_inferred")),
            user_approved=is_user_authorized(authority),
            status="active" if idx == 0 else "pending",
        )
        for idx, spec in enumerate(plan.steps)
    ]


def _seed_initial_plan_if_fresh(ctx, plan) -> None:
    """Seed an external plan only for a new run, never over resumed state."""
    if plan is not None and not ctx.resumed:
        _seed_state_from_plan(ctx.state, plan)


def _seed_initial_edit_evidence(ctx: ExecutionContext, enabled: bool) -> None:
    """Carry an exact prior target edit across a scheduler Task boundary."""
    if not enabled:
        return
    ctx.state.task_has_edits = True
    active = ctx.state.plan.active_step
    if active is not None:
        ctx.state.edited_step_indices.add(active.index)


def _is_planning_mode(ctx: ExecutionContext) -> bool:
    """Whether the next model call must use the plan-management protocol."""
    return (
        getattr(ctx, "allow_plan_mutation", True)
        and not ctx.skip_plan
        and ctx.state.plan.active_step is None
    )


def _apply_exploration_policy(
    ctx: ExecutionContext, step_result: StepResult,
) -> bool:
    """Keep bounded engines on their own direct-tool execution path."""
    if step_result.status != "explore" or ctx.allow_explore:
        return False
    step_result.status = "continue"
    step_result.interrupted = True
    step_result.summary = (
        f"{step_result.summary}\n\n"
        "Tree delegation is disabled for this bounded run. Continue on this "
        "step with direct repository tools and the evidence already gathered."
    ).strip()
    return True


_EDIT_REQUIRING_TASK_KINDS = frozenset({
    "feature", "bugfix", "refactor", "performance", "docs", "test", "chore",
    "config", "migration", "security",
})


def _has_substantive_done_evidence(
    ctx: ExecutionContext, step_result: StepResult,
) -> bool:
    """Whether a first-step close has enough machine activity to be credible."""
    if not step_result.summary.strip() or step_result.action_tool_calls <= 0:
        return False
    task_kind = str(getattr(getattr(ctx, "task", None), "kind", "") or "").casefold()
    if task_kind in _EDIT_REQUIRING_TASK_KINDS and not ctx.state.task_has_edits:
        return False
    return True


def _task_requires_edits(ctx: ExecutionContext) -> bool:
    task_kind = str(getattr(getattr(ctx, "task", None), "kind", "") or "").casefold()
    return task_kind in _EDIT_REQUIRING_TASK_KINDS


def _resource_exhaustion_reason(ctx: ExecutionContext) -> str | None:
    """Return the first reached resource fuse, without inferring success."""
    if (
        ctx.max_total_calls is not None
        and ctx.state.total_tool_calls >= ctx.max_total_calls
    ):
        return (
            "Step interrupted: global tool call limit reached "
            f"({ctx.state.total_tool_calls}/{ctx.max_total_calls} total calls)."
        )
    if (
        ctx.max_prompt_tokens is not None
        and ctx.state.total_prompt_tokens >= ctx.max_prompt_tokens
    ):
        return (
            "Step interrupted: prompt token limit reached "
            f"({ctx.state.total_prompt_tokens}/{ctx.max_prompt_tokens} tokens)."
        )
    return None


def _step_progress_marker(
    ctx: ExecutionContext,
    tracker: BehaviorTracker,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    """Return deterministic evidence that material Step state advanced."""
    outcomes = tuple(
        sorted(
            (command, tuple(fingerprints))
            for command, fingerprints in ctx.state.test_outcome_history.items()
        )
    )
    file_tracker = getattr(ctx, "file_tracker", None)
    if file_tracker is not None and hasattr(file_tracker, "change_fingerprint"):
        changes = file_tracker.change_fingerprint()
    else:
        # Compatibility for narrow unit contexts and third-party engines that
        # have not adopted FileChangeTracker. Production uses net file state.
        changes = (
            (("@successful-edit-calls", str(tracker.successful_edit_count)),)
            if tracker.successful_edit_count
            else ()
        )
    return changes, outcomes


def _renew_step_tool_window(
    ctx: ExecutionContext,
    tracker: BehaviorTracker,
    previous: tuple[
        tuple[tuple[str, str], ...],
        tuple[tuple[str, tuple[str, ...]], ...],
    ],
    renewals_used: int,
) -> tuple[
    int | None,
    tuple[
        tuple[tuple[str, str], ...],
        tuple[tuple[str, tuple[str, ...]], ...],
    ],
    int,
]:
    """Renew one Step window after edits or genuinely new test evidence.

    Reads, searches, and an identical rerun leave the marker unchanged, so a
    stalled episode still reaches its normal boundary. The renewal changes no
    Task-level budget and makes no model call.
    """
    current = _step_progress_marker(ctx, tracker)
    if ctx.step_tool_limit is None:
        return None, current, renewals_used
    if (
        not getattr(ctx, "renew_step_budget_on_progress", False)
        or current == previous
        or renewals_used >= _MAX_STEP_PROGRESS_RENEWALS
    ):
        return ctx.step_tool_limit, current, renewals_used
    ctx.step_tool_limit += ctx.max_per_action
    return ctx.step_tool_limit, current, renewals_used + 1


def _configure_progress_recovery(
    ctx: ExecutionContext,
    messages: list[dict[str, Any]],
) -> None:
    """Latch an implementation Step into action-only mode until it advances."""
    from infinidev.engine.loop.semantic_stagnation import (
        SEMANTIC_RECOVERY_CONTEXT_CALLS,
    )

    active = ctx.state.plan.active_step
    step_index = active.index if active is not None else -1
    no_progress_windows = ctx.state.no_progress_windows_by_step.get(step_index, 0)
    transient = bool(getattr(ctx.state, "discovery_suppression_steps", 0))
    latched = no_progress_windows >= 2
    ctx.suppress_discovery_this_step = transient or latched
    if transient:
        ctx.state.discovery_suppression_steps = max(
            0, ctx.state.discovery_suppression_steps - 1
        )
    ctx.semantic_recovery_context_calls = (
        SEMANTIC_RECOVERY_CONTEXT_CALLS
        if ctx.suppress_discovery_this_step and no_progress_windows <= 2
        else 0
    )
    if not ctx.suppress_discovery_this_step:
        return

    if ctx.semantic_recovery_context_calls:
        read_policy = (
            f"You may use at most {ctx.semantic_recovery_context_calls} direct "
            "read_file calls for the exact edit target."
        )
    else:
        read_policy = "The recovery read allowance is exhausted."
    notice = (
        '<progress-recovery authority="SYSTEM">\n'
        "Repeated implementation windows produced no net workspace change or "
        "new test outcome. Broad discovery is now mechanically unavailable for "
        "this Step and remains unavailable until concrete progress occurs. "
        f"{read_policy} Choose the most plausible reversible edit now and let its "
        "focused test decide; recovery is not a request for more certainty. Low "
        "confidence or several local candidates is not an external blocker. Do "
        "not restart repository orientation.\n"
        "</progress-recovery>"
    )
    for message in reversed(messages):
        if message.get("role") == "user":
            message["content"] = f"{message.get('content', '')}\n\n{notice}"
            break


def _apply_context_pressure(
    ctx: ExecutionContext,
    messages: list[dict[str, Any]],
    *,
    announced: bool,
) -> bool:
    """Compact the live transcript while preserving the active Step."""
    used = int(getattr(ctx.state, "last_prompt_tokens", 0) or 0)
    limit = int(getattr(ctx, "max_context_tokens", 0) or 0)
    if not ContextManager.under_context_pressure(used, limit):
        return False

    ContextManager.compact_for_pressure(messages)
    if announced:
        return True

    remaining = max(0, limit - used)
    percent = min(100.0, (used / limit) * 100)
    _emit_loop_event(
        "loop_context_compaction",
        ctx.project_id,
        ctx.agent_id,
        {
            "prompt_tokens": used,
            "context_limit": limit,
            "remaining_tokens": remaining,
            "percent_used": percent,
        },
    )
    notice = (
        '<context-compaction authority="SYSTEM">\n'
        f"Automatic compaction activated at {percent:.1f}% context usage "
        f"with {remaining} tokens free. Older tool results already consumed "
        "by the model were shortened; the current Step remains active and has "
        "no tool-call budget. Continue from notes, opened files, and recent "
        "evidence. Use recall_context only if exact older output is needed.\n"
        "</context-compaction>"
    )
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str):
            message["content"] = f"{content}\n\n{notice}"
            break
    return True


def _should_advance_plan(step_result: StepResult) -> bool:
    """Whether this result closes the active Step rather than pausing it."""
    return (
        step_result.status in {"continue", "done", "blocked"}
        and not step_result.interrupted
    )


def _reconcile_step_result(
    ctx: ExecutionContext,
    step_result: StepResult,
    step_mgr: StepManager,
) -> tuple[StepResult, bool]:
    """Apply Task-scope reconciliation before whole-Task completion gates."""
    step_result = step_mgr.reconcile_task_completion(ctx, step_result)
    plan = getattr(ctx.state, "plan", None)
    active = getattr(plan, "active_step", None)
    pending_after_active = (
        plan.undischarged(exclude_index=active.index)
        if plan is not None and active is not None
        else []
    )
    if (
        step_result.status == "continue"
        and not step_result.interrupted
        and active is not None
        and not pending_after_active
    ):
        step_result = step_result.model_copy(update={
            "interrupted": True,
            "summary": (
                f"{step_result.summary.rstrip()} No follow-up Step was scheduled, "
                "so execution remains on this Step."
            ).strip(),
        })
    if (
        not getattr(ctx, "allow_plan_mutation", True)
        and step_result.status == "continue"
        and not step_result.interrupted
        and len(getattr(ctx.state.plan, "steps", [])) <= 1
    ):
        step_result = step_result.model_copy(update={"interrupted": True})
    return step_result, _enforce_edit_requirement(ctx, step_result)


def _enforce_edit_requirement(
    ctx: ExecutionContext, step_result: StepResult,
) -> bool:
    """Pause a write Task that claims completion before its first edit."""
    state = getattr(ctx, "state", None)
    plan = getattr(state, "plan", None)
    active = getattr(plan, "active_step", None)
    verified_noop = (
        active is not None
        and _step_phase(active.title) == "verify"
        and active.index
        in getattr(state, "objectively_verified_step_indices", set())
    )
    if (
        step_result.status != "done"
        or not _task_requires_edits(ctx)
        or ctx.state.task_has_edits
        or verified_noop
    ):
        return False
    step_result.status = "continue"
    step_result.final_answer = None
    step_result.interrupted = True
    step_result.summary = (
        f"{step_result.summary.rstrip()} "
        "Completion was deferred because this write Task has not made a "
        "successful edit yet."
    ).strip()
    return True


def _enforce_step_effect(ctx: ExecutionContext, step_result: StepResult) -> bool:
    """Keep a change Step active until it has concrete edit evidence.

    The model owns its plan, but a Step titled ``Implement`` or ``Fix`` is an
    observable commitment. Reads and more planning can refine that work; they
    cannot satisfy it. Evidence survives budget-boundary continuations through
    ActionRecord.changes_made, so a later test-only turn may still close a Step
    that edited successfully on its previous attempt.
    """
    if step_result.interrupted or step_result.status not in {"continue", "done"}:
        return False
    plan = getattr(ctx.state, "plan", None)
    active = getattr(plan, "active_step", None)
    active_phase = _step_phase(active.title) if active is not None else ""
    if active is None or active_phase not in {"change", "test_change"}:
        return False
    tracker = getattr(step_result, "behavior_tracker", None)
    file_tracker = getattr(ctx, "file_tracker", None)
    entry = getattr(ctx.state, "step_entry_change_fingerprints", {}).get(
        active.index
    )
    if (
        entry is not None
        and file_tracker is not None
        and hasattr(file_tracker, "change_fingerprint")
    ):
        has_net_effect = file_tracker.change_fingerprint(reconcile=True) != entry
    else:
        # Backwards-compatible fallback for resumed legacy state and focused
        # tests without a workspace tracker.
        edited_now = bool(getattr(tracker, "files_edited", ()))
        edited_before = any(
            record.step_index == active.index and bool(record.changes_made)
            for record in getattr(ctx.state, "history", ())
        ) or active.index in getattr(ctx.state, "edited_step_indices", set())
        has_net_effect = edited_now or edited_before
    if has_net_effect:
        return False

    # A broad earlier change can already satisfy a later model-authored Step.
    # Do not force a cosmetic second edit when the task has concrete edit
    # evidence and either the Step's declared verifier or a recognised test in
    # this Step proves the current workspace. User-approved steps remain
    # literal commitments, and a green baseline alone is insufficient because
    # ``task_has_edits`` must also be true.
    verified_by_declared_check = active.index in getattr(
        ctx.state, "objectively_verified_step_indices", set(),
    )
    verified_by_current_test = bool(
        getattr(tracker, "successful_test_commands", ())
    )
    verified_existing_effect = (
        not active.user_approved
        and ctx.state.task_has_edits
        and (verified_by_declared_check or verified_by_current_test)
    )
    if verified_existing_effect:
        evidence = (
            "its deterministic declared verification"
            if verified_by_declared_check
            else "a successful test command in this Step"
        )
        step_result.summary = (
            f"{step_result.summary.rstrip()} "
            f"{evidence.capitalize()} confirmed that an earlier task edit "
            "already supplied its requested effect."
        ).strip()
        return False

    # A model-authored change step sometimes turns itself into an umbrella:
    # it adds concrete change steps, then closes so those steps can run. A
    # strict sequential edit gate kept the umbrella active forever, making its
    # children unreachable. Supersede only model-owned containers, and only
    # when no user-approved frontier would be reordered. Task-level edit and
    # completion gates still require the concrete work to happen.
    pending = [step for step in plan.steps if step.status == "pending"]
    active_targets = _step_targets(active.title)
    active_concepts = _step_concepts(active.title)

    def refines_active(step: Any) -> bool:
        """Whether a pending Step is a concrete version of this same work."""
        if step.user_approved:
            return False
        phase = _step_phase(step.title)
        if phase and phase != active_phase:
            return False
        if active_targets & _step_targets(step.title):
            return True
        shared = active_concepts & _step_concepts(step.title)
        return len(shared) >= 2

    refinements = [step for step in pending if refines_active(step)]
    pending_changes = [
        step for step in pending
        if _step_phase(step.title) == active_phase or step in refinements
    ]

    # A user-authorized Step is the durable scope record and cannot be skipped
    # in favour of model-owned work. It can, however, absorb a more concrete
    # child for the same target. Keeping both made the prompt say that the
    # child's edit was future work and therefore forbidden in the parent,
    # while the edit gate refused to advance to that child: a deterministic
    # deadlock. Refine the active execution label and retire only the duplicate
    # model child; the authoritative Task description remains unchanged.
    if active.user_approved and refinements:
        child = refinements[0]
        original_title = active.title
        active.title = child.title
        if child.explanation:
            active.explanation = child.explanation
        if child.expected_output:
            active.expected_output = child.expected_output
        child.status = "skipped"
        step_result.status = "continue"
        step_result.interrupted = True
        step_result.final_answer = None
        step_result.summary = (
            f"{step_result.summary.rstrip()} "
            f"Concrete model Step {child.index} was folded into the active "
            f"user scope ({original_title!r}) so its edit can execute next."
        ).strip()
        return True

    if (
        not active.user_approved
        and pending_changes
        and not any(step.user_approved for step in pending)
        and plan.consecutive_decompositions < _MAX_CONSECUTIVE_DECOMPOSITIONS
    ):
        step_result.decomposed_phase = active_phase
        step_result.summary = (
            f"{step_result.summary.rstrip()} "
            "The model-authored change step was superseded by its concrete "
            "pending change steps; advancing through that execution frontier."
        ).strip()
        return False

    step_result.status = "continue"
    step_result.interrupted = True
    step_result.final_answer = None
    step_result.summary = (
        f"{step_result.summary.rstrip()} "
        "This implementation Step remains active because no successful "
        "workspace edit was observed yet."
    ).strip()
    return True


class LoopEngine(AgentEngine):
    """Plan-execute-summarize loop engine.

    Each iteration rebuilds the prompt from scratch with only:
    system prompt + task + plan + compact summaries + current step.
    Raw tool output exists only temporarily during a step, then is
    discarded and replaced with a ~50-token summary.
    """

    def __init__(self) -> None:
        self._last_file_tracker: FileChangeTracker | None = None
        self._last_state: LoopState | None = None
        self._last_status: str = ""  # terminal status of the last execute()
        self._last_total_tool_calls: int = 0
        self._nudge_threshold_override: int | None = None
        self._summarizer_override: bool | None = None
        self._supports_vision_cached: bool | None = None
        self._cancel_event = threading.Event()
        self._tool_cancel_event = threading.Event()
        self._tool_running_event = threading.Event()
        self._tool_state_lock = threading.Lock()
        # Optional OrchestrationHooks. When set by the caller (typically
        # the pipeline before execute()), the engine forwards file-change
        # and step-start callbacks so a UI can render live progress.
        # Plumbing via attribute keeps execute()'s signature stable.
        self._hooks: object | None = None
        self.session_notes: list[str] = []  # Persist across tasks within a session
        self._session_notes_hydrated: bool = False  # one-shot DB reload on resume
        # User-message injection: thread-safe queue + inject/drain logic.
        self._user_message_injector = UserMessageInjector(
            self._queue_context_rank_guidance
        )
        self._guidance = GuidanceHandler()
        # The pair-programming critic — advisory while tools run, able to
        # veto a step_complete — and the chain of gates a step must clear
        # before it is allowed to close.
        self._critic = CriticLiaison()
        self._step_gate = StepCompleteGate(self)
        self._tool_runner = ToolRunner(self)
        from infinidev.engine.context_rank.hooks import ContextRankHooks
        self._cr_hooks = ContextRankHooks()
        self._cr_cached_result: Any | None = None
        self._cr_last_pivot_key: tuple[int, str] | None = None
        self._cr_pending_user_guidance: list[str] = []
        self._cr_delivered_targets: set[str] = set()
        # Fetched on the first iteration of a run, rendered on all of them.
        self._project_knowledge: list[dict] = []
        self._explore_attempts: dict[int, int] = {}

    def inject_message(
        self, message: str, attachments: list[Any] | None = None,
    ) -> None:
        """Inject user guidance and optional images into the running loop."""
        self._user_message_injector.inject(message, attachments)

    def _drain_user_messages(self) -> list[str]:
        return self._user_message_injector.drain()

    def _inject_mid_step_user_messages(
        self, ctx: "ExecutionContext", messages: list[dict[str, Any]],
    ) -> None:
        self._user_message_injector.inject_mid_step(ctx, messages)

    def _queue_context_rank_guidance(self, messages: list[str]) -> None:
        """Refresh automatic retrieval after live user guidance changes the task."""
        self._cr_pending_user_guidance.extend(messages)
        self._cr_last_pivot_key = None

    def _reject_step_complete_on_late_message(
        self,
        ctx: "ExecutionContext",
        messages: list[dict[str, Any]],
        step_complete_id: str,
    ) -> bool:
        return self._user_message_injector.reject_step_complete_on_late_message(
            ctx, messages, step_complete_id,
        )

    @staticmethod
    def _overwrite_step_complete_tool_result(
        messages: list[dict[str, Any]],
        step_complete_id: str,
        new_body: str,
    ) -> None:
        UserMessageInjector._overwrite_step_complete_tool_result(
            messages, step_complete_id, new_body,
        )

    def _objective_gate_blocks(
        self,
        ctx: "ExecutionContext",
        step_complete_call: Any,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Run the active step's deterministic verification on step_complete.

        Returns True to BLOCK closure (the caller must ``continue`` the inner
        loop for a correction turn). The logic lives in ``StepCompleteGate``,
        which owns the whole chain of reasons a step may not close; this
        stays as the name the objective-verification tests reach for.
        """
        return self._step_gate.objective_unmet(ctx, step_complete_call, messages)

    @staticmethod
    def _step_complete_status(step_complete_call: Any) -> str:
        """Best-effort read of the status arg from a step_complete tool call."""
        return step_complete_status(step_complete_call)

    def cancel(self) -> None:
        """Signal the engine to stop after the current tool call.

        Stays set until ``begin_turn`` clears it, so every phase of the
        turn that follows — review, rework, the end-of-task hooks — can
        see that the user asked to stop.
        """
        self._cancel_event.set()
        # Wake a cooperative foreground tool immediately as well. The task
        # flag remains set after the tool returns, so the loop still stops.
        self._tool_cancel_event.set()

    def cancel_active_tool(self) -> bool:
        """Ask the current foreground tool batch to stop, without ending the task."""
        with self._tool_state_lock:
            if not self._tool_running_event.is_set():
                return False
            self._tool_cancel_event.set()
            return True

    @property
    def has_active_tool(self) -> bool:
        """Return whether the foreground loop is currently executing a tool batch."""
        return self._tool_running_event.is_set()

    def _begin_tool_batch(self) -> None:
        """Open a foreground-tool cancellation scope."""
        with self._tool_state_lock:
            self._tool_cancel_event.clear()
            self._tool_running_event.set()

    def _finish_tool_batch(self) -> None:
        """Close the current foreground-tool cancellation scope."""
        with self._tool_state_lock:
            self._tool_running_event.clear()
            self._tool_cancel_event.clear()

    def begin_turn(self) -> None:
        """Open a new user turn: forget any cancellation from the last one.

        The engine instance outlives a single ``execute()`` (the TUI keeps
        one per session, and the review's rework loop re-enters it), so a
        user turn is the only boundary at which "the user asked to stop"
        stops being true.
        """
        self._cancel_event.clear()
        self._tool_cancel_event.clear()
        self._tool_running_event.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    # ── What the run has to say for itself ─────────────────────────────
    # The reporting lives in ``run_report`` as free functions over the two
    # artefacts a run leaves behind (the tracker and the final state).
    # These stay as the engine's public surface because that is what the
    # pipeline and the review engine call.

    def get_changed_files_summary(self) -> str:
        """Diff-per-file digest of the last run, for the code reviewer."""
        return run_report.changed_files_summary(self._last_file_tracker)

    def has_file_changes(self) -> bool:
        """Whether the last execution modified any files."""
        return run_report.has_file_changes(self._last_file_tracker)

    def get_file_change_reasons(self) -> dict[str, list[str]]:
        """Return path → list of reasons for each changed file."""
        return run_report.file_change_reasons(self._last_file_tracker)

    def get_file_tracker(self) -> FileChangeTracker | None:
        """Expose the tracker from the last task for downstream checks."""
        return self._last_file_tracker

    def get_file_contents(self) -> dict[str, str]:
        """Return path → current content for each changed file."""
        return run_report.file_contents(self._last_file_tracker)

    def get_plan_steps(self) -> list[dict]:
        """Return the final plan steps for the post-loop reviewer."""
        return run_report.plan_steps(self._last_state)

    def get_objective_checks(self) -> list[tuple[int, str, Any]]:
        """``(step_index, title, StepVerification)`` per verifiable step."""
        return run_report.objective_checks(self._last_state)

    def build_work_summary(self, final_answer: str, status: str) -> str | None:
        """Distil the just-finished task into a hidden hand-off summary."""
        return run_report.work_summary(
            self._last_state, self._last_file_tracker,
            final_answer=final_answer, status=status,
        )

    def execute(
        self,
        agent: Any,
        task_prompt: tuple[str, str],
        *,
        verbose: bool = True,
        guardrail: Any | None = None,
        guardrail_max_retries: int = 5,
        output_pydantic: type | None = None,
        task_tools: list | None = None,
        event_id: int | None = None,
        resume_state: dict | None = None,
        max_iterations: int | None = None,
        max_total_tool_calls: int | None = None,
        max_prompt_tokens: int | None = None,
        max_tool_calls_per_action: int | None = None,
        nudge_threshold: int | None = None,
        nudge_message_template: str | None = None,
        summarizer_enabled: bool | None = None,
        identity_override: str | None = None,
        initial_plan: Any | None = None,
        initial_attachments: list[Any] | None = None,
        task: Any | None = None,
        context_corpus: str | None = None,
        allow_llm_retries: bool = True,
        allow_explore: bool = True,
        preserve_file_tracker: bool = False,
        preserve_task_state: bool = False,
        initial_edit_evidence: bool = False,
        skip_plan: bool = False,
        allow_plan_mutation: bool = True,
    ) -> str:
        """Plan-execute-summarize loop.

        Delegates to composition components: LLMCaller, ToolProcessor,
        LoopGuard, StepManager. See class docstrings for details.

        When ``initial_plan`` (an ``infinidev.engine.analysis.plan.Plan``
        instance) is provided, the loop starts with a planner-authored plan:
        plan.overview seeds
        ``state.plan.overview`` (rendered every iteration as
        ``<plan-overview>``), and each Step preserves its explicit authority.
        Model-inferred Steps remain adaptable as evidence changes. The bootstrap
        branch that asks "No plan yet — call add_step" is naturally
        suppressed because state.plan.steps is non-empty.
        """
        if preserve_task_state and resume_state is None and self._last_state is not None:
            resume_state = self._last_state.model_dump(mode="json")

        ctx = self._build_context(
            agent, task_prompt,
            verbose=verbose, guardrail=guardrail,
            guardrail_max_retries=guardrail_max_retries,
            output_pydantic=output_pydantic, task_tools=task_tools,
            event_id=event_id, resume_state=resume_state,
            max_iterations=max_iterations,
            max_total_tool_calls=max_total_tool_calls,
            max_prompt_tokens=max_prompt_tokens,
            max_tool_calls_per_action=max_tool_calls_per_action,
            nudge_threshold=nudge_threshold,
            nudge_message_template=nudge_message_template,
            summarizer_enabled=summarizer_enabled,
            identity_override=identity_override,
            initial_plan=initial_plan,
            task=task,
            context_corpus=context_corpus,
            allow_llm_retries=allow_llm_retries,
            allow_explore=allow_explore,
            preserve_file_tracker=preserve_file_tracker,
            skip_plan=skip_plan,
            allow_plan_mutation=allow_plan_mutation,
        )
        self._cr_delivered_targets.clear()
        # On a resumed session the engine is brand-new (lazily created in
        # the TUI) but the session_id is the SAME as yesterday's. Re-load
        # the persisted session notes once so the developer loop starts
        # with the memory it built up before exit. Reusing the session_id
        # is what makes this self-wiring — no plumbing from the UI needed.
        if not self._session_notes_hydrated and not self.session_notes:
            self._session_notes_hydrated = True
            try:
                from infinidev.tools.base.context import get_current_session_id
                from infinidev.db.service import get_session_notes
                _sid = get_current_session_id()
                if _sid:
                    self.session_notes = get_session_notes(_sid)[-10:]
            except Exception:
                pass

        _seed_initial_plan_if_fresh(ctx, initial_plan)
        _seed_initial_edit_evidence(ctx, initial_edit_evidence)
        # Stash attachments on the engine instance for the first
        # iteration only — subsequent turns rebuild the prompt from
        # compact summaries and don't need the raw payload.
        self._initial_attachments = list(initial_attachments or [])
        # Reset cached vision-capability probe: the configured model can
        # change between execute() calls.
        self._supports_vision_cached = None
        llm_caller, tool_proc, guard, step_mgr = self._init_execution(ctx, task_prompt)
        consecutive_all_done = 0
        self._step_gate.reset_run()
        self._critic.reset_run()
        self._explore_attempts: dict[int, int] = {}

        # Ken's session lifecycle is NOT opened here. A developer run is one
        # task inside a conversation, and /sessions/end snapshots scores and
        # closes the cr_sessions row — doing that per run turned a single
        # conversation into a row per task with the per-turn decay counter
        # restarting each time. The pipeline owns start/turn-end (per user
        # turn) and the host owns end (per process); what this loop still
        # feeds Ken is the reactive channel, one event per tool call, from
        # ``ToolRunner._report_to_ken``.
        iteration = ctx.start_iteration
        while ctx.max_iterations is None or iteration < ctx.max_iterations:
            if self._cancel_event.is_set():
                return self._finish_cancelled(ctx, step_mgr, iteration - 1)

            messages, step_messages_start = self._run_iteration_preamble(ctx, iteration)

            step_result = self._run_inner_loop(
                ctx, messages, iteration, llm_caller, tool_proc, guard,
                step_messages_start=step_messages_start,
            )

            step_result, edit_requirement_blocked = _reconcile_step_result(
                ctx, step_result, step_mgr,
            )
            if edit_requirement_blocked:
                _emit_log(
                    "warning",
                    f"{_YELLOW}⚠ Edit-requiring Task declared done before any "
                    f"successful edit — resuming the same Step{_RESET}",
                    project_id=ctx.project_id,
                    agent_id=ctx.agent_id,
                )
            if _apply_exploration_policy(ctx, step_result):
                _emit_log(
                    "warning",
                    f"{_YELLOW}⚠ Tree delegation disabled for this run; "
                    f"continuing with direct repository tools{_RESET}",
                    project_id=ctx.project_id,
                    agent_id=ctx.agent_id,
                )
            if _enforce_step_effect(ctx, step_result):
                _emit_log(
                    "warning",
                    f"{_YELLOW}⚠ Implementation Step closed without an edit — "
                    f"resuming the same Step{_RESET}",
                    project_id=ctx.project_id,
                    agent_id=ctx.agent_id,
                )

            # A step interrupted mid-flight did not finish, so the plan must
            # not advance over it and the summariser must not be paid for
            # (a 30s LLM call, started after the user asked to stop). The
            # step stays ``active``, which is what the reviewer and the next
            # turn's work summary should see.
            if self._cancel_event.is_set():
                return self._finish_cancelled(ctx, step_mgr, iteration)

            # Track consecutive text-only iterations. The question is
            # whether the model produced function calls at all, which is
            # not what ``action_tool_calls`` measures — a step closed with
            # think + step_complete has zero of those and is not a stall.
            if not step_result.saw_tool_calls:
                guard.mark_text_only_iteration()
                if guard.text_only_iterations >= 3:
                    _emit_log("error",
                              f"{_RED}\u26a0 Model failed to produce tool calls for "
                              f"{guard.text_only_iterations} consecutive iterations "
                              f"\u2014 aborting task{_RESET}",
                              project_id=ctx.project_id, agent_id=ctx.agent_id)
                    return step_mgr.finish(ctx, "blocked", iteration,
                                           "Model unable to produce function calls after multiple attempts.")
            else:
                guard.mark_productive_iteration()

            self._run_post_step(ctx, step_result, step_mgr, messages, step_messages_start, iteration)

            resource_exhaustion = _resource_exhaustion_reason(ctx)
            if (
                resource_exhaustion is not None
                and step_result.status not in {"done", "blocked"}
            ):
                return step_mgr.finish(
                    ctx, "exhausted", iteration, resource_exhaustion,
                )

            term = self._check_termination(
                ctx, step_result, step_mgr, iteration, consecutive_all_done
            )
            if term is not None:
                return term

            # A terminal claim may be demoted by a completion gate. Honour
            # the already-observed resource fuse before starting another
            # iteration in that case.
            if resource_exhaustion is not None:
                return step_mgr.finish(
                    ctx, "exhausted", iteration, resource_exhaustion,
                )

            # Update consecutive all-done counter
            if step_result.status == "explore":
                consecutive_all_done = 0
            elif ctx.state.plan.steps and not ctx.state.plan.has_pending:
                consecutive_all_done += 1
            else:
                consecutive_all_done = 0
            iteration += 1

        # Only explicitly bounded auxiliary loops can reach this point.
        assert ctx.max_iterations is not None
        return step_mgr.finish(
            ctx, "exhausted", max(ctx.start_iteration - 1, ctx.max_iterations - 1)
        )

    # ── Extracted phases of execute() ──────────────────────────────────

    def _finish_cancelled(
        self, ctx: ExecutionContext, step_mgr: StepManager, iteration: int,
    ) -> str:
        """Close a run the user stopped, as its own terminal status.

        ``cancelled`` is a status and not just a flag on purpose: it is what
        ``finish`` writes to ``_last_status``, and from there the pipeline,
        the reviewer, the ``task_end_*`` hooks and the next turn's work
        summary all read it. Falling through to the exhausted-iterations
        ending told every one of them the loop had run out of budget.
        """
        logger.info("LoopEngine: cancelled by user")
        _emit_log("info", f"{_YELLOW}⚠ Task cancelled by user{_RESET}",
                  project_id=ctx.project_id, agent_id=ctx.agent_id)
        return step_mgr.finish(ctx, "cancelled", iteration)

    def _init_execution(
        self, ctx: ExecutionContext, task_prompt: tuple[str, str],
    ) -> tuple[LLMCaller, ToolProcessor, LoopGuard, StepManager]:
        """Set up components and hooks for a new execution run."""
        def _on_thinking(text: str) -> None:
            _emit_loop_event("loop_thinking_chunk", ctx.project_id, ctx.agent_id, {"text": text})

        def _on_stream_status(phase: str, token_count: int, tool_name: str | None) -> None:
            _emit_loop_event("loop_stream_status", ctx.project_id, ctx.agent_id, {
                "phase": phase, "token_count": token_count, "tool_name": tool_name,
            })

        llm_caller = LLMCaller(on_thinking_chunk=_on_thinking, on_stream_status=_on_stream_status)
        tool_proc = ToolProcessor()
        guard = LoopGuard(is_small=ctx.is_small)
        step_mgr = StepManager(self)

        # The cancel flag is NOT cleared here. A user turn can enter
        # execute() more than once — the review's rework loop re-enters the
        # same engine up to three times — and clearing it per call meant a
        # cancelled run was resurrected by the very next phase, which then
        # kept writing to the user's repository. Clearing belongs to the
        # start of a turn; see ``begin_turn``.
        self._last_state = ctx.state

        set_loop_state(ctx.agent_id, ctx.state)
        set_file_tracker(ctx.agent_id, ctx.file_tracker)

        def _request_capability(capability: str, rationale: str) -> str:
            if not settings.DYNAMIC_TOOL_ROUTING_ENABLED:
                return json.dumps({"error": "Dynamic tool routing is disabled"})
            from infinidev.engine.schema_sanitizer import build_tool_schemas
            from infinidev.engine.tool_dispatch import build_tool_dispatch
            from infinidev.engine.tool_routing import expand_capability_tools

            before = {tool.name for tool in ctx.tools}
            expanded = expand_capability_tools(
                ctx.tools,
                list(getattr(ctx.agent, "tools", []) or []),
                capability,
            )
            if not ctx.allow_plan_mutation:
                from infinidev.engine.loop.context_builder import (
                    _filter_plan_free_tools,
                )

                expanded = _filter_plan_free_tools(expanded)
            added = [tool for tool in expanded if tool.name not in before]
            if not added:
                return json.dumps({
                    "capability": capability,
                    "added_tools": [],
                    "message": "Capability already available or unsupported by this model/config.",
                })
            bind_tools_to_agent(added, ctx.agent_id)
            ctx.tools[:] = expanded
            ctx.tool_dispatch.clear()
            ctx.tool_dispatch.update(build_tool_dispatch(ctx.tools))
            ctx.tool_schemas[:] = build_tool_schemas(
                ctx.tools,
                small_model=ctx.compact_tool_schemas,
            )
            if ctx.manual_tc:
                from infinidev.engine.loop.context import build_tools_prompt_section

                ctx.system_prompt += (
                    "\n\n## Newly granted capability\n"
                    + build_tools_prompt_section(
                        build_tool_schemas(
                            added,
                            small_model=ctx.compact_tool_schemas,
                        ),
                        small_model=ctx.is_small,
                    )
                )
            logger.info(
                "Capability %s added tools %s; rationale=%s",
                capability, [tool.name for tool in added], rationale[:200],
            )
            return json.dumps({
                "capability": capability,
                "added_tools": [tool.name for tool in added],
                "permission_granted": False,
                "message": (
                    "Tools are available on the next model turn. Their effect permissions "
                    "and user-authority requirements still apply."
                ),
            })

        set_capability_requester(ctx.agent_id, _request_capability)
        _hook_manager.dispatch(_HookContext(
            event=_HookEvent.LOOP_START,
            metadata={"task_prompt": task_prompt, "tools": ctx.tools, "state": ctx.state},
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        ))

        with best_effort("ContextRank start failed"):
            from infinidev.tools.base.context import get_current_session_id, get_current_agent_run_id
            _cr_session = get_current_session_id() or ctx.agent_id
            _cr_task = get_current_agent_run_id() or ctx.agent_id
            self._cr_hooks.start(_cr_session, _cr_task, ctx.desc)
            self._cr_cached_result = None
            self._cr_last_pivot_key = None

        with best_effort("static_analysis_timer reset failed"):
            from infinidev.engine.static_analysis_timer import reset as _sa_reset
            _sa_reset()

        with best_effort("_trace_run_start failed"):
            _trace_run_start(
                model=str(ctx.llm_params.get("model", "?")),
                task=ctx.desc, expected=ctx.expected,
                settings_snapshot={
                    "is_small": ctx.is_small, "manual_tc": ctx.manual_tc,
                    "model_policy": ctx.model_policy_name,
                    "compact_tool_schemas": ctx.compact_tool_schemas,
                    "max_iterations": ctx.max_iterations, "max_per_action": ctx.max_per_action,
                    "max_total_calls": ctx.max_total_calls, "history_window": ctx.history_window,
                    "max_context_tokens": ctx.max_context_tokens,
                },
            )

        return llm_caller, tool_proc, guard, step_mgr

    def _run_iteration_preamble(
        self, ctx: ExecutionContext, iteration: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Build messages, log step start, dispatch PRE_STEP hook."""
        ctx.state.iteration_count = iteration + 1
        messages = self._build_iteration_messages(ctx, iteration)
        # Drain any critic followups carried over from the previous
        # step (e.g. step_complete review). These were captured AFTER
        # the previous step finished and need a fresh messages list
        # to attach to.
        messages.extend(self._critic.drain_pending())
        with best_effort("step_start hook failed"):
            self._apply_step_start_hook(ctx, messages)
        with best_effort("_trace_iter_prompt failed"):
            _trace_iter_prompt(iteration + 1, messages[0].get("content", ""), messages[1].get("content", ""))

        active = ctx.state.plan.active_step
        if active:
            active_desc = active.title
        elif not ctx.state.plan.steps:
            active_desc = "Planning..."
        else:
            done_steps = [s for s in ctx.state.plan.steps if s.status == "done"]
            active_desc = f"Continuing ({done_steps[-1].title})" if done_steps else "Working..."
        if ctx.verbose:
            _log_step_start(iteration + 1, active_desc)

        _hook_manager.dispatch(_HookContext(
            event=_HookEvent.PRE_STEP,
            metadata={"iteration": iteration, "state": ctx.state, "plan": ctx.state.plan, "agent_name": ctx.agent_name},
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        ))

        with best_effort("ContextRank pre-step activation failed"):
            _cr_pre = ctx.state.plan.active_step
            if _cr_pre:
                self._cr_hooks.on_step_activated(
                    _cr_pre.title, _cr_pre.explanation or "", iteration, _cr_pre.index,
                )

        return messages, len(messages)

    def _run_post_step(
        self, ctx: ExecutionContext, step_result: StepResult,
        step_mgr: StepManager, messages: list[dict[str, Any]],
        step_messages_start: int, iteration: int,
    ) -> None:
        """Advance plan, summarize, log, dispatch hooks after a step completes."""
        _hook_manager.dispatch(_HookContext(
            event=_HookEvent.STEP_TRANSITION,
            metadata={"step_result": step_result, "plan": ctx.state.plan, "iteration": iteration},
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        ))
        # ``explore`` asks for the step to be broken down, not finished.
        # Advancing here marked it done before TreeEngine had even run, and
        # nothing reactivates a step — ``apply_operations`` will not even
        # re-add it when it is user_approved. The step stays active and the
        # exploration's findings land on it.
        if _should_advance_plan(step_result):
            step_mgr.advance_plan(ctx, step_result)

        with best_effort("ContextRank step activation failed"):
            _cr_active = ctx.state.plan.active_step
            if _cr_active:
                self._cr_hooks.on_step_activated(
                    _cr_active.title, _cr_active.explanation or "", iteration, _cr_active.index,
                )

        action_tool_calls = step_result.action_tool_calls
        step_mgr.summarize_and_record(ctx, step_result, messages, action_tool_calls, iteration)

        if ctx.verbose:
            _log_step_done(iteration + 1, step_result.status, step_result.summary, action_tool_calls, ctx.state.total_tokens)
            _log_plan(ctx.state.plan)
        try:
            _trace_step_done(iteration + 1, step_result.status, step_result.summary, action_tool_calls)
            _trace_plan(iteration + 1, ctx.state.plan)
        except Exception:
            pass

        self._guidance.try_queue(ctx, messages, step_messages_start, mid_step=False)

        _hook_manager.dispatch(_HookContext(
            event=_HookEvent.POST_STEP,
            metadata={
                "iteration": iteration, "step_result": step_result,
                "record": ctx.state.history[-1] if ctx.state.history else None,
                "state": ctx.state, "agent_name": ctx.agent_name,
                "action_tool_calls": action_tool_calls,
                "messages": messages, "step_messages_start": step_messages_start,
            },
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        ))
        if ctx.event_id:
            self._checkpoint(ctx.event_id, ctx.state)

    def _check_termination(
        self, ctx: ExecutionContext, step_result: StepResult,
        step_mgr: StepManager, iteration: int,
        consecutive_all_done: int,
    ) -> str | None:
        """Check if the task should terminate. Returns result string or None."""
        if step_result.status == "explore":
            # Now that the plan no longer advances past an explored step,
            # nothing else stops the model from asking to explore the same
            # step forever. The second attempt runs the exploration and
            # then demotes the step to a normal one, so its findings are
            # kept and the step has to be executed.
            key = ctx.state.plan.active_step.index if ctx.state.plan.active_step else -1
            self._explore_attempts[key] = self._explore_attempts.get(key, 0) + 1
            self._handle_explore(ctx, step_result, iteration)
            if self._explore_attempts[key] >= _MAX_EXPLORE_PER_STEP:
                _emit_log("warning",
                          f"{_YELLOW}⚠ step {key} asked to explore "
                          f"{self._explore_attempts[key]}x — continuing with "
                          f"what the exploration found{_RESET}",
                          project_id=ctx.project_id, agent_id=ctx.agent_id)
                step_result.status = "continue"
            return None

        if step_result.status == "done":
            if (
                not step_result.final_answer
                and iteration == ctx.start_iteration
                and not _has_substantive_done_evidence(ctx, step_result)
            ):
                _emit_log("warning",
                          f"{_YELLOW}\u26a0 LLM declared done on first step without "
                          f"evidence or final_answer \u2014 forcing continue{_RESET}",
                          project_id=ctx.project_id, agent_id=ctx.agent_id)
                step_result.status = "continue"
                return None
            result = step_result.final_answer or step_result.summary
            result = step_mgr.finish(ctx, "done", iteration, result)
            return self._apply_guardrail(
                ctx, result, ctx.guardrail, ctx.guardrail_max_retries,
                ctx.llm_params, ctx.system_prompt, ctx.desc, ctx.expected,
                ctx.state, ctx.tool_schemas, ctx.tool_dispatch,
                max_per_action=ctx.max_per_action,
            )

        if step_result.status == "blocked":
            if ctx.state.plan.has_pending:
                active = ctx.state.plan.active_step
                _emit_log(
                    "info",
                    "↪ Blocked Step recorded; continuing with the recovery "
                    f"Step already in the plan: {getattr(active, 'title', 'next Step')}",
                    project_id=ctx.project_id,
                    agent_id=ctx.agent_id,
                )
                return None
            return step_mgr.finish(ctx, "blocked", iteration, step_result.summary)

        if consecutive_all_done >= 2 and ctx.state.plan.steps and not ctx.state.plan.has_pending:
            result = step_mgr.finish(ctx, "done", iteration, step_result.summary)
            return self._apply_guardrail(
                ctx, result, ctx.guardrail, ctx.guardrail_max_retries,
                ctx.llm_params, ctx.system_prompt, ctx.desc, ctx.expected,
                ctx.state, ctx.tool_schemas, ctx.tool_dispatch,
                max_per_action=ctx.max_per_action,
            )

        return None

    # ── Private helpers for execute() ───────────────────────────────────

    def _build_context(
        self, agent: Any, task_prompt: tuple[str, str], **kwargs: Any,
    ) -> ExecutionContext:
        """Resolve everything this run needs and freeze it into one context."""
        return build_execution_context(self, agent, task_prompt, **kwargs)

    def _build_iteration_messages(
        self, ctx: ExecutionContext, iteration: int,
    ) -> list[dict[str, Any]]:
        """Build the fresh two-message conversation for one iteration."""
        return build_iteration_messages(self, ctx, iteration)

    @staticmethod
    def _apply_step_start_hook(
        ctx: ExecutionContext, messages: list[dict[str, Any]],
    ) -> None:
        """Fold a ``step_start`` hook's output into this step's prompt.

        Appended to the existing user message rather than added as a new
        one. The iteration prompt is exactly ``[system, user]``, and a
        second consecutive user message is rejected outright by Anthropic's
        API, which requires strict role alternation.

        Nothing here survives the step: the next iteration rebuilds the
        prompt from summaries, so a start-of-step hook re-runs and speaks
        again rather than accumulating.
        """
        from infinidev.engine.user_hooks import (
            UserHookEvent, context_block, run_hooks, step_payload,
        )

        output = run_hooks(
            UserHookEvent.STEP_START,
            step_payload(ctx),
            workspace_path=getattr(ctx, "workspace_path", None),
        )
        if not output:
            return

        block = context_block(UserHookEvent.STEP_START, output.text)
        for message in reversed(messages):
            if message.get("role") == "user":
                message["content"] = f"{message.get('content', '')}\n\n{block}"
                return
        messages.append({"role": "user", "content": block})

    def _run_inner_loop(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        iteration: int,
        llm_caller: LLMCaller, tool_proc: ToolProcessor, guard: LoopGuard,
        *,
        step_messages_start: int = 0,
    ) -> StepResult:
        """Run the inner tool-calling loop for one step.

        Returns the StepResult for this step.

        *step_messages_start* is the index into *messages* where this
        step's contribution begins. Used by the mid-step guidance
        check so detectors see only the current step's history, not
        the cumulative conversation.
        """
        step_result: StepResult | None = None
        action_tool_calls = 0
        saw_tool_calls = False
        completion_turns_used = 0

        llm_caller.reset()
        guard.reset()
        tracker = BehaviorTracker(set(ctx.state.opened_files.keys()))
        tracker.task_has_edits = ctx.state.task_has_edits
        file_tracker = getattr(ctx, "file_tracker", None)
        if file_tracker is not None and hasattr(file_tracker, "change_fingerprint"):
            guard.seed_workspace_fingerprint(file_tracker.change_fingerprint())
        active_step = ctx.state.plan.active_step
        if active_step is not None:
            entry_fingerprints = ctx.state.step_entry_change_fingerprints
            if file_tracker is not None and hasattr(
                file_tracker, "change_fingerprint"
            ):
                entry_fingerprints.setdefault(
                    active_step.index,
                    file_tracker.change_fingerprint(reconcile=True),
                )
        # Delivery notices are valid only while their source text remains in
        # the current Step transcript. After summarization, large-file slices
        # may no longer be model-visible, so one later read must be deliverable.
        ctx.state.read_delivery_revisions.clear()
        _configure_progress_recovery(ctx, messages)
        ctx.step_tool_limit = ctx.max_per_action or None
        progress_marker = _step_progress_marker(ctx, tracker)
        progress_renewals = 0

        # Tracks the wall-clock time of the previous LLM call's return
        # so we can measure the python-only gap until the next call.
        # None means "no previous call yet in this step".
        import time as _time
        from infinidev.engine.static_analysis_timer import add_elapsed as _sa_add
        _last_llm_call_end: float | None = None
        context_pressure_announced = False

        while (
            _resource_exhaustion_reason(ctx) is None
            and (
                ctx.step_tool_limit is None
                or action_tool_calls < ctx.step_tool_limit
                or (
                    saw_tool_calls
                    and completion_turns_used < _MAX_COMPLETION_ONLY_TURNS
                )
            )
        ):
            completion_only = (
                ctx.step_tool_limit is not None
                and action_tool_calls >= ctx.step_tool_limit
            )
            context_pressure_announced = _apply_context_pressure(
                ctx, messages, announced=context_pressure_announced
            )

            # Plan pseudo-tools mutate the state in this same inner loop.
            # Recompute the mode for every model turn so an ``add_step`` is
            # followed by execution tools, rather than leaving the model in
            # planning mode until the step budget happens to expire.
            # Plan-free adapters are always in execution mode.
            is_planning = _is_planning_mode(ctx)
            # If a previous LLM call ran in this step, record how much
            # wall-clock elapsed between its return and now (the moment
            # right before we dispatch the next one). This is the
            # "between LLM calls" cost the user sees on the GPU monitor
            # as idle GPU time.
            if _last_llm_call_end is not None:
                _sa_add("between_llm_calls", _time.perf_counter() - _last_llm_call_end)

            # Drain user messages BETWEEN LLM calls within a step. The
            # outer iteration drain (in _build_iteration_messages) only
            # fires at step boundaries, which can be 1-3 minutes apart
            # for long inner loops. Draining here gives the user
            # near-immediate visibility of mid-step messages — the next
            # LLM call will see the message as the most recent ``user``
            # turn in the conversation, and the strong wording in the
            # ``<urgent-user-message>`` block (rendered in the iteration
            # prompt) primes the model to acknowledge before continuing.
            #
            # We append a fresh ``user`` turn rather than rebuilding the
            # whole iteration prompt: the in-flight conversation context
            # is preserved, and the model sees the new message as a
            # natural follow-up.
            self._inject_mid_step_user_messages(ctx, messages)

            # Signal UI that LLM call is starting
            progress_step = ctx.state.plan.active_step
            step_limit = getattr(ctx, "step_tool_limit", ctx.max_per_action)
            _emit_loop_event("loop_llm_call_start", ctx.project_id, ctx.agent_id, {
                "iteration": iteration + 1,
                "phase": (
                    "closing" if completion_only
                    else "planning" if is_planning
                    else "recovery" if ctx.suppress_discovery_this_step
                    else "deciding"
                ),
                "step_title": progress_step.title if progress_step is not None else "",
                "tool_calls_step": action_tool_calls,
                "tool_calls_step_limit": step_limit,
                "tool_calls_total": ctx.state.total_tool_calls,
                "tool_calls_total_limit": ctx.max_total_calls,
            })

            result = llm_caller.call(
                ctx, messages, is_planning, action_tool_calls,
                completion_only=completion_only,
            )
            _last_llm_call_end = _time.perf_counter()

            # Checked here rather than only inside the regular-tool branch:
            # a turn that asks for nothing but pseudo-tools never reached
            # that branch, so cancelling during one was ignored until the
            # model happened to call a real tool again.
            if self._cancel_event.is_set():
                break

            try:
                _trace_llm_response(
                    iteration + 1,
                    reasoning=getattr(result, "reasoning_content", None),
                    content=getattr(result, "raw_content", None) or (
                        getattr(result.message, "content", None) if getattr(result, "message", None) else None
                    ),
                    tool_calls=list(getattr(result, "tool_calls", None) or []),
                )
            except Exception as _trace_err:
                logger.warning("reasoning trace emit failed: %s", _trace_err)

            if result.should_retry:
                continue
            if completion_only:
                completion_turns_used += 1
            if result.forced_step_result:
                step_result = result.forced_step_result
                break

            if result.tool_calls:
                # In FC mode (no streaming), signal the detected tool names immediately
                if not ctx.manual_tc:
                    first_tc = result.tool_calls[0]
                    tc_name = getattr(first_tc, "name", None) or getattr(getattr(first_tc, "function", None), "name", None)
                    if tc_name:
                        _emit_loop_event("loop_stream_status", ctx.project_id, ctx.agent_id, {
                            "phase": "tool_detected",
                            "token_count": 0,
                            "tool_name": tc_name,
                        })
                guard.text_retries = 0
                saw_tool_calls = True
                classified = tool_proc.classify(result.tool_calls)
                tool_proc.process_pseudo_tools(ctx, classified, self)
                # Reset read-without-note counter when notes are added
                if classified.notes:
                    guard.reset_read_counter()

                if completion_only and classified.regular:
                    # Native FC providers cannot choose these because their
                    # schemas were hidden. Plan mutations are the exception:
                    # they record the next unit of work without touching the
                    # workspace and are bounded by the two closing turns.
                    invalid = [
                        call.function.name for call in classified.regular
                        if call.function.name not in COMPLETION_PLAN_TOOL_NAMES
                    ]
                    if invalid:
                        names = ", ".join(invalid)
                        step_result = StepResult(
                            summary=(
                                "Step interrupted at its execution-tool budget; "
                                f"the completion-only turn attempted: {names}."
                            ),
                            status="continue",
                            interrupted=True,
                        )
                        break
                    # ToolRunner normally truncates at the Step boundary. Give
                    # exactly these state-only calls room to execute; after the
                    # batch action_tool_calls again equals step_tool_limit, so
                    # the following turn remains completion-only.
                    ctx.step_tool_limit += len(classified.regular)
                if classified.regular:
                    action_tool_calls = self._critic.review_alongside(
                        ctx, messages, classified.regular,
                        getattr(result, "reasoning_content", None),
                        lambda: self._execute_regular_tools(
                            ctx, classified, messages, result,
                            action_tool_calls, iteration, guard, tracker,
                        ),
                    )
                    if self._cancel_event.is_set():
                        break
                    _, progress_marker, progress_renewals = _renew_step_tool_window(
                        ctx,
                        tracker,
                        progress_marker,
                        progress_renewals,
                    )
                    # Expire old thinking content to save context window
                    ContextManager.expire_thinking(messages)
                    # Check guard conditions
                    forced = guard.check_repetition(ctx, messages)
                    if forced:
                        step_result = forced
                        break
                    guard.check_error_circuit_breaker(ctx, messages)
                    guard.check_note_discipline(ctx, messages)
                    guard.check_progress_drift(ctx, messages)

                    # A2 — Mid-step guidance: run detectors right after
                    # each successful tool execution so patterns fire on
                    # the NEXT LLM call rather than waiting for end-of-
                    # step. Shares state with the end-of-step call so
                    # guidance is never emitted twice per step.
                    self._guidance.try_queue(
                        ctx, messages, step_messages_start, mid_step=True,
                    )
                elif classified.step_complete or classified.notes or classified.session_notes or classified.thinks :
                    # Only pseudo-tools, no regular tools
                    self._build_pseudo_only_messages(ctx, classified, messages, result)
                    ContextManager.expire_thinking(messages)
                    # A turn that closes the step is going somewhere; one
                    # that only thinks is the case that can spin forever,
                    # because nothing here spends the step's budget.
                    if not classified.step_complete:
                        forced = guard.handle_pseudo_only(ctx, messages)
                        if forced:
                            step_result = forced
                            break

                if classified.regular:
                    guard.pseudo_only_rounds = 0

                if classified.step_complete:
                    # Several cheap-to-expensive gates can override the
                    # model's claim that this step is finished. Each sends it
                    # back for one more turn; see ``step_complete_gate`` for
                    # the complete chain and its order.
                    if self._step_gate.blocks(
                        ctx, classified.step_complete, messages,
                        action_tool_calls=action_tool_calls,
                        reasoning=getattr(result, "reasoning_content", None),
                    ):
                        continue  # Re-enter the loop, don't break.

                    step_result = _parse_step_complete_args(classified.step_complete.function.arguments)
                    break
            else:
                # Text-only response
                content = (result.message.content or "").strip() if result.message else result.raw_content
                forced = guard.handle_text_only(ctx, messages, content)
                if forced:
                    step_result = forced
                    break
                continue
        else:
            # Inner loop exhausted (while condition became false)
            if step_result is None:
                resource_exhaustion = _resource_exhaustion_reason(ctx)
                if resource_exhaustion is not None:
                    limit_msg = resource_exhaustion.removeprefix(
                        "Step interrupted: "
                    ).removesuffix(".")
                else:
                    limit_msg = (
                        "per-step tool call limit reached "
                        f"({action_tool_calls}/{ctx.step_tool_limit} calls)"
                    )
                step_result = StepResult(
                    summary=f"Step interrupted: {limit_msg}.",
                    status="continue",
                    interrupted=True,
                )
                _emit_log("error", f"{_RED}⚠ Inner loop exhausted: {limit_msg}{_RESET}",
                          project_id=ctx.project_id, agent_id=ctx.agent_id)

        return self._finalize_inner_loop(
            ctx, step_result, action_tool_calls, tracker,
            saw_tool_calls=saw_tool_calls,
        )

    def _finalize_inner_loop(
        self, ctx: ExecutionContext, step_result: StepResult | None,
        action_tool_calls: int, tracker: BehaviorTracker,
        *, saw_tool_calls: bool = False,
    ) -> StepResult:
        """Default step_result, propagate edit state, attach metadata."""
        if step_result is None:
            step_result = StepResult(
                summary="Step interrupted without a completion result.",
                status="continue",
                interrupted=True,
            )

        tracker.on_step_end()
        active = ctx.state.plan.active_step
        file_tracker = getattr(ctx, "file_tracker", None)
        current_fingerprint = (
            file_tracker.change_fingerprint(reconcile=True)
            if file_tracker is not None
            and hasattr(file_tracker, "change_fingerprint")
            else None
        )
        entry_fingerprint = (
            ctx.state.step_entry_change_fingerprints.get(active.index)
            if active is not None
            else None
        )
        tracker.net_workspace_changed = bool(
            current_fingerprint != entry_fingerprint
            if current_fingerprint is not None and entry_fingerprint is not None
            else tracker.files_edited
        )
        if tracker.net_workspace_changed:
            ctx.state.task_has_edits = True

        if active is not None:
            if tracker.net_workspace_changed:
                ctx.state.edited_step_indices.add(active.index)
            else:
                ctx.state.edited_step_indices.discard(active.index)

            if (
                getattr(ctx, "semantic_stagnation_control", False)
                and _step_phase(active.title) in {"change", "test_change"}
            ):
                test_marker = tuple(sorted(
                    fingerprint
                    for fingerprints in ctx.state.test_outcome_history.values()
                    for fingerprint in fingerprints
                ))
                previous_tests = ctx.state.last_test_outcomes_by_step.get(
                    active.index, ()
                )
                made_progress = (
                    tracker.net_workspace_changed or test_marker != previous_tests
                )
                windows = (
                    0 if made_progress
                    else ctx.state.no_progress_windows_by_step.get(active.index, 0) + 1
                )
                ctx.state.no_progress_windows_by_step[active.index] = windows
                ctx.state.last_test_outcomes_by_step[active.index] = test_marker
                if windows == 2:
                    ctx.state.discovery_suppression_steps = max(
                        ctx.state.discovery_suppression_steps, 1
                    )
                    _emit_log(
                        "warning",
                        "Two complete implementation windows produced no net "
                        "workspace or test-outcome change; narrowing discovery "
                        "tools for the next Step.",
                        project_id=ctx.project_id,
                        agent_id=ctx.agent_id,
                    )

        if tracker.files_edited:
            warned = set(ctx.state.similarity_warned_files)
            new_paths = [p for p in tracker.files_edited if p not in warned]
            if new_paths:
                existing = set(ctx.state.recently_written_files)
                for p in new_paths:
                    if p not in existing:
                        ctx.state.recently_written_files.append(p)
                        existing.add(p)

        step_result.action_tool_calls = action_tool_calls
        step_result.saw_tool_calls = saw_tool_calls
        step_result.behavior_tracker = tracker
        return step_result

    # ── Tool execution ──────────────────────────────────────────────────
    # The protocol of turning tool calls into conversation the model can
    # read — batching, budget counters, image turns, provider ordering
    # constraints — lives in ``ToolRunner``. These stay because the inner
    # loop and the critic's parallel branch both reach for them.

    def _execute_regular_tools(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], llm_result: LLMCallResult,
        action_tool_calls: int, iteration: int, guard: LoopGuard,
        tracker: BehaviorTracker,
    ) -> int:
        """Execute regular tool calls and build messages. Returns updated count."""
        return self._tool_runner.run_regular(
            ctx, classified, messages, llm_result, action_tool_calls,
            iteration, guard, tracker,
        )

    def _build_pseudo_only_messages(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], llm_result: LLMCallResult,
    ) -> None:
        """Build messages when only pseudo-tools were called."""
        self._tool_runner.run_pseudo_only(ctx, classified, messages, llm_result)

    def _handle_explore(
        self, ctx: ExecutionContext, step_result: StepResult, iteration: int,
    ) -> None:
        """Delegate sub-problem to TreeEngine."""
        step_index = ctx.state.plan.active_step.index if ctx.state.plan.active_step else iteration + 1
        _emit_log("warning",
                   f"{_YELLOW}🌳 Delegating to exploration tree: {step_result.summary[:120]}{_RESET}",
                   project_id=ctx.project_id, agent_id=ctx.agent_id)
        try:
            from infinidev.engine.tree import TreeEngine
            tree_engine = TreeEngine()
            explore_result = tree_engine.explore_subproblem(ctx.agent, step_result.summary)
            if len(ctx.state.notes) < 20:
                ctx.state.notes.append(f"Exploration result: {explore_result[:500]}")
            ctx.state.history.append(ActionRecord(
                step_index=step_index,
                summary=f"Explored via tree: {explore_result[:200]}",
                tool_calls_count=0,
            ))
        except Exception as exc:
            logger.warning("TreeEngine exploration failed: %s", exc)
            if len(ctx.state.notes) < 20:
                ctx.state.notes.append(f"Exploration failed: {exc}")


    def _checkpoint(self, event_id: int, state: LoopState) -> None:
        """No-op in CLI mode."""
        pass

    def _store_stats(self, state: LoopState) -> None:
        """Store execution stats for external access."""
        self._last_total_tool_calls = state.total_tool_calls
        self._last_state = state

    def _apply_guardrail(
        self,
        ctx: ExecutionContext,
        result: str,
        guardrail: Any | None,
        max_retries: int,
        llm_params: dict[str, Any],
        system_prompt: str,
        desc: str,
        expected: str,
        state: LoopState,
        tool_schemas: list[dict[str, Any]],
        tool_dispatch: dict[str, Any],
        max_per_action: int = 0,
    ) -> str:
        """Validate result with guardrail; retry with feedback if it fails."""
        return apply_guardrail(
            ctx, result, guardrail, max_retries, llm_params, system_prompt,
            desc, expected, state, tool_schemas, tool_dispatch, max_per_action,
            hooks=self._hooks,
        )
