"""LoopEngine — plan-execute-summarize loop orchestrator.

The main engine class that coordinates LLMCaller, ToolProcessor,
LoopGuard, and StepManager to run the agent loop.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from infinidev.engine.base import AgentEngine
from infinidev.engine.hooks.hooks import hook_manager as _hook_manager, HookContext as _HookContext, HookEvent as _HookEvent
from infinidev.engine.llm_client import (
    call_llm as _call_llm,
    is_malformed_tool_call as _is_malformed_tool_call,
    is_transient as _is_transient,
    PERMANENT_ERRORS as _PERMANENT_ERRORS,
)
from infinidev.engine.loop.context import (
    build_iteration_prompt,
    build_system_prompt,
    build_tools_prompt_section,
)
from infinidev.engine.loop.critic import AssistantCritic, CriticVerdict
from infinidev.engine.loop.models import (
    ActionRecord,
    LoopState,
    StepResult,
)
from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.loop.tools import (
    ADD_NOTE_SCHEMA,
    ADD_STEP_SCHEMA,
    MODIFY_STEP_SCHEMA,
    REMOVE_STEP_SCHEMA,
    STEP_COMPLETE_SCHEMA,
    build_tool_dispatch,
    build_tool_schemas,
    execute_tool_call,
)
from infinidev.engine.formats.tool_call_parser import (
    safe_json_loads as _safe_json_loads,
    ManualToolCall as _ManualToolCall,
    parse_text_tool_calls as _parse_text_tool_calls,
    parse_step_complete_args as _parse_step_complete_args,
)
from infinidev.engine.engine_logging import (
    emit_loop_event as _emit_loop_event,
    log as _log,
    emit_log as _emit_log,
    extract_tool_error as _extract_tool_error,
    log_start as _log_start,
    log_step_start as _log_step_start,
    log_step_done as _log_step_done,
    log_plan as _log_plan,
    log_finish as _log_finish,
    YELLOW as _YELLOW,
    RED as _RED,
    RESET as _RESET,
)
from infinidev.engine.tool_executor import (
    update_opened_files_cache as _update_opened_files_cache,
    batch_tool_calls as _batch_tool_calls,
    execute_tool_calls_parallel as _execute_tool_calls_parallel,
    capture_pre_content as _capture_pre_content,
    maybe_emit_file_change as _maybe_emit_file_change,
    WRITE_TOOLS as _WRITE_TOOLS,
)

from infinidev.config.llm import get_litellm_params, _is_small_model
from infinidev.config.model_capabilities import get_model_capabilities
from infinidev.config.settings import settings
from infinidev.tools.base.context import (
    bind_tools_to_agent,
    get_context_for_agent,
    set_loop_state,
)
from infinidev.engine._best_effort import best_effort
from infinidev.engine.loop.context_manager import ContextManager
from infinidev.engine.loop.guidance_handler import GuidanceHandler
from infinidev.engine.loop.model_context import _get_model_max_context
from infinidev.engine.loop.step_summarizer import _summarize_step, _synthesize_final
from infinidev.engine.loop.execution_context import ExecutionContext
from infinidev.engine.loop.llm_caller import LLMCaller, LLMCallResult, ClassifiedCalls
from infinidev.engine.loop.tool_processor import ToolProcessor
from infinidev.engine.loop.loop_guard import LoopGuard
from infinidev.engine.loop.behavior_tracker import BehaviorTracker
from infinidev.engine.loop.step_manager import StepManager, _get_settings
from infinidev.engine.trace_log import (
    trace_run_start as _trace_run_start,
    trace_iteration_prompt as _trace_iter_prompt,
    trace_llm_response as _trace_llm_response,
    trace_plan as _trace_plan,
    trace_step_done as _trace_step_done,
    trace_run_end as _trace_run_end,
)

logger = logging.getLogger(__name__)


def _seed_state_from_plan(state, plan) -> None:
    """Populate ``state.plan`` from an analyst-emitted Plan.

    Each step becomes a user-approved PlanStep (LLM cannot remove or
    modify them via step_complete operations). The first step is set
    active so the developer has somewhere to start.

    Plans with an overview but no steps (analyst fallback path) seed
    only the overview so the developer still has context, and then
    drop into the LoopEngine bootstrap branch where the model is
    instructed to call ``add_step`` to build its own decomposition.
    """
    from infinidev.engine.loop.plan_step import PlanStep

    state.plan.overview = plan.overview or ""
    state.plan.steps = [
        PlanStep(
            index=idx + 1,
            title=spec.title,
            detail=spec.detail,
            expected_output=spec.expected_output,
            user_approved=True,
            status="active" if idx == 0 else "pending",
        )
        for idx, spec in enumerate(plan.steps)
    ]


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
        self._last_total_tool_calls: int = 0
        self._nudge_threshold_override: int | None = None
        self._summarizer_override: bool | None = None
        self._supports_vision_cached: bool | None = None
        self._cancel_event: __import__('threading').Event = __import__('threading').Event()
        # Optional OrchestrationHooks. When set by the caller (typically
        # the pipeline before execute()), the engine forwards file-change
        # and step-start callbacks so a UI can render live progress.
        # Plumbing via attribute keeps execute()'s signature stable.
        self._hooks: object | None = None
        self.session_notes: list[str] = []  # Persist across tasks within a session
        # Thread-safe queue for user messages injected mid-task
        import queue as _queue_mod
        self._user_messages: _queue_mod.Queue[str] = _queue_mod.Queue()
        self._guidance = GuidanceHandler()
        from infinidev.engine.context_rank.hooks import ContextRankHooks
        self._cr_hooks = ContextRankHooks()
        self._cr_cached_result: Any | None = None
        self._cr_last_pivot_key: tuple[int, str] | None = None
        # Pair-programming critic. Lazily built on first use so a
        # disabled feature pays zero cost. Reset between runs is
        # implicit — the descriptions captured at build time are
        # stable across iterations.
        self._critic: AssistantCritic | None = None
        self._critic_init_failed: bool = False
        # Critic observations made on `step_complete` calls survive
        # the inner-loop break by sitting here until the NEXT step's
        # preamble drains them onto its fresh messages list.
        self._pending_critic_messages: list[dict[str, Any]] = []

    def _get_or_build_critic(self, ctx: "ExecutionContext") -> AssistantCritic | None:
        """Lazy-init the assistant critic on first use of the run.

        Returns ``None`` when the feature is disabled OR a previous
        init attempt already failed (we don't retry per-step — a bad
        config should fail loud once and then be silently absent).
        """
        if not settings.ASSISTANT_LLM_ENABLED:
            return None
        if self._critic is not None:
            return self._critic
        if self._critic_init_failed:
            return None
        try:
            descriptions: dict[str, str] = {}
            for name, tool in (ctx.tool_dispatch or {}).items():
                desc = getattr(tool, "description", None) or ""
                descriptions[name] = desc
            self._critic = AssistantCritic(descriptions)
            # Register as active so ``ConsultAssistantTool`` (the
            # principal-facing bridge tool) can reach the critic
            # without dependency-injection through the tool factory.
            try:
                from infinidev.engine.loop.critic import set_active_critic
                set_active_critic(self._critic)
            except Exception:
                logger.debug("set_active_critic failed", exc_info=True)
            return self._critic
        except Exception as exc:
            logger.warning(
                "assistant critic init failed (%s); disabling for this run", exc,
            )
            self._critic_init_failed = True
            return None

    def _inject_critic_message(
        self,
        ctx: "ExecutionContext",
        messages: list[dict[str, Any]],
        verdict: CriticVerdict,
        *,
        source: str,
    ) -> None:
        """Anchor the critic's observation next to the tool that triggered it.

        Models weight nearby tokens more heavily, so we append the
        verdict to the LAST ``role: "tool"`` message's content rather
        than tacking on a separate user-role message at the end. That
        keeps the critique inside the same attentional block as the
        principal's own action output. If no tool message is present
        (defensive — shouldn't happen on the regular-tools path), we
        fall back to a user-role append.

        ``source`` is "tools" for normal tool-call review and
        "step_complete" for end-of-step review; it lets the UI
        annotate the message origin if it cares.
        """
        if verdict.is_silent:
            return
        critic = self._critic
        model_tag = critic.model_short_name if critic else "assistant"
        note = f"\n\n--- critic note ---\n[ASSISTANT ({model_tag}) - {verdict.action}]: {verdict.message}"

        last_tool_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "tool":
                last_tool_idx = i
                break

        if last_tool_idx >= 0:
            existing = messages[last_tool_idx].get("content") or ""
            if not isinstance(existing, str):
                existing = str(existing)
            messages[last_tool_idx]["content"] = existing + note
        else:
            messages.append({"role": "user", "content": note.lstrip("\n")})
        try:
            _emit_loop_event(
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

    def inject_message(self, message: str) -> None:
        """Inject a user message into the running loop (thread-safe).

        The message will be included in the next iteration's prompt as
        a ``<user-message>`` block, giving the LLM live guidance without
        interrupting the current step.
        """
        self._user_messages.put(message)

    def _drain_user_messages(self) -> list[str]:
        """Drain all pending user messages from the queue."""
        messages = []
        while not self._user_messages.empty():
            try:
                messages.append(self._user_messages.get_nowait())
            except Exception:
                break
        return messages

    def _inject_mid_step_user_messages(
        self, ctx: "ExecutionContext", messages: list[dict[str, Any]],
    ) -> None:
        """Drain any pending user messages and inject them as urgent
        ``user``-role turns before the next LLM call.

        No-op if the queue is empty. Used at the top of the inner loop
        so the model always sees the freshest user input even when the
        user speaks while an LLM call is in flight.
        """
        drained = self._drain_user_messages()
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

    def _reject_step_complete_on_late_message(
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
        drained = self._drain_user_messages()
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

    def cancel(self) -> None:
        """Signal the engine to stop after the current tool call."""
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def get_changed_files_summary(self) -> str:
        """Return a summary of files changed in the last execution.

        Used by the code review engine to review changes.
        Returns empty string if no files were changed.
        """
        if self._last_file_tracker is None:
            return ""

        paths = self._last_file_tracker.get_all_paths()
        if not paths:
            return ""

        parts = []
        for path in paths:
            action = self._last_file_tracker.get_action(path)
            diff = self._last_file_tracker.get_diff(path)
            if diff:
                parts.append(f"### {path} ({action})\n```diff\n{diff}\n```")
            else:
                parts.append(f"### {path} ({action}, no diff)")

        return "\n\n".join(parts)

    def has_file_changes(self) -> bool:
        """Whether the last execution modified any files."""
        if self._last_file_tracker is None:
            return False
        return bool(self._last_file_tracker.get_all_paths())

    def get_file_change_reasons(self) -> dict[str, list[str]]:
        """Return path → list of reasons for each changed file."""
        if self._last_file_tracker is None:
            return {}
        result = {}
        for path in self._last_file_tracker.get_all_paths():
            reasons = self._last_file_tracker.get_reasons(path)
            if reasons:
                result[path] = reasons
        return result

    def get_file_tracker(self) -> FileChangeTracker | None:
        """Expose the tracker from the last task for downstream checks."""
        return self._last_file_tracker

    def get_plan_steps(self) -> list[dict]:
        """Loop engine is step-scoped — no multi-step plan of its own."""
        return []

    def get_file_contents(self) -> dict[str, str]:
        """Return path → current content for each changed file."""
        import os as _os
        from infinidev.engine.tool_executor import MAX_TRACK_FILE_SIZE
        if self._last_file_tracker is None:
            return {}
        result = {}
        for path in self._last_file_tracker.get_all_paths():
            with best_effort("failed to read tracked file %s", path):
                if _os.path.isfile(path) and _os.path.getsize(path) <= MAX_TRACK_FILE_SIZE:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        result[path] = f.read()
        return result

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
        max_tool_calls_per_action: int | None = None,
        nudge_threshold: int | None = None,
        nudge_message_template: str | None = None,
        summarizer_enabled: bool | None = None,
        initial_plan: Any | None = None,
        initial_attachments: list[Any] | None = None,
        task: Any | None = None,
    ) -> str:
        """Plan-execute-summarize loop.

        Delegates to composition components: LLMCaller, ToolProcessor,
        LoopGuard, StepManager. See class docstrings for details.

        When ``initial_plan`` (an ``infinidev.engine.analysis.plan.Plan``
        instance) is provided, the loop starts with a pre-approved plan
        populated from the analyst: plan.overview seeds
        ``state.plan.overview`` (rendered every iteration as
        ``<plan-overview>``), and each step becomes a ``user_approved``
        PlanStep that the LLM cannot remove or modify. The bootstrap
        branch that asks "No plan yet — call add_step" is naturally
        suppressed because state.plan.steps is non-empty.
        """
        ctx = self._build_context(
            agent, task_prompt,
            verbose=verbose, guardrail=guardrail,
            guardrail_max_retries=guardrail_max_retries,
            output_pydantic=output_pydantic, task_tools=task_tools,
            event_id=event_id, resume_state=resume_state,
            max_iterations=max_iterations,
            max_total_tool_calls=max_total_tool_calls,
            max_tool_calls_per_action=max_tool_calls_per_action,
            nudge_threshold=nudge_threshold,
            nudge_message_template=nudge_message_template,
            summarizer_enabled=summarizer_enabled,
            task=task,
        )
        if initial_plan is not None:
            _seed_state_from_plan(ctx.state, initial_plan)
        # Stash attachments on the engine instance for the first
        # iteration only — subsequent turns rebuild the prompt from
        # compact summaries and don't need the raw payload.
        self._initial_attachments = list(initial_attachments or [])
        # Reset cached vision-capability probe: the configured model can
        # change between execute() calls.
        self._supports_vision_cached = None
        llm_caller, tool_proc, guard, step_mgr = self._init_execution(ctx, task_prompt)
        consecutive_all_done = 0

        for iteration in range(ctx.start_iteration, ctx.max_iterations):
            if self._cancel_event.is_set():
                logger.info("LoopEngine: cancelled by user")
                _emit_log("info", f"{_YELLOW}\u26a0 Task cancelled by user{_RESET}",
                          project_id=ctx.project_id, agent_id=ctx.agent_id)
                break

            messages, step_messages_start = self._run_iteration_preamble(ctx, iteration)

            step_result = self._run_inner_loop(
                ctx, messages, iteration, llm_caller, tool_proc, guard,
                step_messages_start=step_messages_start,
            )

            # Track consecutive text-only iterations
            if step_result.action_tool_calls == 0:
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

            term = self._check_termination(ctx, step_result, step_mgr, iteration, consecutive_all_done)
            if term is not None:
                return term

            # Update consecutive all-done counter
            if step_result.status == "explore":
                consecutive_all_done = 0
            elif ctx.state.plan.steps and not ctx.state.plan.has_pending:
                consecutive_all_done += 1
            else:
                consecutive_all_done = 0

        # Outer loop exhausted
        return step_mgr.finish(ctx, "exhausted", ctx.max_iterations - 1)

    # ── Extracted phases of execute() ──────────────────────────────────

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

        self._cancel_event.clear()
        self._last_state = ctx.state

        set_loop_state(ctx.agent_id, ctx.state)
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
        if self._pending_critic_messages:
            messages.extend(self._pending_critic_messages)
            self._pending_critic_messages = []
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
        step_result = step_mgr.auto_split(ctx, step_result)

        _hook_manager.dispatch(_HookContext(
            event=_HookEvent.STEP_TRANSITION,
            metadata={"step_result": step_result, "plan": ctx.state.plan, "iteration": iteration},
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        ))
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
            self._handle_explore(ctx, step_result, iteration)
            return None

        if step_result.status == "done":
            if not step_result.final_answer and iteration == ctx.start_iteration:
                _emit_log("warning",
                          f"{_YELLOW}\u26a0 LLM declared done on first step without final_answer \u2014 forcing continue{_RESET}",
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
        """Build ExecutionContext from agent, task_prompt, and overrides."""
        llm_params = get_litellm_params()
        if llm_params is None:
            raise RuntimeError("LoopEngine requires LiteLLM parameters. Ensure INFINIDEV_LLM_MODEL is set.")

        max_iterations = kwargs.get('max_iterations') or settings.LOOP_MAX_ITERATIONS
        max_total_calls = kwargs.get('max_total_tool_calls') or settings.LOOP_MAX_TOTAL_TOOL_CALLS
        max_per_action = (kwargs.get('max_tool_calls_per_action') or settings.LOOP_MAX_TOOL_CALLS_PER_ACTION) or max_total_calls
        self._nudge_threshold_override = kwargs.get('nudge_threshold')
        self._summarizer_override = kwargs.get('summarizer_enabled')

        max_context_tokens = _get_model_max_context(llm_params)

        # Resolve tools
        task_tools = kwargs.get('task_tools')
        tools = task_tools if task_tools is not None else getattr(agent, "tools", [])
        if task_tools is not None:
            bind_tools_to_agent(task_tools, agent.agent_id)

        file_tracker = FileChangeTracker()
        self._last_file_tracker = file_tracker
        self._last_total_tool_calls = 0

        caps = get_model_capabilities()
        manual_tc = not caps.supports_function_calling
        is_small = _is_small_model()

        if is_small:
            logger.info("LoopEngine: small model detected — using simplified prompts and reduced tools")

        if is_small and task_tools is None:
            from infinidev.tools import get_tools_for_role
            tools = get_tools_for_role("developer", small_model=True)
            # CRITICAL: bind the freshly-created tools to this agent's id.
            # Without this, tool.agent_id falls back to thread-local lookup,
            # which is unreliable when hooks/threads change context — causing
            # "No active plan context" errors in plan tools intermittently.
            bind_tools_to_agent(tools, agent.agent_id)

        tool_schemas = build_tool_schemas(tools, small_model=is_small) if tools else [STEP_COMPLETE_SCHEMA]
        tool_dispatch = build_tool_dispatch(tools) if tools else {}

        system_prompt = build_system_prompt(
            agent.backstory,
            tech_hints=getattr(agent, '_tech_hints', None),
            session_summaries=getattr(agent, '_session_summaries', None),
            identity_override=getattr(agent, '_system_prompt_identity', None),
            protocol_override=getattr(agent, '_system_prompt_protocol', None),
            small_model=is_small,
        )
        if manual_tc:
            tools_section = build_tools_prompt_section(tool_schemas, small_model=is_small)
            system_prompt = f"{system_prompt}\n\n{tools_section}"
            logger.info("LoopEngine [%s]: manual tool calling mode", getattr(agent, "agent_id", "?"))

        desc, expected = task_prompt

        # Read event_id / resume_state from tool context if not passed
        event_id = kwargs.get('event_id')
        resume_state = kwargs.get('resume_state')
        if event_id is None or resume_state is None:
            tool_ctx = get_context_for_agent(agent.agent_id)
            if tool_ctx:
                event_id = event_id or tool_ctx.event_id
                resume_state = resume_state or tool_ctx.resume_state

        if resume_state:
            state = LoopState.model_validate(resume_state)
            if state.plan.steps and not state.plan.active_step:
                for s in state.plan.steps:
                    if s.status == "pending":
                        s.status = "active"
                        break
            logger.info("LoopEngine: resuming from iteration %d", state.iteration_count)
        else:
            state = LoopState()

        if kwargs.get('verbose', True):
            _log_start(agent.agent_id, getattr(agent, "name", agent.agent_id),
                       getattr(agent, "role", "agent"), desc, len(tools))

        return ExecutionContext(
            llm_params=llm_params, manual_tc=manual_tc, is_small=is_small,
            system_prompt=system_prompt, tool_schemas=tool_schemas,
            tool_dispatch=tool_dispatch,
            planning_schemas=[
                ADD_STEP_SCHEMA, MODIFY_STEP_SCHEMA, REMOVE_STEP_SCHEMA,
                ADD_NOTE_SCHEMA, STEP_COMPLETE_SCHEMA,
            ],
            tools=tools, max_iterations=max_iterations, max_per_action=max_per_action,
            max_total_calls=max_total_calls, history_window=settings.LOOP_HISTORY_WINDOW,
            max_context_tokens=max_context_tokens,
            verbose=kwargs.get('verbose', True),
            guardrail=kwargs.get('guardrail'), guardrail_max_retries=kwargs.get('guardrail_max_retries', 5),
            output_pydantic=kwargs.get('output_pydantic'),
            agent=agent, agent_name=getattr(agent, "name", agent.agent_id),
            agent_role=getattr(agent, "role", "agent"),
            desc=desc, expected=expected, event_id=event_id,
            skip_plan=False,
            nudge_message_template=kwargs.get('nudge_message_template'),
            state=state, file_tracker=file_tracker,
            start_iteration=state.iteration_count,
            task=kwargs.get('task'),
        )

    def _build_iteration_messages(
        self, ctx: ExecutionContext, iteration: int,
    ) -> list[dict[str, Any]]:
        """Build the messages list for one iteration."""
        effective_state = ctx.state
        if ctx.history_window > 0 and len(ctx.state.history) > ctx.history_window:
            effective_state = ctx.state.model_copy(deep=True)
            effective_state.history = ctx.state.history[-ctx.history_window:]

        if iteration == ctx.start_iteration:
            try:
                from infinidev.db.service import get_project_knowledge
                self._project_knowledge = get_project_knowledge(project_id=ctx.project_id)
            except Exception:
                self._project_knowledge = []

        # Drain any user messages injected mid-task
        injected = self._drain_user_messages()

        # ContextRank: compute + inject only at pivot points
        # (iteration 0 + step transitions). Between pivots, the model
        # already saw the <context-rank> so we skip it to save tokens.
        # The result itself is cached per-step for fast lookup.
        cr_result = None
        with best_effort("ContextRank ranking failed"):
            from infinidev.config.settings import settings as _cr_settings
            if _cr_settings.CONTEXT_RANK_ENABLED and self._cr_hooks._enabled:
                active = ctx.state.plan.active_step
                active_key = (active.index, active.title) if active else (-1, "")
                is_pivot = (
                    iteration == ctx.start_iteration
                    or active_key != getattr(self, "_cr_last_pivot_key", None)
                )
                if is_pivot:
                    from infinidev.engine.context_rank.ranker import rank as _cr_rank
                    cr_result = _cr_rank(
                        ctx.desc,
                        self._cr_hooks._session_id,
                        self._cr_hooks._task_id,
                        iteration,
                        cached_embedding=self._cr_hooks._task_embedding,
                        cached_simplified_embedding=self._cr_hooks._task_embedding_simplified,
                        project_id=ctx.project_id,
                    )
                    self._cr_cached_result = cr_result
                    self._cr_last_pivot_key = active_key
                # Non-pivot iterations: don't inject (keep cr_result = None)

        from infinidev.engine.static_analysis_timer import measure as _sa_measure
        with _sa_measure("prompt_build"):
            user_prompt = build_iteration_prompt(
                ctx.desc, ctx.expected, effective_state,
                project_knowledge=self._project_knowledge if iteration == ctx.start_iteration else None,
                context_rank_result=cr_result,
                max_context_tokens=ctx.max_context_tokens,
                session_notes=self.session_notes if self.session_notes else None,
                user_messages=injected if injected else None,
                skip_plan=ctx.skip_plan,
                task=ctx.task,
                small_model=ctx.is_small,
            )
        user_content: Any = user_prompt
        # Attachments only travel with the first iteration's user turn:
        # subsequent turns rebuild the prompt from compact summaries, and
        # re-sending the base64 payload bloats context and (for billed
        # vision providers) multiplies cost per iteration.
        _atts = getattr(self, "_initial_attachments", None) or []
        if _atts and iteration == ctx.start_iteration:
            if self._supports_vision_cached is None:
                try:
                    from infinidev.config.model_capabilities import _detect_vision_support
                    self._supports_vision_cached = _detect_vision_support()
                except Exception:
                    self._supports_vision_cached = False
            from infinidev.engine.multimodal import (
                build_user_content, mention_paths_as_text,
            )
            if self._supports_vision_cached:
                user_content = build_user_content(user_prompt, _atts)
            else:
                user_content = mention_paths_as_text(user_prompt, _atts)
        return [
            {"role": "system", "content": ctx.system_prompt},
            {"role": "user", "content": user_content},
        ]

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
        is_planning = not ctx.state.plan.steps

        llm_caller.reset()
        guard.reset()
        tracker = BehaviorTracker(set(ctx.state.opened_files.keys()))
        tracker.task_has_edits = ctx.state.task_has_edits

        # Tracks the wall-clock time of the previous LLM call's return
        # so we can measure the python-only gap until the next call.
        # None means "no previous call yet in this step".
        import time as _time
        from infinidev.engine.static_analysis_timer import add_elapsed as _sa_add
        _last_llm_call_end: float | None = None

        while action_tool_calls < ctx.max_per_action and ctx.state.total_tool_calls < ctx.max_total_calls:
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
            _emit_loop_event("loop_llm_call_start", ctx.project_id, ctx.agent_id, {})

            result = llm_caller.call(ctx, messages, is_planning, action_tool_calls)
            _last_llm_call_end = _time.perf_counter()

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
                classified = tool_proc.classify(result.tool_calls)
                tool_proc.process_pseudo_tools(ctx, classified, self)
                # Reset read-without-note counter when notes are added
                if classified.notes:
                    guard.reset_read_counter()

                if classified.regular:
                    critic = self._get_or_build_critic(ctx)
                    if critic is not None:
                        # Run the critic in parallel with tool execution.
                        # The verdict is purely informative — never blocks
                        # or alters the tool result. ThreadPoolExecutor
                        # keeps both the principal-side I/O and the
                        # critic LLM call (on a different GPU/endpoint)
                        # happening at the same wall clock, so the
                        # critic adds ~0 latency in the steady state.
                        import concurrent.futures as _futures
                        # Snapshot messages BEFORE _execute_regular_tools
                        # mutates them — the critic should see exactly
                        # what the principal saw, not the post-execution
                        # state.
                        crit_msgs_snapshot = list(messages)
                        crit_calls_snapshot = list(classified.regular)
                        # Thread the principal's current-turn reasoning to the
                        # critic out-of-band — it's stripped from message
                        # history (see ContextManager.expire_thinking) so the
                        # critic would otherwise only see the actions, not
                        # the thinking that produced them.
                        crit_reasoning_snapshot = getattr(result, "reasoning_content", None)
                        with _futures.ThreadPoolExecutor(max_workers=2) as _ex:
                            _tools_fut = _ex.submit(
                                self._execute_regular_tools,
                                ctx, classified, messages, result, action_tool_calls, iteration, guard, tracker,
                            )
                            _critic_fut = _ex.submit(
                                critic.review,
                                crit_msgs_snapshot,
                                crit_calls_snapshot,
                                crit_reasoning_snapshot,
                            )
                            action_tool_calls = _tools_fut.result()
                            try:
                                verdict = _critic_fut.result()
                            except Exception as _crit_exc:
                                logger.warning("assistant critic raised: %s", _crit_exc)
                                verdict = None
                        if verdict is not None and not verdict.is_silent:
                            self._inject_critic_message(
                                ctx, messages, verdict, source="tools",
                            )
                    else:
                        action_tool_calls = self._execute_regular_tools(
                            ctx, classified, messages, result, action_tool_calls, iteration, guard, tracker,
                        )
                    if self._cancel_event.is_set():
                        break
                    # Expire old thinking content to save context window
                    ContextManager.expire_thinking(messages)
                    # Check guard conditions
                    forced = guard.check_repetition(ctx, messages)
                    if forced:
                        step_result = forced
                        break
                    guard.check_error_circuit_breaker(ctx, messages)
                    guard.check_note_discipline(ctx, messages)

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

                if classified.step_complete:
                    # For small models: gate step_complete on having notes
                    # (prevents context loss from models that never use add_note).
                    # Gate ONCE per step — second attempt is always honored, so the
                    # model can't get stuck in a "must add note → does add fake
                    # note → blocked again" loop. Disable globally with
                    # INFINIBAY_LOOP_REQUIRE_NOTE_BEFORE_COMPLETE=false.
                    _require_note = getattr(_get_settings(), "LOOP_REQUIRE_NOTE_BEFORE_COMPLETE", True)
                    if (_require_note
                        and ctx.is_small
                        and not ctx.state.notes
                        and action_tool_calls >= 2
                        and not getattr(self, '_step_complete_gated', False)):
                        self._step_complete_gated = True
                        nudge = (
                            "Hold on — before you complete this step, save the key "
                            "facts you discovered with add_note (file paths, function "
                            "names, line numbers, decisions). Anything not saved is "
                            "discarded when this step ends. Example: "
                            "add_note(note='auth.py:42 verify_token() uses JWT, no exp check'). "
                            "Then call step_complete again — the second call will be honored."
                        )
                        if ctx.manual_tc:
                            messages.append({"role": "user", "content": nudge})
                        else:
                            # Replace the existing tool_result stub instead of
                            # appending a second one — Anthropic requires exactly
                            # one tool_result per tool_use id.
                            _replaced = False
                            for msg in reversed(messages):
                                if (
                                    msg.get("role") == "tool"
                                    and msg.get("tool_call_id") == classified.step_complete.id
                                ):
                                    msg["content"] = nudge
                                    _replaced = True
                                    break
                            if not _replaced:
                                messages.append({"role": "tool", "tool_call_id": classified.step_complete.id,
                                                 "content": nudge})
                        continue  # Don't break — let model add notes first
                    self._step_complete_gated = False

                    # Mid-step user message arrived during the LLM
                    # call we just finished? Don't honor step_complete
                    # yet — the user is waiting for an immediate
                    # response and the very next thing they should
                    # see is an acknowledgement, not a step boundary.
                    #
                    # The drain at the top of this while loop only
                    # catches messages that arrived BEFORE the LLM
                    # call. Messages enqueued WHILE the model was
                    # generating land here. If we ``break`` now,
                    # they'd sit in the queue until the next outer
                    # iteration's drain — i.e. the next step — which
                    # is exactly the latency the user asked us to
                    # eliminate.
                    #
                    # We inject the message AS A TOOL RESULT on the
                    # step_complete tool_call_id (same pattern as
                    # the note-discipline gate above). The model
                    # already saw an "acknowledged" tool result for
                    # step_complete from _execute_regular_tools /
                    # _build_pseudo_only_messages, but a second tool
                    # result on the same id reads as "your previous
                    # close was overridden by this feedback" — which
                    # is exactly the framing we want. Following the
                    # tool-result is the model's natural mode after
                    # a tool call, so this format gets honored far
                    # more often than a bare user-role message.
                    if self._reject_step_complete_on_late_message(
                        ctx, messages, classified.step_complete.id,
                    ):
                        continue  # Re-enter the loop, don't break.

                    # Pair-programming critic on step_complete. Synchronous
                    # because there's no tool execution to parallelise — the
                    # principal is about to break out of the inner loop.
                    #
                    # ``reject`` is special here: it actually BLOCKS the
                    # step from closing. We overwrite the step_complete
                    # tool_result with the critic's objection (same
                    # mechanism used for late mid-step user messages —
                    # see ``_reject_step_complete_on_late_message``) and
                    # ``continue`` the inner loop so the model gets one
                    # more turn to address the concern. Other verdicts
                    # (information / recommendation) are advisory: queued
                    # for the NEXT step's preamble, the current messages
                    # list dies with the break.
                    if settings.ASSISTANT_LLM_INCLUDE_STEP_COMPLETE:
                        critic = self._get_or_build_critic(ctx)
                        if critic is not None:
                            try:
                                verdict = critic.review(
                                    messages,
                                    [classified.step_complete],
                                    getattr(result, "reasoning_content", None),
                                )
                            except Exception as _crit_exc:
                                logger.warning("assistant critic (step_complete) raised: %s", _crit_exc)
                                verdict = None
                            if verdict is not None and not verdict.is_silent:
                                model_tag = critic.model_short_name
                                if verdict.action == "reject":
                                    rejection_body = (
                                        f"step_complete REJECTED by the assistant critic "
                                        f"({model_tag}). Address the objection below before "
                                        f"closing this step — either fix the issue and call "
                                        f"step_complete again, or push back with a brief "
                                        f"explanation if you disagree.\n\n"
                                        f"Critic objection:\n{verdict.message}"
                                    )
                                    self._overwrite_step_complete_tool_result(
                                        messages, classified.step_complete.id, rejection_body,
                                    )
                                    try:
                                        _emit_loop_event(
                                            "loop_assistant_message",
                                            ctx.project_id, ctx.agent_id,
                                            {
                                                "action": verdict.action,
                                                "message": verdict.message,
                                                "model": model_tag,
                                                "source": "step_complete",
                                                "blocked": True,
                                            },
                                        )
                                    except Exception:
                                        pass
                                    _emit_log(
                                        "info",
                                        f"⚠ step_complete blocked by assistant critic "
                                        f"({model_tag}): {verdict.message[:120]}",
                                        project_id=ctx.project_id, agent_id=ctx.agent_id,
                                    )
                                    continue  # Re-enter the loop, don't break.
                                prefix = f"[ASSISTANT ({model_tag}) - {verdict.action}] (re: step_complete): "
                                self._pending_critic_messages.append({
                                    "role": "user",
                                    "content": prefix + verdict.message,
                                })
                                try:
                                    _emit_loop_event(
                                        "loop_assistant_message",
                                        ctx.project_id, ctx.agent_id,
                                        {
                                            "action": verdict.action,
                                            "message": verdict.message,
                                            "model": model_tag,
                                            "source": "step_complete",
                                        },
                                    )
                                except Exception:
                                    pass

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
                if ctx.state.total_tool_calls >= ctx.max_total_calls:
                    limit_msg = f"global tool call limit reached ({ctx.state.total_tool_calls}/{ctx.max_total_calls} total calls)"
                else:
                    limit_msg = f"per-step tool call limit reached ({action_tool_calls}/{ctx.max_per_action} calls)"
                step_result = StepResult(summary=f"Step interrupted: {limit_msg}.", status="continue")
                _emit_log("error", f"{_RED}⚠ Inner loop exhausted: {limit_msg}{_RESET}",
                          project_id=ctx.project_id, agent_id=ctx.agent_id)

        return self._finalize_inner_loop(ctx, step_result, action_tool_calls, tracker)

    def _finalize_inner_loop(
        self, ctx: ExecutionContext, step_result: StepResult | None,
        action_tool_calls: int, tracker: BehaviorTracker,
    ) -> StepResult:
        """Default step_result, propagate edit state, attach metadata."""
        if step_result is None:
            step_result = StepResult(summary="Step completed.", status="continue")

        tracker.on_step_end()
        if tracker.task_has_edits:
            ctx.state.task_has_edits = True

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
        step_result.behavior_tracker = tracker
        return step_result

    def _capture_test_command_output(
        self, ctx: ExecutionContext, arguments: str, result: str,
    ) -> str:
        """Side-effects + auto-annotation for a test-runner execute_command.

        Three responsibilities, all best-effort (any exception is
        swallowed — this is an optimization, not correctness):

        1. Cache raw stdout on ``ctx.state`` so the ``tail_test_output``
           meta-tool can serve a filtered view without re-running.
        2. Record the outcome fingerprint per *normalised* test command
           so ``regression_after_edit`` compares apples to apples.
           Keeps the last two entries per command to bound state size.
        3. Parse structured failures and append them inline to the
           result so the model sees them next to the raw stdout.
        """
        with best_effort("test command capture failed for %s", arguments[:80]):
            from infinidev.engine.guidance import (
                is_test_command,
                test_outcome_fingerprint,
                normalize_test_command,
            )
            if not is_test_command(arguments, ctx.state):
                return result

            ctx.state.last_test_output = result
            try:
                import json as _json
                _args = _json.loads(arguments) if arguments else {}
                cmd_str = str(_args.get("command", ""))
            except Exception:
                cmd_str = arguments
            ctx.state.last_test_command = cmd_str[:300]

            new_fp = test_outcome_fingerprint(result)
            if new_fp:
                key = normalize_test_command(cmd_str)
                history = ctx.state.test_outcome_history.get(key, [])
                if not history or history[-1] != new_fp:
                    history.append(new_fp)
                    ctx.state.test_outcome_history[key] = history[-2:]

            try:
                from infinidev.engine.test_parsers import parse_test_failures
                failures = parse_test_failures(result)
            except Exception:
                failures = []
            if failures:
                import json as _json2
                _max = 8
                payload = [f.to_dict() for f in failures[:_max]]
                suffix = (
                    f"\n\n[auto-extracted structured_failures "
                    f"({len(failures)} total"
                    f"{', showing first ' + str(_max) if len(failures) > _max else ''}):]\n"
                    + _json2.dumps(payload, indent=2)
                )
                result = result + suffix

        return result

    def _build_assistant_message(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], llm_result: LLMCallResult,
    ) -> None:
        """Append the assistant message with tool_calls to the conversation."""
        message = llm_result.message
        raw_content = llm_result.raw_content
        if ctx.manual_tc:
            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", "") or raw_content,
            })
        else:
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            all_calls = list(classified.regular)
            for pseudo_tc in classified.thinks + classified.notes + classified.session_notes + ([classified.step_complete] if classified.step_complete else []):
                all_calls.append(pseudo_tc)
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in all_calls
            ]
            messages.append(assistant_msg)

    def _append_pseudo_tool_results(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], tool_results_text: list[str] | None = None,
    ) -> None:
        """Append result messages for pseudo-tools (think, notes, step_complete)."""
        if ctx.manual_tc:
            texts = tool_results_text if tool_results_text is not None else []
            for nc in classified.notes:
                texts.append('[Tool: add_note] Result:\n{"status": "noted"}')
            for snc in classified.session_notes:
                texts.append('[Tool: add_session_note] Result:\n{"status": "noted"}')
            for tk in classified.thinks:
                texts.append('[Tool: think] Result:\n{"status": "acknowledged"}')
            if texts:
                messages.append({"role": "user", "content": "\n\n".join(texts)})
        else:
            for tk in classified.thinks:
                messages.append({"role": "tool", "tool_call_id": tk.id, "content": '{"status": "acknowledged"}'})
            for nc in classified.notes:
                messages.append({"role": "tool", "tool_call_id": nc.id, "content": '{"status": "noted"}'})
            for snc in classified.session_notes:
                messages.append({"role": "tool", "tool_call_id": snc.id, "content": '{"status": "noted"}'})
            if classified.step_complete:
                messages.append({"role": "tool", "tool_call_id": classified.step_complete.id, "content": '{"status": "acknowledged"}'})

    def _execute_tool_batches(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]],
        action_tool_calls: int, iteration: int, guard: LoopGuard,
        tracker: BehaviorTracker, tool_results_text: list[str],
    ) -> int:
        """Execute tool batches and process results. Returns updated action_tool_calls."""
        _tool_hook_meta = {
            "agent_name": ctx.agent_name, "iteration": iteration,
            "verbose": ctx.verbose, "tokens_total": ctx.state.total_tokens,
            "prompt_tokens": ctx.state.last_prompt_tokens,
            "completion_tokens": ctx.state.last_completion_tokens,
        }
        batches = _batch_tool_calls(classified.regular)

        # Collected per-batch: maps tc.id -> list[ImageAttachment] for tools
        # (e.g. view_image) that returned a multimodal ToolResult. These get
        # pushed as a follow-up role=user multimodal message in
        # _process_tool_results when the model supports vision.
        attachments_by_tc: dict[str, list] = {}

        for batch in batches:
            is_parallel = len(batch) > 1 and batch[0].function.name not in _WRITE_TOOLS

            if is_parallel:
                _tool_hook_meta["call_num"] = action_tool_calls + 1
                _tool_hook_meta["total_calls"] = ctx.state.total_tool_calls + 1
                _tool_hook_meta["project_id"] = ctx.project_id
                _tool_hook_meta["agent_id"] = ctx.agent_id
                batch_results = _execute_tool_calls_parallel(
                    batch, ctx.tool_dispatch,
                    hook_metadata=_tool_hook_meta,
                    attachments_by_tc=attachments_by_tc,
                )
            else:
                batch_results = []
                for _bi, tc in enumerate(batch):
                    _pre = _capture_pre_content(tc.function.name, tc.function.arguments, ctx.file_tracker)
                    _tool_hook_meta["call_num"] = action_tool_calls + _bi + 1
                    _tool_hook_meta["total_calls"] = ctx.state.total_tool_calls + _bi + 1
                    _tool_hook_meta["project_id"] = ctx.project_id
                    _tool_hook_meta["agent_id"] = ctx.agent_id
                    att_list: list = []
                    attachments_by_tc[tc.id] = att_list
                    result = execute_tool_call(
                        ctx.tool_dispatch, tc.function.name, tc.function.arguments,
                        hook_metadata=_tool_hook_meta,
                        attachments_out=att_list,
                    )
                    _maybe_emit_file_change(tc.function.name, tc.function.arguments, result, _pre, ctx.file_tracker, ctx.project_id, ctx.agent_id, self._hooks)
                    batch_results.append((tc, result))

            if self._cancel_event.is_set():
                if not ctx.manual_tc:
                    _executed_ids = {tc.id for tc, _ in batch_results}
                    for tc in batch:
                        if tc.id not in _executed_ids:
                            messages.append({"role": "tool", "tool_call_id": tc.id,
                                             "content": '{"error": "cancelled"}'})
                break

            action_tool_calls = self._process_tool_results(
                ctx, batch_results, messages, action_tool_calls,
                iteration, is_parallel, guard, tracker, tool_results_text,
                attachments_by_tc=attachments_by_tc,
            )
            ctx.state.tick_opened_files(1)

        # Cancelled: placeholder results for unreached batches
        if self._cancel_event.is_set() and not ctx.manual_tc:
            _has_result = {
                msg["tool_call_id"]
                for msg in messages
                if msg.get("role") == "tool" and "tool_call_id" in msg
            }
            for tc in classified.regular:
                if tc.id not in _has_result:
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": '{"error": "cancelled"}'})

        return action_tool_calls

    def _process_tool_results(
        self, ctx: ExecutionContext,
        batch_results: list[tuple], messages: list[dict[str, Any]],
        action_tool_calls: int, iteration: int, is_parallel: bool,
        guard: LoopGuard, tracker: BehaviorTracker,
        tool_results_text: list[str],
        *,
        attachments_by_tc: dict | None = None,
    ) -> int:
        """Process results from one batch. Returns updated action_tool_calls.

        Image-attachment messages are collected during the per-tool loop
        and flushed AFTER all tool results are appended. Some providers
        (e.g. Minimax) reject conversations where a ``user``-role message
        appears between an ``assistant`` with ``tool_calls`` and the
        ``tool`` result for one of those calls — strict
        "tool result must follow tool call" ordering. Deferring keeps
        the assistant→tool block contiguous regardless of provider.
        """
        _pending_nudge: str | None = None
        _pending_image_msgs: list[dict[str, Any]] = []
        for tc, result in batch_results:
            _tool_error = _extract_tool_error(result)
            guard.on_tool_result(tc.function.name, tc.function.arguments, bool(_tool_error))
            tracker.on_tool_call(tc.function.name, tc.function.arguments, bool(_tool_error))

            if not _tool_error:
                _update_opened_files_cache(ctx.state, tc.function.name, tc.function.arguments, result)
                ToolProcessor.auto_note_for_small(ctx, tc.function.name, tc.function.arguments, result)

            counter_tag = f"\n[Tool call {action_tool_calls + 1}/{ctx.max_per_action} for this step]"
            if is_parallel:
                counter_tag += " (parallel)"

            if tc.function.name == "execute_command":
                result = self._capture_test_command_output(ctx, tc.function.arguments, result)

            if not _tool_error:
                with best_effort("memory annotation failed for %s", tc.function.name):
                    from infinidev.engine.tool_executor import annotate_with_memory
                    result = annotate_with_memory(
                        tc.function.name, tc.function.arguments, result,
                        project_id=ctx.project_id,
                    )

            behavior_feedback = tracker.drain_feedback()
            result_with_counter = result + counter_tag
            if behavior_feedback:
                result_with_counter += f"\n{behavior_feedback}"

            if ctx.manual_tc:
                tool_results_text.append(f"[Tool: {tc.function.name}] Result:\n{result_with_counter}")
            else:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_with_counter})

            # Push any image attachments produced by this tool as a
            # follow-up role=user multimodal message. Many providers
            # reject content-blocks inside role=tool messages, so a
            # separate user message is the most portable path. Only
            # emitted when the model supports vision and we're not in
            # manual (text-only) tool-calling mode.
            if (
                not ctx.manual_tc
                and attachments_by_tc is not None
            ):
                _atts = attachments_by_tc.get(tc.id) or []
                if _atts:
                    try:
                        from infinidev.config.model_capabilities import (
                            _detect_vision_support,
                        )
                        _supports_vision = _detect_vision_support()
                    except Exception:
                        _supports_vision = False
                    if _supports_vision:
                        _blocks = [
                            {
                                "type": "text",
                                "text": (
                                    f"[Images attached by tool "
                                    f"`{tc.function.name}` — {len(_atts)} "
                                    f"image(s)]"
                                ),
                            }
                        ]
                        for _att in _atts:
                            _blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _att.data_url},
                                }
                            )
                        # Defer until after all tool results in this
                        # batch are appended — see docstring.
                        _pending_image_msgs.append(
                            {"role": "user", "content": _blocks}
                        )

            action_tool_calls += 1
            ctx.state.total_tool_calls += 1
            ctx.state.tool_calls_since_last_note += 1

            with best_effort("ContextRank tool call log failed"):
                self._cr_hooks.on_tool_call(
                    tc.function.name, tc.function.arguments, iteration,
                    was_error=bool(_tool_error),
                )

            # Budget nudge (deferred until after all results in batch)
            _default_nudge = 4 if ctx.is_small else _get_settings().LOOP_STEP_NUDGE_THRESHOLD
            _nudge_threshold = self._nudge_threshold_override if self._nudge_threshold_override is not None else _default_nudge
            if _nudge_threshold > 0 and action_tool_calls == _nudge_threshold:
                if ctx.nudge_message_template:
                    _pending_nudge = ctx.nudge_message_template.format(
                        used=action_tool_calls, threshold=_nudge_threshold,
                    )
                else:
                    _active_desc = ctx.state.plan.active_step.title if ctx.state.plan.active_step else ""
                    _pending_nudge = (
                        f"You have used {action_tool_calls}/{ctx.max_per_action} tool calls for this step. "
                        f"Step scope: \"{_active_desc}\". "
                        f"Call step_complete now. If the step is not finished, set status=\'continue\' "
                        f"and add/modify next_steps to capture the remaining work."
                    )

        # Flush any deferred image-attachment messages now that ALL
        # tool results in this batch have been appended. This preserves
        # the strict assistant(tool_calls) → tool(...) ordering required
        # by Minimax and similar providers.
        for _img_msg in _pending_image_msgs:
            messages.append(_img_msg)

        if _pending_nudge is not None:
            if ctx.manual_tc:
                tool_results_text.append(f"\n⚠ STEP BUDGET: {_pending_nudge}")
            else:
                messages.append({"role": "user", "content": _pending_nudge})

        return action_tool_calls

    def _execute_regular_tools(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], llm_result: LLMCallResult,
        action_tool_calls: int, iteration: int, guard: LoopGuard,
        tracker: BehaviorTracker,
    ) -> int:
        """Execute regular tool calls and build messages. Returns updated action_tool_calls."""
        self._build_assistant_message(ctx, classified, messages, llm_result)

        tool_results_text: list[str] = []
        action_tool_calls = self._execute_tool_batches(
            ctx, classified, messages, action_tool_calls, iteration,
            guard, tracker, tool_results_text,
        )

        if ctx.is_small:
            ContextManager.compact_for_small(messages)

        self._append_pseudo_tool_results(ctx, classified, messages, tool_results_text)
        return action_tool_calls

    def _build_pseudo_only_messages(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], llm_result: LLMCallResult,
    ) -> None:
        """Build messages when only pseudo-tools were called (no regular tools)."""
        self._build_assistant_message(ctx, classified, messages, llm_result)
        self._append_pseudo_tool_results(ctx, classified, messages)

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
        if guardrail is None:
            return result

        for attempt in range(max_retries):
            try:
                validation = guardrail(result)
                # CrewAI guardrail convention: returns (success, result_or_feedback)
                if isinstance(validation, tuple):
                    success, feedback = validation
                    if success:
                        return result
                    # Retry with feedback
                    logger.info(
                        "Guardrail failed (attempt %d/%d): %s",
                        attempt + 1, max_retries, str(feedback)[:200],
                    )
                    feedback_prompt = (
                        f"Your previous output was rejected by validation.\n"
                        f"Feedback: {feedback}\n\n"
                        f"Please fix your output and try again.\n\n"
                        f"Previous output:\n{result}"
                    )
                    messages: list[dict[str, Any]] = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": feedback_prompt},
                    ]

                    # Run inner loop for retry
                    step_text = ""
                    action_tool_calls = 0
                    while action_tool_calls < max_per_action:
                        response = _call_llm(
                            llm_params, messages,
                            tool_schemas if tool_schemas else None,
                        )
                        choice = response.choices[0]
                        msg = choice.message
                        tc_list = getattr(msg, "tool_calls", None)
                        if tc_list:
                            assistant_msg: dict[str, Any] = {
                                "role": "assistant",
                                "content": msg.content or "",
                            }
                            assistant_msg["tool_calls"] = [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tc_list
                            ]
                            messages.append(assistant_msg)
                            for tc in tc_list:
                                if tc.function.name == "step_complete":
                                    # Parse final answer from step_complete
                                    sr = _parse_step_complete_args(tc.function.arguments)
                                    step_text = sr.final_answer or sr.summary
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": '{"status": "acknowledged"}',
                                    })
                                    break
                                _pre_content_g = _capture_pre_content(
                                    tc.function.name, tc.function.arguments, ctx.file_tracker,
                                )
                                tc_result = execute_tool_call(
                                    tool_dispatch,
                                    tc.function.name,
                                    tc.function.arguments,
                                )
                                _maybe_emit_file_change(
                                    tc.function.name, tc.function.arguments, tc_result,
                                    _pre_content_g, ctx.file_tracker,
                                    ctx.project_id, ctx.agent_id, self._hooks,
                                )
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tc_result,
                                })
                                action_tool_calls += 1
                            if step_text:
                                break
                        else:
                            step_text = msg.content or ""
                            break

                    result = step_text or result
                else:
                    # Simple bool guardrail
                    if validation:
                        return result
            except Exception as exc:
                logger.warning("Guardrail raised exception: %s", exc)

        return result
