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
        self._cancel_event: __import__('threading').Event = __import__('threading').Event()
        self.session_notes: list[str] = []  # Persist across tasks within a session
        # Thread-safe queue for user messages injected mid-task
        import queue as _queue_mod
        self._user_messages: _queue_mod.Queue[str] = _queue_mod.Queue()

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

    def get_file_contents(self) -> dict[str, str]:
        """Return path → current content for each changed file."""
        import os as _os
        if self._last_file_tracker is None:
            return {}
        result = {}
        for path in self._last_file_tracker.get_all_paths():
            try:
                if _os.path.isfile(path) and _os.path.getsize(path) <= _MAX_TRACK_FILE_SIZE:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        result[path] = f.read()
            except Exception:
                pass
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
        summarizer_enabled: bool | None = None,
    ) -> str:
        """Plan-execute-summarize loop.

        Delegates to composition components: LLMCaller, ToolProcessor,
        LoopGuard, StepManager. See class docstrings for details.
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
            summarizer_enabled=summarizer_enabled,
        )
        # Streaming callback: emit thinking chunks to the UI via event bus
        def _on_thinking(text: str) -> None:
            _emit_loop_event("loop_thinking_chunk", ctx.project_id, ctx.agent_id, {
                "text": text,
            })

        def _on_stream_status(phase: str, token_count: int, tool_name: str | None) -> None:
            _emit_loop_event("loop_stream_status", ctx.project_id, ctx.agent_id, {
                "phase": phase,
                "token_count": token_count,
                "tool_name": tool_name,
            })

        llm_caller = LLMCaller(
            on_thinking_chunk=_on_thinking,
            on_stream_status=_on_stream_status,
        )
        tool_proc = ToolProcessor()
        guard = LoopGuard(is_small=ctx.is_small)
        step_mgr = StepManager(self)

        self._cancel_event.clear()
        self._last_state = ctx.state  # Make state available for live introspection

        # Attach loop state to tool context so plan tools can modify the plan
        from infinidev.tools.base.context import set_loop_state
        set_loop_state(ctx.agent_id, ctx.state)
        _hook_manager.dispatch(_HookContext(
            event=_HookEvent.LOOP_START,
            metadata={"task_prompt": task_prompt, "tools": ctx.tools, "state": ctx.state},
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        ))

        consecutive_all_done = 0

        # Reset the static-analysis latency accumulator at the start of
        # each run so the final summary reports just this task's costs.
        try:
            from infinidev.engine.static_analysis_timer import reset as _sa_reset
            _sa_reset()
        except Exception:
            pass

        try:
            _trace_run_start(
                model=str(ctx.llm_params.get("model", "?")),
                task=ctx.desc,
                expected=ctx.expected,
                settings_snapshot={
                    "is_small": ctx.is_small,
                    "manual_tc": ctx.manual_tc,
                    "max_iterations": ctx.max_iterations,
                    "max_per_action": ctx.max_per_action,
                    "max_total_calls": ctx.max_total_calls,
                    "history_window": ctx.history_window,
                    "max_context_tokens": ctx.max_context_tokens,
                },
            )
        except Exception:
            pass

        for iteration in range(ctx.start_iteration, ctx.max_iterations):
            if self._cancel_event.is_set():
                logger.info("LoopEngine: cancelled by user")
                _emit_log("info", f"{_YELLOW}⚠ Task cancelled by user{_RESET}",
                          project_id=ctx.project_id, agent_id=ctx.agent_id)
                break

            ctx.state.iteration_count = iteration + 1
            messages = self._build_iteration_messages(ctx, iteration)
            try:
                _trace_iter_prompt(iteration + 1, messages[0].get("content", ""), messages[1].get("content", ""))
            except Exception:
                pass

            # Log step start
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

            # ── Inner loop ──────────────────────────────────────────
            # Capture the message-buffer offset *before* this step runs so
            # POST_STEP consumers (e.g. the behavior scorer) can slice out
            # exactly the messages produced during this step.
            step_messages_start = len(messages)
            step_result = self._run_inner_loop(ctx, messages, iteration, llm_caller, tool_proc, guard)

            # Track consecutive text-only iterations across the outer loop
            action_tc = step_result.action_tool_calls
            if action_tc == 0:
                guard.mark_text_only_iteration()
                if guard.text_only_iterations >= 3:
                    _emit_log("error",
                              f"{_RED}⚠ Model failed to produce tool calls for "
                              f"{guard.text_only_iterations} consecutive iterations "
                              f"— aborting task{_RESET}",
                              project_id=ctx.project_id, agent_id=ctx.agent_id)
                    return step_mgr.finish(ctx, "blocked", iteration,
                                           "Model unable to produce function calls after multiple attempts.")
            else:
                guard.mark_productive_iteration()

            # ── Post-step processing ────────────────────────────────
            step_result = step_mgr.auto_split(ctx, step_result)

            _hook_manager.dispatch(_HookContext(
                event=_HookEvent.STEP_TRANSITION,
                metadata={"step_result": step_result, "plan": ctx.state.plan, "iteration": iteration},
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            ))

            step_mgr.advance_plan(ctx, step_result)

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

            # Reactive guidance: at the end of each step, look at the
            # messages produced during this step and queue pre-baked
            # how-to advice if a stuck-pattern is detected. Only fires
            # for small models; never costs an LLM call. The advice
            # is rendered into the *next* iteration's prompt.
            try:
                _settings = _get_settings()
                if (ctx.is_small
                    and getattr(_settings, "LOOP_GUIDANCE_ENABLED", True)):
                    # We always run the detector, regardless of step
                    # status. The original guard skipped done/blocked
                    # steps to avoid wasting guidance on a finished
                    # task, but that bypassed the *proactive* detectors
                    # (e.g. first_test_run) on the very last step where
                    # the model would still benefit from seeing the
                    # advice if a continuation or rework loop fires.
                    # Reactive detectors (stuck_on_*) self-suppress on
                    # done/blocked because their patterns can't match
                    # in a single completed step's history anyway.
                    from infinidev.engine.guidance import maybe_queue_guidance
                    queued = maybe_queue_guidance(
                        ctx.state,
                        messages[step_messages_start:],
                        is_small=True,
                        max_per_task=int(getattr(_settings, "LOOP_GUIDANCE_MAX_PER_TASK", 3)),
                    )
                    if queued and ctx.verbose:
                        _log(f"  {_YELLOW}↪ guidance queued: {queued}{_RESET}")
            except Exception:
                pass

            _hook_manager.dispatch(_HookContext(
                event=_HookEvent.POST_STEP,
                metadata={
                    "iteration": iteration, "step_result": step_result,
                    "record": ctx.state.history[-1] if ctx.state.history else None,
                    "state": ctx.state, "agent_name": ctx.agent_name,
                    "action_tool_calls": action_tool_calls,
                    "messages": messages,
                    "step_messages_start": step_messages_start,
                },
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            ))

            if ctx.event_id:
                self._checkpoint(ctx.event_id, ctx.state)

            # ── Check termination ───────────────────────────────────
            if step_result.status == "explore":
                self._handle_explore(ctx, step_result, iteration)
                consecutive_all_done = 0
                continue

            if step_result.status == "done":
                if not step_result.final_answer and iteration == ctx.start_iteration:
                    _emit_log("warning",
                              f"{_YELLOW}⚠ LLM declared done on first step without final_answer — forcing continue{_RESET}",
                              project_id=ctx.project_id, agent_id=ctx.agent_id)
                    step_result = StepResult(summary=step_result.summary, status="continue", next_steps=step_result.next_steps)
                else:
                    result = step_result.final_answer or step_result.summary
                    result = step_mgr.finish(ctx, "done", iteration, result)
                    return self._apply_guardrail(
                        result, ctx.guardrail, ctx.guardrail_max_retries,
                        ctx.llm_params, ctx.system_prompt, ctx.desc, ctx.expected,
                        ctx.state, ctx.tool_schemas, ctx.tool_dispatch,
                        max_per_action=ctx.max_per_action,
                    )

            if step_result.status == "blocked":
                return step_mgr.finish(ctx, "blocked", iteration, step_result.summary)

            # Safety: consecutive all-done detection
            if ctx.state.plan.steps and not ctx.state.plan.has_pending:
                consecutive_all_done += 1
                if consecutive_all_done >= 2:
                    result = step_mgr.finish(ctx, "done", iteration, step_result.summary)
                    return self._apply_guardrail(
                        result, ctx.guardrail, ctx.guardrail_max_retries,
                        ctx.llm_params, ctx.system_prompt, ctx.desc, ctx.expected,
                        ctx.state, ctx.tool_schemas, ctx.tool_dispatch,
                        max_per_action=ctx.max_per_action,
                    )
            else:
                consecutive_all_done = 0

        # Outer loop exhausted
        return step_mgr.finish(ctx, "exhausted", ctx.max_iterations - 1)

    # ── Private helpers for execute() ───────────────────────────────────

    def _build_context(
        self, agent: Any, task_prompt: tuple[str, str], **kwargs: Any,
    ) -> ExecutionContext:
        """Build ExecutionContext from agent, task_prompt, and overrides."""
        from infinidev.config.llm import get_litellm_params, _is_small_model
        from infinidev.config.settings import settings
        from infinidev.config.model_capabilities import get_model_capabilities

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
            from infinidev.tools.base.context import bind_tools_to_agent
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
            from infinidev.tools.base.context import bind_tools_to_agent
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
            from infinidev.tools.base.context import get_context_for_agent
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
            skip_plan=bool(getattr(agent, '_system_prompt_protocol', None)),
            state=state, file_tracker=file_tracker,
            start_iteration=state.iteration_count,
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

        from infinidev.engine.static_analysis_timer import measure as _sa_measure
        with _sa_measure("prompt_build"):
            user_prompt = build_iteration_prompt(
                ctx.desc, ctx.expected, effective_state,
                project_knowledge=self._project_knowledge if iteration == ctx.start_iteration else None,
                max_context_tokens=ctx.max_context_tokens,
                session_notes=self.session_notes if self.session_notes else None,
                user_messages=injected if injected else None,
                skip_plan=ctx.skip_plan,
                small_model=ctx.is_small,
            )
        return [
            {"role": "system", "content": ctx.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _run_inner_loop(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        iteration: int,
        llm_caller: LLMCaller, tool_proc: ToolProcessor, guard: LoopGuard,
    ) -> StepResult:
        """Run the inner tool-calling loop for one step.

        Returns the StepResult for this step.
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
            except Exception:
                pass

            # Emit reasoning content in FC mode (no streaming available).
            # Send full reasoning to both THINKING panel and chat.
            if result.reasoning_content and not ctx.manual_tc:
                _emit_loop_event("loop_think", ctx.project_id, ctx.agent_id, {
                    "reasoning": result.reasoning_content,
                })

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
                    action_tool_calls = self._execute_regular_tools(
                        ctx, classified, messages, result, action_tool_calls, iteration, guard, tracker,
                    )
                    if self._cancel_event.is_set():
                        break
                    # Expire old thinking content to save context window
                    self._expire_thinking(messages)
                    # Check guard conditions
                    forced = guard.check_repetition(ctx, messages)
                    if forced:
                        step_result = forced
                        break
                    guard.check_error_circuit_breaker(ctx, messages)
                    guard.check_note_discipline(ctx, messages)
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
                            messages.append({"role": "tool", "tool_call_id": classified.step_complete.id,
                                             "content": nudge})
                        continue  # Don't break — let model add notes first
                    self._step_complete_gated = False
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

        if step_result is None:
            step_result = StepResult(summary="Step completed.", status="continue")

        # Run end-of-step behavior checks and propagate edit flag
        tracker.on_step_end()
        if tracker.task_has_edits:
            ctx.state.task_has_edits = True

        # Attach metadata for post-step processing
        step_result.action_tool_calls = action_tool_calls
        step_result.behavior_tracker = tracker
        return step_result

    def _execute_regular_tools(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], llm_result: LLMCallResult,
        action_tool_calls: int, iteration: int, guard: LoopGuard,
        tracker: BehaviorTracker,
    ) -> int:
        """Execute regular tool calls and build messages. Returns updated action_tool_calls."""
        message = llm_result.message
        raw_content = llm_result.raw_content

        if ctx.manual_tc:
            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", "") or raw_content,
            })
        else:
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in classified.regular
            ]
            for pseudo_tc in classified.thinks + classified.notes +([classified.step_complete] if classified.step_complete else []):
                assistant_msg["tool_calls"].append({
                    "id": pseudo_tc.id, "type": "function",
                    "function": {"name": pseudo_tc.function.name, "arguments": pseudo_tc.function.arguments},
                })
            messages.append(assistant_msg)

        tool_results_text: list[str] = []
        _tool_hook_meta = {
            "agent_name": ctx.agent_name, "iteration": iteration,
            "verbose": ctx.verbose, "tokens_total": ctx.state.total_tokens,
            "prompt_tokens": ctx.state.last_prompt_tokens,
            "completion_tokens": ctx.state.last_completion_tokens,
        }
        batches = _batch_tool_calls(classified.regular)

        for batch in batches:
            is_parallel = len(batch) > 1 and batch[0].function.name not in _WRITE_TOOLS

            if is_parallel:
                _tool_hook_meta["call_num"] = action_tool_calls + 1
                _tool_hook_meta["total_calls"] = ctx.state.total_tool_calls + 1
                _tool_hook_meta["project_id"] = ctx.project_id
                _tool_hook_meta["agent_id"] = ctx.agent_id
                batch_results = _execute_tool_calls_parallel(batch, ctx.tool_dispatch, hook_metadata=_tool_hook_meta)
            else:
                batch_results = []
                for _bi, tc in enumerate(batch):
                    _pre = _capture_pre_content(tc.function.name, tc.function.arguments, ctx.file_tracker)
                    _tool_hook_meta["call_num"] = action_tool_calls + _bi + 1
                    _tool_hook_meta["total_calls"] = ctx.state.total_tool_calls + _bi + 1
                    _tool_hook_meta["project_id"] = ctx.project_id
                    _tool_hook_meta["agent_id"] = ctx.agent_id
                    result = execute_tool_call(ctx.tool_dispatch, tc.function.name, tc.function.arguments, hook_metadata=_tool_hook_meta)
                    _maybe_emit_file_change(tc.function.name, tc.function.arguments, result, _pre, ctx.file_tracker, ctx.project_id, ctx.agent_id)
                    batch_results.append((tc, result))

            if self._cancel_event.is_set():
                break
            for tc, result in batch_results:
                _tool_error = _extract_tool_error(result)
                guard.on_tool_result(tc.function.name, tc.function.arguments, bool(_tool_error))
                tracker.on_tool_call(tc.function.name, tc.function.arguments, bool(_tool_error))

                if not _tool_error:
                    _update_opened_files_cache(ctx.state, tc.function.name, tc.function.arguments, result)
                    # Auto-note for small models on successful reads
                    ToolProcessor.auto_note_for_small(ctx, tc.function.name, tc.function.arguments, result)

                counter_tag = f"\n[Tool call {action_tool_calls + 1}/{ctx.max_per_action} for this step]"
                if is_parallel:
                    counter_tag += " (parallel)"

                # Capture test-runner output so the tail_test_output meta
                # tool can serve a filtered view without re-running the
                # tests. Also record the outcome fingerprint per
                # *normalised* test command so the regression_after_edit
                # detector can compare apples to apples (same target set,
                # different time) instead of mixing unrelated test runs.
                if tc.function.name == "execute_command":
                    try:
                        from infinidev.engine.guidance import (
                            is_test_command,
                            test_outcome_fingerprint,
                            normalize_test_command,
                        )
                        if is_test_command(tc.function.arguments, ctx.state):
                            ctx.state.last_test_output = result
                            try:
                                import json as _json
                                _args = _json.loads(tc.function.arguments) if tc.function.arguments else {}
                                cmd_str = str(_args.get("command", ""))
                            except Exception:
                                cmd_str = tc.function.arguments
                            ctx.state.last_test_command = cmd_str[:300]
                            # Record the outcome under the normalised
                            # command key so regression_after_edit can
                            # compare against the previous run of the
                            # SAME target set. We keep the LAST TWO
                            # entries per command — older history is
                            # dropped to keep state small. Identical
                            # consecutive outcomes are not duplicated
                            # (e.g. running the same test twice in a
                            # row without an edit between).
                            new_fp = test_outcome_fingerprint(result)
                            if new_fp:
                                key = normalize_test_command(cmd_str)
                                history = ctx.state.test_outcome_history.get(key, [])
                                if not history or history[-1] != new_fp:
                                    history.append(new_fp)
                                    ctx.state.test_outcome_history[key] = history[-2:]
                    except Exception:
                        pass

                # Inject behavioral feedback into tool result
                behavior_feedback = tracker.drain_feedback()
                result_with_counter = result + counter_tag
                if behavior_feedback:
                    result_with_counter += f"\n{behavior_feedback}"

                if ctx.manual_tc:
                    tool_results_text.append(f"[Tool: {tc.function.name}] Result:\n{result_with_counter}")
                else:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_with_counter})

                action_tool_calls += 1
                ctx.state.total_tool_calls += 1
                ctx.state.tool_calls_since_last_note += 1

                # Budget nudge
                _default_nudge = 4 if ctx.is_small else _get_settings().LOOP_STEP_NUDGE_THRESHOLD
                _nudge_threshold = self._nudge_threshold_override if self._nudge_threshold_override is not None else _default_nudge
                if _nudge_threshold > 0 and action_tool_calls == _nudge_threshold:
                    _active_desc = ctx.state.plan.active_step.title if ctx.state.plan.active_step else ""
                    _nudge_msg = (
                        f"You have used {action_tool_calls}/{ctx.max_per_action} tool calls for this step. "
                        f"Step scope: \"{_active_desc}\". "
                        f"Call step_complete now. If the step is not finished, set status=\'continue\' "
                        f"and add/modify next_steps to capture the remaining work."
                    )
                    if ctx.manual_tc:
                        tool_results_text.append(f"\n⚠ STEP BUDGET: {_nudge_msg}")
                    else:
                        messages.append({"role": "user", "content": _nudge_msg})

                ctx.state.tick_opened_files(1)

        # Small model: compact old messages to prevent context bloat
        if ctx.is_small:
            self._compact_messages_for_small(messages)

        # Manual mode: send all results as single user message
        if ctx.manual_tc:
            for nc in classified.notes:
                tool_results_text.append('[Tool: add_note] Result:\n{"status": "noted"}')
            for snc in classified.session_notes:
                tool_results_text.append('[Tool: add_session_note] Result:\n{"status": "noted"}')
            for tk in classified.thinks:
                tool_results_text.append('[Tool: think] Result:\n{"status": "acknowledged"}')
            if tool_results_text:
                messages.append({"role": "user", "content": "\n\n".join(tool_results_text)})

        # FC mode: pseudo-tool results
        if not ctx.manual_tc:
            for tk in classified.thinks:
                messages.append({"role": "tool", "tool_call_id": tk.id, "content": '{"status": "acknowledged"}'})
            for nc in classified.notes:
                messages.append({"role": "tool", "tool_call_id": nc.id, "content": '{"status": "noted"}'})
            for snc in classified.session_notes:
                messages.append({"role": "tool", "tool_call_id": snc.id, "content": '{"status": "noted"}'})

            if classified.step_complete:
                messages.append({"role": "tool", "tool_call_id": classified.step_complete.id, "content": '{"status": "acknowledged"}'})

        return action_tool_calls

    def _build_pseudo_only_messages(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], llm_result: LLMCallResult,
    ) -> None:
        """Build messages when only pseudo-tools were called (no regular tools)."""
        message = llm_result.message
        raw_content = llm_result.raw_content

        if ctx.manual_tc:
            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", "") or raw_content,
            })
        else:
            assistant_msg = {"role": "assistant", "content": message.content or ""}
            pseudo_calls = classified.thinks + classified.notes + classified.session_notes +([classified.step_complete] if classified.step_complete else [])
            assistant_msg["tool_calls"] = [
                {"id": pc.id, "type": "function",
                 "function": {"name": pc.function.name, "arguments": pc.function.arguments}}
                for pc in pseudo_calls
            ]
            messages.append(assistant_msg)
            for tk in classified.thinks:
                messages.append({"role": "tool", "tool_call_id": tk.id, "content": '{"status": "acknowledged"}'})
            for nc in classified.notes:
                messages.append({"role": "tool", "tool_call_id": nc.id, "content": '{"status": "noted"}'})
            for snc in classified.session_notes:
                messages.append({"role": "tool", "tool_call_id": snc.id, "content": '{"status": "noted"}'})

            if classified.step_complete:
                messages.append({"role": "tool", "tool_call_id": classified.step_complete.id, "content": '{"status": "acknowledged"}'})

    # How many tool call rounds before thinking content is truncated
    _THINKING_TTL = 3

    @staticmethod
    def _expire_thinking(messages: list[dict[str, Any]]) -> None:
        """Truncate old assistant thinking to save context window.

        Assistant messages carry a ``_thinking_age`` counter that increments
        each time this method is called.  Once a message is older than
        ``_THINKING_TTL`` rounds, its ``content`` (reasoning text) is
        replaced with a short placeholder — the tool_calls structure stays
        intact so the API conversation remains valid.

        For manual-TC mode (no tool_calls), the entire assistant content
        is the reasoning, so we truncate it to the first line.
        """
        ttl = LoopEngine._THINKING_TTL

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if not content or len(content) < 80:
                continue  # Already short, skip

            # Initialize or bump age counter
            age = msg.get("_thinking_age", 0) + 1
            msg["_thinking_age"] = age

            if age <= ttl:
                continue

            # Truncate — keep first line as summary
            first_line = content.split("\n", 1)[0][:120]
            if msg.get("tool_calls"):
                # FC mode: content is optional reasoning alongside tool calls
                msg["content"] = f"[thinking truncated] {first_line}"
            else:
                # Manual mode or text-only: content IS the reasoning
                msg["content"] = f"[thinking truncated] {first_line}"

    @staticmethod
    def _compact_messages_for_small(messages: list[dict[str, Any]]) -> None:
        """Compact old messages in the inner loop for small models.

        Small models have limited context.  This truncates tool result
        messages older than the last 2 assistant rounds to their first
        200 chars, preventing context bloat from large tool outputs.
        The system and first user message are always preserved.
        """
        # Count assistant messages from the end to find the cutoff
        assistant_count = 0
        cutoff_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                assistant_count += 1
                if assistant_count >= 2:
                    cutoff_idx = i
                    break

        # Truncate tool results before the cutoff (skip system + first user)
        for i in range(2, cutoff_idx):
            msg = messages[i]
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > 200:
                    msg["content"] = content[:200] + "\n[truncated for context]"
            elif msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content and len(content) > 100:
                    first_line = content.split("\n", 1)[0][:100]
                    msg["content"] = f"[compacted] {first_line}"

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
                                    tc.function.name, tc.function.arguments, file_tracker,
                                )
                                tc_result = execute_tool_call(
                                    tool_dispatch,
                                    tc.function.name,
                                    tc.function.arguments,
                                )
                                _maybe_emit_file_change(
                                    tc.function.name, tc.function.arguments, tc_result,
                                    _pre_content_g, file_tracker,
                                    agent.project_id, agent.agent_id,
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
