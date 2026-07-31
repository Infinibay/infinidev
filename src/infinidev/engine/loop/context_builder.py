"""Assembling what a run needs, and what each iteration says.

Two jobs that look alike and are not.

``build_execution_context`` runs **once per task**: it resolves the model's
capabilities, picks the toolset, composes the system prompt and either
restores a persisted ``LoopState`` or starts a fresh one. Everything it
decides is fixed for the rest of the run.

``build_iteration_messages`` runs **once per step**, and its whole job is
deciding what a fresh two-message conversation should contain. The loop is
plan-execute-summarize, not ReAct: each iteration rebuilds the prompt from
nothing rather than appending to a growing history, which is why this reads
as construction rather than mutation.

What that second function mostly encodes is *what to leave out*. History is
trimmed to a window. ContextRank only recomputes at pivots. Attachments ride
along on the first turn only. Each of those is a place where sending
everything would be correct and wasteful.
"""

from __future__ import annotations

import logging
from typing import Any

from infinidev.config.llm import _is_small_model, get_litellm_params
from infinidev.config.model_capabilities import get_model_capabilities
from infinidev.config.settings import settings
from infinidev.engine._best_effort import best_effort
from infinidev.engine.engine_logging import log_start
from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.loop.context import (
    build_iteration_prompt,
    build_system_prompt,
    build_tools_prompt_section,
)
from infinidev.engine.loop.execution_context import ExecutionContext
from infinidev.engine.loop.model_context import _get_model_max_context
from infinidev.engine.loop.models import LoopState
from infinidev.engine.tool_dispatch import (
    ADD_NOTE_SCHEMA,
    ADD_STEP_SCHEMA,
    MODIFY_STEP_SCHEMA,
    REMOVE_STEP_SCHEMA,
    STEP_COMPLETE_SCHEMA,
    build_tool_dispatch,
    build_tool_schemas,
)
from infinidev.tools.base.context import bind_tools_to_agent, get_context_for_agent

logger = logging.getLogger(__name__)


# ── once per run ─────────────────────────────────────────────────────────


def build_execution_context(
    engine: Any, agent: Any, task_prompt: tuple[str, str], **kwargs: Any,
) -> ExecutionContext:
    """Resolve everything a run needs and freeze it into one context.

    Mutates *engine* only for the state a run owns: the file tracker the
    reporting later reads, and the two overrides the loop consults while it
    runs.
    """
    llm_params = get_litellm_params()
    if llm_params is None:
        raise RuntimeError(
            "LoopEngine requires LiteLLM parameters. "
            "Ensure INFINIDEV_LLM_MODEL is set."
        )

    max_iterations = kwargs.get("max_iterations") or settings.LOOP_MAX_ITERATIONS
    max_total_calls = (
        kwargs.get("max_total_tool_calls") or settings.LOOP_MAX_TOTAL_TOOL_CALLS
    )
    # An unset per-step budget means "the whole task budget", not "zero".
    max_per_action = (
        kwargs.get("max_tool_calls_per_action")
        or settings.LOOP_MAX_TOOL_CALLS_PER_ACTION
    ) or max_total_calls

    engine._nudge_threshold_override = kwargs.get("nudge_threshold")
    engine._summarizer_override = kwargs.get("summarizer_enabled")

    file_tracker = FileChangeTracker()
    engine._last_file_tracker = file_tracker
    engine._last_total_tool_calls = 0

    caps = get_model_capabilities()
    manual_tc = not caps.supports_function_calling
    is_small = _is_small_model()

    tools = _resolve_tools(agent, kwargs.get("task_tools"), is_small)
    tool_schemas = (
        build_tool_schemas(tools, small_model=is_small)
        if tools
        else [STEP_COMPLETE_SCHEMA]
    )
    tool_dispatch = build_tool_dispatch(tools) if tools else {}

    system_prompt = build_system_prompt(
        agent.backstory,
        tech_hints=getattr(agent, "_tech_hints", None),
        session_summaries=getattr(agent, "_session_summaries", None),
        identity_override=getattr(agent, "_system_prompt_identity", None),
        protocol_override=getattr(agent, "_system_prompt_protocol", None),
        small_model=is_small,
        workspace_path=getattr(agent, "workspace_path", None),
    )
    if manual_tc:
        # No function-calling support: the schemas have to be described in
        # prose, because there is no channel to pass them through.
        system_prompt += "\n\n" + build_tools_prompt_section(
            tool_schemas, small_model=is_small,
        )
        logger.info(
            "LoopEngine [%s]: manual tool calling mode",
            getattr(agent, "agent_id", "?"),
        )

    desc, expected = task_prompt
    event_id, resume_state = _resolve_resume(agent, kwargs)
    state = _restore_or_start(resume_state)

    if kwargs.get("verbose", True):
        log_start(
            agent.agent_id,
            getattr(agent, "name", agent.agent_id),
            getattr(agent, "role", "agent"),
            desc,
            len(tools),
        )

    return ExecutionContext(
        llm_params=llm_params, manual_tc=manual_tc, is_small=is_small,
        system_prompt=system_prompt, tool_schemas=tool_schemas,
        tool_dispatch=tool_dispatch,
        planning_schemas=[
            ADD_STEP_SCHEMA, MODIFY_STEP_SCHEMA, REMOVE_STEP_SCHEMA,
            ADD_NOTE_SCHEMA, STEP_COMPLETE_SCHEMA,
        ],
        tools=tools, max_iterations=max_iterations,
        max_per_action=max_per_action, max_total_calls=max_total_calls,
        history_window=settings.LOOP_HISTORY_WINDOW,
        max_context_tokens=_get_model_max_context(llm_params),
        verbose=kwargs.get("verbose", True),
        guardrail=kwargs.get("guardrail"),
        guardrail_max_retries=kwargs.get("guardrail_max_retries", 5),
        output_pydantic=kwargs.get("output_pydantic"),
        agent=agent, agent_name=getattr(agent, "name", agent.agent_id),
        agent_role=getattr(agent, "role", "agent"),
        desc=desc, expected=expected, event_id=event_id,
        skip_plan=False,
        nudge_message_template=kwargs.get("nudge_message_template"),
        state=state, file_tracker=file_tracker,
        start_iteration=state.iteration_count,
        task=kwargs.get("task"),
    )


def _resolve_tools(agent: Any, task_tools: list | None, is_small: bool) -> list:
    """Pick the toolset and make sure it is bound to this agent.

    Binding matters more than it looks: without it a tool falls back to a
    thread-local lookup for its agent id, which is unreliable once hooks or
    worker threads change context — the symptom was intermittent "No active
    plan context" errors from the plan tools.
    """
    if task_tools is not None:
        bind_tools_to_agent(task_tools, agent.agent_id)
        return task_tools

    if is_small:
        logger.info(
            "LoopEngine: small model detected — "
            "using simplified prompts and reduced tools"
        )
        from infinidev.tools import get_tools_for_role

        tools = get_tools_for_role("developer", small_model=True)
        bind_tools_to_agent(tools, agent.agent_id)
        return tools

    return getattr(agent, "tools", [])


def _resolve_resume(agent: Any, kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """Read resume coordinates from the call, falling back to tool context."""
    event_id = kwargs.get("event_id")
    resume_state = kwargs.get("resume_state")
    if event_id is not None and resume_state is not None:
        return event_id, resume_state
    tool_ctx = get_context_for_agent(agent.agent_id)
    if tool_ctx:
        event_id = event_id or tool_ctx.event_id
        resume_state = resume_state or tool_ctx.resume_state
    return event_id, resume_state


def _restore_or_start(resume_state: dict | None) -> LoopState:
    """Rebuild a persisted state, or begin a new one.

    A resumed plan can come back with every step pending — the run died
    between finishing one step and activating the next — so the first
    pending step is re-activated. Without that the loop would resume with
    nowhere to work.
    """
    if not resume_state:
        return LoopState()

    state = LoopState.model_validate(resume_state)
    if state.plan.steps and not state.plan.active_step:
        for step in state.plan.steps:
            if step.status == "pending":
                step.status = "active"
                break
    logger.info("LoopEngine: resuming from iteration %d", state.iteration_count)
    return state


# ── once per iteration ───────────────────────────────────────────────────


def build_iteration_messages(
    engine: Any, ctx: ExecutionContext, iteration: int,
) -> list[dict[str, Any]]:
    """Build the fresh two-message conversation for one iteration."""
    effective_state = ctx.state
    if ctx.history_window > 0 and len(ctx.state.history) > ctx.history_window:
        effective_state = ctx.state.model_copy(deep=True)
        effective_state.history = ctx.state.history[-ctx.history_window :]

    first_turn = iteration == ctx.start_iteration
    if first_turn:
        try:
            from infinidev.db.service import get_project_knowledge

            engine._project_knowledge = get_project_knowledge(
                project_id=ctx.project_id,
            )
        except Exception:
            engine._project_knowledge = []

    injected = engine._drain_user_messages()

    from infinidev.engine.static_analysis_timer import measure

    with measure("prompt_build"):
        user_prompt = build_iteration_prompt(
            ctx.desc, ctx.expected, effective_state,
            # Fetched once (hence the cache on the engine), rendered every
            # iteration. Each iteration builds a brand-new two-message
            # conversation, so "the model already saw it" is never true
            # here — dropping the block after turn one simply deleted the
            # project's facts from the model's context.
            project_knowledge=engine._project_knowledge,
            context_rank_result=_rank_at_pivot(engine, ctx, iteration),
            max_context_tokens=ctx.max_context_tokens,
            session_notes=engine.session_notes or None,
            user_messages=injected or None,
            skip_plan=ctx.skip_plan,
            task=ctx.task,
            small_model=ctx.is_small,
        )

    return [
        {"role": "system", "content": ctx.system_prompt},
        {"role": "user", "content": _with_attachments(engine, ctx, user_prompt, first_turn)},
    ]


def _rank_at_pivot(engine: Any, ctx: ExecutionContext, iteration: int) -> Any | None:
    """Recompute the context ranking, but only where it can have changed.

    A pivot is the first iteration or a change of active step. Between
    pivots the ranking would come back the same, so recomputing it is
    wasted work — but the *block* is still re-sent from cache, because
    each iteration builds a fresh conversation and a block that is not
    rendered is a block the model cannot see.
    """
    result = None
    with best_effort("ContextRank ranking failed"):
        if not (settings.CONTEXT_RANK_ENABLED and engine._cr_hooks._enabled):
            return None
        active = ctx.state.plan.active_step
        pivot_key = (active.index, active.title) if active else (-1, "")
        if iteration != ctx.start_iteration and pivot_key == engine._cr_last_pivot_key:
            return engine._cr_cached_result

        from infinidev.engine.context_rank.ranker import rank

        result = rank(
            ctx.desc,
            engine._cr_hooks._session_id,
            engine._cr_hooks._task_id,
            iteration,
            cached_embedding=engine._cr_hooks._task_embedding,
            cached_simplified_embedding=engine._cr_hooks._task_embedding_simplified,
            project_id=ctx.project_id,
        )
        engine._cr_cached_result = result
        engine._cr_last_pivot_key = pivot_key
    return result


def _with_attachments(
    engine: Any, ctx: ExecutionContext, user_prompt: str, first_turn: bool,
) -> Any:
    """Attach images to the first user turn, and only that one.

    Later turns rebuild the prompt from compact summaries; re-sending the
    base64 payload every iteration would bloat the context and, on a billed
    vision provider, multiply the cost of the whole run. A model that cannot
    see gets the paths mentioned as text instead, which at least lets it
    reason about which files were offered.
    """
    attachments = getattr(engine, "_initial_attachments", None) or []
    if not attachments or not first_turn:
        return user_prompt

    if engine._supports_vision_cached is None:
        try:
            from infinidev.config.model_capabilities import _detect_vision_support

            engine._supports_vision_cached = _detect_vision_support()
        except Exception:
            engine._supports_vision_cached = False

    from infinidev.engine.multimodal import build_user_content, mention_paths_as_text

    if engine._supports_vision_cached:
        return build_user_content(user_prompt, attachments)
    return mention_paths_as_text(user_prompt, attachments)
