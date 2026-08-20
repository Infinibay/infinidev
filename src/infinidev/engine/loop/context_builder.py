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
from infinidev.engine.model_execution_policy import resolve_model_execution_policy
from infinidev.engine.tool_dispatch import (
    ADD_NOTE_SCHEMA,
    ADD_STEP_SCHEMA,
    MODIFY_STEP_SCHEMA,
    REMOVE_STEP_SCHEMA,
    STEP_COMPLETE_SCHEMA,
    build_tool_dispatch,
    build_tool_schemas,
)
from infinidev.prompts.profiles import EffectivePromptConfiguration
from infinidev.tools.base.context import bind_tools_to_agent, get_context_for_agent

logger = logging.getLogger(__name__)

_PLAN_MUTATION_TOOLS = {"add_step", "modify_step", "remove_step"}


# ── once per run ─────────────────────────────────────────────────────────


def _normalize_total_tool_budget(value: int) -> int | None:
    """Translate the public zero-is-unlimited convention into loop state."""
    budget = int(value)
    return budget if budget > 0 else None


def _resolve_identity_override(
    agent: Any,
    tools: list[Any],
    explicit_override: str | None,
) -> str | None:
    """Keep the developer core tool-aware without replacing custom identities."""
    if explicit_override is not None:
        return explicit_override

    existing = getattr(agent, "_system_prompt_identity", None)
    if getattr(agent, "role", "") != "developer":
        return existing

    from infinidev.prompts.flows.develop import DEVELOP_IDENTITY, get_develop_identity

    if existing not in {None, DEVELOP_IDENTITY}:
        return existing
    return get_develop_identity({tool.name for tool in tools})


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

    configured_iterations = kwargs.get("max_iterations")
    max_iterations = (
        settings.LOOP_MAX_ITERATIONS
        if configured_iterations is None
        else int(configured_iterations)
    )
    max_iterations = max_iterations if max_iterations > 0 else None
    configured_total_calls = kwargs.get("max_total_tool_calls")
    if configured_total_calls is None:
        configured_total_calls = settings.LOOP_MAX_TOTAL_TOOL_CALLS
    max_total_calls = _normalize_total_tool_budget(configured_total_calls)
    max_prompt_tokens = kwargs.get("max_prompt_tokens")
    if max_prompt_tokens is not None and max_prompt_tokens <= 0:
        max_prompt_tokens = None
    # Zero means no call-count boundary for a Step. Keep it explicit rather
    # than replacing it with a huge integer that leaks into prompts and UI.
    configured_per_action = kwargs.get("max_tool_calls_per_action")
    if configured_per_action is None:
        configured_per_action = settings.LOOP_MAX_TOOL_CALLS_PER_ACTION
    max_per_action = max(0, int(configured_per_action))
    step_tool_limit = max_per_action or None

    model_policy = resolve_model_execution_policy(
        settings.LLM_PROVIDER,
        str(llm_params.get("model", settings.LLM_MODEL)),
    )
    explicit_nudge = kwargs.get("nudge_threshold")
    engine._nudge_threshold_override = (
        explicit_nudge
        if explicit_nudge is not None
        else model_policy.step_nudge_threshold(
            max_tool_calls=max_per_action,
            configured_threshold=settings.LOOP_STEP_NUDGE_THRESHOLD,
        )
    )
    engine._summarizer_override = kwargs.get("summarizer_enabled")

    previous = getattr(engine, "_last_file_tracker", None)
    if kwargs.get("preserve_file_tracker") and previous is not None:
        baseline = previous.baseline
    else:
        workspace = getattr(agent, "workspace_path", None)
        if not workspace:
            from infinidev.tools.base.context import get_current_workspace_path

            workspace = get_current_workspace_path()
        from infinidev.engine.workspace_baseline import WorkspaceBaseline

        baseline = WorkspaceBaseline.capture(workspace)
    file_tracker = FileChangeTracker(baseline=baseline)
    # Opt-in carry-forward, for callers that re-enter execute() inside one
    # user turn (the review's rework loop). Never automatic: the engine is
    # reused across turns, so an unconditional merge would report last
    # turn's files as changed in this one.
    if kwargs.get("preserve_file_tracker"):
        if previous is not None:
            file_tracker.merge_from(previous)
    engine._last_file_tracker = file_tracker
    engine._last_total_tool_calls = 0

    caps = get_model_capabilities()
    manual_tc = not caps.supports_function_calling
    is_small = _is_small_model()

    tools = _resolve_tools(
        agent,
        kwargs.get("task_tools"),
        is_small,
        description=task_prompt[0],
        initial_plan=kwargs.get("initial_plan"),
    )
    if (
        kwargs.get("skip_plan", False)
        or not kwargs.get("allow_plan_mutation", True)
    ):
        # Plan-free runs have no state machine to mutate. Fixed-plan runs are
        # owned by an outer scheduler. In both cases, exposing these tools
        # creates work the local loop has no authority to schedule.
        tools = _filter_plan_free_tools(tools)
    compact_tool_schemas = is_small or model_policy.compact_tool_schemas
    tool_schemas = (
        build_tool_schemas(tools, small_model=compact_tool_schemas)
        if tools
        else [STEP_COMPLETE_SCHEMA]
    )
    tool_dispatch = build_tool_dispatch(tools) if tools else {}

    task_profile = getattr(kwargs.get("task"), "task_profile", None)
    prompt_configuration = (
        kwargs.get("prompt_configuration")
        or EffectivePromptConfiguration.compile()
    )
    identity_override = _resolve_identity_override(
        agent,
        tools,
        kwargs.get("identity_override"),
    )

    system_prompt = build_system_prompt(
        agent.backstory,
        tech_hints=getattr(agent, "_tech_hints", None),
        session_summaries=getattr(agent, "_session_summaries", None),
        identity_override=identity_override,
        protocol_override=getattr(agent, "_system_prompt_protocol", None),
        small_model=is_small,
        workspace_path=getattr(agent, "workspace_path", None),
        prompt_configuration=prompt_configuration,
    )
    if model_policy.prompt_addendum:
        system_prompt = f"{system_prompt}\n\n{model_policy.prompt_addendum}"
    from infinidev.engine.prompt_profile import apply_calibrated_guidance

    system_prompt = apply_calibrated_guidance(system_prompt, "developer")
    from infinidev.engine.task_policies.rendering import (
        compose_task_aware_system_prompt,
    )

    system_prompt = compose_task_aware_system_prompt(
        system_prompt,
        task_profile,
        role="developer",
        phase="execute",
        max_utf8_bytes=settings.TASK_POLICIES_MAX_UTF8_BYTES,
        cache_boundary=True,
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
    resumed = bool(resume_state)

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
        model_policy_name=model_policy.name,
        compact_tool_schemas=compact_tool_schemas,
        require_step_orientation=model_policy.require_step_orientation,
        renew_step_budget_on_progress=model_policy.renew_step_budget_on_progress,
        semantic_stagnation_control=model_policy.semantic_stagnation_control,
        phase_boundary_control=model_policy.phase_boundary_control,
        recovery_direct_reads_only=model_policy.recovery_direct_reads_only,
        unlimited_recovery_reads=model_policy.unlimited_recovery_reads,
        reuse_unchanged_test_results=model_policy.reuse_unchanged_test_results,
        freeze_plan_growth_in_recovery=model_policy.freeze_plan_growth_in_recovery,
        prompt_configuration=prompt_configuration,
        recovery_requires_workspace_change=(
            model_policy.recovery_requires_workspace_change),
        system_prompt=system_prompt, tool_schemas=tool_schemas,
        tool_dispatch=tool_dispatch,
        planning_schemas=[
            ADD_STEP_SCHEMA, MODIFY_STEP_SCHEMA, REMOVE_STEP_SCHEMA,
            ADD_NOTE_SCHEMA, STEP_COMPLETE_SCHEMA,
        ],
        tools=tools, max_iterations=max_iterations,
        max_per_action=max_per_action, step_tool_limit=step_tool_limit,
        max_total_calls=max_total_calls,
        max_prompt_tokens=max_prompt_tokens,
        history_window=settings.LOOP_HISTORY_WINDOW,
        max_context_tokens=_get_model_max_context(llm_params),
        verbose=kwargs.get("verbose", True),
        guardrail=kwargs.get("guardrail"),
        guardrail_max_retries=kwargs.get("guardrail_max_retries", 5),
        output_pydantic=kwargs.get("output_pydantic"),
        agent=agent, agent_name=getattr(agent, "name", agent.agent_id),
        agent_role=getattr(agent, "role", "agent"),
        desc=desc, expected=expected, event_id=event_id,
        skip_plan=kwargs.get("skip_plan", False),
        allow_plan_mutation=kwargs.get("allow_plan_mutation", True),
        allow_explore=kwargs.get("allow_explore", True),
        nudge_message_template=kwargs.get("nudge_message_template"),
        state=state, file_tracker=file_tracker,
        start_iteration=state.iteration_count,
        resumed=resumed,
        task=kwargs.get("task"),
        context_corpus=kwargs.get("context_corpus"),
        allow_llm_retries=kwargs.get("allow_llm_retries", True),
    )


def _resolve_tools(
    agent: Any,
    task_tools: list | None,
    is_small: bool,
    *,
    description: str = "",
    initial_plan: Any | None = None,
    task_profile: Any | None = None,
) -> list:
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

    tools = list(getattr(agent, "tools", []) or [])
    if not tools:
        # A cold/unavailable MCP index must never turn a developer Task into a
        # no-tool conversation. ``agent.tools`` is normally populated with the
        # local toolbox, but lightweight hosts and partially initialized agents
        # can expose an empty list. Rebuild the local developer set here; MCP
        # remains optional and joins normally when it becomes available.
        logger.warning(
            "LoopEngine received an empty developer toolbox; restoring local tools"
        )
        from infinidev.tools import get_tools_for_role

        tools = get_tools_for_role("developer", small_model=False)

    if settings.DYNAMIC_TOOL_ROUTING_ENABLED:
        from infinidev.engine.tool_routing import select_developer_tools

        tools = select_developer_tools(
            tools, description, initial_plan, task_profile=task_profile,
        )

        # Dynamic routing may be handed a partial external toolbox containing
        # only MCP tools. The developer loop cannot inspect or verify a Task
        # in that state, and the request_capability escape hatch is absent too.
        # Merge the local core back in, preserving any configured MCP tools.
        required = {"read_file", "list_directory", "code_search", "execute_command"}
        if not required.issubset({getattr(tool, "name", "") for tool in tools}):
            logger.warning(
                "Developer toolbox has no local inspection tools; restoring core fallback"
            )
            from infinidev.tools import get_tools_for_role

            fallback = select_developer_tools(
                get_tools_for_role("developer", small_model=False),
                description,
                initial_plan,
                task_profile=task_profile,
            )
            known = {getattr(tool, "name", "") for tool in tools}
            tools.extend(
                tool for tool in fallback if getattr(tool, "name", "") not in known
            )
        bind_tools_to_agent(tools, agent.agent_id)
    return tools


def _filter_plan_free_tools(tools: list) -> list:
    """Remove tools whose state machine only exists in planned runs."""
    return [tool for tool in tools if tool.name not in _PLAN_MUTATION_TOOLS]


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

_MAX_RANK_QUERY_CHARS = 8_000


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
    pending_rank_guidance = list(getattr(engine, "_cr_pending_user_guidance", ()))
    if pending_rank_guidance:
        engine._cr_pending_user_guidance.clear()
    rank_guidance = [*pending_rank_guidance, *injected]

    from infinidev.engine.static_analysis_timer import measure

    prompt_configuration = getattr(ctx, "prompt_configuration", None)
    if prompt_configuration is None:
        # Compatibility for standalone callers that construct a lightweight
        # context instead of entering through build_execution_context().
        prompt_configuration = EffectivePromptConfiguration.compile()

    with measure("prompt_build"):
        user_prompt = build_iteration_prompt(
            ctx.desc, ctx.expected, effective_state,
            # Fetched once (hence the cache on the engine), rendered every
            # iteration. Each iteration builds a brand-new two-message
            # conversation, so "the model already saw it" is never true
            # here — dropping the block after turn one simply deleted the
            # project's facts from the model's context.
            project_knowledge=engine._project_knowledge,
            context_corpus=getattr(ctx, "context_corpus", None),
            context_rank_result=_rank_at_pivot(
                engine, ctx, iteration, user_messages=rank_guidance or None
            ),
            max_context_tokens=ctx.max_context_tokens,
            session_notes=engine.session_notes or None,
            user_messages=injected or None,
            skip_plan=ctx.skip_plan,
            task=ctx.task,
            small_model=ctx.is_small,
            require_step_orientation=getattr(ctx, "require_step_orientation", True),
            prompt_configuration=prompt_configuration,
        )

    from infinidev.engine.prompt_composition import measure_prompt_composition

    composition = measure_prompt_composition(
        ctx.system_prompt,
        user_prompt,
        getattr(ctx, "tool_schemas", None),
        iteration=iteration,
    )
    ctx.state.prompt_composition_history.append(composition)
    # A runaway loop should not turn diagnostics into another source of bloat.
    if len(ctx.state.prompt_composition_history) > 100:
        del ctx.state.prompt_composition_history[:-100]

    return [
        {"role": "system", "content": ctx.system_prompt},
        {"role": "user", "content": _with_attachments(engine, ctx, user_prompt, first_turn)},
    ]


def _rank_at_pivot(
    engine: Any,
    ctx: ExecutionContext,
    iteration: int,
    *,
    user_messages: list[str] | None = None,
) -> Any | None:
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
        if (
            not user_messages
            and iteration != ctx.start_iteration
            and pivot_key == engine._cr_last_pivot_key
        ):
            return engine._cr_cached_result

        from infinidev.engine.context_rank.ranker import rank

        result = rank(
            _rank_query(ctx.desc, active, user_messages),
            engine._cr_hooks._session_id,
            engine._cr_hooks._task_id,
            iteration,
            cached_embedding=engine._cr_hooks._task_embedding,
            cached_simplified_embedding=engine._cr_hooks._task_embedding_simplified,
            project_id=ctx.project_id,
        )
        engine._cr_cached_result = result
        engine._cr_last_pivot_key = pivot_key
        delivered = getattr(engine, "_cr_delivered_targets", None)
        if delivered is not None:
            for collection in (result.files, result.symbols, result.findings):
                delivered.update(str(item.target) for item in collection)
    return result


def _rank_query(
    description: str,
    active_step: Any | None,
    user_messages: list[str] | None,
) -> str:
    """Build a bounded retrieval query from the task's current decision point."""
    parts = [description.strip()]
    if active_step is not None:
        step_parts = [
            str(getattr(active_step, field, "")).strip()
            for field in ("title", "explanation", "detail", "expected_output")
        ]
        step_text = "\n".join(value for value in step_parts if value)
        if step_text:
            parts.append(f"Current step:\n{step_text}")
    if user_messages:
        messages = "\n".join(message.strip() for message in user_messages if message.strip())
        if messages:
            parts.append(f"New user guidance:\n{messages}")
    return "\n\n".join(part for part in parts if part)[:_MAX_RANK_QUERY_CHARS]


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
            from infinidev.config.model_capabilities import get_capability_snapshot

            engine._supports_vision_cached = get_capability_snapshot().supports_vision
        except Exception:
            engine._supports_vision_cached = False

    from infinidev.engine.multimodal import build_user_content, mention_paths_as_text

    if engine._supports_vision_cached:
        return build_user_content(user_prompt, attachments)
    return mention_paths_as_text(user_prompt, attachments)
