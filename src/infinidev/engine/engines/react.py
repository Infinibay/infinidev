"""ReactAdapter — a budgeted, plan-free execution loop.

ReAct is the answer to Staged's overhead for tasks where building a plan
costs more than executing it (docs/GRAPH_ENGINE_BETA_DESIGN.md §8.1): a
single local change, a quick investigation, one or a few tools. It runs the
existing LoopEngine directly — no Stage Planner, no Task Planner — with a
tight iteration/tool-call budget and the same completion gate (step_complete
plus objective verification) the developer loop already enforces.

The budget fuses are resource ceilings, never success conditions: hitting
``REACT_MAX_ITERATIONS`` or ``REACT_MAX_TOOL_CALLS`` closes the run as
*blocked* carrying a :class:`TransitionRequest` toward the staged engine, so
a task that outgrows ReAct is escalated, not falsely completed.
"""

from __future__ import annotations

from typing import Any

from infinidev.config.settings import settings
from infinidev.engine.engines.base import (
    EngineResult,
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    TransitionRequest,
    get_loop_status,
    loop_observed_metrics,
    normalize_loop_status,
)


def _build_task_prompt(
    escalation: Any,
    turn_context: str,
) -> tuple[str, str]:
    """Compose the developer task prompt for a plan-free run."""
    from infinidev.engine.orchestration.staged_pipeline import (
        _goal_from_escalation,
    )
    from infinidev.prompts.flows import get_flow_config

    goal = _goal_from_escalation(escalation)
    checks = list(goal.derived_verification_criteria)
    checks_block = (
        "\n".join(f"- {item}" for item in checks) if checks else "- none"
    )
    context = f"\n\n{turn_context}" if turn_context else ""
    description = (
        "<goal authority=\"USER_LITERAL\">\n"
        f"{goal.user_request}\n"
        "</goal>\n\n"
        "<approach authority=\"DERIVED\">\n"
        "Direct execution without a staged plan. Work incrementally with the "
        "tools, verify what you produce, and finish with step_complete once "
        "the request is satisfied or you are genuinely blocked. Do not expand "
        "the goal.\n"
        f"Derived checks (guide verification, not scope):\n{checks_block}\n"
        "</approach>"
        f"{context}"
    )
    flow_config = get_flow_config("develop")
    return description, flow_config.expected_output


def _budget_transition(engine: Any) -> TransitionRequest:
    metrics = loop_observed_metrics(engine)
    tool_calls = metrics["observed_tool_calls"]
    prompt_tokens = metrics["observed_prompt_tokens"]
    iterations = metrics["observed_iterations"]

    if (
        settings.REACT_MAX_TOOL_CALLS > 0
        and tool_calls >= settings.REACT_MAX_TOOL_CALLS
    ):
        detail = (
            "tool-call budget reached "
            f"({tool_calls}/{settings.REACT_MAX_TOOL_CALLS})"
        )
    elif (
        settings.REACT_MAX_PROMPT_TOKENS > 0
        and prompt_tokens >= settings.REACT_MAX_PROMPT_TOKENS
    ):
        detail = (
            "prompt token budget reached "
            f"({prompt_tokens}/{settings.REACT_MAX_PROMPT_TOKENS})"
        )
    elif (
        settings.REACT_MAX_ITERATIONS > 0
        and iterations >= settings.REACT_MAX_ITERATIONS
    ):
        detail = (
            "iteration budget reached "
            f"({iterations}/{settings.REACT_MAX_ITERATIONS})"
        )
    else:
        detail = (
            "the task did not converge within the configured iteration, "
            "tool-call, or prompt-token budget"
        )

    return TransitionRequest(
        target="staged",
        reason=f"react_budget_exhausted: {detail}.",
    )


def _result_text(result: Any) -> str:
    if isinstance(result, str) and result.strip():
        return result
    return "Done. (no additional output)"


class ReactAdapter:
    """Execute an escalated task as one budgeted, plan-free loop."""

    name = "react"

    def run(self, **kwargs: Any) -> EngineResult:
        from infinidev.engine.orchestration import pipeline as pipeline_mod
        from infinidev.engine.orchestration.staged_pipeline import (
            _goal_from_escalation,
        )
        from infinidev.engine.orchestration.task_schema import task_from_free_text

        escalation = kwargs["escalation"]
        agent = kwargs["agent"]
        engine = kwargs["engine"]
        reviewer = kwargs["reviewer"]
        hooks = kwargs["hooks"]
        session_id = kwargs["session_id"]
        force_gather = kwargs.get("force_gather", False)
        turn_context = kwargs.get("turn_context", "")
        from infinidev.prompts.profiles import EffectivePromptConfiguration

        prompt_configuration = (
            kwargs.get("prompt_configuration")
            or EffectivePromptConfiguration.compile()
        )

        goal = _goal_from_escalation(escalation)
        task_prompt = _build_task_prompt(escalation, turn_context)

        task_prompt = pipeline_mod._run_gather_phase(
            user_input=goal.user_request,
            agent=agent,
            task_prompt=task_prompt,
            session_id=session_id,
            force_gather=force_gather,
            hooks=hooks,
            prompt_configuration=prompt_configuration,
        )

        literal_description = goal.user_request
        if len(literal_description.strip()) < 20:
            literal_description = f"User request (verbatim): {literal_description}"
        structured_task = task_from_free_text(
            literal_description,
            title=_schema_safe_title(goal.title),
            acceptance_criteria=list(goal.acceptance_criteria) or None,
            derived_verification_criteria=list(goal.derived_verification_criteria),
            task_profile=escalation.task_profile,
        )

        hooks.on_phase("execute")
        hooks.on_status(
            "info",
            f"ReAct direct execution (budget {settings.REACT_MAX_ITERATIONS} "
            f"iterations / {settings.REACT_MAX_TOOL_CALLS} tool calls / "
            f"{settings.REACT_MAX_PROMPT_TOKENS} prompt tokens)",
        )

        agent.activate_context(session_id=session_id)
        try:
            result = engine.execute(
                agent=agent,
                task_prompt=task_prompt,
                verbose=True,
                initial_plan=None,
                initial_attachments=(
                    list(escalation.attachments) if escalation.attachments else None
                ),
                task=structured_task,
                max_iterations=settings.REACT_MAX_ITERATIONS,
                max_total_tool_calls=settings.REACT_MAX_TOOL_CALLS,
                max_prompt_tokens=settings.REACT_MAX_PROMPT_TOKENS,
                skip_plan=True,
                allow_explore=False,
                prompt_configuration=prompt_configuration,
            )
        finally:
            agent.deactivate()

        result = _result_text(result)
        loop_status = get_loop_status(engine)
        status = normalize_loop_status(loop_status)
        closing_loop_status = loop_status
        transition_request = None

        if getattr(engine, "is_cancelled", False):
            status = STATUS_CANCELLED
        elif loop_status == "exhausted":
            transition_request = _budget_transition(engine)
            hooks.on_status(
                "warn",
                "ReAct budget exhausted — marking blocked and suggesting the "
                "staged engine.",
            )
        elif status == STATUS_FAILED and loop_status != "failed":
            hooks.on_status(
                "error",
                "ReAct returned an empty or unknown terminal status; failing "
                "closed instead of reporting completion.",
            )

        # Run the same closing review Staged uses, so ReAct does not bypass
        # semantic verification.
        if status == STATUS_COMPLETED:
            result = pipeline_mod._run_review_phase(
                engine=engine,
                agent=agent,
                session_id=session_id,
                task_prompt=task_prompt,
                result=result,
                reviewer=reviewer,
                hooks=hooks,
                acceptance_criteria=list(goal.acceptance_criteria) or None,
                derived_verification_criteria=list(
                    goal.derived_verification_criteria
                ),
                task=structured_task,
                max_iterations=settings.REACT_MAX_ITERATIONS,
                max_total_tool_calls=settings.REACT_MAX_TOOL_CALLS,
                rework_execute_kwargs={
                    "skip_plan": True,
                    "max_prompt_tokens": settings.REACT_MAX_PROMPT_TOKENS,
                },
                prompt_configuration=prompt_configuration,
            )
            result = _result_text(result)
            review_status = get_loop_status(engine)
            closing_loop_status = review_status
            status = normalize_loop_status(review_status)

            if getattr(engine, "is_cancelled", False):
                status = STATUS_CANCELLED
            elif review_status == "exhausted":
                transition_request = _budget_transition(engine)
                hooks.on_status(
                    "warn",
                    "ReAct review rework exhausted its budget — marking blocked "
                    "and suggesting the staged engine.",
                )
            elif status == STATUS_FAILED and review_status != "failed":
                hooks.on_status(
                    "error",
                    "ReAct review returned an empty or unknown terminal status; "
                    "failing closed.",
                )

        return EngineResult(
            engine_name=self.name,
            status=status,
            user_message=result,
            summary=(
                f"ReAct run closed {status} (initial loop status: "
                f"{loop_status or '<empty>'}; closing loop status: "
                f"{closing_loop_status or '<empty>'})."
            ),
            engine=engine,
            state=getattr(engine, "_last_state", None),
            resume_token=session_id,
            transition_request=transition_request,
            metrics={
                "max_iterations": settings.REACT_MAX_ITERATIONS,
                "max_tool_calls": settings.REACT_MAX_TOOL_CALLS,
                "max_prompt_tokens": settings.REACT_MAX_PROMPT_TOKENS,
                **loop_observed_metrics(engine),
            },
        )


def _schema_safe_title(title: str) -> str:
    cleaned = (title or "").strip()[:120]
    return cleaned if len(cleaned) >= 5 else f"{cleaned} task"[:120]


__all__ = ["ReactAdapter"]
