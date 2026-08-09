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
    TransitionRequest,
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

        goal = _goal_from_escalation(escalation)
        task_prompt = _build_task_prompt(escalation, turn_context)

        task_prompt = pipeline_mod._run_gather_phase(
            user_input=goal.user_request,
            agent=agent,
            task_prompt=task_prompt,
            session_id=session_id,
            force_gather=force_gather,
            hooks=hooks,
        )

        literal_description = goal.user_request
        if len(literal_description.strip()) < 20:
            literal_description = f"User request (verbatim): {literal_description}"
        structured_task = task_from_free_text(
            literal_description,
            title=_schema_safe_title(goal.title),
            acceptance_criteria=list(goal.acceptance_criteria) or None,
            derived_verification_criteria=list(goal.derived_verification_criteria),
        )

        hooks.on_phase("execute")
        hooks.on_status(
            "info",
            f"ReAct direct execution (budget {settings.REACT_MAX_ITERATIONS} "
            f"iterations / {settings.REACT_MAX_TOOL_CALLS} tool calls)",
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
            )
        finally:
            agent.deactivate()

        if not result or not result.strip():
            result = "Done. (no additional output)"

        if getattr(engine, "is_cancelled", False):
            return EngineResult(
                engine_name=self.name,
                status=STATUS_CANCELLED,
                user_message=result,
                summary="ReAct run cancelled by the user.",
                engine=engine,
                resume_token=session_id,
            )

        loop_status = getattr(engine, "_last_status", "") or "completed"
        transition_request = None

        if loop_status == "exhausted":
            # Budget fuse blown: this is NOT success. Escalate to Staged.
            status = STATUS_BLOCKED
            transition_request = TransitionRequest(
                target="staged",
                reason=(
                    "react_budget_exhausted: the task did not converge within "
                    f"{settings.REACT_MAX_ITERATIONS} iterations / "
                    f"{settings.REACT_MAX_TOOL_CALLS} tool calls."
                ),
            )
            hooks.on_status(
                "warn",
                "ReAct budget exhausted — marking blocked and suggesting the "
                "staged engine.",
            )
        elif loop_status in {"blocked", "failed"}:
            status = STATUS_BLOCKED
        else:
            status = STATUS_COMPLETED

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
                rework_execute_kwargs={"skip_plan": True},
            )
            if getattr(engine, "is_cancelled", False):
                return EngineResult(
                    engine_name=self.name,
                    status=STATUS_CANCELLED,
                    user_message=result,
                    summary="ReAct run cancelled during review.",
                    engine=engine,
                    resume_token=session_id,
                )
            # The rework loop re-runs the engine; a blocked close there must
            # surface as blocked, mirroring the staged per-task handling.
            review_status = getattr(engine, "_last_status", "") or "completed"
            if review_status in {"blocked", "failed", "exhausted"}:
                status = STATUS_BLOCKED

        return EngineResult(
            engine_name=self.name,
            status=status,
            user_message=result,
            summary=f"ReAct run closed {status} (loop status: {loop_status}).",
            engine=engine,
            resume_token=session_id,
            transition_request=transition_request,
            metrics={
                "max_iterations": settings.REACT_MAX_ITERATIONS,
                "max_tool_calls": settings.REACT_MAX_TOOL_CALLS,
                "max_prompt_tokens": settings.REACT_MAX_PROMPT_TOKENS,
                "observed_iterations": getattr(
                    getattr(engine, "_last_state", None), "iteration_count", 0
                ),
                "observed_tool_calls": getattr(
                    getattr(engine, "_last_state", None), "total_tool_calls", 0
                ),
                "observed_prompt_tokens": getattr(
                    getattr(engine, "_last_state", None), "total_prompt_tokens", 0
                ),
                "observed_completion_tokens": getattr(
                    getattr(engine, "_last_state", None),
                    "total_completion_tokens",
                    0,
                ),
            },
        )


def _schema_safe_title(title: str) -> str:
    cleaned = (title or "").strip()[:120]
    return cleaned if len(cleaned) >= 5 else f"{cleaned} task"[:120]


__all__ = ["ReactAdapter"]
