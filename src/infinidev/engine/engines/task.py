"""One durable Task executed through a rolling, developer-owned Step plan."""

from __future__ import annotations

from typing import Any

from infinidev.config.settings import settings
from infinidev.engine.analysis.plan import Plan, PlanStepSpec
from infinidev.engine.engines.base import (
    EngineResult,
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
)


def _build_task_prompt(escalation: Any, turn_context: str) -> tuple[str, str]:
    """Render the durable Task contract and the rolling-plan protocol."""
    from infinidev.engine.orchestration.staged_pipeline import _goal_from_escalation
    from infinidev.prompts.flows import get_flow_config

    goal = _goal_from_escalation(escalation)
    checks = "\n".join(
        f"- {item}" for item in goal.derived_verification_criteria
    ) or "- none"
    handoff = "\n".join(part for part in (
        f"Chat handoff: {goal.understanding}" if goal.understanding else "",
        goal.planning_context,
    ) if part) or "- none"
    context = f"\n\n{turn_context}" if turn_context else ""
    description = (
        "<task authority=\"USER_LITERAL\">\n"
        f"{goal.user_request}\n"
        f"Requested result kind: {goal.intent}\n"
        "</task>\n\n"
        "<rolling-step-policy authority=\"SYSTEM\">\n"
        "You own both the execution plan and the implementation for this one "
        "durable Task. Start with only the next 1-3 Steps you can justify now; "
        "do not plan the whole project. A Step may be discovery when one fact "
        "will change the route, but do not split ordinary repository orientation "
        "into separate Tasks. Add, modify, or remove model-inferred Steps whenever "
        "new evidence changes the tactic. After the final planned Step, compare "
        "the observed result with this Task: add the next small horizon if work "
        "remains, finish only when it is satisfied, or block only on a real "
        "obstacle. Prior Step summaries are context; use history tools only when "
        "you need older detail.\n"
        f"Derived checks (guide verification, not scope):\n{checks}\n"
        "</rolling-step-policy>"
        "\n\n<task-context authority=\"DERIVED\">\n"
        "This is prior grounded context and constraints. Use it to preserve "
        "the user's intent, but do not treat it as permission to expand scope.\n"
        f"{handoff}\n"
        "</task-context>"
        f"{context}"
    )
    return description, get_flow_config("develop").expected_output


def _bootstrap_step(task: Any) -> PlanStepSpec:
    """Create one executable frontier step without another model round-trip."""
    from infinidev.engine.loop.loop_plan import _step_phase

    title = str(task.title).strip()
    phase = _step_phase(title)
    if task.kind == "investigation":
        if phase not in {"discover", "verify"}:
            title = f"Investigate {title}"
    elif phase not in {"change", "test_change"}:
        title = f"Implement {title}"
    return PlanStepSpec(
        title=title,
        expected_output=(
            "Make concrete progress toward the Task and leave the relevant "
            "verification passing; refine the rolling plan if evidence "
            "reveals distinct remaining work."
        ),
    )


class TaskAdapter:
    """Run one user Task with a compact, rolling Step horizon.

    ``LoopEngine`` already owns mutable Steps, per-Step summaries, context
    compaction, and deterministic verification. This adapter deliberately adds
    no planner role between the Task and that loop.
    """

    name = "task"

    def run(self, **kwargs: Any) -> EngineResult:
        from infinidev.engine.orchestration import pipeline as pipeline_mod
        from infinidev.engine.orchestration.staged_pipeline import (
            _goal_from_escalation,
            _schema_safe_title,
            _task_kind,
        )
        from infinidev.engine.orchestration.task_schema import task_from_free_text

        escalation = kwargs["escalation"]
        agent = kwargs["agent"]
        engine = kwargs["engine"]
        reviewer = kwargs["reviewer"]
        hooks = kwargs["hooks"]
        session_id = kwargs["session_id"]
        goal = _goal_from_escalation(escalation)
        task_prompt = _build_task_prompt(escalation, kwargs.get("turn_context", ""))

        structured_task = task_from_free_text(
            goal.user_request if len(goal.user_request.strip()) >= 20 else (
                f"User request (verbatim): {goal.user_request}"
            ),
            title=_schema_safe_title(goal.title),
            kind=_task_kind(goal.intent, "delivery"),
            acceptance_criteria=list(goal.acceptance_criteria) or None,
            derived_verification_criteria=list(goal.derived_verification_criteria),
            out_of_scope=list(getattr(escalation.grounded_spec, "out_of_scope", []) or []),
            constraints=list(goal.constraints),
            references=list(escalation.opened_files),
        )
        task_prompt = pipeline_mod._run_gather_phase(
            user_input=goal.user_request,
            agent=agent,
            task_prompt=task_prompt,
            session_id=session_id,
            force_gather=kwargs.get("force_gather", False),
            hooks=hooks,
        )

        hooks.on_status(
            "info",
            "Task execution with a rolling Step horizon "
            f"({settings.TASK_MAX_ITERATIONS} iterations / "
            f"{settings.TASK_MAX_TOOL_CALLS} tool calls)",
        )
        rolling_plan = Plan(
            overview=(
                "Developer-owned rolling plan. Keep only the next 1-3 "
                "evidence-backed Steps and extend it after they complete."
            ),
            # Starting empty forces an otherwise deterministic model call just
            # to name the first Step. Seed one executable frontier from the
            # already-structured Task; the developer can still modify or
            # decompose it as repository evidence arrives.
            steps=[_bootstrap_step(structured_task)],
            rolling_horizon_limit=3,
        )
        result, used_engine = pipeline_mod._run_execution_phase(
            agent=agent,
            engine=engine,
            task_prompt=task_prompt,
            plan=rolling_plan,
            session_id=session_id,
            use_phase_engine=False,
            hooks=hooks,
            initial_attachments=(
                list(escalation.attachments) if escalation.attachments else None
            ),
            task=structured_task,
            max_iterations=settings.TASK_MAX_ITERATIONS,
            max_total_tool_calls=settings.TASK_MAX_TOOL_CALLS,
            max_tool_calls_per_action=settings.TASK_MAX_TOOL_CALLS_PER_STEP,
            # Task already has a durable developer with repository tools and
            # compact step summaries. Spawning a second exploration engine
            # duplicates that investigation and discards most of its context.
            # Dedicated /explore remains available for tree-shaped research.
            allow_explore=False,
        )
        if getattr(used_engine, "is_cancelled", False):
            return EngineResult(
                engine_name=self.name,
                status=STATUS_CANCELLED,
                user_message=result,
                summary="Task execution cancelled by the user.",
                engine=used_engine,
                state=getattr(used_engine, "_last_state", None),
                resume_token=session_id,
            )

        loop_status = getattr(used_engine, "_last_status", "") or "completed"
        status = STATUS_BLOCKED if loop_status in {"blocked", "failed", "exhausted"} else STATUS_COMPLETED
        if status == STATUS_COMPLETED:
            result = pipeline_mod._run_review_phase(
                engine=used_engine,
                agent=agent,
                session_id=session_id,
                task_prompt=task_prompt,
                result=result,
                reviewer=reviewer,
                hooks=hooks,
                acceptance_criteria=list(goal.acceptance_criteria) or None,
                derived_verification_criteria=list(goal.derived_verification_criteria),
                task=structured_task,
                max_iterations=settings.TASK_MAX_ITERATIONS,
                max_total_tool_calls=settings.TASK_MAX_TOOL_CALLS,
            )
            review_status = getattr(used_engine, "_last_status", "") or "completed"
            if review_status in {"blocked", "failed", "exhausted"}:
                status = STATUS_BLOCKED

        return EngineResult(
            engine_name=self.name,
            status=status,
            user_message=result,
            summary=f"Task loop closed {status} (loop status: {loop_status}).",
            engine=used_engine,
            state=getattr(used_engine, "_last_state", None),
            resume_token=session_id,
            metrics={
                "max_iterations": settings.TASK_MAX_ITERATIONS,
                "max_tool_calls": settings.TASK_MAX_TOOL_CALLS,
            },
        )


__all__ = ["TaskAdapter"]
