"""Adaptive Stage -> Task -> Step orchestration for one durable Goal."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from infinidev.engine.analysis.plan import Plan
from infinidev.engine.analysis.staged_planning import (
    BlockGoalDecision,
    CompleteGoalDecision,
    EmitStageDecision,
    EvidenceEntry,
    GoalSpec,
    GoalTerminalState,
    StageExecutionRecord,
    StagedPlanningState,
    TaskExecutionRecord,
    TaskPlanningHandoff,
    plan_snapshot,
)
from infinidev.engine.orchestration.escalation_packet import EscalationPacket

logger = logging.getLogger(__name__)

# This is a resource fuse, not a success condition or a semantic estimate of
# Goal size. Hitting it produces an explicit incomplete/blocked outcome.
_DEFAULT_MAX_STAGE_TRANSITIONS = 24


@dataclass(slots=True)
class StagedRunResult:
    """Result returned to the ordinary pipeline closing path."""

    text: str
    engine: Any
    state: StagedPlanningState


def run_staged_goal(
    *,
    escalation: EscalationPacket,
    agent: Any,
    engine: Any,
    reviewer: Any,
    hooks: Any,
    session_id: str,
    project_id: int | None,
    workspace_path: str | None,
    turn_context: str = "",
    use_phase_engine: bool = False,
    force_gather: bool = False,
    max_stage_transitions: int = _DEFAULT_MAX_STAGE_TRANSITIONS,
    max_execution_tool_calls_per_task: int | None = None,
) -> StagedRunResult:
    """Run as many evidence-dependent Stages as the Goal requires.

    A Stage finishing only sends control back to the Stage Planner. It never
    closes the Goal by queue exhaustion. Tasks execute sequentially in DAG
    order so persistence and evidence have a single unambiguous writer.
    """
    from infinidev.engine.analysis.stage_planner import run_stage_planner

    state = _load_or_create_state(session_id, escalation)
    _add_guidance(state, escalation.user_request)
    _restore_interrupted_task(state)
    _publish_state(state, session_id, hooks)

    last_result = ""
    used_engine = engine

    while True:
        active_stage = state.active_stage
        if active_stage is not None and active_stage.status != "evaluating":
            last_result, used_engine = _execute_stage(
                state=state,
                stage=active_stage,
                escalation=escalation,
                agent=agent,
                engine=engine,
                reviewer=reviewer,
                hooks=hooks,
                session_id=session_id,
                project_id=project_id,
                workspace_path=workspace_path,
                turn_context=turn_context,
                use_phase_engine=use_phase_engine,
                force_gather=force_gather,
                max_execution_tool_calls_per_task=max_execution_tool_calls_per_task,
            )
            _publish_state(state, session_id, hooks)
            if getattr(used_engine, "is_cancelled", False):
                return _cancelled_result(state, used_engine, last_result, session_id, hooks)

        hooks.on_phase("analysis")
        hooks.on_status("info", "Evaluating Goal and planning the next Stage...")
        decision = run_stage_planner(
            state,
            session_id=session_id,
            project_id=project_id,
            workspace_path=workspace_path,
            hooks=hooks,
        )
        _publish_state(state, session_id, hooks)

        if isinstance(decision, EmitStageDecision):
            prior = state.active_stage
            if prior is not None and prior.status == "evaluating":
                prior.status = (
                    "completed"
                    if all(task.status == "completed" for task in prior.tasks)
                    else "blocked"
                )
            if len(state.stages) >= max_stage_transitions:
                return _resource_blocked_result(
                    state, used_engine, max_stage_transitions, session_id, hooks
                )
            if _repeats_without_evidence(state, decision):
                return _stagnation_result(state, used_engine, session_id, hooks)
            stage = state.add_stage(decision.stage)
            hooks.on_status("info", f"Stage {stage.number}: {stage.spec.title}")
            notify = getattr(hooks, "notify", None)
            if callable(notify):
                notify(
                    "Planner",
                    _stage_preview(stage),
                    "agent",
                )
            _publish_state(state, session_id, hooks)
            continue

        if isinstance(decision, CompleteGoalDecision):
            prior = state.active_stage
            completion_error = _completion_error(state, prior, decision)
            if completion_error:
                state.status = "blocked"
                state.terminal = GoalTerminalState(
                    kind="goal_blocked",
                    summary=completion_error,
                    missing=(
                        "A new Stage Planner decision grounded in completed Task "
                        "and evidence records."
                    ),
                )
                state.revision += 1
                _set_engine_status(used_engine, "blocked")
                _publish_state(state, session_id, hooks)
                return StagedRunResult(
                    text=_blocked_text(
                        completion_error,
                        state.terminal.missing,
                        [],
                    ),
                    engine=used_engine,
                    state=state,
                )
            if prior is not None and prior.status == "evaluating":
                prior.status = (
                    "completed"
                    if all(task.status == "completed" for task in prior.tasks)
                    else prior.status
                )
            state.status = "complete"
            state.terminal = GoalTerminalState(
                kind="goal_complete",
                summary="The Stage Planner established the Goal from observed evidence.",
                evidence=list(decision.evidence),
            )
            state.revision += 1
            _set_engine_status(used_engine, "completed")
            _publish_state(state, session_id, hooks)
            return StagedRunResult(
                text=_completed_text(last_result, decision.evidence),
                engine=used_engine,
                state=state,
            )

        assert isinstance(decision, BlockGoalDecision)
        prior = state.active_stage
        if prior is not None and prior.status == "evaluating":
            prior.status = "blocked"
        state.status = "blocked"
        state.terminal = GoalTerminalState(
            kind="goal_blocked",
            summary=decision.reason,
            evidence=list(decision.evidence),
            missing=decision.missing,
        )
        state.revision += 1
        _set_engine_status(used_engine, "blocked")
        _publish_state(state, session_id, hooks)
        return StagedRunResult(
            text=_blocked_text(decision.reason, decision.missing, decision.evidence),
            engine=used_engine,
            state=state,
        )


def _execute_stage(
    *,
    state: StagedPlanningState,
    stage: StageExecutionRecord,
    escalation: EscalationPacket,
    agent: Any,
    engine: Any,
    reviewer: Any,
    hooks: Any,
    session_id: str,
    project_id: int | None,
    workspace_path: str | None,
    turn_context: str,
    use_phase_engine: bool,
    force_gather: bool,
    max_execution_tool_calls_per_task: int | None,
) -> tuple[str, Any]:
    from infinidev.engine.analysis.planner import run_planner
    from infinidev.engine.orchestration import pipeline as pipeline_mod
    from infinidev.engine.orchestration.task_schema import task_from_free_text
    from infinidev.prompts.flows import get_flow_config

    stage.status = "active"
    state.revision += 1
    _publish_state(state, session_id, hooks)
    last_result = ""
    used_engine = engine

    while True:
        task = _next_ready_task(stage)
        if task is None:
            pending = [item for item in stage.tasks if item.status == "pending"]
            if pending:
                _block_unrunnable_tasks(stage, pending)
            break

        task.status = "active"
        task.attempts += 1
        state.revision += 1
        hooks.on_status("info", f"Stage {stage.number} / Task: {task.spec.title}")
        _publish_state(state, session_id, hooks)

        handoff = TaskPlanningHandoff(
            goal=state.goal,
            stage_id=stage.id,
            stage_number=stage.number,
            stage=stage.spec,
            task=task.spec,
            dependency_results=_dependency_results(stage, task),
            evidence=list(state.evidence),
        )
        hooks.on_phase("analysis")
        hooks.on_status("info", f"Planning Task: {task.spec.title}")
        plan = run_planner(
            escalation,
            task_handoff=handoff,
            session_id=session_id,
            project_id=project_id,
            workspace_path=workspace_path,
            hooks=hooks,
        )
        plan = _scope_task_plan(plan, stage, task)
        task.plan = plan_snapshot(plan)
        state.revision += 1
        notify = getattr(hooks, "notify", None)
        if callable(notify):
            notify("Planner", plan.overview, "agent")
        _publish_state(state, session_id, hooks)

        flow_config = get_flow_config("develop")
        task_prompt = (
            _render_developer_task(state, stage, task, turn_context),
            flow_config.expected_output,
        )
        task_checks = _task_checks(task, plan)
        structured_task = task_from_free_text(
            _render_structured_task_description(state, stage, task),
            title=_schema_safe_title(task.spec.title),
            kind=_task_kind(state.goal.intent, stage.spec.purpose),
            acceptance_criteria=list(state.goal.acceptance_criteria) or None,
            derived_verification_criteria=task_checks,
        )
        task_prompt = pipeline_mod._run_gather_phase(
            user_input=task.spec.title,
            agent=agent,
            task_prompt=task_prompt,
            session_id=session_id,
            force_gather=force_gather,
            hooks=hooks,
        )
        try:
            result, used_engine = pipeline_mod._run_execution_phase(
                agent=agent,
                engine=engine,
                task_prompt=task_prompt,
                plan=plan,
                session_id=session_id,
                use_phase_engine=use_phase_engine,
                hooks=hooks,
                initial_attachments=(
                    list(escalation.attachments) if escalation.attachments else None
                ),
                task=structured_task,
                preserve_file_tracker=(
                    stage.number > 1
                    or any(
                        candidate is not task and candidate.attempts > 0
                        for candidate in stage.tasks
                    )
                    or task.attempts > 1
                ),
                max_total_tool_calls=_staged_execution_tool_budget(
                    max_execution_tool_calls_per_task
                ),
            )
        except Exception as exc:
            logger.exception("Stage Task execution failed: %s", task.spec.title)
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
            _record_task_evidence(state, stage, task, task.error, used_engine)
            _publish_state(state, session_id, hooks)
            continue

        last_result = result
        if getattr(used_engine, "is_cancelled", False):
            task.status = "cancelled"
            task.result = result
            stage.status = "cancelled"
            _record_task_evidence(state, stage, task, result, used_engine)
            return result, used_engine

        loop_status = getattr(used_engine, "_last_status", "") or "completed"
        if loop_status == "blocked" or _has_blocked_steps(used_engine):
            task.status = "blocked"
            task.result = result
            task.error = "The Task execution closed with blocked work."
            _record_task_evidence(state, stage, task, result, used_engine)
            _publish_state(state, session_id, hooks)
            continue

        result = pipeline_mod._run_review_phase(
            engine=used_engine,
            agent=agent,
            session_id=session_id,
            task_prompt=task_prompt,
            result=result,
            reviewer=reviewer,
            hooks=hooks,
            acceptance_criteria=None,
            derived_verification_criteria=task_checks,
        )
        last_result = result
        if getattr(used_engine, "is_cancelled", False):
            task.status = "cancelled"
            task.result = result
            stage.status = "cancelled"
            _record_task_evidence(state, stage, task, result, used_engine)
            return result, used_engine
        review_status = getattr(used_engine, "_last_status", "") or "completed"
        if review_status == "blocked" or _has_blocked_steps(used_engine):
            task.status = "blocked"
            task.result = result
            task.error = "Review/rework closed with blocked work."
            _record_task_evidence(state, stage, task, result, used_engine)
            _publish_state(state, session_id, hooks)
            continue
        task.status = "completed"
        task.result = result
        _record_task_evidence(state, stage, task, result, used_engine)
        _publish_state(state, session_id, hooks)

    stage.evidence_after = [entry.id for entry in state.evidence]
    statuses = ", ".join(f"{task.spec.id}={task.status}" for task in stage.tasks)
    stage.outcome_summary = f"Stage {stage.number} task outcomes: {statuses}."
    if any(task.status == "cancelled" for task in stage.tasks):
        stage.status = "cancelled"
    elif all(task.status == "completed" for task in stage.tasks):
        stage.status = "evaluating"
    else:
        stage.status = "evaluating"
    state.add_evidence(EvidenceEntry(
        kind="stage_outcome",
        summary=stage.outcome_summary,
        stage_id=stage.id,
        details={"exit_criteria": stage.spec.exit_criteria},
    ))
    state.revision += 1
    return last_result, used_engine


def _load_or_create_state(
    session_id: str,
    escalation: EscalationPacket,
) -> StagedPlanningState:
    from infinidev.db.service import get_session_runtime_state

    raw = get_session_runtime_state(session_id).get("staged_planning")
    if isinstance(raw, dict) and raw:
        try:
            existing = StagedPlanningState.model_validate(raw)
            if existing.status != "complete":
                existing.status = "active"
                existing.terminal = None
                return existing
        except Exception:
            logger.warning("Ignoring invalid persisted staged-planning state", exc_info=True)
    return StagedPlanningState(goal=_goal_from_escalation(escalation))


def _goal_from_escalation(escalation: EscalationPacket) -> GoalSpec:
    request = escalation.user_request.strip()
    first_line = next((line.strip() for line in request.splitlines() if line.strip()), request)
    title = (first_line or "Active goal")[:120]
    planning_context: list[str] = []
    derived_checks: list[str] = []
    confirmed_constraints: list[str] = []
    grounded = getattr(escalation, "grounded_spec", None)
    if grounded is not None:
        try:
            planning_context.append(grounded.render_for_planner())
        except Exception:
            logger.debug("Could not render grounded spec for Goal", exc_info=True)
        deliverable = str(getattr(grounded, "deliverable", "") or "").strip()
        if deliverable:
            derived_checks.append(deliverable)
        derived_checks.extend(
            str(item).strip()
            for item in getattr(grounded, "in_scope", []) or []
            if str(item).strip()
        )
        confirmed_constraints.extend(
            str(item).strip()
            for item in getattr(grounded, "confirmed_decisions", []) or []
            if str(item).strip()
        )
    brief = getattr(escalation, "design_brief", None)
    if brief is not None:
        try:
            planning_context.append(brief.render_for_planner())
        except Exception:
            logger.debug("Could not render design brief for Goal", exc_info=True)
    return GoalSpec(
        title=title,
        user_request=request,
        understanding=escalation.understanding,
        derived_verification_criteria=derived_checks,
        constraints=confirmed_constraints,
        planning_context="\n\n".join(planning_context),
        intent=_infer_goal_intent(escalation),
    )


def _infer_goal_intent(escalation: EscalationPacket) -> str:
    """Classify the requested result without making discovery an error."""
    parts = [escalation.user_request, escalation.understanding]
    text = "\n".join(parts).lower()
    implementation_pattern = re.compile(
        r"\b(implement|build|create|add|update|change|fix|write|modify|"
        r"crear|agregar|añadir|implementar|programar|modificar|corregir|"
        r"construir)\b|quiero arrancar|arranca con",
    )
    informational_pattern = re.compile(
        r"what do you think|qué te parece|\b(opinion|opinión|explain|explica|"
        r"investig\w*|analiz\w*|research|review|revisa)\b",
    )
    if implementation_pattern.search(text):
        return "implementation"
    if informational_pattern.search(text):
        return "informational"
    return "mixed"


def _add_guidance(state: StagedPlanningState, guidance: str) -> None:
    text = guidance.strip()
    if text and text != state.goal.user_request and text not in state.guidance:
        state.guidance.append(text)
        state.revision += 1


def _restore_interrupted_task(state: StagedPlanningState) -> None:
    stage = state.active_stage
    if stage is None:
        return
    for task in stage.tasks:
        if task.status == "active":
            task.status = "pending"
            task.error = "Execution was interrupted; resuming from persisted state."
            state.revision += 1


def _next_ready_task(stage: StageExecutionRecord) -> TaskExecutionRecord | None:
    completed = {
        task.spec.id for task in stage.tasks if task.status == "completed"
    }
    for task in stage.tasks:
        if task.status == "pending" and all(
            dependency in completed for dependency in task.spec.depends_on
        ):
            return task
    return None


def _block_unrunnable_tasks(
    stage: StageExecutionRecord,
    pending: list[TaskExecutionRecord],
) -> None:
    statuses = {task.spec.id: task.status for task in stage.tasks}
    for task in pending:
        unavailable = [
            dependency
            for dependency in task.spec.depends_on
            if statuses.get(dependency) != "completed"
        ]
        task.status = "blocked"
        task.error = f"Dependencies did not complete: {', '.join(unavailable)}"


def _dependency_results(
    stage: StageExecutionRecord,
    task: TaskExecutionRecord,
) -> dict[str, str]:
    wanted = set(task.spec.depends_on)
    return {
        candidate.spec.id: candidate.result
        for candidate in stage.tasks
        if candidate.spec.id in wanted and candidate.status == "completed"
    }


def _record_task_evidence(
    state: StagedPlanningState,
    stage: StageExecutionRecord,
    task: TaskExecutionRecord,
    result: str,
    engine: Any,
) -> None:
    plan_steps = _engine_plan_steps(engine)
    entry = EvidenceEntry(
        kind="task_result",
        summary=(result or task.error or f"Task {task.spec.id} produced no text result")[:6000],
        stage_id=stage.id,
        task_id=task.spec.id,
        details={
            "task_status": task.status,
            "derived_verification_criteria": list(dict.fromkeys([
                *task.spec.derived_verification_criteria,
                *((task.plan or {}).get("derived_verification_criteria") or []),
            ])),
            "plan_steps": plan_steps,
            "error": task.error,
            "workspace_changed": _engine_has_file_changes(engine),
        },
    )
    if state.add_evidence(entry):
        task.evidence_ids.append(entry.id)


def _engine_plan_steps(engine: Any) -> list[dict[str, Any]]:
    getter = getattr(engine, "get_plan_steps", None)
    if not callable(getter):
        return []
    try:
        return [dict(item) for item in getter() if isinstance(item, dict)]
    except Exception:
        return []


def _engine_has_file_changes(engine: Any) -> bool | None:
    getter = getattr(engine, "has_file_changes", None)
    if not callable(getter):
        return None
    try:
        return bool(getter())
    except Exception:
        return None


def _has_blocked_steps(engine: Any) -> bool:
    return any(step.get("status") == "blocked" for step in _engine_plan_steps(engine))


def _render_developer_task(
    state: StagedPlanningState,
    stage: StageExecutionRecord,
    task: TaskExecutionRecord,
    turn_context: str,
) -> str:
    dependency_text = "\n".join(
        f"- {key}: {value}" for key, value in _dependency_results(stage, task).items()
    ) or "- none"
    checks = "\n".join(f"- {item}" for item in task.spec.acceptance_criteria)
    context = f"\n\n{turn_context}" if turn_context else ""
    return (
        "<goal authority=\"USER_LITERAL\">\n"
        f"{state.goal.user_request}\n"
        f"Requested result kind: {state.goal.intent}\n"
        "</goal>\n\n"
        "<active-stage authority=\"DERIVED\">\n"
        f"Title: {stage.spec.title}\nPurpose: {stage.spec.purpose}\n"
        f"Outcome: {stage.spec.outcome}\n"
        "</active-stage>\n\n"
        "<current-task authority=\"DERIVED\">\n"
        f"Title: {task.spec.title}\nOutcome: {task.spec.outcome}\n"
        f"Checks proposed by the Stage Planner:\n{checks}\n"
        f"Completed dependency outputs:\n{dependency_text}\n"
        "Execute only this Task. Its checks guide verification but do not expand "
        "the Goal.\n</current-task>"
        f"{context}"
    )


def _render_structured_task_description(
    state: StagedPlanningState,
    stage: StageExecutionRecord,
    task: TaskExecutionRecord,
) -> str:
    """Keep Goal authority while making the executable Task unambiguous."""
    return (
        f"User-authorized Goal:\n{state.goal.user_request}\n\n"
        "Current derived execution scope:\n"
        f"Stage ({stage.spec.purpose}): {stage.spec.title}\n"
        f"Task: {task.spec.title}\n"
        f"Required outcome: {task.spec.outcome}"
    )


def _task_kind(goal_intent: str, stage_purpose: str) -> str:
    if stage_purpose == "discovery":
        return "investigation"
    if goal_intent == "implementation":
        return "feature"
    return "investigation" if goal_intent == "informational" else "chore"


def _staged_execution_tool_budget(override: int | None) -> int:
    if override is not None:
        return override
    from infinidev.config.settings import settings

    return settings.STAGED_MAX_EXECUTION_TOOL_CALLS_PER_TASK


def _schema_safe_title(title: str) -> str:
    cleaned = title.strip()[:120]
    return cleaned if len(cleaned) >= 5 else f"{cleaned} task"[:120]


def _scope_task_plan(
    plan: Plan,
    stage: StageExecutionRecord,
    task: TaskExecutionRecord,
) -> Plan:
    """Keep the active derived scope visible without promoting it to Goal text."""
    overview = (
        "DERIVED active execution scope:\n"
        f"Stage: {stage.spec.title}\n"
        f"Task: {task.spec.title}\n"
        f"Task outcome: {task.spec.outcome}\n\n"
        f"{plan.overview}"
    )
    return Plan(
        overview=overview,
        steps=list(plan.steps),
        acceptance_criteria=list(plan.acceptance_criteria),
        acceptance_criteria_authority=plan.acceptance_criteria_authority,
    )


def _task_checks(task: TaskExecutionRecord, plan: Plan) -> list[str]:
    return list(dict.fromkeys([
        *task.spec.derived_verification_criteria,
        *(plan.acceptance_criteria or []),
    ]))


def _repeats_without_evidence(
    state: StagedPlanningState,
    decision: EmitStageDecision,
) -> bool:
    if not state.stages:
        return False
    prior = state.stages[-1]
    return (
        prior.spec.fingerprint() == decision.stage.fingerprint()
        and prior.evidence_after == prior.evidence_before
    )


def _completion_error(
    state: StagedPlanningState,
    prior: StageExecutionRecord | None,
    decision: CompleteGoalDecision,
) -> str:
    if not state.evidence:
        return "Goal completion was rejected because the evidence ledger is empty."
    task_evidence = [
        entry for entry in state.evidence if entry.kind == "task_result"
    ]
    if (
        state.goal.intent == "implementation"
        and task_evidence
        and all(entry.details.get("workspace_changed") is False for entry in task_evidence)
    ):
        return (
            "Goal completion was rejected because this implementation Goal has "
            "no observed workspace change. Continue with a delivery Stage or "
            "block on the concrete obstacle."
        )
    if prior is not None and any(task.status != "completed" for task in prior.tasks):
        return (
            "Goal completion was rejected because the latest Stage contains "
            "blocked, failed, cancelled, or pending Tasks."
        )
    literal_count = len(state.goal.acceptance_criteria)
    if literal_count and len(decision.evidence) < literal_count:
        return (
            "Goal completion was rejected because not every USER_LITERAL "
            "acceptance condition has a corresponding evidence statement."
        )
    known_ids = {entry.id for entry in state.evidence}
    if any(
        not any(evidence_id in statement for evidence_id in known_ids)
        for statement in decision.evidence
    ):
        return (
            "Goal completion was rejected because an evidence statement does "
            "not cite an exact observed evidence-ledger ID."
        )
    return ""


def _stage_preview(stage: StageExecutionRecord) -> str:
    tasks = "\n".join(
        f"- {task.spec.title}" for task in stage.tasks
    )
    return (
        f"Stage {stage.number}: {stage.spec.title}\n"
        f"Outcome: {stage.spec.outcome}\nTasks:\n{tasks}"
    )


def _completed_text(last_result: str, evidence: list[str]) -> str:
    if last_result.strip():
        return last_result
    proof = "\n".join(f"- {item}" for item in evidence)
    return f"Goal completed. Evidence:\n{proof}"


def _blocked_text(reason: str, missing: str, evidence: list[str]) -> str:
    text = f"Goal blocked: {reason}\n\nNeeded to continue: {missing}"
    if evidence:
        text += "\n\nObserved evidence:\n" + "\n".join(f"- {item}" for item in evidence)
    return text


def _cancelled_result(
    state: StagedPlanningState,
    engine: Any,
    last_result: str,
    session_id: str,
    hooks: Any,
) -> StagedRunResult:
    state.status = "cancelled"
    state.terminal = GoalTerminalState(
        kind="cancelled",
        summary="The user cancelled execution before the Goal was evaluated as complete.",
    )
    state.revision += 1
    _set_engine_status(engine, "cancelled")
    _publish_state(state, session_id, hooks)
    return StagedRunResult(
        text=last_result or "Execution cancelled; the Goal remains incomplete.",
        engine=engine,
        state=state,
    )


def _resource_blocked_result(
    state: StagedPlanningState,
    engine: Any,
    limit: int,
    session_id: str,
    hooks: Any,
) -> StagedRunResult:
    reason = (
        f"The runtime reached its safety limit of {limit} Stage transitions. "
        "This is a resource stop, not evidence that the Goal is complete."
    )
    missing = "Resume the Goal with a larger Stage resource budget or inspect the history."
    state.status = "blocked"
    state.terminal = GoalTerminalState(
        kind="goal_blocked", summary=reason, missing=missing
    )
    state.revision += 1
    _set_engine_status(engine, "blocked")
    _publish_state(state, session_id, hooks)
    return StagedRunResult(
        text=_blocked_text(reason, missing, []), engine=engine, state=state
    )


def _stagnation_result(
    state: StagedPlanningState,
    engine: Any,
    session_id: str,
    hooks: Any,
) -> StagedRunResult:
    reason = "The proposed Stage repeats the same route without new observed evidence."
    missing = "A changed input, changed method, user decision, or other in-scope evidence route."
    state.status = "blocked"
    state.terminal = GoalTerminalState(
        kind="goal_blocked", summary=reason, missing=missing
    )
    state.revision += 1
    _set_engine_status(engine, "blocked")
    _publish_state(state, session_id, hooks)
    return StagedRunResult(
        text=_blocked_text(reason, missing, []), engine=engine, state=state
    )


def _set_engine_status(engine: Any, status: str) -> None:
    try:
        setattr(engine, "_last_status", status)
    except Exception:
        pass


def _publish_state(
    state: StagedPlanningState,
    session_id: str,
    hooks: Any,
) -> None:
    from infinidev.db.service import persist_staged_planning_state

    snapshot = state.snapshot()
    try:
        persist_staged_planning_state(
            session_id,
            snapshot,
            task_description=state.goal.user_request,
        )
    except Exception:
        logger.debug("staged-planning persistence failed", exc_info=True)
    callback = getattr(hooks, "on_stage_update", None)
    if callable(callback):
        try:
            callback(snapshot)
        except Exception:
            logger.debug("stage update hook failed", exc_info=True)


__all__ = ["StagedRunResult", "run_staged_goal"]
