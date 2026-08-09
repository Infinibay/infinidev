"""End-to-end state-machine tests for Stage -> Task -> Step orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from infinidev.config.settings import settings
from infinidev.engine.analysis.plan import Plan, PlanStepSpec
from infinidev.engine.analysis.staged_planning import (
    BlockGoalDecision,
    CompleteGoalDecision,
    EmitStageDecision,
    EvidenceEntry,
    GoalSpec,
    StageSpec,
    StageTaskSpec,
    StagedPlanningState,
)
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.engine.orchestration.staged_pipeline import (
    _goal_from_escalation,
    _scope_task_plan,
    run_staged_goal,
)


@dataclass
class _Agent:
    project_id: int | None = 1
    workspace_path: str | None = "/workspace"


class _Engine:
    def __init__(self) -> None:
        self._last_status = "completed"
        self.is_cancelled = False
        self.steps: list[dict[str, Any]] = []
        self.has_changes = True

    def get_plan_steps(self) -> list[dict[str, Any]]:
        return list(self.steps)

    def has_file_changes(self) -> bool:
        return self.has_changes


class _Hooks:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, str]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.notifications: list[tuple[str, str, str]] = []

    def on_phase(self, phase: str) -> None:
        pass

    def on_status(self, level: str, message: str) -> None:
        self.statuses.append((level, message))

    def notify(self, speaker: str, message: str, kind: str = "agent") -> None:
        self.notifications.append((speaker, message, kind))

    def on_stage_update(self, snapshot: dict[str, Any]) -> None:
        self.snapshots.append(snapshot)


def _escalation(text: str = "Implement the complete staged planning behavior.") -> EscalationPacket:
    return EscalationPacket(user_request=text, understanding=text)


def _task(
    task_id: str,
    *,
    depends_on: list[str] | None = None,
) -> StageTaskSpec:
    return StageTaskSpec(
        id=task_id,
        title=f"Task {task_id}",
        outcome=f"Outcome {task_id} exists",
        acceptance_criteria=[f"Outcome {task_id} is observed"],
        depends_on=depends_on or [],
    )


def _stage(title: str, tasks: list[StageTaskSpec]) -> EmitStageDecision:
    return EmitStageDecision(stage=StageSpec(
        title=title,
        outcome=f"{title} outcome",
        exit_criteria=[f"{title} exit is observed"],
        tasks=tasks,
    ))


def _complete(summary: str):
    def decision(state: StagedPlanningState) -> CompleteGoalDecision:
        assert state.evidence
        return CompleteGoalDecision(
            evidence=[f"{state.evidence[-1].id}: {summary}"]
        )
    return decision


@pytest.fixture
def runtime(monkeypatch):
    calls: dict[str, list[Any]] = {"task_plans": [], "executions": [], "reviews": []}

    def task_planner(_escalation, *, task_handoff=None, **_kwargs):
        calls["task_plans"].append(task_handoff)
        return Plan(
            overview=f"Plan {task_handoff.task.id}",
            steps=[PlanStepSpec(title=f"Step {task_handoff.task.id}")],
            acceptance_criteria=list(task_handoff.task.acceptance_criteria),
        )

    def gather(**kwargs):
        return kwargs["task_prompt"]

    def execute(**kwargs):
        calls["executions"].append(kwargs)
        engine = kwargs["engine"]
        engine._last_status = "completed"
        engine.steps = [{"title": "done", "status": "done"}]
        handoff_text = kwargs["task_prompt"][0]
        return f"executed {handoff_text.split('Title: ')[-1].splitlines()[0]}", engine

    def review(**kwargs):
        calls["reviews"].append(kwargs)
        return kwargs["result"]

    monkeypatch.setattr("infinidev.engine.analysis.planner.run_planner", task_planner)
    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline._run_gather_phase", gather
    )
    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline._run_execution_phase", execute
    )
    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline._run_review_phase", review
    )
    return calls


def _install_stage_planner(monkeypatch, decisions):
    queue = list(decisions)
    seen: list[StagedPlanningState] = []

    def planner(state, **_kwargs):
        seen.append(state.model_copy(deep=True))
        if not queue:
            raise AssertionError("Unexpected extra Stage Planner call")
        decision = queue.pop(0)
        return decision(state) if callable(decision) else decision

    monkeypatch.setattr(
        "infinidev.engine.analysis.stage_planner.run_stage_planner", planner
    )
    return seen


def test_small_goal_uses_one_stage_one_task_then_evidence_completion(
    temp_db, monkeypatch, runtime,
):
    seen = _install_stage_planner(monkeypatch, [
        _stage("One slice", [_task("only")]),
        _complete("The Task result and focused step are observed"),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="small", project_id=1, workspace_path="/workspace",
    )

    assert result.state.status == "complete"
    assert len(result.state.stages) == 1
    assert result.state.stages[0].tasks[0].status == "completed"
    assert len(runtime["executions"]) == 1
    structured = runtime["executions"][0]["task"]
    assert "User-authorized Goal" in structured.description
    assert "Current derived execution scope" in structured.description
    assert "Task: Task only" in structured.description
    assert structured.title == "Task only"
    assert structured.kind == "feature"
    assert structured.acceptance_criteria == [
        "The user's request as written in <description> is satisfied to the user's confirmation."
    ]
    assert "Outcome only is observed" in structured.derived_verification_criteria
    assert runtime["executions"][0]["max_total_tool_calls"] == 40
    assert runtime["executions"][0]["allow_plan_mutation"] is False
    assert runtime["reviews"][0]["task"] is structured
    assert runtime["reviews"][0]["max_total_tool_calls"] == 40
    assert seen[1].evidence
    assert result.text.startswith("executed Task only")


def test_completed_stage_recovers_when_only_terminal_planner_protocol_fails(
    temp_db, monkeypatch, runtime,
):
    protocol_failure = BlockGoalDecision(
        reason="Stage Planner exhausted its iteration budget without a valid decision.",
        missing="A valid Stage Planner decision on a later retry.",
        evidence=[],
    )
    _install_stage_planner(monkeypatch, [
        _stage("One slice", [_task("only")]),
        protocol_failure,
    ])
    engine = _Engine()
    engine._last_state = SimpleNamespace(last_test_command="pytest focused.py -q")
    engine.get_file_contents = lambda: {"src/fixed.py": "fixed"}
    engine.get_file_tracker = lambda: None
    monkeypatch.setattr(
        "infinidev.engine.analysis.verification_engine.VerificationEngine.verify",
        lambda self, **kwargs: SimpleNamespace(
            passed=True,
            summary="All 1 verification command(s) passed",
        ),
    )

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=engine, reviewer=object(),
        hooks=_Hooks(), session_id="planner-protocol-recovery", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "complete"
    assert result.engine._last_status == "completed"


def test_empty_task_plan_gets_bounded_steps_from_structured_task() -> None:
    state = StagedPlanningState(goal=GoalSpec(
        title="Fix widget",
        user_request="Fix the widget and verify it.",
        intent="implementation",
    ))
    stage = state.add_stage(_stage("Delivery", [_task("only")]).stage)
    task = stage.tasks[0]

    scoped = _scope_task_plan(Plan(overview="No structured plan", steps=[]), stage, task)

    assert [step.title.split()[0] for step in scoped.steps] == [
        "Implement", "Verify",
    ]
    assert task.spec.acceptance_criteria[0] in scoped.steps[0].detail
    assert task.spec.acceptance_criteria[0] in scoped.steps[1].detail
    assert "fallback" in scoped.overview.lower()
    assert scoped.rolling_horizon_limit == 1


def test_empty_task_plan_splits_behavior_invalidation_and_tests() -> None:
    state = StagedPlanningState(goal=GoalSpec(
        title="Deduplicate reads",
        user_request="Suppress duplicate reads and test invalidation.",
        intent="implementation",
    ))
    spec = StageTaskSpec(
        id="dedup",
        title="Implement read result deduplication",
        outcome="Repeated unchanged reads are compact and edits invalidate them",
        acceptance_criteria=[
            "An exact repeated range returns a compact structured result",
            "A file edit invalidates the prior read revision",
            "pytest tests/test_read_dedup.py passes",
        ],
    )
    stage = state.add_stage(_stage("Delivery", [spec]).stage)

    scoped = _scope_task_plan(
        Plan(overview="No structured plan", steps=[]), stage, stage.tasks[0]
    )

    assert len(scoped.steps) == 3
    assert scoped.steps[0].title.startswith("Implement ")
    assert scoped.steps[1].title.startswith("Integrate ")
    assert scoped.steps[2].title.startswith("Verify ")
    assert "pytest tests/test_read_dedup.py passes" in scoped.steps[2].detail


def test_narrow_valid_plan_is_repaired_for_uncovered_task_checks() -> None:
    state = StagedPlanningState(goal=GoalSpec(
        title="Deduplicate reads",
        user_request="Suppress duplicate reads and test invalidation.",
        intent="implementation",
    ))
    spec = StageTaskSpec(
        id="dedup",
        title="Implement read result deduplication",
        outcome="Repeated unchanged reads are compact and edits invalidate them",
        acceptance_criteria=[
            "A bounded read repetition cache exists",
            "The cache is integrated into the read_file execution pipeline",
            "pytest tests/test_read_dedup.py passes",
        ],
    )
    stage = state.add_stage(_stage("Delivery", [spec]).stage)
    emitted = Plan(
        overview="Create the cache storage.",
        steps=[PlanStepSpec(
            title="Add bounded read repetition cache",
            detail="Define the cache and its maximum size.",
            expected_output="A bounded read repetition cache exists",
        )],
    )

    scoped = _scope_task_plan(emitted, stage, stage.tasks[0])

    assert len(scoped.steps) == 3
    assert scoped.steps[0].title == "Add bounded read repetition cache"
    assert "execution pipeline" in scoped.steps[1].detail
    assert scoped.steps[2].title.startswith("Verify ")
    assert "pytest tests/test_read_dedup.py passes" in scoped.steps[2].detail
    assert "coverage repair" in scoped.overview.lower()


def test_first_bounded_path_task_uses_local_plan(
    temp_db, monkeypatch, runtime,
):
    bounded = StageTaskSpec(
        id="bounded",
        title="Update src/widget.py",
        outcome="src/widget.py exposes the corrected widget behavior",
        acceptance_criteria=[
            "src/widget.py contains the corrected implementation",
            "No unrelated files change",
        ],
    )
    _install_stage_planner(monkeypatch, [
        _stage("Delivery", [bounded]),
        _complete("The bounded file change is observed"),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="bounded-local-plan", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "complete"
    assert runtime["task_plans"] == []
    seeded_plan = runtime["executions"][0]["plan"]
    assert [step.title for step in seeded_plan.steps] == ["Update src/widget.py"]
    assert "Local routing" in seeded_plan.overview


def test_later_path_task_keeps_evidence_aware_task_planner(
    temp_db, monkeypatch, runtime,
):
    first = StageTaskSpec(
        id="first",
        title="Update src/widget.py",
        outcome="src/widget.py is updated",
        acceptance_criteria=["src/widget.py contains the implementation"],
    )
    later = StageTaskSpec(
        id="later",
        title="Add tests/test_widget.py",
        outcome="tests/test_widget.py covers the implementation",
        acceptance_criteria=["tests/test_widget.py contains focused coverage"],
    )
    _install_stage_planner(monkeypatch, [
        _stage("Delivery", [first, later]),
        _complete("Both bounded file Tasks are observed"),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="later-planner", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "complete"
    assert [handoff.task.id for handoff in runtime["task_plans"]] == ["later"]


def test_later_task_inherits_edit_evidence_only_for_its_exact_target(
    temp_db, monkeypatch, runtime,
):
    first = StageTaskSpec(
        id="first",
        title="Update src/widget.py",
        outcome="src/widget.py is updated",
        acceptance_criteria=["src/widget.py contains the implementation"],
    )
    later = StageTaskSpec(
        id="later",
        title="Add tests/test_widget.py",
        outcome="tests/test_widget.py covers the implementation",
        acceptance_criteria=["tests/test_widget.py contains focused coverage"],
    )
    _install_stage_planner(monkeypatch, [
        _stage("Delivery", [first, later]),
        _complete("Both file Tasks are observed"),
    ])
    engine = _Engine()
    engine.get_file_tracker = lambda: SimpleNamespace(
        get_all_paths=lambda: ["/workspace/tests/test_widget.py"],
    )

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=engine, reviewer=object(),
        hooks=_Hooks(), session_id="prior-target-edit", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "complete"
    assert [
        call["initial_edit_evidence"] for call in runtime["executions"]
    ] == [False, True]


def test_verification_only_stage_task_does_not_require_an_edit() -> None:
    from infinidev.engine.orchestration.staged_pipeline import _task_kind

    assert _task_kind(
        "implementation",
        "delivery",
        "Run the new tests and confirm pass",
        "pytest exits successfully without changing files",
    ) == "investigation"
    assert _task_kind(
        "implementation",
        "delivery",
        "Create focused tests",
        "A new test module exists",
    ) == "feature"


def test_deterministic_verification_task_skips_developer_execution(
    temp_db, monkeypatch, runtime,
):
    from infinidev.engine.analysis.verification_result import VerificationResult
    from infinidev.engine.analysis.step_verification import StepVerification

    stage = _stage("Delivery", [_task("change"), _task("verify")])
    _install_stage_planner(monkeypatch, [
        stage,
        _complete("The change and direct verification are observed"),
    ])

    def task_planner(_escalation, *, task_handoff=None, **_kwargs):
        if task_handoff.task.id == "verify":
            return Plan(
                overview="Verify the completed change",
                steps=[PlanStepSpec(
                    title="Run pytest and confirm exit 0",
                    verify=StepVerification(kind="command", spec="pytest -q"),
                )],
            )
        return Plan(
            overview="Make the change",
            steps=[PlanStepSpec(title="Implement the change")],
        )

    monkeypatch.setattr(
        "infinidev.engine.analysis.planner.run_planner", task_planner,
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.objective_verifier.ObjectiveVerifier.verify",
        lambda self, check: VerificationResult(
            passed=True,
            summary="verification passed",
            commands_run=[],
        ),
    )

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="direct-verify-pass", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "complete"
    assert len(runtime["executions"]) == 1
    verification_task = result.state.stages[0].tasks[1]
    assert verification_task.status == "completed"
    assert "without a developer turn" in verification_task.result
    evidence = next(
        entry for entry in result.state.evidence
        if entry.task_id == verification_task.spec.id
    )
    assert evidence.details["workspace_changed"] is False
    assert evidence.details["plan_steps"][0]["verify"]["spec"] == "pytest -q"


def test_failed_deterministic_verification_falls_through_to_developer(
    temp_db, monkeypatch, runtime,
):
    from infinidev.engine.analysis.verification_result import VerificationResult
    from infinidev.engine.analysis.step_verification import StepVerification

    _install_stage_planner(monkeypatch, [
        _stage("Verification", [_task("verify")]),
        _complete("The repaired verification Task completed"),
    ])
    monkeypatch.setattr(
        "infinidev.engine.analysis.planner.run_planner",
        lambda *_args, **_kwargs: Plan(
            overview="Verify and repair if needed",
            steps=[PlanStepSpec(
                title="Run pytest and confirm exit 0",
                verify=StepVerification(kind="command", spec="pytest -q"),
            )],
        ),
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.objective_verifier.ObjectiveVerifier.verify",
        lambda self, check: VerificationResult(
            passed=False,
            summary="verification failed",
            commands_run=[],
        ),
    )

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="direct-verify-fail", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "complete"
    assert len(runtime["executions"]) == 1


def test_task_dag_executes_only_dependency_ready_tasks(
    temp_db, monkeypatch, runtime,
):
    _install_stage_planner(monkeypatch, [
        _stage("DAG", [_task("producer"), _task("consumer", depends_on=["producer"])]),
        _complete("Both dependency-linked outcomes are observed"),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="dag", project_id=1, workspace_path="/workspace",
    )

    assert [handoff.task.id for handoff in runtime["task_plans"]] == [
        "producer", "consumer"
    ]
    consumer = runtime["task_plans"][1]
    assert "producer" in consumer.dependency_results
    assert result.state.status == "complete"


def test_stage_evidence_can_change_the_next_stage_strategy(
    temp_db, monkeypatch, runtime,
):
    def second_stage(state: StagedPlanningState):
        assert any("Task measure" in entry.summary for entry in state.evidence)
        return _stage("Optimize measured cause", [_task("optimize")])

    _install_stage_planner(monkeypatch, [
        _stage("Measure", [_task("measure")]),
        second_stage,
        _complete("Measurement and optimization results are observed"),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="multi", project_id=1, workspace_path="/workspace",
    )

    assert [stage.spec.title for stage in result.state.stages] == [
        "Measure", "Optimize measured cause"
    ]
    assert len(runtime["executions"]) == 2


def test_blocked_task_prevents_false_goal_completion(
    temp_db, monkeypatch, runtime,
):
    def blocked_execute(**kwargs):
        runtime["executions"].append(kwargs)
        engine = kwargs["engine"]
        engine._last_status = "blocked"
        engine.steps = [{"title": "cannot continue", "status": "blocked"}]
        return "missing authority", engine

    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline._run_execution_phase",
        blocked_execute,
    )
    _install_stage_planner(monkeypatch, [
        _stage("Attempt", [_task("attempt")]),
        _complete("The queue is empty"),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="blocked-task", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "blocked"
    assert result.state.stages[0].tasks[0].status == "blocked"
    assert "Task attempt" in result.text


def test_exhausted_task_prevents_false_goal_completion(
    temp_db, monkeypatch, runtime,
):
    def exhausted_execute(**kwargs):
        runtime["executions"].append(kwargs)
        engine = kwargs["engine"]
        engine._last_status = "exhausted"
        engine.steps = [{"title": "budget exhausted", "status": "active"}]
        return "global tool call limit reached", engine

    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline._run_execution_phase",
        exhausted_execute,
    )
    _install_stage_planner(monkeypatch, [
        _stage("Attempt", [_task("attempt")]),
        _complete("The queue is empty"),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="exhausted-task", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "blocked"
    assert result.state.stages[0].tasks[0].status == "blocked"
    assert len(runtime["task_plans"]) == 1
    assert len(runtime["executions"]) == 2
    assert runtime["executions"][1]["plan"].overview == (
        runtime["executions"][0]["plan"].overview
    )
    assert runtime["executions"][1]["plan"].steps == (
        runtime["executions"][0]["plan"].steps
    )
    assert runtime["executions"][1]["preserve_task_state"] is True
    assert runtime["executions"][1]["max_total_tool_calls"] == 80
    assert (
        runtime["executions"][1]["max_prompt_tokens"]
        == settings.STAGED_MAX_PROMPT_TOKENS_PER_TASK
    )
    assert "Task attempt" in result.text


def test_prompt_exhausted_task_does_not_retry(
    temp_db, monkeypatch, runtime,
):
    def exhausted_execute(**kwargs):
        runtime["executions"].append(kwargs)
        engine = kwargs["engine"]
        engine._last_status = "exhausted"
        engine._last_state = SimpleNamespace(
            total_prompt_tokens=settings.STAGED_MAX_PROMPT_TOKENS_PER_TASK,
        )
        engine.steps = [{"title": "budget exhausted", "status": "active"}]
        return "prompt token limit reached", engine

    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline._run_execution_phase",
        exhausted_execute,
    )
    _install_stage_planner(monkeypatch, [
        _stage("Attempt", [_task("attempt")]),
        _complete("The queue is empty"),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="prompt-exhausted-task", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "blocked"
    assert len(runtime["executions"]) == 1
    assert (
        runtime["executions"][0]["max_prompt_tokens"]
        == settings.STAGED_MAX_PROMPT_TOKENS_PER_TASK
    )


def test_blocked_task_does_not_suppress_independent_ready_task(
    temp_db, monkeypatch, runtime,
):
    executed: list[str] = []

    def mixed_execute(**kwargs):
        engine = kwargs["engine"]
        description = kwargs["task_prompt"][0]
        current = description.split("<current-task", 1)[1]
        task_title = current.split("Title: ", 1)[1].splitlines()[0]
        executed.append(task_title)
        if task_title == "Task blocked":
            engine._last_status = "blocked"
            engine.steps = [{"title": "blocked", "status": "blocked"}]
            return "blocked result", engine
        engine._last_status = "completed"
        engine.steps = [{"title": "done", "status": "done"}]
        return "independent result", engine

    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline._run_execution_phase",
        mixed_execute,
    )
    _install_stage_planner(monkeypatch, [
        _stage("Mixed DAG", [
            _task("blocked"),
            _task("independent"),
            _task("dependent", depends_on=["blocked"]),
        ]),
        BlockGoalDecision(
            reason="One required dependency is blocked",
            missing="The blocked dependency",
            evidence=[],
        ),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="mixed-dag", project_id=1,
        workspace_path="/workspace",
    )

    assert executed == ["Task blocked", "Task independent"]
    statuses = {
        task.spec.id: task.status for task in result.state.stages[0].tasks
    }
    assert statuses == {
        "blocked": "blocked",
        "independent": "completed",
        "dependent": "blocked",
    }


def test_block_goal_does_not_execute_a_task(temp_db, monkeypatch, runtime):
    _install_stage_planner(monkeypatch, [BlockGoalDecision(
        reason="The singular target has two candidates",
        missing="The user's target choice",
        evidence=["Candidates A and B were observed"],
    )])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="blocked-goal", project_id=1,
        workspace_path="/workspace",
    )

    assert result.state.status == "blocked"
    assert runtime["executions"] == []
    assert "user's target choice" in result.text


def test_resume_mid_stage_skips_completed_dependency(
    temp_db, monkeypatch, runtime,
):
    from infinidev.db.service import persist_staged_planning_state, register_session

    spec = StageSpec(
        title="Resume",
        outcome="Both Tasks complete",
        exit_criteria=["Both results are observed"],
        tasks=[_task("first"), _task("second", depends_on=["first"])],
    )
    state = StagedPlanningState(goal=GoalSpec(
        title="Resume staged work",
        user_request="Resume the persisted staged work until it is complete.",
    ))
    stage = state.add_stage(spec)
    stage.status = "active"
    stage.tasks[0].status = "completed"
    stage.tasks[0].result = "first result"
    state.add_evidence(EvidenceEntry(
        kind="task_result", summary="first result", stage_id=stage.id, task_id="first"
    ))
    register_session("resume", "/workspace")
    persist_staged_planning_state("resume", state.snapshot())
    _install_stage_planner(monkeypatch, [
        _complete("The resumed second Task completed"),
    ])

    result = run_staged_goal(
        escalation=_escalation("continue the active goal"), agent=_Agent(),
        engine=_Engine(), reviewer=object(), hooks=_Hooks(), session_id="resume",
        project_id=1, workspace_path="/workspace",
    )

    assert [handoff.task.id for handoff in runtime["task_plans"]] == ["second"]
    assert result.state.status == "complete"
    assert "continue the active goal" in result.state.guidance


def test_stage_resource_limit_is_incomplete_not_success(
    temp_db, monkeypatch, runtime,
):
    _install_stage_planner(monkeypatch, [
        _stage("First", [_task("first")]),
        _stage("Second", [_task("second")]),
    ])

    result = run_staged_goal(
        escalation=_escalation(), agent=_Agent(), engine=_Engine(), reviewer=object(),
        hooks=_Hooks(), session_id="limit", project_id=1, workspace_path="/workspace",
        max_stage_transitions=1,
    )

    assert result.state.status == "blocked"
    assert result.state.terminal is not None
    assert "resource stop" in result.state.terminal.summary


def test_implementation_goal_cannot_complete_from_read_only_evidence(
    temp_db, monkeypatch, runtime,
):
    _install_stage_planner(monkeypatch, [
        _stage("Inspect", [_task("inspect")]),
        _complete("The inspection result is observed"),
    ])
    engine = _Engine()
    engine.has_changes = False

    result = run_staged_goal(
        escalation=_escalation("Implement a new feedback tool."), agent=_Agent(),
        engine=engine, reviewer=object(), hooks=_Hooks(), session_id="read-only-impl",
        project_id=1, workspace_path="/workspace",
    )

    assert result.state.status == "blocked"
    assert "no observed workspace change" in result.text


def test_informational_goal_can_complete_from_read_only_evidence(
    temp_db, monkeypatch, runtime,
):
    _install_stage_planner(monkeypatch, [
        _stage("Inspect", [_task("inspect")]),
        _complete("The inspection result is observed"),
    ])
    engine = _Engine()
    engine.has_changes = False

    result = run_staged_goal(
        escalation=_escalation("Analiza la arquitectura actual y explica el flujo."),
        agent=_Agent(), engine=engine, reviewer=object(), hooks=_Hooks(),
        session_id="read-only-info", project_id=1, workspace_path="/workspace",
    )

    assert result.state.status == "complete"


def test_reviewing_an_existing_implementation_stays_informational():
    goal = _goal_from_escalation(_escalation(
        "Revisa la implementación actual y explica los riesgos."
    ))

    assert goal.intent == "informational"
