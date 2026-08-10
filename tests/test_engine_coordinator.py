"""Tests for the engine coordinator and all live engine adapters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.config.settings import settings
from infinidev.engine.analysis.staged_planning import (
    EvidenceEntry,
    GoalSpec,
    GoalTerminalState,
    StageSpec,
    StageTaskSpec,
    StagedPlanningState,
)
from infinidev.engine.engines import run_selected_engine
from infinidev.engine.engines.base import (
    EngineResult,
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    TransitionRequest,
)
from infinidev.engine.engines.react import ReactAdapter
from infinidev.engine.engines.staged_adapter import StagedAdapter
from infinidev.engine.engines.task import TaskAdapter, _bootstrap_step
from infinidev.engine.history import store
from infinidev.engine.orchestration import staged_pipeline as staged_pipeline_mod
from infinidev.engine.orchestration.escalation_packet import EscalationPacket


# ── Fakes ────────────────────────────────────────────────────────────────────


class _Hooks:
    def __init__(self):
        self.statuses = []

    def on_phase(self, phase):
        pass

    def on_status(self, level, message):
        self.statuses.append((level, message))

    def notify(self, *a, **k):
        pass


class _Agent:
    project_id = 1
    workspace_path = "/workspace"

    def activate_context(self, session_id=None):
        pass

    def deactivate(self):
        pass


class _LoopEngine:
    """Minimal LoopEngine double: execute() returns text + sets _last_status."""

    def __init__(self, result_text: str, status: str):
        self._result_text = result_text
        self._status = status
        self._last_status = ""
        self.is_cancelled = False
        self.execute_kwargs = None

    def execute(self, **kwargs):
        self.execute_kwargs = kwargs
        self._last_status = self._status
        return self._result_text

    def has_file_changes(self):
        return False

    def build_work_summary(self, result, status):
        return ""


def _packet(text: str = "Add JWT to all endpoints") -> EscalationPacket:
    return EscalationPacket(user_request=text, understanding=text)


@pytest.fixture
def mode(monkeypatch):
    def _set(value: str):
        monkeypatch.setattr(settings, "TASK_ENGINE_MODE", value)
    original = settings.TASK_ENGINE_MODE
    yield _set
    settings.TASK_ENGINE_MODE = original


# ── Coordinator: staged route ────────────────────────────────────────────────


def _completed_staged_state() -> StagedPlanningState:
    state = StagedPlanningState(
        goal=GoalSpec(title="Add JWT", user_request="Add JWT to all endpoints")
    )
    spec = StageSpec(
        title="Implement", outcome="JWT works", exit_criteria=["ok"],
        tasks=[StageTaskSpec(id="t1", title="Add middleware", outcome="mw",
                             acceptance_criteria=["mw works"])],
    )
    record = state.add_stage(spec)
    record.tasks[0].status = "completed"
    record.tasks[0].result = "done"
    record.status = "completed"
    state.add_evidence(EvidenceEntry(kind="task_result", summary="middleware added"))
    state.status = "complete"
    state.terminal = GoalTerminalState(
        kind="goal_complete", summary="done", evidence=["e1"]
    )
    return state


class TestCoordinatorStaged:
    def test_staged_mode_dispatches_and_records(self, temp_db, monkeypatch, mode):
        mode("staged")
        engine = _LoopEngine("ok", "done")

        def fake_run_staged_goal(**kwargs):
            return staged_pipeline_mod.StagedRunResult(
                text="Goal complete.", engine=engine,
                state=_completed_staged_state(),
            )

        monkeypatch.setattr(
            staged_pipeline_mod, "run_staged_goal", fake_run_staged_goal
        )

        result = run_selected_engine(
            escalation=_packet(), agent=_Agent(), engine=engine, reviewer=None,
            hooks=_Hooks(), session_id="sess-1", project_id=1,
            workspace_path="/workspace",
        )

        assert result.engine_name == "staged"
        assert result.status == STATUS_COMPLETED
        assert result.user_message == "Goal complete."
        assert result.engine is engine

        run = store.get_run(result.run_id)
        assert run["engine"] == "staged"
        assert run["status"] == "completed"
        types = {e["event_type"] for e in store.list_run_events(result.run_id)}
        assert {"run_started", "engine_selected", "task_closed",
                "run_completed", "digest_created"} <= types

    def test_graph_beta_dispatches_graph_and_records(
        self, temp_db, monkeypatch, mode, patched_pipeline
    ):
        mode("graph_beta")
        monkeypatch.setattr(settings, "AUTO_ENGINE_ALLOW_GRAPH", False)
        engine = _LoopEngine("graph leaf done", "done")

        def unexpected_staged(**kwargs):
            raise AssertionError("explicit graph_beta entered the staged planner")

        monkeypatch.setattr(
            staged_pipeline_mod, "run_staged_goal", unexpected_staged
        )
        result = run_selected_engine(
            escalation=_packet(), agent=_Agent(), engine=engine, reviewer=None,
            hooks=_Hooks(), session_id="sess-graph", project_id=1,
            workspace_path="/workspace",
        )

        assert result.engine_name == "graph_beta"
        assert result.status == STATUS_COMPLETED
        assert result.user_message == "graph leaf done"
        assert len(engine.execute_kwargs["initial_plan"].steps) == 1
        assert engine.execute_kwargs["skip_plan"] is False

        run = store.get_run(result.run_id)
        assert run["engine"] == "graph_beta"
        assert run["status"] == "completed"
        event_types = {
            event["event_type"]
            for event in store.list_run_events(result.run_id)
        }
        assert {
            "engine_selected",
            "graph_patched",
            "node_resolved",
            "run_completed",
            "digest_created",
        } <= event_types

    def test_auto_can_dispatch_graph(
        self, temp_db, monkeypatch, mode, patched_pipeline
    ):
        mode("auto")
        monkeypatch.setattr(settings, "AUTO_ENGINE_ALLOW_GRAPH", True)

        def unexpected_staged(**kwargs):
            raise AssertionError("auto-selected Graph entered Staged")

        monkeypatch.setattr(
            staged_pipeline_mod, "run_staged_goal", unexpected_staged
        )
        engine = _LoopEngine("investigation complete", "done")
        result = run_selected_engine(
            escalation=_packet(
                "Investigate alternatives and compare their trade-offs."
            ),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-auto-graph", project_id=1,
            workspace_path="/workspace",
        )

        assert result.engine_name == "graph_beta"
        assert result.status == STATUS_COMPLETED
        assert store.get_run(result.run_id)["mode"] == "auto"


# ── ReactAdapter ─────────────────────────────────────────────────────────────


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Neutralise gather + review so the react loop runs in isolation."""
    from infinidev.engine.orchestration import pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod, "_run_gather_phase",
        lambda **kwargs: kwargs["task_prompt"],
    )
    monkeypatch.setattr(
        pipeline_mod, "_run_review_phase",
        lambda **kwargs: kwargs["result"],
    )


class TestReactAdapter:
    def test_done_maps_to_completed(self, temp_db, patched_pipeline):
        engine = _LoopEngine("Implemented the fix.", "done")
        adapter = ReactAdapter()
        result = adapter.run(
            escalation=_packet("Rename the helper function"),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-1", project_id=1, workspace_path="/workspace",
        )
        assert result.status == STATUS_COMPLETED
        assert result.user_message == "Implemented the fix."
        assert result.transition_request is None
        # The loop ran plan-free with the react budget.
        assert engine.execute_kwargs["initial_plan"] is None
        assert engine.execute_kwargs["skip_plan"] is True
        assert engine.execute_kwargs["allow_explore"] is False
        assert (
            engine.execute_kwargs["max_prompt_tokens"]
            == settings.REACT_MAX_PROMPT_TOKENS
        )
        assert engine.execute_kwargs["max_iterations"] == settings.REACT_MAX_ITERATIONS

    def test_review_rework_preserves_plan_free_mode_and_budget(
        self, temp_db, monkeypatch
    ):
        from infinidev.engine.orchestration import pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod, "_run_gather_phase",
            lambda **kwargs: kwargs["task_prompt"],
        )
        captured = {}

        def review(**kwargs):
            captured.update(kwargs)
            return kwargs["result"]

        monkeypatch.setattr(pipeline_mod, "_run_review_phase", review)
        engine = _LoopEngine("Implemented the fix.", "done")

        ReactAdapter().run(
            escalation=_packet("Rename the helper function"),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-review", project_id=1, workspace_path="/workspace",
        )

        assert captured["task"] is not None
        assert captured["max_iterations"] == settings.REACT_MAX_ITERATIONS
        assert captured["max_total_tool_calls"] == settings.REACT_MAX_TOOL_CALLS
        assert captured["rework_execute_kwargs"] == {"skip_plan": True}

    def test_exhausted_maps_to_blocked_with_escalation(self, temp_db, patched_pipeline):
        engine = _LoopEngine("still working", "exhausted")
        adapter = ReactAdapter()
        result = adapter.run(
            escalation=_packet("Do a big thing"),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-1", project_id=1, workspace_path="/workspace",
        )
        assert result.status == STATUS_BLOCKED
        assert result.transition_request is not None
        assert result.transition_request.target == "staged"
        assert "budget" in result.transition_request.reason

    def test_cancelled_maps_to_cancelled(self, temp_db, patched_pipeline):
        engine = _LoopEngine("partial", "done")
        engine.is_cancelled = True
        adapter = ReactAdapter()
        result = adapter.run(
            escalation=_packet("Do a thing"),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-1", project_id=1, workspace_path="/workspace",
        )
        assert result.status == "cancelled"

    def test_review_rework_closing_blocked_maps_to_blocked(
        self, temp_db, monkeypatch
    ):
        from infinidev.engine.orchestration import pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod, "_run_gather_phase",
            lambda **kwargs: kwargs["task_prompt"],
        )

        def review_that_blocks(**kwargs):
            kwargs["engine"]._last_status = "blocked"
            return kwargs["result"]

        monkeypatch.setattr(pipeline_mod, "_run_review_phase", review_that_blocks)

        engine = _LoopEngine("did it", "done")
        adapter = ReactAdapter()
        result = adapter.run(
            escalation=_packet("Do a thing"),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-1", project_id=1, workspace_path="/workspace",
        )
        assert result.status == STATUS_BLOCKED


class TestTaskAdapter:
    def test_bootstrap_step_is_executable_without_reclassifying_investigation(self):
        implementation = _bootstrap_step(SimpleNamespace(
            title="Transaction restore ordering",
            kind="bugfix",
        ))
        investigation = _bootstrap_step(SimpleNamespace(
            title="Compare queue backends",
            kind="investigation",
        ))
        test_change = _bootstrap_step(SimpleNamespace(
            title="Add focused regression tests",
            kind="bugfix",
        ))
        verification = _bootstrap_step(SimpleNamespace(
            title="Integrated transport outcome",
            kind="verification",
        ))

        assert implementation.title == "Implement Transaction restore ordering"
        assert investigation.title == "Investigate Compare queue backends"
        assert test_change.title == "Add focused regression tests"
        assert verification.title == "Verify Integrated transport outcome"

    def test_task_uses_one_rolling_plan_without_an_analyst(self, temp_db, patched_pipeline):
        engine = _LoopEngine("Implemented the fix.", "done")
        adapter = TaskAdapter()
        hooks = _Hooks()

        result = adapter.run(
            escalation=_packet("Implement the feedback tool"), agent=_Agent(),
            engine=engine, reviewer=None, hooks=hooks, session_id="task-1",
            project_id=1, workspace_path="/workspace",
        )

        assert result.status == STATUS_COMPLETED
        assert result.engine_name == "task"
        assert len(engine.execute_kwargs["initial_plan"].steps) == 1
        assert (
            engine.execute_kwargs["initial_plan"].steps[0].title
            == "Implement the feedback tool"
        )
        assert "rolling" in engine.execute_kwargs["initial_plan"].overview.lower()
        assert engine.execute_kwargs["initial_plan"].rolling_horizon_limit == 3
        assert engine.execute_kwargs["max_iterations"] == settings.TASK_MAX_ITERATIONS
        assert engine.execute_kwargs["max_total_tool_calls"] == 0
        assert (
            engine.execute_kwargs["max_tool_calls_per_action"]
            == settings.TASK_MAX_TOOL_CALLS_PER_STEP
        )
        assert engine.execute_kwargs["allow_explore"] is False
        assert result.metrics["max_tool_calls"] is None
        status_messages = "\n".join(message for _level, message in hooks.statuses)
        assert "unlimited total tool calls" in status_messages
        assert "160 tool calls" not in status_messages

    def test_task_reports_an_explicit_opt_in_total_budget(self, temp_db, patched_pipeline, monkeypatch):
        monkeypatch.setattr(settings, "TASK_MAX_TOOL_CALLS", 240)
        engine = _LoopEngine("Implemented the fix.", "done")
        hooks = _Hooks()

        result = TaskAdapter().run(
            escalation=_packet("Implement the feedback tool"), agent=_Agent(),
            engine=engine, reviewer=None, hooks=hooks, session_id="task-bounded",
            project_id=1, workspace_path="/workspace",
        )

        assert engine.execute_kwargs["max_total_tool_calls"] == 240
        assert result.metrics["max_tool_calls"] == 240
        assert "240 total tool calls" in "\n".join(
            message for _level, message in hooks.statuses
        )


class TestCoordinatorReactRoute:
    def test_react_mode_dispatches_to_react(self, temp_db, monkeypatch, mode,
                                            patched_pipeline):
        mode("react")
        engine = _LoopEngine("done quickly", "done")
        result = run_selected_engine(
            escalation=_packet("Rename the helper function"),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-1", project_id=1, workspace_path="/workspace",
        )
        assert result.engine_name == "react"
        assert result.status == STATUS_COMPLETED
        run = store.get_run(result.run_id)
        assert run["engine"] == "react"

    def test_phase_engine_does_not_override_explicit_react(
        self, temp_db, monkeypatch, mode, patched_pipeline
    ):
        mode("react")
        engine = _LoopEngine("react done", "done")

        def unexpected_staged(**kwargs):
            raise AssertionError("explicit react entered the staged planner")

        monkeypatch.setattr(
            staged_pipeline_mod, "run_staged_goal", unexpected_staged
        )
        result = run_selected_engine(
            escalation=_packet("Do a thing"),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-react-think", project_id=1,
            workspace_path="/workspace", use_phase_engine=True,
        )

        assert result.engine_name == "react"
        assert result.user_message == "react done"

    def test_phase_flag_does_not_bypass_auto_task(
        self, temp_db, monkeypatch, mode, patched_pipeline
    ):
        mode("auto")
        engine = _LoopEngine("ok", "done")
        result = run_selected_engine(
            escalation=_packet("Rename helper"),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-auto-think", project_id=1,
            workspace_path="/workspace", use_phase_engine=True,
        )

        assert result.engine_name == "task"
        assert result.user_message == "ok"

    def test_budget_transition_continues_once_in_staged(
        self, temp_db, monkeypatch, mode
    ):
        mode("react")
        engine = _LoopEngine("read-only progress", "exhausted")
        captured = {}

        def exhausted_react(_self, **_kwargs):
            return EngineResult(
                engine_name="react",
                status=STATUS_BLOCKED,
                user_message="Inspected parser.py; implementation remains.",
                summary="Repository inspection completed without an edit.",
                engine=engine,
                transition_request=TransitionRequest(
                    target="staged", reason="react_budget_exhausted"
                ),
                metrics={"max_tool_calls": 40},
            )

        def completed_staged(_self, **kwargs):
            captured.update(kwargs)
            return EngineResult(
                engine_name="staged",
                status=STATUS_COMPLETED,
                user_message="Implemented and verified.",
                engine=engine,
                state=_completed_staged_state(),
            )

        monkeypatch.setattr(ReactAdapter, "run", exhausted_react)
        monkeypatch.setattr(StagedAdapter, "run", completed_staged)

        result = run_selected_engine(
            escalation=_packet("Implement the parser fix"),
            agent=_Agent(), engine=engine, reviewer=None, hooks=_Hooks(),
            session_id="sess-react-handoff", project_id=1,
            workspace_path="/workspace", turn_context="Existing context.",
        )

        assert result.engine_name == "staged"
        assert result.status == STATUS_COMPLETED
        assert captured["preserve_file_tracker_from_handoff"] is True
        assert "Existing context." in captured["turn_context"]
        assert (
            '<engine-handoff authority="RUNTIME_EVIDENCE"'
            in captured["turn_context"]
        )
        assert "Repository inspection completed" in captured["turn_context"]

        run = store.get_run(result.run_id)
        assert run["engine"] == "react"
        assert run["status"] == "completed"
        assert run["digest_json"]["engine"]["transitions"][0]["applied"] is True
        events = store.list_run_events(result.run_id)
        switched = [event for event in events if event["event_type"] == "engine_switched"]
        assert len(switched) == 1
        assert switched[0]["payload"] == {
            "from": "react",
            "proposed_target": "staged",
            "reason": "react_budget_exhausted",
            "applied": True,
        }
        terminal = [
            event["event_type"] for event in events
            if event["event_type"] in {"run_completed", "run_blocked", "run_failed"}
        ]
        assert terminal == ["run_completed"]

    def test_staged_adapter_forwards_handoff_tracker_flag(
        self, temp_db, monkeypatch
    ):
        captured = {}
        engine = _LoopEngine("done", "done")

        def fake_run_staged_goal(**kwargs):
            captured.update(kwargs)
            return staged_pipeline_mod.StagedRunResult(
                text="Goal complete.", engine=engine,
                state=_completed_staged_state(),
            )

        monkeypatch.setattr(
            staged_pipeline_mod, "run_staged_goal", fake_run_staged_goal
        )
        result = StagedAdapter().run(
            escalation=_packet(), agent=_Agent(), engine=engine, reviewer=None,
            hooks=_Hooks(), session_id="sess-staged-handoff", project_id=1,
            workspace_path="/workspace",
            preserve_file_tracker_from_handoff=True,
        )

        assert result.status == STATUS_COMPLETED
        assert captured["preserve_file_tracker_from_handoff"] is True
