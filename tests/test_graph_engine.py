"""Tests for the Graph engine adapter, capsule, completion gates and
persistence/replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.engines.base import (
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
)
from infinidev.engine.engines.graph import completion
from infinidev.engine.engines.graph.context import build_capsule, render_capsule
from infinidev.engine.engines.graph.domain import (
    EDGE_REQUIRES,
    EDGE_SATISFIES,
    EDGE_SUPPORTS,
    Freshness,
    GraphState,
    Lifecycle,
)
from infinidev.engine.engines.graph.engine import (
    _LEAF_INTERRUPTED,
    GraphEngineAdapter,
)
from infinidev.engine.engines.graph.ops import (
    AbandonNodeOp,
    ActivateNodeOp,
    EdgeSpec,
    GraphPatchOp,
    NodeSpec,
    ResolveNodeOp,
    ReviseGoalOp,
)
from infinidev.engine.engines.graph.persistence import GraphPersistence
from infinidev.engine.engines.graph.reducer import reduce
from infinidev.engine.history import store
from infinidev.engine.orchestration.escalation_packet import EscalationPacket


def _escalation(text="Add JWT middleware to the auth module") -> EscalationPacket:
    return EscalationPacket(user_request=text, understanding=text)


def _run_live_leaf_with_status(
    monkeypatch,
    *,
    initial_status: str,
    review_status: str | None = None,
) -> str:
    from infinidev.engine.orchestration import pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod, "_run_gather_phase", lambda **kwargs: kwargs["task_prompt"]
    )

    def review(**kwargs):
        if review_status is not None:
            kwargs["engine"]._last_status = review_status
        return kwargs["result"]

    monkeypatch.setattr(pipeline_mod, "_run_review_phase", review)

    class Agent:
        def activate_context(self, **kwargs):
            pass

        def deactivate(self):
            pass

    class Engine:
        _last_status = initial_status
        is_cancelled = False

        def execute(self, **kwargs):
            return "leaf result"

    class Hooks:
        def on_phase(self, phase):
            pass

        def on_status(self, level, message):
            pass

    _result, status = GraphEngineAdapter()._run_live_leaf(
        capsule_text="active graph node",
        budget={"max_tool_calls": 2},
        node=SimpleNamespace(
            title="Implement middleware",
            node_type="work",
            objective="Implement middleware",
            expected_outcome="Middleware works",
            payload={"deferred_scope": []},
        ),
        kwargs={
            "escalation": _escalation(),
            "agent": Agent(),
            "engine": Engine(),
            "hooks": Hooks(),
            "session_id": "s1",
            "reviewer": None,
        },
        preserve_file_tracker=False,
    )
    return status


# ── Adapter: completed path ──────────────────────────────────────────────────


class TestAdapterCompleted:
    def test_run_completes_and_renders_capsule_to_executor(self):
        captured = {}

        def executor(capsule_text, budget):
            captured["text"] = capsule_text
            return "middleware added and tested"

        adapter = GraphEngineAdapter(executor=executor)
        result = adapter.run(escalation=_escalation(), session_id="s1")

        assert result.status == STATUS_COMPLETED
        assert result.engine_name == "graph_beta"
        assert result.user_message == "middleware added and tested"
        assert result.metrics["leaf_runs"] == 1
        assert result.metrics["max_leaf_runs"] == 12
        assert result.metrics["observed_tool_calls"] == 0
        assert result.metrics["visited_nodes"] == 1
        # The executor received an authority-tagged capsule, not the raw graph.
        text = captured["text"]
        assert '<goal authority="USER_LITERAL">' in text
        assert "<focus-node" in text
        assert "Add JWT middleware" in text

    def test_result_carries_graph_state(self):
        adapter = GraphEngineAdapter(executor=lambda t, b: "done")
        result = adapter.run(escalation=_escalation(), session_id="s1")
        assert result.state is not None
        assert completion.is_goal_complete(result.state)

    def test_grounded_multi_scope_becomes_branches_plus_verification(self):
        escalation = EscalationPacket(
            user_request="Ship the complete GPU feature",
            understanding="Implement and verify the GPU feature",
            grounded_spec=SimpleNamespace(in_scope=[
                "Implement transport",
                "Implement rendering",
                "Add conformance tests",
            ]),
        )
        seen: list[str] = []

        def executor(capsule_text, budget):
            seen.append(capsule_text)
            return f"completed node {len(seen)}"

        result = GraphEngineAdapter(executor=executor).run(
            escalation=escalation,
            session_id="s1",
        )

        assert result.status == STATUS_COMPLETED
        assert len(seen) == 4
        assert sum("Type: work" in capsule for capsule in seen) == 3
        assert "Type: verification" in seen[-1]
        assert "Graph completed all work nodes" in result.user_message
        assert all(f"completed node {i}" in result.user_message for i in range(1, 5))

    def test_live_branches_accumulate_changes_for_integration_and_summary(self):
        escalation = EscalationPacket(
            user_request="Implement both branches and verify the result",
            understanding="Implement and verify both branches",
            grounded_spec=SimpleNamespace(
                in_scope=["Implement transport", "Implement rendering"]
            ),
        )

        class Tracker:
            def __init__(self, labels):
                self.labels = list(labels)

            def merge_from(self, other):
                self.labels = list(other.labels) + self.labels

        engine = SimpleNamespace(
            _last_file_tracker=None,
            _last_total_tool_calls=0,
        )

        class RecordingAdapter(GraphEngineAdapter):
            def __init__(self):
                super().__init__()
                self.calls = []

            def _run_live_leaf(self, **kwargs):
                node = kwargs["node"]
                preserve = kwargs["preserve_file_tracker"]
                tracker = Tracker([node.title])
                if preserve and engine._last_file_tracker is not None:
                    tracker.merge_from(engine._last_file_tracker)
                engine._last_file_tracker = tracker
                engine._last_total_tool_calls = len(self.calls) + 1
                self.calls.append((node.node_type, preserve, node.title))
                return f"completed {node.title}", STATUS_COMPLETED

        adapter = RecordingAdapter()
        result = adapter.run(
            escalation=escalation,
            session_id="s1",
            engine=engine,
        )

        assert result.status == STATUS_COMPLETED
        assert [preserve for _, preserve, _ in adapter.calls] == [
            False,
            False,
            True,
        ]
        assert {title for _, _, title in adapter.calls} == set(
            engine._last_file_tracker.labels
        )
        assert len(engine._last_file_tracker.labels) == 3
        assert engine._last_total_tool_calls == 6
        assert result.metrics["leaf_runs"] == 3
        assert result.metrics["observed_tool_calls"] == 6
        assert result.metrics["visited_nodes"] == 3

    def test_explicit_test_scope_is_the_integrating_verification_node(self):
        escalation = EscalationPacket(
            user_request="Fix both bugs and run tests",
            understanding="Fix and verify",
            grounded_spec=SimpleNamespace(in_scope=[
                "Fix path handling",
                "Fix help formatting",
                "Run the focused pytest suite",
            ]),
        )
        state = GraphEngineAdapter()._seed_state("run", "s1", escalation)

        work = [node for node in state.nodes.values() if node.node_type == "work"]
        verification = [
            node for node in state.nodes.values()
            if node.node_type == "verification"
        ]
        assert len(work) == 2
        assert len(verification) == 1
        assert set(state.hard_dependencies(verification[0].node_id)) == {
            node.node_id for node in work
        }

    def test_inspection_scope_is_marked_evidence_only(self):
        escalation = EscalationPacket(
            user_request="Inspect, compare, and report on the utility",
            understanding="Ground the fix before implementing it",
            grounded_spec=SimpleNamespace(in_scope=[
                "Inspect the current utility implementation",
                "Compare it with the documented contract",
                "Report concrete findings",
            ]),
        )
        state = GraphEngineAdapter()._seed_state("run", "s1", escalation)

        evidence_work = next(
            node for node in state.nodes.values()
            if node.payload.get("evidence_only")
        )
        assert evidence_work.title.startswith("Investigate:")

    def test_generic_inspection_preamble_is_folded_into_implementation(self):
        escalation = EscalationPacket(
            user_request="Fix all utility regressions",
            understanding="Inspect, fix, and verify",
            grounded_spec=SimpleNamespace(in_scope=[
                "Inspect the existing utility implementation",
                "Fix path handling",
                "Fix help formatting",
                "Run the focused tests",
            ]),
        )

        state = GraphEngineAdapter()._seed_state("run", "s1", escalation)

        executable = [
            node for node in state.nodes.values()
            if node.node_type in {"work", "verification"}
        ]
        assert len(executable) == 3
        assert not any(node.payload.get("evidence_only") for node in executable)

    def test_derived_git_branch_scope_requires_literal_git_authority(self):
        escalation = EscalationPacket(
            user_request="Fix each bug in an independent Graph branch",
            understanding="Fix and integrate three bugs",
            grounded_spec=SimpleNamespace(in_scope=[
                "Fix path handling",
                "Fix help formatting",
                "Create each fix on an independent Git branch and merge them",
                "Run the focused tests",
            ]),
        )

        state = GraphEngineAdapter()._seed_state("run", "s1", escalation)

        assert not any(
            "git branch" in node.objective.lower()
            for node in state.nodes.values()
        )

    def test_compound_scope_is_split_and_graph_meta_scope_is_not_work(self):
        escalation = EscalationPacket(
            user_request="Fix three independent regressions in separate Graph work nodes",
            understanding="Fix and verify all three",
            grounded_spec=SimpleNamespace(in_scope=[
                "Apply basename only when shortening paths; guard help slicing only "
                "when a paragraph boundary exists; and preserve module names while "
                "shortening directly executed script paths",
                "Represent each regression in its own Graph work node",
                "Run the exact focused pytest suite",
            ]),
        )

        state = GraphEngineAdapter()._seed_state("run", "s1", escalation)
        work = [node for node in state.nodes.values() if node.node_type == "work"]
        verification = [
            node for node in state.nodes.values()
            if node.node_type == "verification"
        ]

        assert len(work) == 3
        assert len(verification) == 1
        assert not any("graph work node" in node.objective.lower() for node in work)

    def test_literal_enumeration_overrides_collapsed_grounded_scope(self):
        escalation = EscalationPacket(
            user_request=(
                "Fix three regressions. First, restore path shortening. "
                "Second, restore paragraph boundary handling. Third, restore "
                "script basename handling. Use idiomatic code; do not change tests. "
                "Final gate: run pytest."
            ),
            understanding="Fix all three",
            grounded_spec=SimpleNamespace(in_scope=[
                "Inspect the three functions and tests",
                "Change the relevant logic in all three functions",
            ]),
        )

        state = GraphEngineAdapter()._seed_state("run", "s1", escalation)
        work = [node for node in state.nodes.values() if node.node_type == "work"]

        assert [node.objective for node in work] == [
            "restore path shortening",
            "restore paragraph boundary handling",
            "restore script basename handling",
        ]

    def test_scope_cap_reserves_a_slot_for_integration_verification(self):
        from infinidev.engine.engines.graph.scheduler import SchedulerLimits

        escalation = EscalationPacket(
            user_request="Fix four independent bugs",
            understanding="Fix and integrate",
            grounded_spec=SimpleNamespace(in_scope=[
                "Fix transport behavior",
                "Fix rendering behavior",
                "Fix scheduling behavior",
                "Fix persistence behavior",
            ]),
        )
        adapter = GraphEngineAdapter(limits=SchedulerLimits(max_open_branches=3))

        state = adapter._seed_state("run", "s1", escalation)
        executable = [
            node for node in state.nodes.values()
            if node.node_type in {"work", "verification"}
        ]

        assert len(executable) == 3
        assert sum(node.node_type == "verification" for node in executable) == 1

    def test_evidence_only_live_leaf_skips_code_review(self, monkeypatch):
        from infinidev.engine.orchestration import pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod, "_run_gather_phase", lambda **kwargs: kwargs["task_prompt"]
        )

        def unexpected_review(**kwargs):
            raise AssertionError("evidence-only leaves must not enter code review")

        monkeypatch.setattr(pipeline_mod, "_run_review_phase", unexpected_review)

        class Agent:
            def activate_context(self, **kwargs):
                pass

            def deactivate(self):
                pass

        class Engine:
            _last_status = "done"
            is_cancelled = False

            def execute(self, **kwargs):
                self.execute_kwargs = kwargs
                return "grounding evidence"

            def has_file_changes(self):
                return False

        class Hooks:
            def on_phase(self, phase):
                pass

            def on_status(self, level, message):
                pass

        engine = Engine()
        result, status = GraphEngineAdapter()._run_live_leaf(
            capsule_text="inspection node",
            budget={"max_tool_calls": 2},
            node=SimpleNamespace(
                title="Inspect utility",
                node_type="work",
                objective="Inspect the current utility implementation",
                expected_outcome="Concrete findings are recorded",
                payload={"evidence_only": True, "deferred_scope": []},
            ),
            kwargs={
                "escalation": _escalation("Inspect before fixing the utility"),
                "agent": Agent(),
                "engine": engine,
                "hooks": Hooks(),
                "session_id": "s1",
                "reviewer": None,
            },
            preserve_file_tracker=False,
        )

        assert (result, status) == ("grounding evidence", STATUS_COMPLETED)
        assert engine.execute_kwargs["task"].kind == "investigation"
        assert engine.execute_kwargs["skip_plan"] is False
        assert engine.execute_kwargs["allow_plan_mutation"] is False
        assert len(engine.execute_kwargs["initial_plan"].steps) == 1
        assert engine.execute_kwargs["initial_plan"].steps[0].title == "Inspect utility"

    def test_live_leaf_uses_local_plan_under_graph_authority(self, monkeypatch):
        from infinidev.engine.orchestration import pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod, "_run_gather_phase", lambda **kwargs: kwargs["task_prompt"]
        )
        review_kwargs = {}

        def capture_review(**kwargs):
            review_kwargs.update(kwargs)
            return kwargs["result"]

        monkeypatch.setattr(pipeline_mod, "_run_review_phase", capture_review)

        class Agent:
            def activate_context(self, **kwargs):
                pass

            def deactivate(self):
                pass

        class Engine:
            _last_status = "done"
            is_cancelled = False

            def execute(self, **kwargs):
                self.execute_kwargs = kwargs
                return "done"

        class Hooks:
            def on_phase(self, phase):
                pass

            def on_status(self, level, message):
                pass

        engine = Engine()
        adapter = GraphEngineAdapter()
        escalation = EscalationPacket(
            user_request=(
                "Complete the transport and rendering overhaul. "
                "Do not modify tests."
            ),
            understanding="Complete the transport and rendering overhaul",
            grounded_spec=SimpleNamespace(
                in_scope=[],
                out_of_scope=["Changing command-line behavior"],
            ),
        )
        result, status = adapter._run_live_leaf(
            capsule_text="active graph node",
            budget={"max_tool_calls": 1, "token_budget": 1_234},
            node=SimpleNamespace(
                title="Do the thing",
                node_type="work",
                objective="Only change the transport layer",
                expected_outcome="Transport works",
                payload={"deferred_scope": ["Change the rendering layer"]},
            ),
            kwargs={
                "escalation": escalation,
                "agent": Agent(),
                "engine": engine,
                "hooks": Hooks(),
                "session_id": "s1",
                "reviewer": None,
            },
            preserve_file_tracker=False,
        )

        assert (result, status) == ("done", STATUS_COMPLETED)
        assert engine.execute_kwargs["skip_plan"] is False
        assert engine.execute_kwargs["allow_plan_mutation"] is False
        assert engine.execute_kwargs["max_prompt_tokens"] == 1_234
        assert engine.execute_kwargs["initial_plan"].rolling_horizon_limit == 3
        assert len(engine.execute_kwargs["initial_plan"].steps) == 1
        task = engine.execute_kwargs["task"]
        assert task.kind == "feature"
        assert "Only change the transport layer" in task.description
        assert "Do the thing" not in task.description
        assert task.out_of_scope == [
            "Changing command-line behavior",
            "Sibling Graph branch; do not implement in this leaf: "
            "Change the rendering layer"
        ]
        assert task.constraints == [
            "Do not modify tests",
            "Work only on the active Graph node; do not execute sibling branches.",
        ]
        assert (
            "Complete the transport and rendering overhaul"
            not in engine.execute_kwargs["task_prompt"][0]
        )
        assert review_kwargs["run_verification"] is False
        assert review_kwargs["rework_execute_kwargs"] == {
            "skip_plan": False,
            "allow_plan_mutation": False,
            "max_prompt_tokens": 1_234,
        }

    def test_live_leaf_reports_budget_exhaustion_as_resumable(self, monkeypatch):
        from infinidev.engine.orchestration import pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod, "_run_gather_phase", lambda **kwargs: kwargs["task_prompt"]
        )

        class Agent:
            def activate_context(self, **kwargs):
                pass

            def deactivate(self):
                pass

        class Engine:
            _last_status = "exhausted"
            is_cancelled = False

            def execute(self, **kwargs):
                return "Step interrupted at its tool-call boundary."

        class Hooks:
            def on_phase(self, phase):
                pass

            def on_status(self, level, message):
                pass

        result, status = GraphEngineAdapter()._run_live_leaf(
            capsule_text="active graph node",
            budget={"max_tool_calls": 2},
            node=SimpleNamespace(
                title="Implement middleware",
                node_type="work",
                objective="Implement middleware",
                expected_outcome="Middleware works",
                payload={"deferred_scope": []},
            ),
            kwargs={
                "escalation": _escalation(),
                "agent": Agent(),
                "engine": Engine(),
                "hooks": Hooks(),
                "session_id": "s1",
                "reviewer": None,
            },
            preserve_file_tracker=False,
        )

        assert result == "Step interrupted at its tool-call boundary."
        assert status == _LEAF_INTERRUPTED


    @pytest.mark.parametrize(
        ("loop_status", "expected_status"),
        [
            ("cancelled", STATUS_CANCELLED),
            ("", STATUS_FAILED),
            ("unknown", STATUS_FAILED),
        ],
    )
    def test_live_leaf_preserves_non_success_terminal_statuses(
        self, monkeypatch, loop_status, expected_status
    ):
        assert _run_live_leaf_with_status(
            monkeypatch, initial_status=loop_status
        ) == expected_status

    @pytest.mark.parametrize(
        ("review_status", "expected_status"),
        [
            ("cancelled", STATUS_CANCELLED),
            ("failed", STATUS_FAILED),
            ("", STATUS_FAILED),
            ("unknown", STATUS_FAILED),
            ("exhausted", _LEAF_INTERRUPTED),
        ],
    )
    def test_live_leaf_preserves_review_terminal_statuses(
        self, monkeypatch, review_status, expected_status
    ):
        assert _run_live_leaf_with_status(
            monkeypatch,
            initial_status="done",
            review_status=review_status,
        ) == expected_status


# ── Adapter: blocked / budget paths ─────────────────────────────────────────


class TestAdapterBlocked:
    def test_leaf_resume_keeps_evidence_but_refreshes_episode_budgets(self):
        from infinidev.engine.loop.loop_state import LoopState

        state = LoopState(
            iteration_count=3,
            total_tool_calls=20,
            total_tokens=900,
            total_prompt_tokens=800,
            total_completion_tokens=100,
            task_has_edits=True,
            edited_step_indices={1},
            notes=["Implemented the requested API"],
            prompt_composition_history=[{"iteration": 1}],
            request_payload_history=[{"message_count": 2}],
        )

        resumed = GraphEngineAdapter._resume_leaf_state(
            SimpleNamespace(_last_state=state)
        )

        assert resumed is not None
        assert resumed["iteration_count"] == 0
        assert resumed["total_tool_calls"] == 0
        assert resumed["total_tokens"] == 0
        assert resumed["task_has_edits"] is True
        assert resumed["edited_step_indices"] == [1]
        assert resumed["notes"] == ["Implemented the requested API"]
        assert resumed["prompt_composition_history"] == []
        assert resumed["request_payload_history"] == []

    def test_budget_interrupted_leaf_is_checkpointed_and_retried(self):
        class RetryAdapter(GraphEngineAdapter):
            calls = 0
            capsules = []
            resume_flags = []

            def _run_live_leaf(self, **kwargs):
                self.calls += 1
                self.capsules.append(kwargs["capsule_text"])
                self.resume_flags.append(
                    (kwargs["resume_leaf"], kwargs["preserve_file_tracker"])
                )
                if self.calls == 1:
                    return "Tests passed at the budget boundary", _LEAF_INTERRUPTED
                return "Work resumed and completed", STATUS_COMPLETED

        adapter = RetryAdapter(max_leaf_runs=2)
        result = adapter.run(escalation=_escalation(), session_id="s1")

        assert result.status == STATUS_COMPLETED
        assert adapter.calls == 2
        work = next(
            node for node in result.state.nodes.values() if node.node_type == "work"
        )
        assert work.lifecycle is Lifecycle.RESOLVED
        assert "Tests passed at the budget boundary" in adapter.capsules[1]
        assert adapter.resume_flags == [(False, False), (True, True)]

    def test_interrupted_leaf_retry_obeys_revisit_fuse(self):
        from infinidev.engine.engines.graph.scheduler import SchedulerLimits

        class AlwaysInterrupted(GraphEngineAdapter):
            calls = 0

            def _run_live_leaf(self, **kwargs):
                self.calls += 1
                return "budget boundary", _LEAF_INTERRUPTED

        adapter = AlwaysInterrupted(
            limits=SchedulerLimits(max_node_revisits=1),
            max_leaf_runs=3,
        )
        result = adapter.run(escalation=_escalation(), session_id="s1")

        assert result.status == STATUS_BLOCKED
        assert adapter.calls == 1
        assert "revisit budget (1)" in result.summary

    def test_leaf_run_fuse_reports_consumed_budget_metrics(self):
        from infinidev.engine.engines.graph.scheduler import SchedulerLimits

        class AlwaysInterrupted(GraphEngineAdapter):
            calls = 0

            def _run_live_leaf(self, **kwargs):
                self.calls += 1
                return "budget boundary", _LEAF_INTERRUPTED

        adapter = AlwaysInterrupted(
            limits=SchedulerLimits(max_node_revisits=10),
            max_leaf_runs=2,
        )
        result = adapter.run(escalation=_escalation(), session_id="s1")

        assert result.status == STATUS_BLOCKED
        assert result.transition_request is not None
        assert result.transition_request.reason == "graph_leaf_budget_exhausted"
        assert adapter.calls == 2
        assert result.metrics["leaf_runs"] == 2
        assert result.metrics["max_leaf_runs"] == 2
        assert result.metrics["node_visits"] == 2
        assert result.metrics["observed_tool_calls"] == 0

    def test_run_tool_budget_is_shared_across_leaves(self, monkeypatch):
        from infinidev.config.settings import settings

        monkeypatch.setattr(settings, "GRAPH_RUN_TOOL_BUDGET", 5)
        engine = SimpleNamespace(
            _last_file_tracker=None,
            _last_total_tool_calls=0,
        )

        class BudgetedAdapter(GraphEngineAdapter):
            def __init__(self):
                super().__init__()
                self.budgets = []

            def _run_live_leaf(self, **kwargs):
                self.budgets.append(kwargs["budget"]["max_tool_calls"])
                engine._last_total_tool_calls = 4 if len(self.budgets) == 1 else 1
                return "leaf completed", STATUS_COMPLETED

        escalation = EscalationPacket(
            user_request="Implement both branches and verify the result",
            understanding="Implement and verify both branches",
            grounded_spec=SimpleNamespace(
                in_scope=["Implement transport", "Implement rendering"]
            ),
        )
        adapter = BudgetedAdapter()
        result = adapter.run(
            escalation=escalation,
            session_id="s1",
            engine=engine,
        )

        assert adapter.budgets == [5, 1]
        assert result.status == STATUS_BLOCKED
        assert result.transition_request is not None
        assert result.transition_request.reason == (
            "graph_run_tool_budget_exhausted: 5/5 tool calls"
        )
        assert result.metrics["leaf_runs"] == 2
        assert result.metrics["observed_tool_calls"] == 5

    def test_zero_run_tool_budget_disables_only_the_global_fuse(
        self, monkeypatch
    ):
        from infinidev.config.settings import settings

        monkeypatch.setattr(settings, "GRAPH_RUN_TOOL_BUDGET", 0)
        budgets = []

        def executor(capsule_text, budget):
            budgets.append(budget["max_tool_calls"])
            return "leaf completed"

        escalation = EscalationPacket(
            user_request="Implement both branches and verify the result",
            understanding="Implement and verify both branches",
            grounded_spec=SimpleNamespace(
                in_scope=["Implement transport", "Implement rendering"]
            ),
        )
        result = GraphEngineAdapter(executor=executor).run(
            escalation=escalation,
            session_id="s1",
        )

        assert result.status == STATUS_COMPLETED
        assert budgets == [settings.REACT_MAX_TOOL_CALLS] * 3
        assert result.metrics["max_tool_calls"] is None
        assert result.metrics["leaf_runs"] == 3

    def test_revisit_fuse_zero_blocks_immediately(self):
        from infinidev.engine.engines.graph.scheduler import SchedulerLimits

        calls = {"n": 0}

        def executor(capsule_text, budget):
            calls["n"] += 1
            return "did it"

        adapter = GraphEngineAdapter(
            executor=executor,
            limits=SchedulerLimits(max_node_revisits=0),
        )
        result = adapter.run(escalation=_escalation(), session_id="s1")
        # The scheduler refuses every node (revisit budget already at zero),
        # so the run blocks without executing a single leaf.
        assert result.status == STATUS_BLOCKED
        assert calls["n"] == 0

    def test_empty_executor_result_still_completes(self):
        adapter = GraphEngineAdapter(executor=lambda t, b: "", max_leaf_runs=2)
        result = adapter.run(escalation=_escalation(), session_id="s1")
        assert result.status == STATUS_COMPLETED
        assert result.user_message == "Goal completed."

    def test_single_leaf_run_suffices_for_seeded_graph(self):
        calls = {"n": 0}

        def executor(capsule_text, budget):
            calls["n"] += 1
            return "done"

        adapter = GraphEngineAdapter(executor=executor)
        result = adapter.run(escalation=_escalation(), session_id="s1")
        assert result.status == STATUS_COMPLETED
        assert calls["n"] == 1


# ── Capsule ──────────────────────────────────────────────────────────────────


class TestCapsule:
    def _graph_with_dependency(self):
        state = GraphState(run_id="run-cap")
        state, _ = reduce(state, ReviseGoalOp(text="Ship the feature"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="e", node_type="evidence", title="Observed result"),
                NodeSpec(node_id="req1", node_type="requirement",
                         title="Feature required"),
                NodeSpec(node_id="dep", node_type="work", title="Prepare schema"),
                NodeSpec(node_id="w1", node_type="work", title="Implement endpoint"),
            ],
            add_edges=[
                EdgeSpec(source="req1", target="w1", edge_type="decomposes_into"),
                EdgeSpec(source="w1", target="dep", edge_type=EDGE_REQUIRES),
            ],
            based_on_revision=state.revision,
        ))
        return state

    def test_capsule_includes_ancestors_and_dependencies(self):
        state = self._graph_with_dependency()
        # Resolve the dependency so it shows an outcome.
        state, _ = reduce(state, ActivateNodeOp(node_id="dep"))
        state, _ = reduce(state, ResolveNodeOp(
            node_id="dep", evidence_ids=["e"], outcome="schema ready"
        ))

        capsule = build_capsule(state, "w1", selection_reason="unblocks req1")
        assert capsule.goal_text == "Ship the feature"
        assert capsule.focus["node_id"] == "w1"
        ancestor_ids = {a["node_id"] for a in capsule.ancestors}
        assert "req1" in ancestor_ids
        dep_ids = {d["node_id"] for d in capsule.dependencies}
        assert "dep" in dep_ids
        dep_entry = next(d for d in capsule.dependencies if d["node_id"] == "dep")
        assert dep_entry["outcome"] == "schema ready"

    def test_semantic_edges_survive_serialization_round_trip(self):
        state = self._graph_with_dependency()
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(
                    node_id="e1",
                    node_type="evidence",
                    title="Observed schema evidence",
                ),
            ],
            add_edges=[
                EdgeSpec(
                    source="e1",
                    target="w1",
                    edge_type=EDGE_SUPPORTS,
                ),
                EdgeSpec(
                    source="w1",
                    target="req1",
                    edge_type=EDGE_SATISFIES,
                ),
            ],
            based_on_revision=state.revision,
        ))
        restored = GraphState.model_validate_json(state.model_dump_json())

        capsule = build_capsule(restored, "w1")

        assert [item["node_id"] for item in capsule.ancestors] == ["req1"]
        assert [item["node_id"] for item in capsule.evidence] == ["e1"]
        assert GraphEngineAdapter()._requirements_satisfied_by(
            restored, "w1"
        ) == ["req1"]

    def test_render_uses_authority_blocks(self):
        state = self._graph_with_dependency()
        state, _ = reduce(state, ActivateNodeOp(node_id="dep"))
        state, _ = reduce(state, ResolveNodeOp(
            node_id="dep", evidence_ids=["e"], outcome="schema ready"
        ))
        capsule = build_capsule(state, "w1")
        text = render_capsule(capsule)
        assert '<goal authority="USER_LITERAL">' in text
        assert '<focus-node authority="DERIVED">' in text
        assert '<dependencies authority="OBSERVED_EVIDENCE">' in text
        assert "schema ready" in text

    def test_unknown_node_raises(self):
        state = GraphState(run_id="run-cap")
        with pytest.raises(KeyError):
            build_capsule(state, "missing")


# ── Completion gates ─────────────────────────────────────────────────────────


class TestCompletion:
    def test_goal_complete_when_requirements_confirmed(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="e", node_type="evidence"),
                NodeSpec(node_id="r1", node_type="requirement", title="r"),
            ],
            based_on_revision=state.revision,
        ))
        assert not completion.is_goal_complete(state)
        state, _ = reduce(state, ResolveNodeOp(
            node_id="r1", evidence_ids=["e"], verdict="confirmed"
        ))
        assert completion.is_goal_complete(state)

    def test_blocker_blocks_goal(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="e", node_type="evidence"),
                NodeSpec(node_id="r1", node_type="requirement", title="r"),
                NodeSpec(node_id="b1", node_type="blocker", title="needs creds"),
            ],
            based_on_revision=state.revision,
        ))
        state, _ = reduce(state, ResolveNodeOp(
            node_id="r1", evidence_ids=["e"], verdict="confirmed"
        ))
        assessment = completion.evaluate_goal(state)
        assert assessment.status == "blocked"
        assert "needs creds" in assessment.missing

    def test_stale_resolved_blocker_still_blocks_goal(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="e", node_type="evidence"),
                NodeSpec(node_id="r1", node_type="requirement", title="r"),
                NodeSpec(node_id="b1", node_type="blocker", title="needs creds"),
            ],
            based_on_revision=state.revision,
        ))
        state, _ = reduce(
            state, ResolveNodeOp(node_id="r1", evidence_ids=["e"])
        )
        state, _ = reduce(state, ResolveNodeOp(node_id="b1", evidence_ids=[]))
        state.nodes["b1"] = state.nodes["b1"].with_updates(
            freshness=Freshness.STALE
        )

        assessment = completion.evaluate_goal(state)

        assert assessment.status == "blocked"
        assert assessment.missing == ["needs creds"]

    def test_resolved_rejected_blocker_is_closed(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="e", node_type="evidence"),
                NodeSpec(node_id="r1", node_type="requirement", title="r"),
                NodeSpec(node_id="b1", node_type="blocker", title="false alarm"),
            ],
            based_on_revision=state.revision,
        ))
        state, _ = reduce(
            state, ResolveNodeOp(node_id="r1", evidence_ids=["e"])
        )
        state, _ = reduce(
            state,
            ResolveNodeOp(node_id="b1", evidence_ids=[], verdict="rejected"),
        )

        assert completion.is_goal_complete(state)

    def test_confirmed_requirement_does_not_hide_open_work(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="e", node_type="evidence"),
                NodeSpec(node_id="r1", node_type="requirement", title="r"),
                NodeSpec(node_id="w1", node_type="work", title="still open"),
            ],
            based_on_revision=state.revision,
        ))
        state, _ = reduce(state, ResolveNodeOp(
            node_id="r1", evidence_ids=["e"], verdict="confirmed"
        ))

        assessment = completion.evaluate_goal(state)

        assert assessment.status == "in_progress"
        assert "still open" in assessment.missing

    def test_abandoned_requirement_blocks_goal(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="r1", node_type="requirement", title="required"),
            ],
            based_on_revision=state.revision,
        ))
        state, _ = reduce(
            state,
            AbandonNodeOp(node_id="r1", reason="could not satisfy it"),
        )

        assessment = completion.evaluate_goal(state)

        assert assessment.status == "blocked"
        assert assessment.missing == ["required"]

    @pytest.mark.parametrize(
        ("freshness", "expected_missing"),
        [
            (Freshness.STALE, ["required"]),
            (Freshness.INVALIDATED, []),
        ],
    )
    def test_confirmed_requirement_must_be_current(
        self, freshness, expected_missing
    ):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="e", node_type="evidence"),
                NodeSpec(node_id="r1", node_type="requirement", title="required"),
            ],
            based_on_revision=state.revision,
        ))
        state, _ = reduce(
            state,
            ResolveNodeOp(node_id="r1", evidence_ids=["e"], verdict="confirmed"),
        )
        state.nodes["r1"] = state.nodes["r1"].with_updates(freshness=freshness)

        assessment = completion.evaluate_goal(state)

        assert assessment.status == "in_progress"
        assert assessment.missing == expected_missing

    def test_invalidated_historical_requirement_does_not_block_current_goal(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="old-evidence", node_type="evidence"),
                NodeSpec(node_id="current-evidence", node_type="code_ref"),
                NodeSpec(node_id="old", node_type="requirement", title="old"),
                NodeSpec(node_id="current", node_type="requirement", title="current"),
            ],
            based_on_revision=state.revision,
        ))
        state, _ = reduce(
            state, ResolveNodeOp(node_id="old", evidence_ids=["old-evidence"])
        )
        state.nodes["old"] = state.nodes["old"].with_updates(
            freshness=Freshness.INVALIDATED
        )
        state, _ = reduce(
            state, ResolveNodeOp(node_id="current", evidence_ids=["current-evidence"])
        )

        assert completion.is_goal_complete(state)

    def test_rejected_resolved_work_is_not_complete(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[NodeSpec(node_id="w1", node_type="work", title="failed")],
            based_on_revision=state.revision,
        ))
        state, _ = reduce(
            state,
            ResolveNodeOp(node_id="w1", evidence_ids=[], verdict="rejected"),
        )

        assessment = completion.evaluate_goal(state)

        assert assessment.status == "in_progress"
        assert assessment.missing == ["failed"]

    def test_evidence_required_node_without_evidence_is_not_complete(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[NodeSpec(node_id="w1", node_type="work", title="unproven")],
            based_on_revision=state.revision,
        ))
        state.nodes["w1"] = state.nodes["w1"].with_updates(
            lifecycle=Lifecycle.RESOLVED,
            verdict="confirmed",
        )

        assessment = completion.evaluate_goal(state)

        assert assessment.status == "in_progress"
        assert assessment.missing == ["unproven"]

    @pytest.mark.parametrize("proof_id", ["missing", "stale-proof"])
    def test_loaded_state_requires_a_current_materialized_proof(self, proof_id):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="w1", node_type="work", title="legacy work"),
                NodeSpec(node_id="stale-proof", node_type="evidence"),
            ],
            based_on_revision=state.revision,
        ))
        state.nodes["stale-proof"] = state.nodes["stale-proof"].with_updates(
            freshness=Freshness.STALE
        )
        state.nodes["w1"] = state.nodes["w1"].with_updates(
            lifecycle=Lifecycle.RESOLVED,
            verdict="confirmed",
            evidence_refs=[proof_id],
        )

        assessment = completion.evaluate_goal(state)

        assert assessment.status == "in_progress"
        assert assessment.missing == ["legacy work"]


    def test_budget_fuses(self):
        node_budget = completion.NodeBudget(tokens=100, tool_calls=5)
        assert node_budget.exhausted(tokens_used=100) is not None
        assert node_budget.exhausted(tool_calls_used=5) is not None
        assert node_budget.exhausted(tokens_used=10, tool_calls_used=1) is None

        run_budget = completion.RunBudget(tool_calls=10)
        assert run_budget.exhausted(tool_calls_used=10) is not None
        assert run_budget.exhausted(tool_calls_used=1) is None


# ── Persistence & replay ─────────────────────────────────────────────────────


class TestPersistence:
    def test_projection_round_trip(self, temp_db):
        run_id = store.create_run(run_id=None, session_id="s1", engine="graph_beta")
        persistence = GraphPersistence(run_id, session_id="s1")

        state = GraphState(run_id=run_id, session_id="s1")
        state, _ = persistence.apply(state, ReviseGoalOp(text="goal"))
        state, _ = persistence.apply(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="w1", node_type="work", title="task one"),
            ],
            based_on_revision=state.revision,
        ))

        loaded = persistence.load_projection()

        assert loaded.model_dump(mode="json") == state.model_dump(mode="json")

    def test_apply_rolls_back_event_when_projection_fails(
        self, temp_db, monkeypatch
    ):
        run_id = store.create_run(session_id="s1", engine="graph_beta")
        persistence = GraphPersistence(run_id, session_id="s1")
        state = GraphState(run_id=run_id, session_id="s1")

        def fail_projection(*args, **kwargs):
            raise RuntimeError("projection write failed")

        monkeypatch.setattr(persistence, "save_projection", fail_projection)

        with pytest.raises(RuntimeError, match="projection write failed"):
            persistence.apply(state, ReviseGoalOp(text="goal"))

        assert store.list_run_events(run_id) == []

    def test_replay_rebuilds_graph(self, temp_db):
        run_id = store.create_run(session_id="s1", engine="graph_beta")
        persistence = GraphPersistence(run_id, session_id="s1")

        state = GraphState(run_id=run_id, session_id="s1")
        state, _ = persistence.apply(state, ReviseGoalOp(text="goal"))
        state, _ = persistence.apply(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="e", node_type="evidence"),
                NodeSpec(node_id="a", node_type="work", title="A"),
                NodeSpec(node_id="b", node_type="work", title="B"),
            ],
            add_edges=[EdgeSpec(source="a", target="b", edge_type="supports")],
            based_on_revision=state.revision,
        ))
        state, _ = persistence.apply(state, ActivateNodeOp(node_id="a"))
        state, _ = persistence.apply(
            state, ResolveNodeOp(node_id="a", evidence_ids=["e"], outcome="ok")
        )

        replayed = persistence.replay()

        assert replayed.model_dump(mode="json") == state.model_dump(mode="json")

    def test_invalid_op_not_persisted(self, temp_db):
        run_id = store.create_run(session_id="s1", engine="graph_beta")
        persistence = GraphPersistence(run_id, session_id="s1")
        state = GraphState(run_id=run_id, session_id="s1")
        state, _ = persistence.apply(state, ReviseGoalOp(text="goal"))

        from infinidev.engine.engines.graph.reducer import GraphInvariantError

        with pytest.raises(GraphInvariantError):
            persistence.apply(
                state,
                GraphPatchOp(based_on_revision=999, add_nodes=[
                    NodeSpec(node_id="x", node_type="work"),
                ]),
            )
        # No graph_patched event should have been recorded for the bad op.
        events = store.list_run_events(run_id)
        assert all(
            e["payload"].get("op", {}).get("based_on_revision") != 999
            for e in events
        )
