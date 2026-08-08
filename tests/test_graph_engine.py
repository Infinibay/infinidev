"""Tests for the Graph engine adapter, capsule, completion gates and
persistence/replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.engines.base import STATUS_BLOCKED, STATUS_COMPLETED
from infinidev.engine.engines.graph import completion
from infinidev.engine.engines.graph.context import build_capsule, render_capsule
from infinidev.engine.engines.graph.domain import (
    EDGE_REQUIRES,
    Freshness,
    GraphState,
    Lifecycle,
)
from infinidev.engine.engines.graph.engine import GraphEngineAdapter
from infinidev.engine.engines.graph.ops import (
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

    def test_live_leaf_skips_loop_plan_management(self, monkeypatch):
        from infinidev.engine.orchestration import pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod, "_run_gather_phase", lambda **kwargs: kwargs["task_prompt"]
        )
        monkeypatch.setattr(
            pipeline_mod, "_run_review_phase", lambda **kwargs: kwargs["result"]
        )

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
        result, status = adapter._run_live_leaf(
            capsule_text="active graph node",
            budget={"max_tool_calls": 1},
            node=SimpleNamespace(title="Do the thing"),
            kwargs={
                "escalation": _escalation("Do the thing"),
                "agent": Agent(),
                "engine": engine,
                "hooks": Hooks(),
                "session_id": "s1",
                "reviewer": None,
            },
            preserve_file_tracker=False,
        )

        assert (result, status) == ("done", STATUS_COMPLETED)
        assert engine.execute_kwargs["skip_plan"] is True


# ── Adapter: blocked / budget paths ─────────────────────────────────────────


class TestAdapterBlocked:
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
            add_nodes=[NodeSpec(node_id="r1", node_type="requirement", title="r")],
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
        assert "w1" in loaded.nodes
        assert loaded.nodes["w1"].title == "task one"
        assert loaded.revision == state.revision

    def test_replay_rebuilds_graph(self, temp_db):
        run_id = store.create_run(session_id="s1", engine="graph_beta")
        persistence = GraphPersistence(run_id, session_id="s1")

        state = GraphState(run_id=run_id, session_id="s1")
        state, _ = persistence.apply(state, ReviseGoalOp(text="goal"))
        state, _ = persistence.apply(state, GraphPatchOp(
            add_nodes=[
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
        assert set(replayed.nodes) == set(state.nodes)
        assert replayed.nodes["a"].lifecycle is Lifecycle.RESOLVED
        assert replayed.revision == state.revision

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
