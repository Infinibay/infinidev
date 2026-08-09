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
                return "grounding evidence"

            def has_file_changes(self):
                return False

        class Hooks:
            def on_phase(self, phase):
                pass

            def on_status(self, level, message):
                pass

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
                "engine": Engine(),
                "hooks": Hooks(),
                "session_id": "s1",
                "reviewer": None,
            },
            preserve_file_tracker=False,
        )

        assert (result, status) == ("grounding evidence", STATUS_COMPLETED)

    def test_live_leaf_skips_loop_plan_management(self, monkeypatch):
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
            budget={"max_tool_calls": 1},
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
        assert engine.execute_kwargs["skip_plan"] is True
        task = engine.execute_kwargs["task"]
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

    def test_confirmed_requirement_does_not_hide_open_work(self):
        state = GraphState(run_id="run-c")
        state, _ = reduce(state, ReviseGoalOp(text="goal"))
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
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
