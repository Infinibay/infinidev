"""Tests for the graph reducer invariants and determinism."""

from __future__ import annotations

import pytest

from infinidev.engine.engines.graph.domain import (
    EDGE_REQUIRES,
    Freshness,
    GraphState,
    Lifecycle,
    Verdict,
)
from infinidev.engine.engines.graph.ops import (
    AbandonNodeOp,
    ActivateNodeOp,
    AttachEvidenceOp,
    CheckpointNodeOp,
    EdgeSpec,
    GraphPatchOp,
    NodeSpec,
    ResolveGoalOp,
    ResolveNodeOp,
    ReviseGoalOp,
    SuspendNodeOp,
)
from infinidev.engine.engines.graph.reducer import GraphInvariantError, reduce


def _state() -> GraphState:
    state = GraphState(run_id="run-test")
    state, _ = reduce(state, ReviseGoalOp(text="Build the feature"))
    return state


def _patch(state, nodes=None, edges=None, revision=None):
    return GraphPatchOp(
        add_nodes=nodes or [],
        add_edges=edges or [],
        based_on_revision=state.revision if revision is None else revision,
    )


class TestGoalRevision:
    def test_revise_goal_increments_revision(self):
        state = GraphState(run_id="r")
        assert state.revision == 0
        state, events = reduce(state, ReviseGoalOp(text="first"))
        assert state.revision == 1
        assert events[0]["event_type"] == "goal_revised"
        state, _ = reduce(state, ReviseGoalOp(text="second"))
        assert state.revision == 2
        assert state.current_goal.text == "second"

    def test_destabilising_revision_marks_open_nodes_stale(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="w1", node_type="work", title="work"),
        ]))
        state, _ = reduce(
            state, ReviseGoalOp(text="changed", classification="replacement")
        )
        assert state.nodes["w1"].freshness is Freshness.STALE

    def test_clarification_does_not_stale_nodes(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="w1", node_type="work", title="work"),
        ]))
        state, _ = reduce(
            state, ReviseGoalOp(text="clarify", classification="clarification")
        )
        assert state.nodes["w1"].freshness is Freshness.CURRENT


class TestGraphPatchInvariants:
    def test_stale_revision_rejected(self):
        state = _state()
        with pytest.raises(GraphInvariantError):
            reduce(state, _patch(state, revision=99, nodes=[
                NodeSpec(node_id="w", node_type="work"),
            ]))

    def test_duplicate_node_rejected(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="w1", node_type="work"),
        ]))
        with pytest.raises(GraphInvariantError):
            reduce(state, _patch(state, nodes=[
                NodeSpec(node_id="w1", node_type="work"),
            ]))

    def test_edge_with_missing_endpoint_rejected(self):
        state = _state()
        with pytest.raises(GraphInvariantError):
            reduce(state, _patch(state, edges=[
                EdgeSpec(source="ghost", target="other", edge_type="supports"),
            ]))

    def test_requires_cycle_rejected(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="a", node_type="work"),
            NodeSpec(node_id="b", node_type="work"),
        ], edges=[
            EdgeSpec(source="a", target="b", edge_type=EDGE_REQUIRES),
        ]))
        with pytest.raises(GraphInvariantError):
            reduce(state, _patch(state, edges=[
                EdgeSpec(source="b", target="a", edge_type=EDGE_REQUIRES),
            ]))

    def test_requires_self_loop_rejected(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="a", node_type="work"),
        ]))
        with pytest.raises(GraphInvariantError):
            reduce(state, _patch(state, edges=[
                EdgeSpec(source="a", target="a", edge_type=EDGE_REQUIRES),
            ]))

    def test_semantic_cycle_allowed(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="h1", node_type="hypothesis"),
            NodeSpec(node_id="h2", node_type="hypothesis"),
        ], edges=[
            EdgeSpec(source="h1", target="h2", edge_type="supports"),
        ]))
        # supports may cycle — it carries meaning, not execution order.
        state, _ = reduce(state, _patch(state, edges=[
            EdgeSpec(source="h2", target="h1", edge_type="supports"),
        ]))
        assert len(state.edges) == 2

    def test_duplicate_edge_is_idempotent(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="a", node_type="work"),
            NodeSpec(node_id="b", node_type="work"),
        ]))
        state, _ = reduce(state, _patch(state, edges=[
            EdgeSpec(source="a", target="b", edge_type="supports"),
        ]))
        state, events = reduce(state, _patch(state, edges=[
            EdgeSpec(source="a", target="b", edge_type="supports"),
        ]))
        assert len(state.edges) == 1
        assert events[0]["payload"]["added_edges"]  # still audited


class TestLifecycleTransitions:
    def test_activate_then_suspend(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="w1", node_type="work"),
        ]))
        state, _ = reduce(state, ActivateNodeOp(node_id="w1"))
        assert state.nodes["w1"].lifecycle is Lifecycle.ACTIVE
        state, _ = reduce(
            state, SuspendNodeOp(node_id="w1", reason="dep elsewhere",
                                 checkpoint="halfway")
        )
        assert state.nodes["w1"].lifecycle is Lifecycle.SUSPENDED
        assert state.nodes["w1"].checkpoint == "halfway"

    def test_resolve_requires_evidence_for_work(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="w1", node_type="work"),
        ]))
        with pytest.raises(GraphInvariantError):
            reduce(state, ResolveNodeOp(node_id="w1", evidence_ids=[]))

    def test_resolve_with_evidence_and_promotes_dependents(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="dep", node_type="work"),
            NodeSpec(node_id="w1", node_type="work"),
        ], edges=[
            EdgeSpec(source="w1", target="dep", edge_type=EDGE_REQUIRES),
        ]))
        # w1 requires dep; w1 stays not-ready until dep resolves.
        state, _ = reduce(state, ActivateNodeOp(node_id="dep"))
        state, _ = reduce(
            state, ResolveNodeOp(node_id="dep", evidence_ids=["e1"], outcome="ok")
        )
        assert state.nodes["dep"].lifecycle is Lifecycle.RESOLVED
        assert state.nodes["dep"].verdict is Verdict.CONFIRMED
        # dependent promoted proposed → ready
        assert state.nodes["w1"].lifecycle is Lifecycle.READY

    def test_terminal_node_is_immutable(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="q1", node_type="question"),
        ]))
        # questions don't require evidence
        state, _ = reduce(state, ResolveNodeOp(node_id="q1", evidence_ids=[]))
        with pytest.raises(GraphInvariantError):
            reduce(state, ActivateNodeOp(node_id="q1"))

    def test_abandon_blocks_later_abandon_of_resolved(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="q1", node_type="question"),
        ]))
        state, _ = reduce(state, ResolveNodeOp(node_id="q1", evidence_ids=[]))
        with pytest.raises(GraphInvariantError):
            reduce(state, AbandonNodeOp(node_id="q1", reason="nope"))

    def test_checkpoint_updates_without_lifecycle_change(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="w1", node_type="work"),
        ]))
        state, _ = reduce(state, ActivateNodeOp(node_id="w1"))
        state, _ = reduce(
            state, CheckpointNodeOp(node_id="w1", checkpoint="read the module",
                                    reason="switching branches")
        )
        assert state.nodes["w1"].checkpoint == "read the module"
        assert state.nodes["w1"].lifecycle is Lifecycle.ACTIVE


class TestAttachEvidenceAndResolveGoal:
    def test_attach_evidence_idempotent(self):
        state = _state()
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="w1", node_type="work"),
        ]))
        state, _ = reduce(state, AttachEvidenceOp(node_id="w1", evidence_id="e1"))
        state, events = reduce(
            state, AttachEvidenceOp(node_id="w1", evidence_id="e1")
        )
        assert state.nodes["w1"].evidence_refs == ["e1"]
        assert events == []

    def test_resolve_goal_requires_matching_revision(self):
        state = _state()
        with pytest.raises(GraphInvariantError):
            reduce(state, ResolveGoalOp(revision_id=5, evidence_ids=["e"]))

    def test_resolve_goal_requires_evidence(self):
        state = _state()
        with pytest.raises(GraphInvariantError):
            reduce(state, ResolveGoalOp(revision_id=state.revision, evidence_ids=[]))

    def test_resolve_goal_emits_event(self):
        state = _state()
        state, events = reduce(
            state, ResolveGoalOp(revision_id=state.revision, evidence_ids=["e1"])
        )
        assert events[0]["event_type"] == "goal_resolved"


class TestDeterminism:
    def test_same_ops_produce_same_graph(self):
        def build():
            state = GraphState(run_id="r")
            state, _ = reduce(state, ReviseGoalOp(text="g"))
            state, _ = reduce(state, _patch(state, nodes=[
                NodeSpec(node_id="a", node_type="work"),
                NodeSpec(node_id="b", node_type="work"),
            ], edges=[
                EdgeSpec(source="a", target="b", edge_type=EDGE_REQUIRES),
            ]))
            return state

        s1 = build()
        s2 = build()
        assert set(s1.nodes) == set(s2.nodes)
        assert set(s1.edges) == set(s2.edges)
        # Edge ids are content-derived, so they match across builds.
        assert list(s1.edges.keys()) == list(s2.edges.keys())

    def test_reduce_does_not_mutate_input_state(self):
        state = _state()
        before = state.model_dump_json()
        reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="w1", node_type="work"),
        ]))
        assert state.model_dump_json() == before

    def test_version_increments_once_per_op(self):
        state = _state()
        v0 = state.version
        state, _ = reduce(state, _patch(state, nodes=[
            NodeSpec(node_id="w1", node_type="work"),
        ]))
        assert state.version == v0 + 1
