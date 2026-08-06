"""Tests for the graph scheduler: frontier, scoring, budget fuses."""

from __future__ import annotations

from infinidev.engine.engines.graph.domain import (
    EDGE_REQUIRES,
    Freshness,
    GraphState,
    Lifecycle,
)
from infinidev.engine.engines.graph.ops import (
    ActivateNodeOp,
    EdgeSpec,
    GraphPatchOp,
    NodeSpec,
    ResolveNodeOp,
    ReviseGoalOp,
)
from infinidev.engine.engines.graph.reducer import reduce
from infinidev.engine.engines.graph.scheduler import (
    SchedulerLimits,
    ready_frontier,
    score_frontier,
    select_next,
)


def _seed_state() -> GraphState:
    state = GraphState(run_id="run-sched")
    state, _ = reduce(state, ReviseGoalOp(text="goal"))
    return state


def _add_work_nodes(state, ids):
    return reduce(state, GraphPatchOp(
        add_nodes=[NodeSpec(node_id=i, node_type="work", title=i) for i in ids],
        based_on_revision=state.revision,
    ))


class TestReadyFrontier:
    def test_only_executable_open_nodes(self):
        state = _seed_state()
        state, _ = _add_work_nodes(state, ["w1"])
        state, _ = reduce(state, GraphPatchOp(
            add_nodes=[
                NodeSpec(node_id="q1", node_type="question", title="q"),
                NodeSpec(node_id="r1", node_type="requirement", title="r"),
            ],
            based_on_revision=state.revision,
        ))
        frontier = ready_frontier(state)
        ids = {n.node_id for n in frontier}
        assert ids == {"w1"}  # question/requirement are not executable

    def test_dependency_blocks_dependent_until_resolved(self):
        state = _seed_state()
        state, _ = _add_work_nodes(state, ["dep", "w1"])
        state, _ = reduce(state, GraphPatchOp(
            add_edges=[EdgeSpec(source="w1", target="dep", edge_type=EDGE_REQUIRES)],
            based_on_revision=state.revision,
        ))
        # dep not resolved → only dep is ready
        assert {n.node_id for n in ready_frontier(state)} == {"dep"}
        state, _ = reduce(state, ActivateNodeOp(node_id="dep"))
        state, _ = reduce(state, ResolveNodeOp(
            node_id="dep", evidence_ids=["e"], outcome="ok"
        ))
        assert {n.node_id for n in ready_frontier(state)} == {"w1"}

    def test_invalidated_nodes_excluded(self):
        state = _seed_state()
        state, _ = _add_work_nodes(state, ["w1"])
        node = state.nodes["w1"]
        state.nodes["w1"] = node.with_updates(freshness=Freshness.INVALIDATED)
        assert ready_frontier(state) == []


class TestScoring:
    def test_priority_and_unlock_explained(self):
        state = _seed_state()
        state, _ = _add_work_nodes(state, ["a", "b"])
        # b has a dependent → unlock value
        state, _ = _add_work_nodes(state, ["c"])
        state, _ = reduce(state, GraphPatchOp(
            add_edges=[EdgeSpec(source="c", target="b", edge_type=EDGE_REQUIRES)],
            based_on_revision=state.revision,
        ))
        frontier = ready_frontier(state)
        scored = {n.node_id: (score, reasons) for n, score, reasons in
                  score_frontier(state, frontier)}
        b_score, b_reasons = scored["b"]
        a_score, _ = scored["a"]
        assert b_score > a_score
        assert any("unblocks" in r for r in b_reasons)

    def test_revisit_penalty(self):
        state = _seed_state()
        state, _ = _add_work_nodes(state, ["a", "b"])
        frontier = ready_frontier(state)
        scored_with_visit = {
            n.node_id: score for n, score, _ in
            score_frontier(state, frontier, visits={"a": 2})
        }
        scored_clean = {
            n.node_id: score for n, score, _ in
            score_frontier(state, frontier)
        }
        assert scored_with_visit["a"] < scored_clean["a"]

    def test_stale_penalty(self):
        state = _seed_state()
        state, _ = _add_work_nodes(state, ["a"])
        node = state.nodes["a"]
        state.nodes["a"] = node.with_updates(freshness=Freshness.STALE)
        frontier = ready_frontier(state)
        scored = score_frontier(state, frontier)
        assert any("stale" in r for r in scored[0][2])


class TestSelectNext:
    def test_returns_rationale(self):
        state = _seed_state()
        state, _ = _add_work_nodes(state, ["w1"])
        node, rationale = select_next(state)
        assert node is not None and node.node_id == "w1"
        assert "w1" in rationale

    def test_no_ready_nodes_reports_blocked_reason(self):
        state = _seed_state()
        node, reason = select_next(state)
        assert node is None
        assert reason

    def test_revisit_budget_exhausted(self):
        state = _seed_state()
        state, _ = _add_work_nodes(state, ["w1"])
        limits = SchedulerLimits(max_node_revisits=1)
        node, reason = select_next(state, visits={"w1": 1}, limits=limits)
        assert node is None
        assert "revisit" in reason

    def test_open_branch_budget_blocks_new_branches(self):
        state = _seed_state()
        state, _ = _add_work_nodes(state, ["active1", "fresh"])
        state, _ = reduce(state, ActivateNodeOp(node_id="active1"))
        limits = SchedulerLimits(max_open_branches=1)
        node, reason = select_next(state, limits=limits)
        # Only one branch may be open; a fresh PROPOSED node cannot start a
        # new one, so nothing new is selected.
        assert node is None or node.node_id != "fresh"
