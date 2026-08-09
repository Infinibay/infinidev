"""Completion gates for the graph engine.

The design is emphatic here (§5.4, §8.3): exhausting a budget is never the
same as completing the goal, and a good gate is what keeps Graph from
producing a false sense of progress. These helpers answer three questions
purely from the graph state:

* is the governing goal complete?
* is the run blocked (something needs the user or an unresolved blocker)?
* has a budget fuse blown?

Each returns an explainable assessment so the caller can persist *why* a run
closed the way it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infinidev.engine.engines.graph.domain import (
    NODE_BLOCKER,
    NODE_REQUIREMENT,
    Freshness,
    GraphState,
    Lifecycle,
    Verdict,
)


@dataclass
class GoalAssessment:
    """Verdict on the goal plus the reasoning behind it."""

    status: str  # "complete" | "in_progress" | "blocked"
    reasons: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _requirement_nodes(state: GraphState):
    return [n for n in state.nodes.values() if n.node_type == NODE_REQUIREMENT]


def _blocker_nodes(state: GraphState):
    return [
        n for n in state.nodes.values()
        if n.node_type == NODE_BLOCKER and n.lifecycle is not Lifecycle.RESOLVED
    ]


def evaluate_goal(state: GraphState) -> GoalAssessment:
    """Decide whether the governing goal is complete, in progress or blocked.

    A goal is **complete** when every requirement is resolved and confirmed
    and no unresolved contradiction or blocker remains. It is **blocked** when
    an unresolved blocker node exists, or a requirement was abandoned without
    being satisfied. Otherwise it is **in progress**.
    """
    reasons: list[str] = []
    missing: list[str] = []

    blockers = _blocker_nodes(state)
    if blockers:
        for blocker in blockers:
            missing.append(blocker.title or blocker.node_id)
        return GoalAssessment(
            status="blocked",
            reasons=[f"{len(blockers)} unresolved blocker node(s)"],
            missing=missing,
        )

    requirements = _requirement_nodes(state)
    if not requirements:
        # No explicit requirements: fall back to executable work. Complete
        # only if every work/verification node resolved.
        work = [
            n for n in state.nodes.values()
            if n.node_type in {"work", "verification"}
        ]
        if not work:
            return GoalAssessment(
                status="in_progress",
                reasons=["no requirements or work nodes yet"],
            )
        open_work = [
            n for n in work if n.lifecycle is not Lifecycle.RESOLVED
        ]
        if not open_work:
            return GoalAssessment(
                status="complete",
                reasons=["all work nodes resolved"],
            )
        missing = [n.title or n.node_id for n in open_work]
        return GoalAssessment(
            status="in_progress",
            reasons=[f"{len(open_work)} work node(s) still open"],
            missing=missing,
        )

    satisfied = []
    for req in requirements:
        if req.lifecycle is Lifecycle.ABANDONED:
            # An abandoned requirement means the goal changed under us; that
            # needs a user decision, not silent completion.
            missing.append(req.title or req.node_id)
            continue
        if req.lifecycle is Lifecycle.RESOLVED and req.verdict is Verdict.CONFIRMED:
            satisfied.append(req)
        else:
            missing.append(req.title or req.node_id)

    open_work = [
        node for node in state.nodes.values()
        if node.node_type in {"work", "verification"}
        and node.lifecycle is not Lifecycle.RESOLVED
    ]
    if open_work:
        missing.extend(node.title or node.node_id for node in open_work)

    if missing:
        return GoalAssessment(
            status="in_progress",
            reasons=[
                f"{len(satisfied)}/{len(requirements)} requirement(s) satisfied; "
                f"{len(open_work)} executable node(s) still open"
            ],
            missing=missing,
        )

    return GoalAssessment(
        status="complete",
        reasons=[f"all {len(requirements)} requirement(s) resolved and confirmed"],
    )


def is_goal_complete(state: GraphState) -> bool:
    return evaluate_goal(state).status == "complete"


class NodeBudget:
    """Per-node budget fuses (§5.4). Resource ceilings, never success."""

    def __init__(self, *, tokens: int | None = None, tool_calls: int | None = None):
        self.token_budget = tokens
        self.tool_call_budget = tool_calls

    def exhausted(self, *, tokens_used: int = 0, tool_calls_used: int = 0):
        """Return a reason string if a fuse blew, else None."""
        if self.token_budget is not None and tokens_used >= self.token_budget:
            return f"node token budget exhausted ({tokens_used}/{self.token_budget})"
        if self.tool_call_budget is not None and tool_calls_used >= self.tool_call_budget:
            return (
                "node tool-call budget exhausted "
                f"({tool_calls_used}/{self.tool_call_budget})"
            )
        return None


class RunBudget:
    """Run-wide budget fuses (§5.4)."""

    def __init__(self, *, tool_calls: int | None = None, tokens: int | None = None):
        self.tool_call_budget = tool_calls
        self.token_budget = tokens

    def exhausted(self, *, tokens_used: int = 0, tool_calls_used: int = 0):
        if self.tool_call_budget is not None and tool_calls_used >= self.tool_call_budget:
            return (
                "run tool-call budget exhausted "
                f"({tool_calls_used}/{self.tool_call_budget})"
            )
        if self.token_budget is not None and tokens_used >= self.token_budget:
            return f"run token budget exhausted ({tokens_used}/{self.token_budget})"
        return None


__all__ = [
    "GoalAssessment",
    "NodeBudget",
    "RunBudget",
    "evaluate_goal",
    "is_goal_complete",
]
