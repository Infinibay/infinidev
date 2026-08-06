"""Best-first scheduler over the graph's executable frontier.

Implements §5 of the design: local freedom with global control. The agent may
move between nodes, but *which* node runs next is chosen here by an
explainable score over the ``ready`` frontier, and every selection carries a
human-readable rationale that the caller persists (``selection_reason``).

The scheduler is read-only over :class:`GraphState`; it never mutates. Budget
fuses (open branches, revisits) are enforced as filters — exhausting them
narrows the frontier, and when nothing is left the caller reports
``blocked``/``needs_user_input`` rather than declaring success (§5.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from infinidev.engine.engines.graph.domain import (
    EXECUTABLE_NODE_TYPES,
    Freshness,
    GraphNode,
    GraphState,
    Lifecycle,
    OPEN_LIFECYCLES,
    TERMINAL_LIFECYCLES,
)


@dataclass(frozen=True)
class SchedulerLimits:
    """Operational fuses for one run (§5.4). Resource ceilings, not goals."""

    max_open_branches: int = 8
    max_node_revisits: int = 4


def _is_schedulable(state: GraphState, node: GraphNode) -> bool:
    """Executable, non-terminal, not invalidated, deps resolved."""
    if node.node_type not in EXECUTABLE_NODE_TYPES:
        return False
    if node.lifecycle not in OPEN_LIFECYCLES:
        return False
    if node.freshness is Freshness.INVALIDATED:
        return False
    for dep_id in state.hard_dependencies(node.node_id):
        dep = state.nodes.get(dep_id)
        if dep is None or dep.lifecycle is not Lifecycle.RESOLVED:
            return False
    return True


def ready_frontier(state: GraphState) -> list[GraphNode]:
    """All nodes that could run right now, ordered by stable id."""
    frontier = [node for node in state.nodes.values() if _is_schedulable(state, node)]
    return sorted(frontier, key=lambda n: n.node_id)


def _open_branch_count(state: GraphState) -> int:
    return sum(
        1 for n in state.nodes.values() if n.lifecycle is Lifecycle.ACTIVE
    )


def _unresolved_dependents(state: GraphState, node_id: str) -> int:
    count = 0
    for dependent_id in state.hard_dependents(node_id):
        dependent = state.nodes.get(dependent_id)
        if dependent is not None and dependent.lifecycle not in TERMINAL_LIFECYCLES:
            count += 1
    return count


def score_frontier(
    state: GraphState,
    frontier: list[GraphNode],
    visits: dict[str, int] | None = None,
) -> list[tuple[GraphNode, float, list[str]]]:
    """Score every frontier node; higher runs sooner.

    Signals (each contributes a reason string so the choice is explainable):
      * explicit ``priority``;
      * unlock value — how many open nodes depend on this one;
      * age — older nodes accrue a bonus to avoid starvation;
      * revisit penalty — already-attempted nodes score lower;
      * staleness penalty — stale grounding needs re-validation first.
    """
    visits = visits or {}
    oldest_first = sorted(frontier, key=lambda n: (n.created_at, n.node_id))
    age_rank = {n.node_id: i for i, n in enumerate(oldest_first)}
    oldest_bonus = max(len(frontier) - 1, 0)

    scored: list[tuple[GraphNode, float, list[str]]] = []
    for node in frontier:
        score = 0.0
        reasons: list[str] = []

        if node.priority:
            score += float(node.priority)
            reasons.append(f"priority={node.priority:g}")

        dependents = _unresolved_dependents(state, node.node_id)
        if dependents:
            score += 1.5 * dependents
            reasons.append(f"unblocks {dependents} node(s)")

        age = oldest_bonus - age_rank[node.node_id]
        if age:
            score += 0.1 * age
            reasons.append(f"waiting (age rank {age_rank[node.node_id]})")

        revisits = visits.get(node.node_id, 0)
        if revisits:
            score -= 0.75 * revisits
            reasons.append(f"already attempted {revisits}x")

        if node.freshness is Freshness.STALE:
            score -= 0.5
            reasons.append("stale — needs re-validation")

        scored.append((node, round(score, 4), reasons))
    return scored


def select_next(
    state: GraphState,
    *,
    visits: dict[str, int] | None = None,
    limits: SchedulerLimits | None = None,
) -> tuple[GraphNode | None, str]:
    """Pick the next node, or ``(None, reason)`` when nothing may run.

    The returned reason is fit to persist as the run's ``selection_reason``
    (§5: "persistir la razón de selección").
    """
    limits = limits or SchedulerLimits()
    visits = visits or {}

    frontier = ready_frontier(state)
    if not frontier:
        active = _open_branch_count(state)
        if active:
            return None, f"no ready nodes; {active} branch(es) still active"
        return None, "no ready nodes and no open branches"

    # Budget fuses. A node already attempted too many times is skipped; too
    # many concurrent branches pauses fresh starts.
    eligible = [
        node for node in frontier
        if visits.get(node.node_id, 0) < limits.max_node_revisits
    ]
    if not eligible:
        return None, (
            "every ready node exceeded its revisit budget "
            f"({limits.max_node_revisits})"
        )

    if _open_branch_count(state) >= limits.max_open_branches:
        # Still allow resuming an already-active/suspended node, but not
        # opening a brand-new branch.
        eligible = [
            node for node in eligible
            if node.lifecycle is not Lifecycle.PROPOSED
        ] or eligible[:0]
        if not eligible:
            return None, (
                f"open-branch budget reached ({limits.max_open_branches}); "
                "resume an existing branch or close one"
            )

    scored = score_frontier(state, eligible, visits)
    scored.sort(key=lambda item: (-item[1], item[0].node_id))
    node, score, reasons = scored[0]
    rationale = f"selected {node.node_id} (score {score}): " + "; ".join(
        reasons or ["only ready node"]
    )
    return node, rationale


__all__ = [
    "SchedulerLimits",
    "ready_frontier",
    "score_frontier",
    "select_next",
]
