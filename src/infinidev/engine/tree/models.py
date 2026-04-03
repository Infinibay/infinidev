"""Pydantic models for the exploration tree engine.

Decomposes complex problems into sub-problems, explores them recursively
with tools, propagates results upward, and synthesizes findings.

Three phases: INIT (decompose) -> EXPLORE (loop) -> SYNTHESIZE (final).

Brainstorming Mode: for very short problem statements (< 100 chars),
the engine enters a speculative exploration mode where hypotheses can be
generated, factorized into subproblems, and explored through an
intermediate factorization phase (Phase 2.5).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


from infinidev.engine.tree.brainstorming_detector import BrainstormingDetector
from infinidev.engine.tree.fact import Fact
from infinidev.engine.tree.question import Question
from infinidev.engine.tree.blocker import Blocker
from infinidev.engine.tree.tree_node import TreeNode
from infinidev.engine.tree.tree_state import TreeState

# State ranking for propagation (higher = better)
STATE_RANK: dict[str, float] = {
    "unsolvable": 0,
    "needs_decision": 1,
    "needs_experiment": 2,
    "hypothesis": 2.5,  # Speculative -- above needs_experiment, below mitigable
    "mitigable": 3,
    "solvable": 4,
    # Non-terminal states (should not appear in propagation of resolved nodes)
    "pending": -1,
    "exploring": -1,
    "discarded": -2,
}

# Confidence ranking (higher = better)
CONF_RANK: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def propagate(root: TreeNode) -> None:
    """Bottom-up propagation of state, confidence, constraints, and blockers.

    Pure Python -- no LLM calls. Modifies the tree in place.
    """
    _propagate_node(root)


def _propagate_node(node: TreeNode) -> None:
    """Recursively propagate child states up to this node."""
    if not node.children:
        return

    # Recurse into children first (bottom-up)
    for child in node.children:
        _propagate_node(child)

    active = [c for c in node.children if c.state != "discarded"]
    if not active:
        node.state = "unsolvable"
        node.confidence = "high"
        return

    # Constraints and blockers always propagate upward
    for child in active:
        for c in child.constraints:
            if c not in node.constraints:
                node.constraints.append(c)
        for b in child.external_blockers:
            if not any(x.description == b.description for x in node.external_blockers):
                node.external_blockers.append(b)

    if node.logic == "AND":
        # Check if any children are still unresolved
        if any(c.state in ("pending", "exploring") for c in active):
            node.state = "exploring"
            return

        # All resolved -- worst state wins, lowest confidence wins
        worst = min(active, key=lambda c: STATE_RANK.get(c.state, -1))
        node.state = worst.state
        lowest_conf = min(active, key=lambda c: CONF_RANK.get(c.confidence, 0))
        node.confidence = lowest_conf.confidence

    else:  # OR
        resolved = [c for c in active if c.state not in ("pending", "exploring")]
        if not resolved:
            node.state = "exploring"
            return

        # Best resolved state wins; among equal states, highest confidence wins
        best = max(
            resolved,
            key=lambda c: (STATE_RANK.get(c.state, -1), CONF_RANK.get(c.confidence, 0)),
        )
        node.state = best.state
        node.confidence = best.confidence


def select_next_node(tree: TreeState) -> TreeNode | None:
    """Select the next node to explore using DFS with priority.

    Priority: pivot questions > shallower depth > more facts > ID order.
    Excludes nodes already in explored_node_ids.
    """
    pending = [
        n for n in tree.get_pending_nodes()
        if n.id not in tree.explored_node_ids
    ]
    if not pending:
        return None

    def sort_key(n: TreeNode) -> tuple[int, int, int, str]:
        has_unanswered_pivot = any(
            q.question_type == "pivot" and not q.answer
            for q in n.questions
        )
        return (
            0 if has_unanswered_pivot else 1,
            n.depth,
            -len(n.facts),
            n.id,
        )

    pending.sort(key=sort_key)
    return pending[0]
