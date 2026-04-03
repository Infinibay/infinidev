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


class BrainstormingDetector:
    """Detects when to enter brainstorming mode based on problem characteristics.

    Triggers brainstorming mode when:
    - Problem statement is very short (< 100 characters)
    - Problem is empty or whitespace-only
    """

    BRAINSTORMING_THRESHOLD = 100  # Maximum problem length to trigger brainstorming

    def should_brainstorm(self, problem: str) -> bool:
        """Determine if the system should enter brainstorming mode.

        Args:
            problem: The problem statement from the user.

        Returns:
            True if brainstorming mode should be activated.
        """
        if not problem or not problem.strip():
            return True

        return len(problem) < self.BRAINSTORMING_THRESHOLD


class Fact(BaseModel):
    """A verified or asserted piece of knowledge within a tree node."""

    content: str
    source: Literal["initial", "discovered", "inherited"] = "discovered"
    evidence: str = ""  # Raw tool output backing this fact
    source_tool: str = ""  # Which tool produced the evidence
    confidence: Literal["high", "medium", "low"] = "medium"


class Question(BaseModel):
    """A question to be answered during node exploration."""

    content: str
    question_type: Literal["pivot", "informational"] = "informational"
    answer: str | None = None
    answered_by_tool: str | None = None
    impact: str | None = None  # What changed when answered (for pivots)


class Blocker(BaseModel):
    """An external blocker preventing node resolution."""

    description: str
    blocker_type: Literal["api", "library", "permission", "infra", "unknown"] = "unknown"
    workaround: str | None = None


class TreeNode(BaseModel):
    """A node in the exploration tree representing a problem or sub-problem."""

    id: str  # Hierarchical: "1", "1.1", "1.1.2"
    problem_statement: str
    problem_reformulated: str | None = None
    facts: list[Fact] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    children: list[TreeNode] = Field(default_factory=list)
    logic: Literal["AND", "OR"] = "AND"
    state: Literal[
        "pending", "exploring", "solvable", "unsolvable",
        "mitigable", "needs_decision", "needs_experiment", "discarded",
        "hypothesis",  # Speculative approach when facing information gaps
    ] = "pending"
    confidence: Literal["high", "medium", "low"] = "low"
    constraints: list[str] = Field(default_factory=list)
    external_blockers: list[Blocker] = Field(default_factory=list)
    discard_reason: str | None = None
    exploration_summary: str | None = None
    hypothesis_content: str | None = None  # Speculative content for brainstorming mode
    tool_calls_count: int = 0
    depth: int = 0

    def add_child(self, problem: str, logic: Literal["AND", "OR"] = "AND") -> TreeNode:
        """Create and append a child node with auto-generated ID."""
        child_index = len(self.children) + 1
        child_id = f"{self.id}.{child_index}"
        child = TreeNode(
            id=child_id,
            problem_statement=problem,
            logic=logic,
            depth=self.depth + 1,
        )
        self.children.append(child)
        return child

    def is_resolved(self) -> bool:
        """Whether this node has reached a terminal state."""
        return self.state in ("solvable", "unsolvable", "mitigable", "discarded", "hypothesis")

    def collect_inheritable_facts(self) -> list[Fact]:
        """Return discovered facts that can be inherited by siblings/parent."""
        return [f for f in self.facts if f.source == "discovered"]


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


class TreeState(BaseModel):
    """Full state of the exploration tree engine across iterations."""

    root: TreeNode | None = None
    current_node_id: str | None = None
    explored_node_ids: list[str] = Field(default_factory=list)
    iteration_count: int = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    synthesis: str | None = None

    def get_node(self, node_id: str) -> TreeNode | None:
        """Find a node by its hierarchical ID."""
        if self.root is None:
            return None
        return self._find_node(self.root, node_id)

    def _find_node(self, node: TreeNode, node_id: str) -> TreeNode | None:
        if node.id == node_id:
            return node
        for child in node.children:
            found = self._find_node(child, node_id)
            if found is not None:
                return found
        return None

    def get_path_to_node(self, node_id: str) -> list[TreeNode]:
        """Return the path from root to the given node (inclusive)."""
        if self.root is None:
            return []
        path: list[TreeNode] = []
        self._find_path(self.root, node_id, path)
        return path

    def _find_path(self, node: TreeNode, node_id: str, path: list[TreeNode]) -> bool:
        path.append(node)
        if node.id == node_id:
            return True
        for child in node.children:
            if self._find_path(child, node_id, path):
                return True
        path.pop()
        return False

    def get_siblings(self, node_id: str) -> list[TreeNode]:
        """Return sibling nodes (same parent, excluding self)."""
        if self.root is None:
            return []
        parent = self._find_parent(self.root, node_id)
        if parent is None:
            return []
        return [c for c in parent.children if c.id != node_id]

    def _find_parent(self, node: TreeNode, child_id: str) -> TreeNode | None:
        for child in node.children:
            if child.id == child_id:
                return node
            found = self._find_parent(child, child_id)
            if found is not None:
                return found
        return None

    def get_pending_nodes(self) -> list[TreeNode]:
        """Return all nodes in pending or needs_experiment state."""
        if self.root is None:
            return []
        result: list[TreeNode] = []
        self._collect_pending(self.root, result)
        return result

    def _collect_pending(self, node: TreeNode, result: list[TreeNode]) -> None:
        if node.state in ("pending", "needs_experiment") and not node.children:
            # Only leaf nodes (no children) should be directly explored.
            # Parent nodes are resolved by propagation from children.
            result.append(node)
        for child in node.children:
            self._collect_pending(child, result)

    def count_nodes(self) -> int:
        """Count total nodes in the tree."""
        if self.root is None:
            return 0
        return self._count(self.root)

    def _count(self, node: TreeNode) -> int:
        return 1 + sum(self._count(c) for c in node.children)


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
