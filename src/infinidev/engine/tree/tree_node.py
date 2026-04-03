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
from infinidev.engine.tree.blocker import Blocker
from infinidev.engine.tree.fact import Fact
from infinidev.engine.tree.question import Question


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


