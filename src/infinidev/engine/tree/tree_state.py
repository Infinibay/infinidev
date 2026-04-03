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
from infinidev.engine.tree.tree_node import TreeNode


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


