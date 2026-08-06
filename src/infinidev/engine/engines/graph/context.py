"""NodeContextCapsule — the slice of graph an activation actually sees.

Implements §5.3 of the design. Context follows the exploration stack
conceptually but is materialized as a rebuildable capsule: when the agent
returns to a node from another path it gets the node's goal, authoritative
ancestors, resolved dependencies, evidence, prior checkpoint and remaining
budget — never the whole graph. Capsules are therefore bounded by
construction, which is what lets the graph grow without the context window.

This module owns the data and its rendering. The rendering uses the same
authority-tagged block vocabulary as the staged/react task prompts so a
developer loop reads a graph capsule exactly like any other task context.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from infinidev.engine.engines.graph.domain import (
    EDGE_DECOMPOSES_INTO,
    EDGE_REQUIRES,
    EDGE_SUPPORTS,
    Freshness,
    GraphNode,
    GraphState,
    Lifecycle,
)

#: How far up the decomposition tree the capsule walks for ancestors.
MAX_ANCESTOR_DEPTH = 5
#: How many sibling/related nodes the capsule surfaces.
MAX_NEIGHBORS = 6
#: How many recent goal revisions the capsule carries.
MAX_RECENT_REVISIONS = 3


class NodeContextCapsule(BaseModel):
    """Everything one node activation needs, nothing it does not."""

    run_id: str
    node_id: str
    goal_text: str = ""
    goal_revision: int = 0
    focus: dict[str, Any] = Field(default_factory=dict)
    ancestors: list[dict[str, Any]] = Field(default_factory=list)
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    prior_checkpoint: str = ""
    recent_revisions: list[dict[str, Any]] = Field(default_factory=list)
    neighbors: list[dict[str, Any]] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    ken_refs: list[str] = Field(default_factory=list)
    selection_reason: str = ""


def _node_summary(node: GraphNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "title": node.title,
        "lifecycle": node.lifecycle.value,
        "verdict": node.verdict.value,
        "freshness": node.freshness.value,
        "checkpoint": node.checkpoint,
    }


def _ancestors(state: GraphState, node_id: str) -> list[GraphNode]:
    """Walk ``decomposes_into`` upward: parents, grandparents, …"""
    found: list[GraphNode] = []
    seen: set[str] = {node_id}
    frontier = [node_id]
    depth = 0
    while frontier and depth < MAX_ANCESTOR_DEPTH:
        next_frontier: list[str] = []
        for current in frontier:
            for edge in state.edges_to(current):
                if edge.edge_type is not EDGE_DECOMPOSES_INTO:
                    continue
                parent = state.nodes.get(edge.source)
                if parent is None or parent.node_id in seen:
                    continue
                seen.add(parent.node_id)
                found.append(parent)
                next_frontier.append(parent.node_id)
        frontier = next_frontier
        depth += 1
    return found


def _dependencies(state: GraphState, node_id: str) -> list[dict[str, Any]]:
    deps: list[dict[str, Any]] = []
    for dep_id in state.hard_dependencies(node_id):
        dep = state.nodes.get(dep_id)
        if dep is None:
            continue
        summary = _node_summary(dep)
        # A resolved dependency contributes its outcome so the node can build
        # on it without re-deriving it.
        summary["outcome"] = dep.checkpoint
        deps.append(summary)
    return deps


def _evidence(state: GraphState, node: GraphNode) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(candidate: GraphNode | None) -> None:
        if candidate is None or candidate.node_id in seen:
            return
        seen.add(candidate.node_id)
        evidence.append(_node_summary(candidate))

    # Evidence the node already holds by reference.
    for ref in node.evidence_refs:
        _add(state.nodes.get(ref))
    # Evidence wired in through ``supports`` edges.
    for edge in state.edges_to(node.node_id):
        if edge.edge_type is not EDGE_SUPPORTS:
            continue
        source = state.nodes.get(edge.source)
        if source is not None and source.node_type == "evidence":
            _add(source)
    return evidence


def _neighbors(
    state: GraphState, node_id: str, exclude: set[str]
) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []
    seen: set[str] = set(exclude)
    for edge in list(state.edges_from(node_id)) + list(state.edges_to(node_id)):
        other_id = edge.target if edge.source == node_id else edge.source
        if other_id in seen or len(neighbors) >= MAX_NEIGHBORS:
            continue
        other = state.nodes.get(other_id)
        if other is None:
            continue
        seen.add(other_id)
        summary = _node_summary(other)
        summary["relation"] = edge.edge_type
        neighbors.append(summary)
    return neighbors


def build_capsule(
    state: GraphState,
    node_id: str,
    *,
    ken_refs: list[str] | None = None,
    budget: dict[str, Any] | None = None,
    selection_reason: str = "",
) -> NodeContextCapsule:
    """Assemble the capsule for *node_id* from the current graph state."""
    node = state.nodes.get(node_id)
    if node is None:
        raise KeyError(f"unknown node_id {node_id!r}")

    goal = state.current_goal
    ancestors = _ancestors(state, node_id)
    dependencies = _dependencies(state, node_id)
    evidence = _evidence(state, node)

    accounted = {node_id}
    accounted.update(a.node_id for a in ancestors)
    accounted.update(d["node_id"] for d in dependencies)
    accounted.update(e["node_id"] for e in evidence)
    neighbors = _neighbors(state, node_id, accounted)

    recent_revisions = [
        {
            "revision": rev.revision,
            "text": rev.text,
            "classification": rev.classification,
        }
        for rev in state.goal_revisions[-MAX_RECENT_REVISIONS:]
    ]

    return NodeContextCapsule(
        run_id=state.run_id,
        node_id=node_id,
        goal_text=goal.text if goal is not None else "",
        goal_revision=state.revision,
        focus=_node_summary(node),
        ancestors=[_node_summary(a) for a in ancestors],
        dependencies=dependencies,
        evidence=evidence,
        prior_checkpoint=node.checkpoint,
        recent_revisions=recent_revisions,
        neighbors=neighbors,
        budget=budget or {},
        ken_refs=ken_refs or [],
        selection_reason=selection_reason,
    )


# ── rendering ────────────────────────────────────────────────────────────────


def _fmt_lines(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        label = item.get("title") or item.get("node_id")
        bits = [f"[{item.get('node_type')}] {label}"]
        lifecycle = item.get("lifecycle")
        if lifecycle:
            bits.append(f"({lifecycle})")
        outcome = item.get("outcome")
        if outcome:
            bits.append(f"— {outcome}")
        relation = item.get("relation")
        if relation:
            bits.append(f"via {relation}")
        lines.append("- " + " ".join(bits))
    return "\n".join(lines) if lines else "- none"


def render_capsule(capsule: NodeContextCapsule) -> str:
    """Render the capsule into the authority-tagged task-context block.

    Mirrors the staged/react prompt blocks: the goal is USER_LITERAL, the
    working context is DERIVED, resolved dependencies and evidence are
    OBSERVED_EVIDENCE. Advisory retrieval (Ken) is labelled advisory.
    """
    focus = capsule.focus
    focus_lines = [
        f"Type: {focus.get('node_type')}",
        f"Title: {focus.get('title')}",
    ]
    if focus.get("checkpoint"):
        focus_lines.append(f"Checkpoint so far: {focus['checkpoint']}")

    parts: list[str] = []
    if capsule.goal_text:
        parts.append(
            '<goal authority="USER_LITERAL">\n'
            f"{capsule.goal_text}\n"
            "</goal>"
        )

    parts.append(
        '<focus-node authority="DERIVED">\n'
        + "\n".join(focus_lines)
        + "\n</focus-node>"
    )

    if capsule.selection_reason:
        parts.append(
            '<selection-reason authority="DERIVED">\n'
            f"{capsule.selection_reason}\n"
            "</selection-reason>"
        )

    if capsule.ancestors:
        parts.append(
            '<ancestors authority="DERIVED">\n'
            f"{_fmt_lines(capsule.ancestors)}\n"
            "</ancestors>"
        )

    if capsule.dependencies:
        parts.append(
            '<dependencies authority="OBSERVED_EVIDENCE">\n'
            f"{_fmt_lines(capsule.dependencies)}\n"
            "</dependencies>"
        )

    if capsule.evidence:
        parts.append(
            '<evidence authority="OBSERVED_EVIDENCE">\n'
            f"{_fmt_lines(capsule.evidence)}\n"
            "</evidence>"
        )

    if capsule.recent_revisions:
        lines = [
            f"- rev {r['revision']} ({r['classification']}): {r['text']}"
            for r in capsule.recent_revisions
        ]
        parts.append(
            '<recent-goal-revisions authority="USER_LITERAL">\n'
            + "\n".join(lines)
            + "\n</recent-goal-revisions>"
        )

    if capsule.neighbors:
        parts.append(
            '<neighbors authority="DERIVED">\n'
            f"{_fmt_lines(capsule.neighbors)}\n"
            "</neighbors>"
        )

    if capsule.budget:
        budget_bits = ", ".join(f"{k}={v}" for k, v in capsule.budget.items())
        parts.append(
            '<budget authority="DERIVED">\n' f"{budget_bits}\n" "</budget>"
        )

    if capsule.ken_refs:
        parts.append(
            '<retrieval-context authority="advisory" scope-effect="none">\n'
            + "\n".join(f"- {ref}" for ref in capsule.ken_refs)
            + "\n</retrieval-context>"
        )

    return "\n\n".join(parts)


__all__ = ["NodeContextCapsule", "build_capsule", "render_capsule"]
