"""Deterministic transactional reducer for the work graph.

The single writer of graph state (docs/GRAPH_ENGINE_BETA_DESIGN.md §5.1).
Every mutation arrives as one :mod:`ops <infinidev.engine.engines.graph.ops>`
object; :func:`reduce` validates the invariants, applies the change to a copy
of the state, and returns the new state plus the audit events it produced.

The function is pure — no I/O, no randomness, no generated ids of its own
(edge ids derive from content) — so replaying the event log through it always
rebuilds the same graph (§10). Persistence is somebody else's job; see
``persistence.py``.

Invariants enforced (§6):

* ids referenced by an operation exist;
* ``graph_patch.based_on_revision`` matches the current goal revision;
* no ``requires`` cycle is created;
* resolved nodes carry evidence when their type demands it;
* terminal nodes are immutable;
* every applied operation bumps the write version exactly once.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from infinidev.engine.engines.graph.domain import (
    EVIDENCE_REQUIRED_NODE_TYPES,
    HARD_EDGE_TYPES,
    KNOWN_EDGE_TYPES,
    KNOWN_NODE_TYPES,
    KNOWN_REVISION_KINDS,
    Freshness,
    GoalRevision,
    GraphEdge,
    GraphNode,
    GraphState,
    Lifecycle,
    Verdict,
)
from infinidev.engine.engines.graph.ops import (
    AbandonNodeOp,
    ActivateNodeOp,
    AttachEvidenceOp,
    CheckpointNodeOp,
    GraphOp,
    GraphPatchOp,
    ResolveGoalOp,
    ResolveNodeOp,
    ReviseGoalOp,
    SuspendNodeOp,
)
from infinidev.engine.history import events as ev

logger = logging.getLogger(__name__)


class GraphInvariantError(ValueError):
    """An operation violated a graph invariant and was rejected."""


def _edge_id(source: str, target: str, edge_type: str) -> str:
    """Deterministic id so replaying the same patch yields the same edge."""
    digest = hashlib.sha256(f"{source}\x00{target}\x00{edge_type}".encode())
    return f"edge_{digest.hexdigest()[:14]}"


def _event(
    event_type: str,
    payload: dict[str, Any],
    *,
    node_id: str | None = None,
    goal_revision: int | None = None,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "payload": payload,
        "node_id": node_id,
        "goal_revision": goal_revision,
    }


def _require_node(state: GraphState, node_id: str) -> GraphNode:
    node = state.nodes.get(node_id)
    if node is None:
        raise GraphInvariantError(f"unknown node_id {node_id!r}")
    return node


def _require_non_terminal(node: GraphNode) -> None:
    if node.lifecycle in {Lifecycle.RESOLVED, Lifecycle.ABANDONED}:
        raise GraphInvariantError(
            f"node {node.node_id!r} is {node.lifecycle.value}; terminal nodes "
            "are immutable — supersede it instead"
        )


def _would_create_hard_cycle(
    state: GraphState, source: str, target: str
) -> bool:
    """True when adding ``source requires target`` closes a hard loop.

    A cycle appears exactly when *target* can already reach *source* through
    existing hard edges.
    """
    stack = [target]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == source:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(state.hard_dependencies(current))
    return False


def _promote_ready(state: GraphState, changed_node_id: str) -> None:
    """Move dependents PROPOSED→READY once all hard deps are resolved."""
    for dependent_id in state.hard_dependents(changed_node_id):
        dependent = state.nodes.get(dependent_id)
        if dependent is None or dependent.lifecycle is not Lifecycle.PROPOSED:
            continue
        deps = [state.nodes.get(d) for d in state.hard_dependencies(dependent_id)]
        if all(d is not None and d.lifecycle is Lifecycle.RESOLVED for d in deps):
            state.nodes[dependent_id] = dependent.with_updates(
                lifecycle=Lifecycle.READY
            )


# ── per-op application ───────────────────────────────────────────────────────


def _apply_graph_patch(
    state: GraphState, op: GraphPatchOp
) -> list[dict[str, Any]]:
    if op.based_on_revision != state.revision:
        raise GraphInvariantError(
            f"graph_patch based on revision {op.based_on_revision} but the "
            f"goal is at revision {state.revision}; re-read the goal first"
        )

    for spec in op.add_nodes:
        if spec.node_id in state.nodes:
            raise GraphInvariantError(f"node {spec.node_id!r} already exists")
        if spec.node_type not in KNOWN_NODE_TYPES:
            logger.debug("Unusual node_type %r accepted", spec.node_type)
        goal_revision = (
            spec.goal_revision if spec.goal_revision is not None else state.revision
        )
        state.nodes[spec.node_id] = GraphNode(
            node_id=spec.node_id,
            node_type=spec.node_type,
            title=spec.title,
            objective=spec.objective,
            expected_outcome=spec.expected_outcome,
            priority=spec.priority,
            goal_revision=goal_revision,
            budget=dict(spec.budget),
            payload=dict(spec.payload),
        )

    for spec in op.add_edges:
        if spec.source not in state.nodes:
            raise GraphInvariantError(f"edge source {spec.source!r} does not exist")
        if spec.target not in state.nodes:
            raise GraphInvariantError(f"edge target {spec.target!r} does not exist")
        if spec.edge_type not in KNOWN_EDGE_TYPES:
            logger.debug("Unusual edge_type %r accepted", spec.edge_type)
        if spec.edge_type in HARD_EDGE_TYPES and spec.source == spec.target:
            raise GraphInvariantError(
                f"hard edge {spec.edge_type!r} cannot be a self-loop"
            )
        edge_id = _edge_id(spec.source, spec.target, spec.edge_type)
        if edge_id in state.edges:
            continue  # idempotent: the same relation was already asserted
        if spec.edge_type in HARD_EDGE_TYPES and _would_create_hard_cycle(
            state, spec.source, spec.target
        ):
            raise GraphInvariantError(
                f"edge {spec.source!r} requires {spec.target!r} would create "
                "a dependency cycle; use an informative edge type instead"
            )
        state.edges[edge_id] = GraphEdge(
            edge_id=edge_id,
            source=spec.source,
            target=spec.target,
            edge_type=spec.edge_type,
            confidence=spec.confidence,
            evidence_ref=spec.evidence_ref,
            payload=dict(spec.payload),
        )

    for update in op.update_nodes:
        node = _require_node(state, update.node_id)
        changes: dict[str, Any] = {}
        if update.title is not None:
            changes["title"] = update.title
        if update.objective is not None:
            changes["objective"] = update.objective
        if update.expected_outcome is not None:
            changes["expected_outcome"] = update.expected_outcome
        if update.priority is not None:
            changes["priority"] = update.priority
        if update.verdict is not None:
            changes["verdict"] = Verdict(update.verdict)
        if update.freshness is not None:
            changes["freshness"] = Freshness(update.freshness)
        if update.payload is not None:
            changes["payload"] = {**node.payload, **update.payload}
        if changes:
            state.nodes[update.node_id] = node.with_updates(**changes)

    return [
        _event(
            ev.GRAPH_PATCHED,
            {
                "added_nodes": [s.node_id for s in op.add_nodes],
                "added_edges": [
                    {"source": s.source, "target": s.target, "type": s.edge_type}
                    for s in op.add_edges
                ],
                "updated_nodes": [u.node_id for u in op.update_nodes],
                "rationale": op.rationale,
                "based_on_revision": op.based_on_revision,
            },
            goal_revision=state.revision,
        )
    ]


def _apply_activate(state: GraphState, op: ActivateNodeOp) -> list[dict[str, Any]]:
    node = _require_node(state, op.node_id)
    _require_non_terminal(node)
    if node.lifecycle is Lifecycle.ACTIVE:
        return []
    state.nodes[op.node_id] = node.with_updates(lifecycle=Lifecycle.ACTIVE)
    return [
        _event(
            ev.NODE_ACTIVATED,
            {"title": node.title, "rationale": op.rationale},
            node_id=op.node_id,
            goal_revision=node.goal_revision,
        )
    ]


def _apply_suspend(state: GraphState, op: SuspendNodeOp) -> list[dict[str, Any]]:
    node = _require_node(state, op.node_id)
    _require_non_terminal(node)
    changes: dict[str, Any] = {"lifecycle": Lifecycle.SUSPENDED}
    if op.checkpoint:
        changes["checkpoint"] = op.checkpoint
    state.nodes[op.node_id] = node.with_updates(**changes)
    return [
        _event(
            ev.NODE_CHECKPOINTED,
            {"reason": op.reason, "checkpoint": op.checkpoint, "suspended": True},
            node_id=op.node_id,
            goal_revision=node.goal_revision,
        )
    ]


def _apply_checkpoint(
    state: GraphState, op: CheckpointNodeOp
) -> list[dict[str, Any]]:
    node = _require_node(state, op.node_id)
    state.nodes[op.node_id] = node.with_updates(checkpoint=op.checkpoint)
    return [
        _event(
            ev.NODE_CHECKPOINTED,
            {"reason": op.reason, "checkpoint": op.checkpoint, "suspended": False},
            node_id=op.node_id,
            goal_revision=node.goal_revision,
        )
    ]


def _apply_abandon(state: GraphState, op: AbandonNodeOp) -> list[dict[str, Any]]:
    node = _require_node(state, op.node_id)
    if node.lifecycle is Lifecycle.RESOLVED:
        raise GraphInvariantError(
            f"node {op.node_id!r} is resolved; invalidate it instead of abandoning"
        )
    state.nodes[op.node_id] = node.with_updates(
        lifecycle=Lifecycle.ABANDONED,
        checkpoint=op.reason or node.checkpoint,
    )
    return [
        _event(
            ev.NODE_INVALIDATED,
            {"reason": op.reason, "abandoned": True},
            node_id=op.node_id,
            goal_revision=node.goal_revision,
        )
    ]


def _apply_resolve_node(
    state: GraphState, op: ResolveNodeOp
) -> list[dict[str, Any]]:
    node = _require_node(state, op.node_id)
    _require_non_terminal(node)
    if node.node_type in EVIDENCE_REQUIRED_NODE_TYPES and not op.evidence_ids:
        raise GraphInvariantError(
            f"node {op.node_id!r} ({node.node_type}) cannot resolve without "
            "evidence; attach evidence first"
        )
    state.nodes[op.node_id] = node.with_updates(
        lifecycle=Lifecycle.RESOLVED,
        verdict=Verdict(op.verdict),
        evidence_refs=list(dict.fromkeys([*node.evidence_refs, *op.evidence_ids])),
        checkpoint=op.outcome or node.checkpoint,
    )
    _promote_ready(state, op.node_id)
    return [
        _event(
            ev.NODE_RESOLVED,
            {
                "evidence_ids": list(op.evidence_ids),
                "outcome": op.outcome,
                "verdict": op.verdict,
            },
            node_id=op.node_id,
            goal_revision=node.goal_revision,
        )
    ]


def _apply_attach_evidence(
    state: GraphState, op: AttachEvidenceOp
) -> list[dict[str, Any]]:
    node = _require_node(state, op.node_id)
    if op.evidence_id in node.evidence_refs:
        return []
    state.nodes[op.node_id] = node.with_updates(
        evidence_refs=[*node.evidence_refs, op.evidence_id]
    )
    return [
        _event(
            ev.EVIDENCE_ATTACHED,
            {"evidence_id": op.evidence_id, "summary": op.summary},
            node_id=op.node_id,
            goal_revision=node.goal_revision,
        )
    ]


def _apply_revise_goal(
    state: GraphState, op: ReviseGoalOp
) -> list[dict[str, Any]]:
    classification = op.classification
    if classification not in KNOWN_REVISION_KINDS:
        logger.debug("Unusual revision classification %r", classification)
    new_revision = state.revision + 1
    state.goal_revisions.append(
        GoalRevision(
            revision=new_revision,
            text=op.text,
            classification=classification,
            author=op.author,
            supersedes=state.revision or None,
        )
    )
    state.revision = new_revision
    # A goal that changed under the work's feet makes unfinished work stale
    # until somebody re-validates it (§7 reverse scope, conservative cut).
    destabilising = {"replacement", "contradiction", "removed_requirement"}
    if classification in destabilising:
        for node_id, node in state.nodes.items():
            if node.lifecycle in {Lifecycle.RESOLVED, Lifecycle.ABANDONED}:
                continue
            if node.freshness is Freshness.CURRENT:
                state.nodes[node_id] = node.with_updates(
                    freshness=Freshness.STALE
                )
    return [
        _event(
            ev.GOAL_REVISED,
            {
                "revision": new_revision,
                "text": op.text,
                "classification": classification,
            },
            goal_revision=new_revision,
        )
    ]


def _apply_resolve_goal(
    state: GraphState, op: ResolveGoalOp
) -> list[dict[str, Any]]:
    if op.revision_id != state.revision:
        raise GraphInvariantError(
            f"resolve_goal targets revision {op.revision_id} but the goal is "
            f"at revision {state.revision}"
        )
    if not op.evidence_ids:
        raise GraphInvariantError("resolve_goal requires at least one evidence id")
    return [
        _event(
            ev.GOAL_RESOLVED,
            {"revision": op.revision_id, "evidence_ids": list(op.evidence_ids)},
            goal_revision=op.revision_id,
        )
    ]


# ── entry point ──────────────────────────────────────────────────────────────


def reduce(state: GraphState, op: GraphOp) -> tuple[GraphState, list[dict[str, Any]]]:
    """Apply one operation; return the new state and the audit events.

    Raises :class:`GraphInvariantError` without mutating *state* when any
    invariant fails.
    """
    working = state.model_copy(deep=True)

    handlers = {
        "graph_patch": _apply_graph_patch,
        "activate_node": _apply_activate,
        "suspend_node": _apply_suspend,
        "checkpoint_node": _apply_checkpoint,
        "abandon_node": _apply_abandon,
        "resolve_node": _apply_resolve_node,
        "attach_evidence": _apply_attach_evidence,
        "revise_goal": _apply_revise_goal,
        "resolve_goal": _apply_resolve_goal,
    }
    handler = handlers.get(op.kind)
    if handler is None:
        raise GraphInvariantError(f"unknown graph op kind {op.kind!r}")

    events = handler(working, op)
    working.version += 1
    return working, events


__all__ = ["GraphInvariantError", "reduce"]
