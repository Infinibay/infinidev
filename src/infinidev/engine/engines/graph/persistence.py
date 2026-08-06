"""Graph persistence: projection to SQLite and replay from the event log.

The event log is canonical; ``graph_nodes``/``graph_edges`` are a projection
the reducer maintains for fast reads (docs/GRAPH_ENGINE_BETA_DESIGN.md §10).
Every applied operation is also written to ``execution_events`` with its full
``op`` payload, so a graph can always be rebuilt by replaying those events
through the reducer — the projection is disposable cache.

Single writer: one :class:`GraphPersistence` owns a run's graph and applies
operations sequentially, matching the reducer's single-writer contract (§5.1).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from infinidev.code_intel._db import execute_with_retry
from infinidev.engine.engines.graph.domain import GraphState
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
from infinidev.engine.engines.graph.reducer import reduce
from infinidev.engine.history import store

logger = logging.getLogger(__name__)

#: Event types that carry a replayable ``op`` payload.
GRAPH_EVENT_TYPES = frozenset({
    "graph_patched", "node_activated", "node_checkpointed",
    "node_invalidated", "node_resolved", "evidence_attached",
    "goal_revised", "goal_resolved",
})

_OP_CLASSES: dict[str, type] = {
    "graph_patch": GraphPatchOp,
    "activate_node": ActivateNodeOp,
    "suspend_node": SuspendNodeOp,
    "checkpoint_node": CheckpointNodeOp,
    "abandon_node": AbandonNodeOp,
    "resolve_node": ResolveNodeOp,
    "attach_evidence": AttachEvidenceOp,
    "resolve_goal": ResolveGoalOp,
    "revise_goal": ReviseGoalOp,
}


def op_from_payload(payload: dict[str, Any]) -> GraphOp | None:
    """Rebuild a typed op from its stored payload, or None if not replayable."""
    op_data = payload.get("op")
    if not isinstance(op_data, dict):
        return None
    cls = _OP_CLASSES.get(op_data.get("kind"))
    if cls is None:
        return None
    try:
        return cls.model_validate(op_data)
    except Exception:
        logger.debug("Could not re-validate graph op %r", op_data.get("kind"))
        return None


class GraphPersistence:
    """Owns the graph for one run: applies ops, projects, and replays."""

    def __init__(self, run_id: str, session_id: str = "") -> None:
        self.run_id = run_id
        self.session_id = session_id

    # ── apply ──────────────────────────────────────────────────────────

    def apply(self, state: GraphState, op: GraphOp) -> tuple[GraphState, list[str]]:
        """Reduce *op*, persist the events, refresh the projection.

        Returns the new state and the event ids written. Raises
        :class:`GraphInvariantError` (propagated from the reducer) without
        touching the log when the operation is invalid.
        """
        new_state, events = reduce(state, op)
        event_ids: list[str] = []
        for event in events:
            payload = dict(event.get("payload") or {})
            # Carry the full op so replay can re-apply it deterministically.
            payload["op"] = op.model_dump(mode="json")
            event_id = store.append_event(
                self.run_id,
                self.session_id,
                event["event_type"],
                payload,
                node_id=event.get("node_id"),
                goal_revision=event.get("goal_revision"),
            )
            event_ids.append(event_id)
        self.save_projection(new_state)
        return new_state, event_ids

    # ── projection ─────────────────────────────────────────────────────

    def save_projection(self, state: GraphState) -> None:
        """Replace this run's projected rows with the current state."""

        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "DELETE FROM graph_nodes WHERE run_id = ?", (self.run_id,)
            )
            conn.execute(
                "DELETE FROM graph_edges WHERE run_id = ?", (self.run_id,)
            )
            for node in state.nodes.values():
                conn.execute(
                    """INSERT INTO graph_nodes
                       (node_id, run_id, session_id, node_type, title,
                        objective, expected_outcome, lifecycle, verdict,
                        freshness, goal_revision, priority, budget_json,
                        author, version, checkpoint, evidence_json,
                        payload_json, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        node.node_id, self.run_id, self.session_id,
                        node.node_type, node.title, node.objective,
                        node.expected_outcome, node.lifecycle.value,
                        node.verdict.value, node.freshness.value,
                        node.goal_revision, node.priority,
                        json.dumps(node.budget, ensure_ascii=False),
                        node.author, node.version, node.checkpoint,
                        json.dumps(node.evidence_refs, ensure_ascii=False),
                        json.dumps(node.payload, ensure_ascii=False),
                        node.created_at, node.updated_at,
                    ),
                )
            for edge in state.edges.values():
                conn.execute(
                    """INSERT INTO graph_edges
                       (edge_id, run_id, source, target, edge_type,
                        confidence, author, version, evidence_ref,
                        payload_json, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        edge.edge_id, self.run_id, edge.source, edge.target,
                        edge.edge_type, edge.confidence, edge.author,
                        edge.version, edge.evidence_ref,
                        json.dumps(edge.payload, ensure_ascii=False),
                        edge.created_at,
                    ),
                )
            conn.commit()

        execute_with_retry(_write)

    def load_projection(self) -> GraphState:
        """Read the projected rows back into a GraphState."""
        from infinidev.engine.engines.graph.domain import (
            Freshness,
            GraphEdge,
            GraphNode,
            Lifecycle,
            Verdict,
        )

        def _read(conn: sqlite3.Connection) -> GraphState:
            state = GraphState(run_id=self.run_id, session_id=self.session_id)
            node_rows = conn.execute(
                "SELECT * FROM graph_nodes WHERE run_id = ?", (self.run_id,)
            ).fetchall()
            for row in node_rows:
                node = GraphNode(
                    node_id=row["node_id"],
                    node_type=row["node_type"],
                    title=row["title"],
                    objective=row["objective"],
                    expected_outcome=row["expected_outcome"],
                    lifecycle=Lifecycle(row["lifecycle"]),
                    verdict=Verdict(row["verdict"]),
                    freshness=Freshness(row["freshness"]),
                    goal_revision=row["goal_revision"],
                    priority=row["priority"],
                    budget=json.loads(row["budget_json"] or "{}"),
                    author=row["author"],
                    version=row["version"],
                    checkpoint=row["checkpoint"],
                    evidence_refs=json.loads(row["evidence_json"] or "[]"),
                    payload=json.loads(row["payload_json"] or "{}"),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                state.nodes[node.node_id] = node
            edge_rows = conn.execute(
                "SELECT * FROM graph_edges WHERE run_id = ?", (self.run_id,)
            ).fetchall()
            for row in edge_rows:
                edge = GraphEdge(
                    edge_id=row["edge_id"],
                    source=row["source"],
                    target=row["target"],
                    edge_type=row["edge_type"],
                    confidence=row["confidence"],
                    author=row["author"],
                    version=row["version"],
                    evidence_ref=row["evidence_ref"],
                    payload=json.loads(row["payload_json"] or "{}"),
                    created_at=row["created_at"],
                )
                state.edges[edge.edge_id] = edge
            return state

        state = execute_with_retry(_read)
        # Recover the goal revision counter from the log so a projection
        # loaded cold still rejects stale graph_patch bases correctly.
        state.revision = self._revision_from_log()
        return state

    # ── replay ─────────────────────────────────────────────────────────

    def replay(self) -> GraphState:
        """Rebuild the graph from the event log (the canonical record)."""
        state = GraphState(run_id=self.run_id, session_id=self.session_id)
        events = store.list_run_events(self.run_id, include_archive=True)
        for event in sorted(events, key=lambda e: e.get("sequence", 0)):
            if event.get("event_type") not in GRAPH_EVENT_TYPES:
                continue
            op = op_from_payload(event.get("payload") or {})
            if op is None:
                continue
            state, _ = reduce(state, op)
        return state

    def _revision_from_log(self) -> int:
        events = store.list_run_events(self.run_id, include_archive=True)
        revision = 0
        for event in events:
            if event.get("event_type") == "goal_revised":
                goal_revision = event.get("goal_revision") or 0
                revision = max(revision, goal_revision)
        return revision


__all__ = ["GRAPH_EVENT_TYPES", "GraphPersistence", "op_from_payload"]
