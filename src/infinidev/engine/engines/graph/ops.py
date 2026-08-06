"""Graph mutation operations — the protocol the model writes through.

The model never edits graph rows directly; it emits one of these domain
operations and the reducer validates + applies it
(docs/GRAPH_ENGINE_BETA_DESIGN.md §6). Each op is a plain Pydantic model so
it serializes cleanly into the event log and can be replayed.

Operations are intentionally coarse enough to express the design's protocol
(``graph_patch``, ``checkpoint_node``, ``resolve_node``, ``resolve_goal``)
plus the lifecycle moves the scheduler needs (``activate``, ``suspend``,
``abandon``, ``revise_goal``, ``attach_evidence``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class NodeSpec(BaseModel):
    """A node to add inside a ``graph_patch``."""

    node_id: str
    node_type: str
    title: str = ""
    objective: str = ""
    expected_outcome: str = ""
    priority: float = 0.0
    goal_revision: int | None = None
    budget: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class EdgeSpec(BaseModel):
    """An edge to add inside a ``graph_patch``."""

    source: str
    target: str
    edge_type: str
    confidence: float = 1.0
    evidence_ref: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class NodeUpdate(BaseModel):
    """A field-level update to an existing node inside a ``graph_patch``."""

    node_id: str
    title: str | None = None
    objective: str | None = None
    expected_outcome: str | None = None
    priority: float | None = None
    verdict: str | None = None
    freshness: str | None = None
    payload: dict[str, Any] | None = None


class GraphPatchOp(BaseModel):
    """Batch structural mutation with optimistic-concurrency on the revision."""

    kind: Literal["graph_patch"] = "graph_patch"
    add_nodes: list[NodeSpec] = Field(default_factory=list)
    add_edges: list[EdgeSpec] = Field(default_factory=list)
    update_nodes: list[NodeUpdate] = Field(default_factory=list)
    rationale: str = ""
    based_on_revision: int = 0


class ActivateNodeOp(BaseModel):
    kind: Literal["activate_node"] = "activate_node"
    node_id: str
    rationale: str = ""


class SuspendNodeOp(BaseModel):
    kind: Literal["suspend_node"] = "suspend_node"
    node_id: str
    reason: str = ""
    checkpoint: str = ""


class CheckpointNodeOp(BaseModel):
    kind: Literal["checkpoint_node"] = "checkpoint_node"
    node_id: str
    reason: str = ""
    checkpoint: str = ""


class AbandonNodeOp(BaseModel):
    kind: Literal["abandon_node"] = "abandon_node"
    node_id: str
    reason: str = ""


class ResolveNodeOp(BaseModel):
    kind: Literal["resolve_node"] = "resolve_node"
    node_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    outcome: str = ""
    verdict: str = "confirmed"


class AttachEvidenceOp(BaseModel):
    kind: Literal["attach_evidence"] = "attach_evidence"
    node_id: str
    evidence_id: str
    summary: str = ""


class ResolveGoalOp(BaseModel):
    kind: Literal["resolve_goal"] = "resolve_goal"
    revision_id: int
    evidence_ids: list[str] = Field(default_factory=list)


class ReviseGoalOp(BaseModel):
    kind: Literal["revise_goal"] = "revise_goal"
    text: str
    classification: str = "new_requirement"
    author: str = "user"


GraphOp = (
    GraphPatchOp
    | ActivateNodeOp
    | SuspendNodeOp
    | CheckpointNodeOp
    | AbandonNodeOp
    | ResolveNodeOp
    | AttachEvidenceOp
    | ResolveGoalOp
    | ReviseGoalOp
)


__all__ = [
    "AbandonNodeOp",
    "ActivateNodeOp",
    "AttachEvidenceOp",
    "CheckpointNodeOp",
    "EdgeSpec",
    "GraphOp",
    "GraphPatchOp",
    "NodeSpec",
    "NodeUpdate",
    "ResolveGoalOp",
    "ResolveNodeOp",
    "ReviseGoalOp",
    "SuspendNodeOp",
]
