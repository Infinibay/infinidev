"""Typed, versioned work-graph domain model.

Implements the conceptual layers of docs/GRAPH_ENGINE_BETA_DESIGN.md §4:
a small set of node and edge types, three orthogonal status dimensions
(lifecycle / verdict / freshness), and the goal-revision sequence that
governs intent. The models are pure data — no I/O, no LLM — so the reducer
and scheduler can stay deterministic and replayable.

Design constraints honoured here:

* **Few types, extend deliberately.** Node and edge vocabularies are open
  strings with a curated constant set, mirroring ``task_schema.Task.kind``:
  unknown values pass validation but are logged, never raised on.
* **Hard vs. semantic edges.** ``requires`` is the one edge kind that must
  stay a DAG; every other edge may participate in cycles because it carries
  meaning, not execution order (§4.4).
* **Three status axes.** A node may be ``resolved`` yet ``stale`` after a
  user revision; lifecycle, verdict and freshness never collapse into one
  field (§4.3).
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now() -> float:
    return time.time()


def new_node_id(prefix: str = "node") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:14]}"


def new_edge_id() -> str:
    return f"edge_{uuid.uuid4().hex[:14]}"


# ── Node vocabulary (§4.1) ──────────────────────────────────────────────────

NODE_GOAL_REVISION = "goal_revision"
NODE_REQUIREMENT = "requirement"
NODE_QUESTION = "question"
NODE_HYPOTHESIS = "hypothesis"
NODE_DECISION = "decision"
NODE_WORK = "work"
NODE_VERIFICATION = "verification"
NODE_EVIDENCE = "evidence"
NODE_BLOCKER = "blocker"
NODE_ARTIFACT_REF = "artifact_ref"
NODE_CODE_REF = "code_ref"

KNOWN_NODE_TYPES: frozenset[str] = frozenset({
    NODE_GOAL_REVISION, NODE_REQUIREMENT, NODE_QUESTION, NODE_HYPOTHESIS,
    NODE_DECISION, NODE_WORK, NODE_VERIFICATION, NODE_EVIDENCE,
    NODE_BLOCKER, NODE_ARTIFACT_REF, NODE_CODE_REF,
})

#: Node types that represent executable work the scheduler may select.
EXECUTABLE_NODE_TYPES: frozenset[str] = frozenset({NODE_WORK, NODE_VERIFICATION})

#: Node types that must carry evidence before they may resolve (§6).
EVIDENCE_REQUIRED_NODE_TYPES: frozenset[str] = frozenset({
    NODE_WORK, NODE_VERIFICATION, NODE_HYPOTHESIS, NODE_REQUIREMENT,
})


# ── Edge vocabulary (§4.2) ──────────────────────────────────────────────────

EDGE_DECOMPOSES_INTO = "decomposes_into"
EDGE_REQUIRES = "requires"
EDGE_ALTERNATIVE_TO = "alternative_to"
EDGE_BLOCKS = "blocks"
EDGE_SATISFIES = "satisfies"
EDGE_SUPPORTS = "supports"
EDGE_CONTRADICTS = "contradicts"
EDGE_PRODUCED_BY = "produced_by"
EDGE_TARGETS = "targets"
EDGE_SUPERSEDES = "supersedes"
EDGE_INVALIDATES = "invalidates"

KNOWN_EDGE_TYPES: frozenset[str] = frozenset({
    EDGE_DECOMPOSES_INTO, EDGE_REQUIRES, EDGE_ALTERNATIVE_TO, EDGE_BLOCKS,
    EDGE_SATISFIES, EDGE_SUPPORTS, EDGE_CONTRADICTS, EDGE_PRODUCED_BY,
    EDGE_TARGETS, EDGE_SUPERSEDES, EDGE_INVALIDATES,
})

#: Edges that impose execution order and therefore must stay acyclic (§4.4).
#: ``requires`` reads source→target as "source depends on target".
HARD_EDGE_TYPES: frozenset[str] = frozenset({EDGE_REQUIRES})


# ── Status dimensions (§4.3) ────────────────────────────────────────────────


class Lifecycle(str, Enum):
    """Where a node is in its executable life."""

    PROPOSED = "proposed"
    READY = "ready"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


#: Lifecycles that are terminal — the scheduler never selects them.
TERMINAL_LIFECYCLES: frozenset[Lifecycle] = frozenset({
    Lifecycle.RESOLVED, Lifecycle.ABANDONED,
})

#: Lifecycles the scheduler may pick up.
OPEN_LIFECYCLES: frozenset[Lifecycle] = frozenset({
    Lifecycle.PROPOSED, Lifecycle.READY, Lifecycle.SUSPENDED,
})


class Verdict(str, Enum):
    """What we currently believe about a node's claim."""

    UNKNOWN = "unknown"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class Freshness(str, Enum):
    """Whether a node's grounding is still valid relative to the goal/repo."""

    CURRENT = "current"
    STALE = "stale"
    INVALIDATED = "invalidated"


#: User-revision classifications (§7).
REVISION_CLARIFICATION = "clarification"
REVISION_NEW_REQUIREMENT = "new_requirement"
REVISION_REMOVED_REQUIREMENT = "removed_requirement"
REVISION_PRIORITY_CHANGE = "priority_change"
REVISION_CONSTRAINT = "constraint"
REVISION_CONTRADICTION = "contradiction"
REVISION_REPLACEMENT = "replacement"
REVISION_PAUSE_OR_CANCEL = "pause_or_cancel"

KNOWN_REVISION_KINDS: frozenset[str] = frozenset({
    REVISION_CLARIFICATION, REVISION_NEW_REQUIREMENT,
    REVISION_REMOVED_REQUIREMENT, REVISION_PRIORITY_CHANGE,
    REVISION_CONSTRAINT, REVISION_CONTRADICTION, REVISION_REPLACEMENT,
    REVISION_PAUSE_OR_CANCEL,
})


# ── Models ───────────────────────────────────────────────────────────────────


class GraphNode(BaseModel):
    """One node of the work graph (§4.1)."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    node_type: str
    title: str = ""
    objective: str = ""
    expected_outcome: str = ""
    lifecycle: Lifecycle = Lifecycle.PROPOSED
    verdict: Verdict = Verdict.UNKNOWN
    freshness: Freshness = Freshness.CURRENT
    goal_revision: int | None = None
    priority: float = 0.0
    # Per-node budgets: {"tokens": int, "tool_calls": int}. Empty = unbounded
    # here; the scheduler applies the run-level caps.
    budget: dict[str, Any] = Field(default_factory=dict)
    author: str = "model"
    version: int = 1
    checkpoint: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)

    def with_updates(self, **changes: Any) -> "GraphNode":
        """Return a copy with fields replaced and version/updated_at bumped."""
        data = self.model_dump()
        data.update(changes)
        data["version"] = self.version + 1
        data["updated_at"] = _now()
        return GraphNode.model_validate(data)


class GraphEdge(BaseModel):
    """One directed edge of the work graph (§4.2)."""

    model_config = ConfigDict(frozen=True)

    edge_id: str
    source: str
    target: str
    edge_type: str
    confidence: float = 1.0
    author: str = "model"
    version: int = 1
    evidence_ref: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=_now)

    @property
    def is_hard(self) -> bool:
        return self.edge_type in HARD_EDGE_TYPES


class GoalRevision(BaseModel):
    """One revision of the governing objective (§4, §7)."""

    model_config = ConfigDict(frozen=True)

    revision: int
    text: str
    classification: str = REVISION_NEW_REQUIREMENT
    author: str = "user"
    supersedes: int | None = None
    created_at: float = Field(default_factory=_now)


class GraphState(BaseModel):
    """The whole in-memory graph for one run.

    ``version`` is the single-writer write counter: every applied operation
    increments it, so a ``graph_patch`` can carry ``based_on_revision`` and
    the reducer can reject stale writes (§6).
    """

    run_id: str
    session_id: str = ""
    nodes: dict[str, GraphNode] = Field(default_factory=dict)
    edges: dict[str, GraphEdge] = Field(default_factory=dict)
    goal_revisions: list[GoalRevision] = Field(default_factory=list)
    revision: int = 0
    version: int = 0

    # ── convenience accessors ──────────────────────────────────────────

    @property
    def current_goal(self) -> GoalRevision | None:
        return self.goal_revisions[-1] if self.goal_revisions else None

    def node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def edges_from(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges.values() if e.source == node_id]

    def edges_to(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges.values() if e.target == node_id]

    def hard_dependencies(self, node_id: str) -> list[str]:
        """Node ids this node ``requires`` (must resolve before it runs)."""
        return [
            e.target for e in self.edges_from(node_id) if e.is_hard
        ]

    def hard_dependents(self, node_id: str) -> list[str]:
        """Node ids that ``require`` this node."""
        return [
            e.source for e in self.edges_to(node_id) if e.is_hard
        ]


__all__ = [
    "EDGE_ALTERNATIVE_TO",
    "EDGE_BLOCKS",
    "EDGE_CONTRADICTS",
    "EDGE_DECOMPOSES_INTO",
    "EDGE_INVALIDATES",
    "EDGE_PRODUCED_BY",
    "EDGE_REQUIRES",
    "EDGE_SATISFIES",
    "EDGE_SUPERSEDES",
    "EDGE_SUPPORTS",
    "EDGE_TARGETS",
    "EVIDENCE_REQUIRED_NODE_TYPES",
    "EXECUTABLE_NODE_TYPES",
    "Freshness",
    "GoalRevision",
    "GraphEdge",
    "GraphNode",
    "GraphState",
    "HARD_EDGE_TYPES",
    "KNOWN_EDGE_TYPES",
    "KNOWN_NODE_TYPES",
    "KNOWN_REVISION_KINDS",
    "Lifecycle",
    "NODE_ARTIFACT_REF",
    "NODE_BLOCKER",
    "NODE_CODE_REF",
    "NODE_DECISION",
    "NODE_EVIDENCE",
    "NODE_GOAL_REVISION",
    "NODE_HYPOTHESIS",
    "NODE_QUESTION",
    "NODE_REQUIREMENT",
    "NODE_VERIFICATION",
    "NODE_WORK",
    "OPEN_LIFECYCLES",
    "REVISION_CLARIFICATION",
    "REVISION_CONSTRAINT",
    "REVISION_CONTRADICTION",
    "REVISION_NEW_REQUIREMENT",
    "REVISION_PAUSE_OR_CANCEL",
    "REVISION_PRIORITY_CHANGE",
    "REVISION_REMOVED_REQUIREMENT",
    "REVISION_REPLACEMENT",
    "TERMINAL_LIFECYCLES",
    "Verdict",
    "new_edge_id",
    "new_node_id",
]
