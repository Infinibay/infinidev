"""Engine contracts shared by every task-execution adapter.

The coordinator dispatches escalated work through one adapter; every adapter
returns the same :class:`EngineResult` so the pipeline closing path (runtime
bookkeeping, work summaries, end-of-task hooks) never needs to know which
engine ran. See docs/GRAPH_ENGINE_BETA_DESIGN.md §12 for the normalisation
this buys us: Staged keeps its Goal/Stage/Task domain, ReAct stays a plain
budgeted loop, and a future Graph engine can present stage-shaped *views*
without making Stage or Step mandatory domain objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

#: Normalized terminal statuses every adapter must produce.
Status = Literal["completed", "blocked", "cancelled", "failed"]

STATUS_COMPLETED = "completed"
STATUS_BLOCKED = "blocked"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class TransitionRequest:
    """A persisted, explainable request to switch engines.

    Adapters propose; the coordinator (or the user) decides. Proposing a
    transition never changes the current run — that is how oscillation is
    kept out of the loop until hysteresis limits exist (§8.5).
    """

    target: str
    reason: str


@dataclass
class EngineResult:
    """Normalized outcome of one engine run.

    Attributes:
        engine_name: Adapter that produced the result (``staged``/``react``).
        status: One of the normalized terminal statuses.
        user_message: The text shown to the user as the turn's reply.
        summary: Short internal description for events and digests.
        engine: The underlying execution engine instance (LoopEngine or
            PhaseEngine). The pipeline reads ``is_cancelled``,
            ``_last_status``, ``has_file_changes`` and
            ``build_work_summary`` off it, so adapters must pass the real
            instance through, not a wrapper.
        state: Optional structured final state (e.g. StagedPlanningState).
        artifacts: Paths/refs of artifacts produced by the run.
        evidence: Evidence statements established by the run.
        resume_token: Handle a later turn can use to resume this run.
        transition_request: Optional engine-switch proposal.
        metrics: Run counters for the event log.
        run_id: Event-log run this result belongs to.
    """

    engine_name: str
    status: str
    user_message: str
    summary: str = ""
    engine: Any = None
    state: Any = None
    artifacts: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    resume_token: str | None = None
    transition_request: TransitionRequest | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None


@runtime_checkable
class EngineAdapter(Protocol):
    """What every engine adapter must provide.

    ``run`` receives the same keyword bundle the coordinator gets from the
    pipeline; adapters ignore what they do not need. Keeping one fat keyword
    contract (instead of per-engine signatures) is what lets the coordinator
    stay engine-agnostic.
    """

    name: str

    def run(self, **kwargs: Any) -> EngineResult:
        ...


__all__ = [
    "EngineAdapter",
    "EngineResult",
    "STATUS_BLOCKED",
    "STATUS_CANCELLED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "TransitionRequest",
]
