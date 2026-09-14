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

_LOOP_STATUS_MAP: dict[str, Status] = {
    "done": "completed",
    "completed": "completed",
    "blocked": "blocked",
    "exhausted": "blocked",
    "cancelled": "cancelled",
    "failed": "failed",
}


def get_loop_status(engine: Any) -> str:
    """Return a normalized raw LoopEngine terminal label."""
    return str(getattr(engine, "_last_status", "") or "").strip().lower()


def normalize_loop_status(loop_status: str) -> Status:
    """Map known loop labels and fail closed on missing or unknown values."""
    return _LOOP_STATUS_MAP.get(loop_status, "failed")


def normalize_terminal_message(
    message: Any,
    status: Status,
) -> str:
    """Make an empty/generic result agree with the terminal status."""
    text = message.strip() if isinstance(message, str) else ""
    if text and not (
        status != STATUS_COMPLETED
        and text in {"Done.", "Done. (no additional output)"}
    ):
        return text

    return {
        STATUS_COMPLETED: "Done. (no additional output)",
        STATUS_BLOCKED: (
            "Execution stopped before completion. "
            "No additional output was produced."
        ),
        STATUS_CANCELLED: (
            "Execution was cancelled. No additional output was produced."
        ),
        STATUS_FAILED: (
            "Execution failed. No additional output was produced."
        ),
    }[status]


def loop_observed_metrics(engine: Any) -> dict[str, int]:
    """Return stable counters from the LoopEngine-compatible result surface."""
    state = getattr(engine, "_last_state", None)

    def counter(source: Any, name: str) -> int:
        value = getattr(source, name, 0)
        return value if type(value) is int and value >= 0 else 0

    observed_tool_calls = getattr(engine, "_last_total_tool_calls", None)
    if type(observed_tool_calls) is not int or observed_tool_calls < 0:
        observed_tool_calls = counter(state, "total_tool_calls")

    return {
        "observed_iterations": counter(state, "iteration_count"),
        "observed_tool_calls": observed_tool_calls,
        "observed_prompt_tokens": counter(state, "total_prompt_tokens"),
        "observed_completion_tokens": counter(state, "total_completion_tokens"),
        # Calls the model issued with an invented shape. This is the only
        # counter that moves when the engine gets better at preventing
        # hallucinated calls rather than merely surviving them.
        "observed_malformed_tool_calls": counter(state, "malformed_tool_calls"),
        # Prompt-cache accounting, which the loop already collected from the
        # provider's usage and then only printed. A cache read is the cheapest
        # input token there is, so a change that costs tokens can still be a
        # win if it moves this — and nothing was measuring it.
        "observed_cache_read_tokens": counter(state, "cache_read_tokens"),
        "observed_cache_creation_tokens": counter(state, "cache_creation_tokens"),
        "observed_cached_prefix_tokens": counter(state, "cached_tokens"),
    }


@dataclass(frozen=True)
class TransitionRequest:
    """A persisted, explainable request to switch engines.

    Adapters propose; the coordinator (or the user) decides. The coordinator
    may apply one monotonic ReAct/Graph-to-Staged recovery within the current
    run. Unsupported or second-order transitions remain proposals, preventing
    oscillation (§8.5).
    """

    target: str
    reason: str


@dataclass
class EngineResult:
    """Normalized outcome of one engine run.

    Attributes:
        engine_name: Adapter that produced the result (normally ``task``;
            compatibility adapters include ``staged`` and ``react``).
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
    "get_loop_status",
    "loop_observed_metrics",
    "normalize_loop_status",
]
