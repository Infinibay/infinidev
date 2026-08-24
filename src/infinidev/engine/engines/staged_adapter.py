"""StagedAdapter — wraps the existing Goal/Stage/Task pipeline unchanged.

The design's Phase-2 instruction is explicit: wrap Staged in an adapter
without altering its behaviour
(docs/GRAPH_ENGINE_BETA_DESIGN.md §13). This adapter is therefore a thin
translation layer: it forwards the coordinator's keyword bundle to
:func:`run_staged_goal` and maps the returned :class:`StagedRunResult` onto
the normalized :class:`EngineResult`. No planning logic lives here.
"""

from __future__ import annotations

from typing import Any

from infinidev.engine.engines.base import (
    EngineResult,
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
)


def _map_status(staged_status: str) -> str:
    return {
        "complete": STATUS_COMPLETED,
        "blocked": STATUS_BLOCKED,
        "cancelled": STATUS_CANCELLED,
    }.get(staged_status, STATUS_FAILED)


def _staged_state_metrics(state: Any) -> dict[str, Any]:
    """Summarize durable staged progress.

    The last LoopEngine sub-run is not reported as the whole staged execution.
    """
    stages = list(getattr(state, "stages", ()) or ())
    tasks = [
        task
        for stage in stages
        for task in (getattr(stage, "tasks", ()) or ())
    ]

    def status_counts(records: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            status = str(getattr(record, "status", "") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts

    attempts = sum(
        value
        for task in tasks
        if type(value := getattr(task, "attempts", 0)) is int and value >= 0
    )
    revision = getattr(state, "revision", 0)
    if type(revision) is not int or revision < 0:
        revision = 0

    return {
        "goal_status": str(getattr(state, "status", "") or "unknown"),
        "stages": len(stages),
        "stage_status_counts": status_counts(stages),
        "tasks": len(tasks),
        "task_status_counts": status_counts(tasks),
        "task_attempts": attempts,
        "evidence_entries": len(getattr(state, "evidence", ()) or ()),
        "state_revision": revision,
    }


class StagedAdapter:
    """Dispatch escalated work through the durable Goal/Stage/Task engine."""

    name = "staged"

    def run(self, **kwargs: Any) -> EngineResult:
        from infinidev.engine.orchestration.staged_pipeline import run_staged_goal

        staged_run = run_staged_goal(
            escalation=kwargs["escalation"],
            agent=kwargs["agent"],
            engine=kwargs["engine"],
            reviewer=kwargs["reviewer"],
            hooks=kwargs["hooks"],
            session_id=kwargs["session_id"],
            project_id=kwargs.get("project_id"),
            workspace_path=kwargs.get("workspace_path"),
            turn_context=kwargs.get("turn_context", ""),
            use_phase_engine=kwargs.get("use_phase_engine", False),
            force_gather=kwargs.get("force_gather", False),
            prompt_configuration=kwargs.get("prompt_configuration"),
            max_execution_tool_calls_per_task=kwargs.get(
                "max_execution_tool_calls_per_task"
            ),
            preserve_file_tracker_from_handoff=kwargs.get(
                "preserve_file_tracker_from_handoff", False
            ),
        )

        state = staged_run.state
        status = _map_status(state.status)
        evidence = [entry.summary for entry in state.evidence]
        terminal = state.terminal
        summary = terminal.summary if terminal is not None else (
            f"Staged goal {state.status} after {len(state.stages)} stage(s)."
        )

        return EngineResult(
            engine_name=self.name,
            status=status,
            user_message=staged_run.text,
            summary=summary,
            engine=staged_run.engine,
            state=state,
            evidence=evidence,
            resume_token=kwargs.get("session_id"),
            metrics=_staged_state_metrics(state),
        )


__all__ = ["StagedAdapter"]
