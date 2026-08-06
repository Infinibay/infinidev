"""Engine coordinator: selects an engine, runs it, and records the event log.

One entry point — :func:`run_selected_engine` — replaces the pipeline's
direct call to ``run_staged_goal``. The coordinator is deliberately thin:

1. resolve the configured mode into an :class:`EngineSelection`;
2. show the selection reason when enabled;
3. open an ``engine_runs`` row and emit the lifecycle events (best-effort —
   the event log must never sink a task run);
4. dispatch to the selected adapter;
5. append the terminal event plus a RunDigest and close the run.

Adapters keep their own domain (Staged its Goal/Stage/Task state, ReAct its
plain loop), so the coordinator never inspects engine internals beyond the
normalized :class:`EngineResult`. See
docs/GRAPH_ENGINE_BETA_DESIGN.md §8.4, §10 and §16.
"""

from __future__ import annotations

import logging
from typing import Any

from infinidev.engine._best_effort import best_effort
from infinidev.engine.engines.base import (
    EngineResult,
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
)
from infinidev.engine.engines.routing import (
    ENGINE_GRAPH_BETA,
    ENGINE_REACT,
    ENGINE_STAGED,
    EngineSelection,
    select_engine,
)
from infinidev.engine.engines.staged_adapter import StagedAdapter
from infinidev.engine.history import events as ev
from infinidev.engine.history import store
from infinidev.engine.history.digest import (
    digest_from_outcome,
    digest_from_staged_state,
)

logger = logging.getLogger(__name__)


def _show_selection(selection: EngineSelection, hooks: Any) -> None:
    from infinidev.config.settings import settings

    if not settings.ENGINE_SHOW_SELECTION_REASON:
        return
    reason = "; ".join(selection.reasons) or "no reason recorded"
    message = (
        f"Engine: {selection.engine} (mode: {selection.requested_mode}) — {reason}"
    )
    if selection.fallback_note:
        message += f" [{selection.fallback_note}]"
    with best_effort("engine selection notice failed"):
        hooks.on_status("info", message)


def _staged_projection_events(state: Any) -> list[tuple[str, str | None, dict[str, Any]]]:
    """Stage/Task lifecycle events derived from the final staged state.

    Emitted post-hoc (the staged pipeline itself stays untouched); they give
    ``history_search`` something structured to find per Stage and Task.
    Returns (event_type, node_id, payload) triples.
    """
    triples: list[tuple[str, str | None, dict[str, Any]]] = []
    for stage in getattr(state, "stages", []) or []:
        triples.append((
            ev.STAGE_OPENED,
            stage.id,
            {
                "stage_number": stage.number,
                "title": stage.spec.title,
                "outcome": stage.spec.outcome,
            },
        ))
        for task in stage.tasks:
            triples.append((
                ev.TASK_CLOSED,
                task.spec.id,
                {
                    "stage_number": stage.number,
                    "task_id": task.spec.id,
                    "title": task.spec.title,
                    "status": task.status,
                    "attempts": task.attempts,
                    "error": task.error,
                    "result_excerpt": (task.result or "")[:2000],
                },
            ))
        triples.append((
            ev.STAGE_CLOSED,
            stage.id,
            {
                "stage_number": stage.number,
                "title": stage.spec.title,
                "status": stage.status,
                "outcome_summary": stage.outcome_summary,
            },
        ))
    return triples


def run_selected_engine(
    *,
    escalation: Any,
    agent: Any,
    engine: Any,
    reviewer: Any,
    hooks: Any,
    session_id: str,
    project_id: int | None,
    workspace_path: str | None,
    turn_context: str = "",
    use_phase_engine: bool = False,
    force_gather: bool = False,
    mode_override: str | None = None,
) -> EngineResult:
    """Select an engine for the escalated task and run it."""
    from infinidev.config.settings import settings as _settings

    selection = select_engine(escalation, mode_override)

    # Auto may treat --think as a Staged signal. Explicit selections stay
    # pinned: choosing react, staged or graph_beta never dispatches another
    # adapter behind the user's back.
    if (
        use_phase_engine
        and selection.requested_mode == "auto"
        and selection.engine == ENGINE_REACT
    ):
        selection = EngineSelection(
            engine=ENGINE_STAGED,
            requested_mode=selection.requested_mode,
            confidence=selection.confidence,
            reasons=[*selection.reasons, "phase_engine_requires_staged"],
            risks=list(selection.risks),
            reconsider_if=list(selection.reconsider_if),
            estimated_overhead="medium",
            fallback_note="use_phase_engine forced the staged adapter.",
        )

    _show_selection(selection, hooks)

    dispatch: dict[str, Any] = {
        "escalation": escalation,
        "agent": agent,
        "engine": engine,
        "reviewer": reviewer,
        "hooks": hooks,
        "session_id": session_id,
        "project_id": project_id,
        "workspace_path": workspace_path,
        "turn_context": turn_context,
        "use_phase_engine": use_phase_engine,
        "force_gather": force_gather,
    }

    # ── Event log: open the run ────────────────────────────────────────────
    run_id: str | None = None
    with best_effort("engine run registration failed"):
        run_id = store.create_run(
            session_id=session_id,
            engine=selection.engine,
            mode=selection.requested_mode,
            goal_title=(escalation.user_request.strip().splitlines() or [""])[0][:120],
            goal_request=escalation.user_request,
            project_id=project_id,
            selection=selection.to_payload(),
        )
        store.append_event(
            run_id, session_id, ev.RUN_STARTED,
            {"mode": selection.requested_mode, "engine": selection.engine},
        )
        store.append_event(
            run_id, session_id, ev.GOAL_REVISED,
            {
                "title": (escalation.user_request.strip().splitlines() or [""])[0][:120],
                "revision": 1,
                "understanding": escalation.understanding,
            },
            goal_revision=1,
        )
        store.append_event(
            run_id, session_id, ev.ENGINE_SELECTED,
            selection.to_payload(),
        )

    dispatch["run_id"] = run_id

    # ── Dispatch ───────────────────────────────────────────────────────────
    if selection.engine == ENGINE_REACT:
        from infinidev.engine.engines.react import ReactAdapter

        adapter = ReactAdapter()
    elif selection.engine == ENGINE_GRAPH_BETA:
        from infinidev.engine.engines.graph import (
            GraphEngineAdapter,
            GraphPersistence,
            SchedulerLimits,
        )

        persistence = (
            GraphPersistence(run_id, session_id=session_id)
            if run_id is not None
            else None
        )
        adapter = GraphEngineAdapter(
            persistence=persistence,
            limits=SchedulerLimits(
                max_open_branches=_settings.GRAPH_MAX_OPEN_BRANCHES,
                max_node_revisits=_settings.GRAPH_MAX_NODE_REVISITS,
            ),
        )
    else:
        adapter = StagedAdapter()

    try:
        result = adapter.run(**dispatch)
    except Exception as exc:
        logger.exception("Engine adapter %s failed", selection.engine)
        result = EngineResult(
            engine_name=selection.engine,
            status=STATUS_FAILED,
            user_message=(
                f"The {selection.engine} engine failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            summary=f"adapter exception: {type(exc).__name__}",
            engine=engine,
            resume_token=session_id,
        )

    result.run_id = run_id

    # ── Event log: close the run ───────────────────────────────────────────
    if run_id is None:
        return result

    terminal_event = {
        STATUS_COMPLETED: ev.RUN_COMPLETED,
        STATUS_BLOCKED: ev.RUN_BLOCKED,
        STATUS_CANCELLED: ev.RUN_CANCELLED,
        STATUS_FAILED: ev.RUN_FAILED,
    }.get(result.status, ev.RUN_FAILED)

    with best_effort("engine run closing events failed"):
        if selection.engine == ENGINE_STAGED and result.state is not None:
            for event_type, node_id, payload in _staged_projection_events(result.state):
                store.append_event(
                    run_id, session_id, event_type, payload, node_id=node_id
                )
        if result.transition_request is not None:
            store.append_event(
                run_id, session_id, ev.ENGINE_SWITCHED,
                {
                    "proposed_target": result.transition_request.target,
                    "reason": result.transition_request.reason,
                    "applied": False,
                },
            )
        store.append_event(
            run_id, session_id, terminal_event,
            {
                "status": result.status,
                "summary": result.summary,
                "result_excerpt": (result.user_message or "")[:2000],
                "metrics": result.metrics,
            },
        )

        if selection.engine == ENGINE_STAGED and result.state is not None:
            digest = digest_from_staged_state(
                result.state,
                run_id=run_id,
                engine_name=result.engine_name,
                mode=selection.requested_mode,
                selection=selection.to_payload(),
                status=result.status,
            )
        else:
            digest = digest_from_outcome(
                run_id=run_id,
                engine_name=result.engine_name,
                mode=selection.requested_mode,
                status=result.status,
                goal_title=(escalation.user_request.strip().splitlines() or [""])[0][:120],
                user_request=escalation.user_request,
                selection=selection.to_payload(),
                result_text=result.user_message,
            )
        store.append_event(run_id, session_id, ev.DIGEST_CREATED, digest)
        store.finish_run(run_id, result.status, digest=digest, metrics=result.metrics)

    return result


__all__ = ["run_selected_engine"]
