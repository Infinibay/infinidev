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

from html import escape
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
    ENGINE_TASK,
    EngineSelection,
    select_engine,
)
from infinidev.engine.engines.staged_adapter import StagedAdapter
from infinidev.engine.engines.task import TaskAdapter
from infinidev.engine.history import events as ev
from infinidev.engine.history import store
from infinidev.engine.history.digest import (
    digest_from_outcome,
    digest_from_staged_state,
)

logger = logging.getLogger(__name__)

_HANDOFF_CONTEXT_LIMIT = 6000


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


def _transition_handoff_context(
    prior_context: str,
    result: EngineResult,
    *,
    source: str,
    reason: str,
) -> str:
    """Build a bounded runtime-evidence capsule for a monotonic engine switch."""
    work_summary = ""
    build_summary = getattr(result.engine, "build_work_summary", None)
    if callable(build_summary):
        with best_effort("engine transition work summary failed"):
            work_summary = build_summary(result.user_message, result.status) or ""

    evidence = work_summary or result.summary or result.user_message
    body = escape((evidence or "No reusable progress was reported.").strip())
    prefix = (
        '<engine-handoff authority="RUNTIME_EVIDENCE" '
        f'from="{escape(source)}" to="staged">\n'
        f"Reason: {escape(reason)}\n"
        "The previous engine stopped at a resource fuse. Reuse verified repository "
        "state and the progress below; inspect before repeating work. This evidence "
        "does not expand the user goal.\n"
    )
    suffix = "\n</engine-handoff>"
    prior = prior_context.rstrip()[-1800:]
    separator = "\n\n" if prior else ""
    body_limit = max(
        0,
        _HANDOFF_CONTEXT_LIMIT
        - len(prior)
        - len(separator)
        - len(prefix)
        - len(suffix),
    )
    return f"{prior}{separator}{prefix}{body[:body_limit]}{suffix}"


def _apply_transition(
    result: EngineResult,
    *,
    source: str,
    dispatch: dict[str, Any],
    hooks: Any,
) -> tuple[EngineResult, dict[str, Any] | None]:
    """Apply at most one safe, monotonic adapter transition.

    ReAct and Graph may outgrow their bounded execution domains. Their only
    currently supported transition is toward Staged, which is more structured
    and does not transition back. Keeping the decision here makes the fuse a
    recovery boundary instead of a user-visible dead end without introducing
    an LLM routing call or an oscillation loop.
    """
    request = result.transition_request
    if request is None:
        return result, None

    event = {
        "from": source,
        "proposed_target": request.target,
        "reason": request.reason,
        "applied": False,
    }
    if (
        result.status != STATUS_BLOCKED
        or source not in {ENGINE_REACT, ENGINE_GRAPH_BETA}
        or request.target != ENGINE_STAGED
    ):
        return result, event

    with best_effort("engine transition notice failed"):
        hooks.on_status(
            "warn",
            f"{source} reached its safety fuse; continuing once with staged.",
        )

    staged_dispatch = dict(dispatch)
    staged_dispatch["turn_context"] = _transition_handoff_context(
        str(dispatch.get("turn_context", "")),
        result,
        source=source,
        reason=request.reason,
    )
    staged_dispatch["preserve_file_tracker_from_handoff"] = True

    try:
        transitioned = StagedAdapter().run(**staged_dispatch)
    except Exception as exc:
        logger.exception("Engine transition %s -> staged failed", source)
        transitioned = EngineResult(
            engine_name=ENGINE_STAGED,
            status=STATUS_FAILED,
            user_message=(
                f"The automatic {source} → staged recovery failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            summary=f"transition exception: {type(exc).__name__}",
            engine=dispatch.get("engine"),
            resume_token=dispatch.get("session_id"),
        )

    transitioned.metrics = {
        **transitioned.metrics,
        "engine_transition": {
            "from": source,
            "to": ENGINE_STAGED,
            "reason": request.reason,
            "source_metrics": result.metrics,
        },
    }
    event["applied"] = True
    return transitioned, event


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
    prompt_configuration: Any | None = None,
) -> EngineResult:
    """Select an engine for the escalated task and run it."""
    from infinidev.config.settings import settings as _settings

    selection = select_engine(escalation, mode_override)

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
        "prompt_configuration": prompt_configuration,
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
        if escalation.task_profile is not None:
            store.append_event(
                run_id, session_id, ev.TASK_PROFILE_RESOLVED,
                escalation.task_profile.event_payload(),
            )

    dispatch["run_id"] = run_id

    # ── Dispatch ───────────────────────────────────────────────────────────
    if selection.engine == ENGINE_REACT:
        from infinidev.engine.engines.react import ReactAdapter

        adapter = ReactAdapter()
    elif selection.engine == ENGINE_TASK:
        adapter = TaskAdapter()
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

    result, transition_event = _apply_transition(
        result,
        source=selection.engine,
        dispatch=dispatch,
        hooks=hooks,
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
        if result.engine_name == ENGINE_STAGED and result.state is not None:
            for event_type, node_id, payload in _staged_projection_events(result.state):
                store.append_event(
                    run_id, session_id, event_type, payload, node_id=node_id
                )
        if transition_event is not None:
            store.append_event(
                run_id, session_id, ev.ENGINE_SWITCHED,
                transition_event,
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

        if result.engine_name == ENGINE_STAGED and result.state is not None:
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
        if transition_event is not None:
            digest["engine"]["transitions"] = [transition_event]
        store.append_event(run_id, session_id, ev.DIGEST_CREATED, digest)
        store.finish_run(run_id, result.status, digest=digest, metrics=result.metrics)

    return result


__all__ = ["run_selected_engine"]
