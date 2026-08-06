"""RunDigest — the structured closing snapshot of one engine run.

Built deterministically from observed state (never by an LLM) whenever a
run pauses, blocks, cancels, or completes (docs/GRAPH_ENGINE_BETA_DESIGN.md
§11). The digest makes resume and quick "what happened?" answers cheap, but
it never replaces the event log — every claim here can be re-derived from
``execution_events``.
"""

from __future__ import annotations

from typing import Any


def _goal_block(
    *,
    title: str,
    user_request: str,
    revision: int,
) -> dict[str, Any]:
    return {
        "title": title,
        "user_request": user_request[:4000],
        "revision": revision,
    }


def digest_from_staged_state(
    state: Any,
    *,
    run_id: str,
    engine_name: str,
    mode: str,
    selection: dict[str, Any] | None = None,
    status: str = "",
) -> dict[str, Any]:
    """Assemble a RunDigest from a ``StagedPlanningState``."""
    goal = state.goal
    completed_work: list[str] = []
    open_active: list[str] = []
    open_blocked: list[str] = []
    errors: list[str] = []

    for stage in state.stages:
        for task in stage.tasks:
            label = f"Stage {stage.number} / {task.spec.title}"
            if task.status == "completed":
                completed_work.append(label)
            elif task.status in {"blocked", "failed"}:
                open_blocked.append(label)
                if task.error:
                    errors.append(f"{label}: {task.error}")
            elif task.status in {"pending", "active", "cancelled"}:
                open_active.append(label)

    terminal = state.terminal
    next_steps: list[str] = []
    if terminal is not None and terminal.missing:
        next_steps.append(terminal.missing)
    if state.status in {"active", "blocked"}:
        next_steps.append(
            "Resume the Goal: re-run the Stage Planner over the persisted state."
        )

    return {
        "goal": _goal_block(
            title=goal.title,
            user_request=goal.user_request,
            revision=state.revision,
        ),
        "engine": {
            "name": engine_name,
            "mode": mode,
            "transitions": [],
        },
        "selection": selection or {},
        "status": status or state.status,
        "completed_work": completed_work,
        "open_work": {
            "active": open_active,
            "suspended": [],
            "blocked": open_blocked,
        },
        "decisions": list(state.guidance),
        "verifications": [
            entry.summary[:1000]
            for entry in state.evidence
            if entry.kind in {"task_result", "stage_outcome"}
        ][-10:],
        "errors_and_risks": errors,
        "next_steps": next_steps,
        "references": {
            "run_id": run_id,
            "evidence_ids": [entry.id for entry in state.evidence],
        },
    }


def digest_from_outcome(
    *,
    run_id: str,
    engine_name: str,
    mode: str,
    status: str,
    goal_title: str,
    user_request: str,
    selection: dict[str, Any] | None = None,
    result_text: str = "",
    next_steps: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a RunDigest for engines without staged state (e.g. ReAct)."""
    return {
        "goal": _goal_block(
            title=goal_title,
            user_request=user_request,
            revision=1,
        ),
        "engine": {
            "name": engine_name,
            "mode": mode,
            "transitions": [],
        },
        "selection": selection or {},
        "status": status,
        "completed_work": [goal_title] if status == "completed" else [],
        "open_work": {
            "active": [],
            "suspended": [],
            "blocked": [goal_title] if status == "blocked" else [],
        },
        "decisions": [],
        "verifications": [],
        "errors_and_risks": [],
        "next_steps": next_steps
        or (["Re-run with the staged engine for multi-step structure."]
            if status == "blocked" else []),
        "references": {"run_id": run_id},
        "result_excerpt": (result_text or "")[:2000],
    }


def render_digest(digest: dict[str, Any]) -> str:
    """Short human-readable rendering used for pause confirmations."""
    goal = digest.get("goal", {})
    lines = [
        f"Run digest — {digest.get('status', 'unknown')} "
        f"(engine: {digest.get('engine', {}).get('name', '?')})",
        f"Goal: {goal.get('title', '')}",
    ]
    completed = digest.get("completed_work") or []
    if completed:
        lines.append("Completed: " + "; ".join(completed[:6]))
    open_work = digest.get("open_work", {})
    blocked = open_work.get("blocked") or []
    active = open_work.get("active") or []
    if blocked:
        lines.append("Blocked: " + "; ".join(blocked[:6]))
    if active:
        lines.append("Open: " + "; ".join(active[:6]))
    next_steps = digest.get("next_steps") or []
    if next_steps:
        lines.append("Next: " + next_steps[0])
    return "\n".join(lines)


__all__ = [
    "digest_from_outcome",
    "digest_from_staged_state",
    "render_digest",
]
