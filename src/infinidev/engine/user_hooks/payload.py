"""The facts a hook is handed about the moment it fired.

One builder per lifecycle scope, so the same field means the same thing
whichever event a user binds to. A hook writer should be able to read
``$INFINIDEV_HOOK_STEP_INDEX`` without first checking which event they
attached to.

Only cheap, already-computed values go in. These builders run on every
step of every run once a hook exists, and a payload that shelled out to
git or walked the plan would put that cost on the loop's critical path.
Anything expensive is the hook's own job — it has the workspace as cwd
and can run whatever it likes.

Every field is best-effort: a payload missing ``step_title`` because the
plan was empty is fine, a payload that raised while being built is not.
"""

from __future__ import annotations

from typing import Any


def _plain(value: Any) -> Any:
    """Coerce to something ``json.dumps`` and ``str()`` both handle well."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def step_payload(ctx: Any, **extra: Any) -> dict[str, Any]:
    """Facts about the step that is starting or ending."""
    payload: dict[str, Any] = {
        "session_id": _plain(getattr(ctx, "session_id", "")),
        "agent_id": _plain(getattr(ctx, "agent_id", "")),
        "agent_name": _plain(getattr(ctx, "agent_name", "")),
        "task": _plain(getattr(ctx, "desc", "")),
        "workspace_path": _plain(getattr(ctx, "workspace_path", "")),
    }

    state = getattr(ctx, "state", None)
    plan = getattr(state, "plan", None) if state is not None else None
    active = getattr(plan, "active_step", None) if plan is not None else None
    if active is not None:
        payload["step_index"] = _plain(getattr(active, "index", 0))
        payload["step_title"] = _plain(getattr(active, "title", ""))
        payload["step_status"] = _plain(getattr(active, "status", ""))
    if plan is not None:
        steps = list(getattr(plan, "steps", ()) or ())
        payload["step_total"] = len(steps)
    if state is not None:
        payload["iteration"] = _plain(getattr(state, "iteration_count", 0))
        payload["files_changed"] = bool(getattr(state, "task_has_edits", False))

    payload.update({key: _plain(value) for key, value in extra.items()})
    return payload


def task_payload(
    *,
    session_id: str = "",
    user_input: str = "",
    workspace_path: str = "",
    project_id: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Facts about the turn that is starting or ending."""
    payload: dict[str, Any] = {
        "session_id": _plain(session_id),
        "task": _plain(user_input),
        "workspace_path": _plain(workspace_path),
    }
    if project_id is not None:
        payload["project_id"] = _plain(project_id)
    payload.update({key: _plain(value) for key, value in extra.items()})
    return payload
