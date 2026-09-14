"""Event subscriptions that suspend an agent stack without calling its model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from infinidev.engine.team.store import member_label, now
from infinidev.engine.tool_progress import is_tool_cancelled
from infinidev.tools.base.context import get_context_for_agent
from infinidev.tools.shell.background_manager import get_background_manager

EventKind = Literal["message", "report", "background_task", "note", "ticket"]

#: How long a subscription with no deadline sleeps before re-checking. The
#: check-then-wait in :func:`wait_for_events` is a lost-wakeup window: a worker
#: that finishes between the check and the wait notifies nobody the waiter has
#: registered for yet, and an unbounded wait then never returns. The loop
#: already re-reads the event log on every iteration, so a periodic wakeup is
#: free — no model call — and it bounds the worst case to this many seconds. It
#: also makes cancellation responsive while idle.
_IDLE_RECHECK_SECONDS = 5.0


def _wait_slice(remaining: float | None) -> float:
    """The timeout to hand ``Condition.wait``, never ``None``.

    ``None`` blocks until notified, which is exactly the case that hangs. A
    subscription with a deadline keeps it; one without re-checks every
    :data:`_IDLE_RECHECK_SECONDS`.
    """
    if remaining is None:
        return _IDLE_RECHECK_SECONDS
    return max(0.0, min(remaining, _IDLE_RECHECK_SECONDS))


class IdleInput(BaseModel):
    events: list[EventKind] = Field(default_factory=lambda: ["message", "report"],
                                    min_length=1, max_length=5)
    sender: str | None = Field(default=None, description="Match a teammate's ID or name")
    reply_to: int | None = Field(default=None, ge=1,
                                 description="Wake for an answer to this message, even if already received")
    task_ids: list[str] = Field(default_factory=list, max_length=50,
                               description="Background process IDs; empty matches any in this workspace")
    ticket_id: str | None = None
    after: int | None = Field(default=None, ge=0,
                              description="Optional event cursor; default is the last delivered update")
    timeout: float | None = Field(default=None, ge=0, le=604800,
                                   description="Seconds; omit to wait until an event or cancellation")
    reason: str = Field(default="Waiting for a subscribed event", max_length=500)

    @model_validator(mode="after")
    def compatible_filters(self) -> IdleInput:
        if self.reply_to is not None and "message" not in self.events:
            raise ValueError("reply_to requires the message event")
        if self.task_ids and "background_task" not in self.events:
            raise ValueError("task_ids requires the background_task event")
        return self


def _background_events(team: Any, actor: str, spec: IdleInput) -> list[dict]:
    if "background_task" not in spec.events:
        return []
    seen = team._background_seen.setdefault(actor, set())
    root = Path(team.workspace_path).resolve()
    result = []
    for task in get_background_manager().list():
        if not Path(task.cwd).resolve().is_relative_to(root):
            continue
        if spec.task_ids and task.id not in spec.task_ids:
            continue
        if not spec.task_ids and task.id in seen:
            continue
        # poll() can finish before the pipe reader has drained the last bytes.
        if task.end_time is not None:
            summary = task.summary()
            summary["stdout"] = summary["stdout"][-8000:]
            summary["stderr"] = summary["stderr"][-8000:]
            result.append(dict(summary, kind="background_task"))
    seen.update(event["id"] for event in result)
    return result


def _matches(event: dict, actor: str, spec: IdleInput, sender: str | None) -> bool:
    if event["author"] == actor:
        return False
    if event["kind"] == "user_guidance":
        return event["author"] == "user"
    kind = "ticket" if event["kind"] in {"review", "delegation"} else event["kind"]
    if kind not in spec.events:
        return False
    if sender and event["author"] != sender:
        return False
    if spec.ticket_id and event["ticket_id"] != spec.ticket_id:
        return False
    if kind == "message":
        if event["recipient"] not in {actor, "all"}:
            return False
        if spec.reply_to is not None and event["reply_to"] != spec.reply_to:
            return False
    return True


def wait_for_events(team: Any, actor: str, spec: IdleInput) -> dict:
    """Register and inspect under one condition lock, avoiding lost wakeups."""
    sender = team.resolve_recipient(spec.sender) if spec.sender else None
    state = team.store.snapshot()
    if actor not in state["agents"]:
        raise ValueError("Agent is not a member of this team")
    if sender == "all":
        raise ValueError("Choose a specific sender or omit sender")
    if spec.ticket_id:
        team._ticket(state, spec.ticket_id)
    if spec.reply_to is not None:
        original = team.store.event(spec.reply_to)
        if original["kind"] != "message" or original["author"] != actor:
            raise ValueError("Wait for a reply to a message you sent")
    for task_id in spec.task_ids:
        task = get_background_manager().get(task_id)
        if task is None or not Path(task.cwd).resolve().is_relative_to(
            Path(team.workspace_path).resolve(),
        ):
            raise ValueError(f"Unknown background task in this workspace: {task_id}")

    agent_id = team.root_agent_id if actor == "orchestrator" else actor
    context = get_context_for_agent(agent_id)
    tracker = context.file_tracker
    leased = actor in team._executing
    before = (tracker.baseline.current_states()
              if leased and tracker and tracker.baseline else None)
    started = time.monotonic()
    deadline = None if spec.timeout is None else started + spec.timeout
    cursor = spec.after if spec.after is not None else (
        spec.reply_to if spec.reply_to is not None else team._cursors.get(actor, 0))
    previous_status = state["agents"][actor]["status"]
    result: dict = {"reason": "cancelled", "events": []}
    sleeping = False
    released = False
    try:
        while True:
            notice = ""
            with team._condition:
                if team._cancelled(actor) or is_tool_cancelled():
                    break
                matches = []
                while True:
                    page = team.store.events(after=cursor, limit=100)
                    received = team._delivered_events.get(actor, set())
                    matches.extend(event for event in page
                                   if (spec.reply_to is not None or event["id"] not in received)
                                   and _matches(event, actor, spec, sender))
                    if page:
                        cursor = page[-1]["id"]
                    if matches or len(page) < 100:
                        break
                matches.extend(_background_events(team, actor, spec))
                if matches:
                    team.acknowledge_events(actor, matches)
                    reason = ("user_guidance" if any(e["kind"] == "user_guidance" for e in matches)
                              else "event")
                    result = {"reason": reason, "events": matches}
                    break
                remaining = None if deadline is None else max(0, deadline - time.monotonic())
                if remaining == 0:
                    result = {"reason": "timeout", "events": []}
                    break
                if not sleeping:
                    def suspend(current, emit):  # noqa: E306 - defined for the store update
                        current["agents"][actor].update(status="waiting", waiting={
                            **spec.model_dump(), "since": now(),
                        })
                        emit("idle", actor, spec.reason)

                    team.store.update(suspend)
                    sleeping = True
                    if leased:
                        if tracker and before is not None:
                            tracker.deactivate()
                        team._release_worker(actor)
                        released = True
                    notice = f"{member_label(state['agents'][actor])}: idle — {spec.reason}"
                else:
                    team._condition.wait(timeout=_wait_slice(remaining))
            # UI callbacks acquire session locks that user-input paths hold
            # before entering the team. Recheck events after releasing this
            # lock rather than calling the renderer inside the subscription.
            if notice:
                team._notice(notice)
    finally:
        if sleeping:
            if released:
                team.store.update(lambda current, emit: current["agents"][actor].update(
                    status="queued"))
                acquired = team._acquire_worker(actor)
                if acquired and tracker and before is not None:
                    changed = tracker.exclude_external_changes(before)
                    result["workspace_changes_while_idle"] = changed
                    if changed:
                        result["workspace_notice"] = (
                            "Files changed while idle. Reread them before editing; peer changes "
                            "are excluded from this worker's rollback and verification evidence.")
                if not acquired:
                    result = {"reason": "cancelled", "events": []}
            # The principal also caches source while specialists change it.
            # Its global diff remains intact, but its read snapshots expire.
            if context.loop_state is not None:
                context.loop_state.opened_files.clear()
                context.loop_state.read_delivery_revisions.clear()
            with team._condition:
                def resume(current, emit):
                    member = current["agents"][actor]
                    member.update(status="idle" if team._stopped.is_set() else previous_status,
                                  last_wake=result)
                    member.pop("waiting", None)
                    emit("wake", actor, result["reason"])

                team.store.update(resume)
                team.wake_waiters()
            team._notice(f"{member_label(state['agents'][actor])}: resumed — {result['reason']}")
    result["waited_seconds"] = round(time.monotonic() - started, 3)
    return result
