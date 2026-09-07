"""Bounded workers, durable tickets and asynchronous peer conversations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from pathlib import Path
from typing import Any, Callable

from infinidev.engine.team.store import TeamStore, member_label, now

logger = logging.getLogger(__name__)
ROOT = "orchestrator"
_RESERVED_TOOLS = {"send_message", "request_capability", "help"}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def clone_tools(tools: list, agent_id: str) -> list:
    """Bindings belong to a worker, never to shared catalog instances."""
    from infinidev.tools.base.context import bind_tools_to_agent

    cloned = [tool.model_copy() if hasattr(tool, "model_copy") else copy(tool) for tool in tools]
    bind_tools_to_agent(cloned, agent_id)
    return cloned


class TeamRuntime:
    """Run a session team; the board survives, process liveness does not."""

    def __init__(self, *, session_id: str, project_id: int, workspace_path: str,
                 root_agent_id: str, catalog: list, max_workers: int = 3,
                 max_agents: int = 12, max_followups: int = 8,
                 worker_runner: Callable | None = None, prompt_configuration: Any = None,
                 repository_path: str | None = None, on_status: Callable | None = None,
                 user_request: str = "", turn_context: str = "",
                 catalog_supplier: Callable | None = None,
                 attachments: list | None = None) -> None:
        key = json.dumps([project_id, str(Path(workspace_path).resolve()), session_id])
        self.store = TeamStore(hashlib.sha256(key.encode()).hexdigest())
        self.session_id = session_id
        self.project_id = project_id
        self.workspace_path = workspace_path
        self.repository_path = repository_path
        self.root_agent_id = root_agent_id
        self.prompt_configuration = prompt_configuration
        self.user_request = user_request
        self.turn_context = turn_context
        self.attachments = list(attachments or [])
        self._guidance_attachments: dict[int, list] = {}
        self._catalog_supplier = catalog_supplier
        self.catalog = {t.name: t for t in catalog
                        if t.name not in _RESERVED_TOOLS and not t.name.startswith("team_")}
        self.max_agents = max_agents
        self.max_followups = max_followups
        self._runner = worker_runner
        self._on_status = on_status
        self._lock = threading.RLock()
        self._changed = threading.Event()
        self._stopped = threading.Event()
        self._writers = threading.Lock()
        self._active: set[str] = set()
        self._engines: dict[str, Any] = {}
        self._worker_metrics: dict[str, int] = {}
        self._cursors: dict[str, int] = {}
        self._followups: dict[str, int] = {}
        self._run_id = uuid.uuid4().hex

        def claim(state, emit):
            if state.get("running") and _pid_alive(state["owner_pid"]):
                raise ValueError("This session already has a running team")
            if not state:
                state.update(agents={}, tickets={}, session_id=session_id,
                             workspace_path=workspace_path)
            for member in state["agents"].values():
                member["status"] = "idle"
                member.setdefault("role", "Specialist")
            for ticket in state["tickets"].values():
                if ticket["status"] == "running":
                    ticket["status"] = "interrupted"
                    ticket["updated_at"] = now()
                    emit("ticket", ROOT, "Worker interrupted; liveness must be revalidated",
                         ticket_id=ticket["id"])
            state.update(running=True, owner_pid=os.getpid(), run_id=self._run_id)
            cursor = state["agents"].get(ROOT, {}).get("cursor", 0)
            state["agents"][ROOT] = {"id": ROOT, "name": "Orchestrator", "status": "running",
                                     "role": "Orchestrator", "cursor": cursor}
            emit("lifecycle", ROOT, "Team attached to this turn; persisted state is not liveness")
            if user_request:
                emit("user_guidance", "user", user_request, recipient="all")

        self.store.update(claim)
        self._cursors = {key: member.get("cursor", 0)
                         for key, member in self.store.snapshot()["agents"].items()}
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_workers),
                                        thread_name_prefix="infinidev-team")

    def actor(self, agent_id: str) -> str:
        if agent_id == self.root_agent_id:
            return ROOT
        if agent_id in self.store.snapshot()["agents"] and agent_id != ROOT:
            return agent_id
        raise ValueError("Agent is not a member of this team")

    @staticmethod
    def _require_root(actor: str) -> None:
        if actor != ROOT:
            raise ValueError("Only the orchestrator can manage tickets and delegate")

    def create_ticket(self, actor: str, *, title: str, objective: str,
                      acceptance: list[str], constraints: list[str],
                      dependencies: list[str]) -> dict:
        self._require_root(actor)
        ticket_id = "t_" + uuid.uuid4().hex[:12]

        def create(state, emit):
            if any(dep not in state["tickets"] for dep in dependencies):
                raise ValueError("Dependencies must name existing tickets in this team")
            ticket = dict(id=ticket_id, title=title, objective=objective,
                          acceptance=acceptance, constraints=constraints,
                          dependencies=list(dict.fromkeys(dependencies)), author=actor,
                          assignee=None, status="pending", result="", review="",
                          created_at=now(), updated_at=now())
            state["tickets"][ticket_id] = ticket
            emit("ticket", actor, json.dumps(ticket), ticket_id=ticket_id)
            return ticket

        return self.store.update(create)

    def delegate(self, actor: str, *, ticket_id: str, name: str,
                 system_prompt: str, tools: list[str], worker_id: str | None = None,
                 role: str = "Specialist") -> dict:
        self._require_root(actor)
        name, role = name.strip(), role.strip()
        if not name or not role:
            raise ValueError("A worker needs a name and a visible role")
        if name.casefold() in {ROOT, "all", "user"}:
            raise ValueError("Choose a worker name distinct from built-in recipients")
        self.refresh_catalog()
        if self._stopped.is_set():
            raise ValueError("Team is stopping")
        unknown = set(tools) - self.catalog.keys()
        if unknown:
            raise ValueError(f"Unavailable tools: {', '.join(sorted(unknown))}")
        member_id = worker_id or "w_" + uuid.uuid4().hex[:12]

        def assign(state, emit):
            if worker_id and (worker_id == ROOT or worker_id not in state["agents"]
                              or worker_id in self._active):
                raise ValueError("Reuse an existing idle worker")
            if not worker_id and len(state["agents"]) - 1 >= self.max_agents:
                raise ValueError("Team agent limit reached; reuse workers by messaging them")
            ticket = self._ticket(state, ticket_id)
            if ticket["status"] not in {"pending", "needs_work", "interrupted", "blocked", "failed"}:
                raise ValueError("Ticket is not ready for delegation")
            if any(state["tickets"][dep]["status"] != "accepted" for dep in ticket["dependencies"]):
                raise ValueError("Dependencies must be reviewed and accepted before delegation")
            if any(m.get("name", "").casefold() == name.casefold() and m["id"] != member_id
                   for m in state["agents"].values()):
                raise ValueError("Choose a unique worker name")
            member = dict(id=member_id, name=name, role=role, ticket_id=ticket_id,
                          system_prompt=system_prompt, tools=list(dict.fromkeys(tools)),
                          status="queued", cursor=self._cursors.get(member_id, 0))
            state["agents"][member_id] = member
            ticket.update(assignee=member_id, status="running", updated_at=now())
            emit("delegation", actor, json.dumps(member), ticket_id=ticket_id)
            return member

        with self._lock:
            member = self.store.update(assign)
            self._schedule(member_id, assignment=True)
        self._notice(f"{member_label(member)}: {self.store.snapshot()['tickets'][ticket_id]['title']}")
        return member

    @staticmethod
    def _ticket(state: dict, ticket_id: str) -> dict:
        if ticket_id not in state["tickets"]:
            raise ValueError("Unknown ticket in this team")
        return state["tickets"][ticket_id]

    def refresh_catalog(self) -> None:
        """Include MCP discovery completed after the turn began; grants stay fixed."""
        if self._catalog_supplier:
            with self._lock:
                for tool in self._catalog_supplier():
                    if tool.name not in _RESERVED_TOOLS and not tool.name.startswith("team_"):
                        self.catalog.setdefault(tool.name, tool)

    def review(self, actor: str, *, ticket_id: str, decision: str, reason: str) -> dict:
        self._require_root(actor)
        with self._lock:
            def change(state, emit):
                ticket = self._ticket(state, ticket_id)
                if decision == "accepted" and ticket["status"] != "review":
                    raise ValueError("Only a delivered report can be accepted")
                if ticket.get("assignee") in self._active and decision != "cancelled":
                    raise ValueError("Worker is still running; wait for its report")
                if decision not in {"accepted", "needs_work", "cancelled"}:
                    raise ValueError("Invalid review decision")
                ticket.update(status=decision, review=reason, updated_at=now())
                emit("review", actor, f"{decision}: {reason}", ticket_id=ticket_id)
                return ticket

            ticket = self.store.update(change)
            if decision == "cancelled":
                engine = self._engines.get(ticket.get("assignee"))
                if engine is not None:
                    engine.cancel()
            self._changed.set()
            return ticket

    def write_note(self, actor: str, *, content: str, kind: str, refs: list[str],
                   ticket_id: str | None = None, supersedes: int | None = None) -> dict:
        with self._lock:
            if supersedes is not None:
                previous = self.store.event(supersedes)
                if previous["kind"] != "note" or previous["superseded_by"]:
                    raise ValueError("Supersede a current note, not a message or historical revision")

            def append(state, emit):
                if ticket_id:
                    self._ticket(state, ticket_id)
                return emit("note", actor, json.dumps({"kind": kind, "text": content}),
                            ticket_id=ticket_id, supersedes=supersedes, refs=refs)

            event_id = self.store.update(append)
            self._changed.set()
            return self.store.event(event_id)

    def send(self, actor: str, *, recipient: str, content: str,
             reply_to: int | None = None, ticket_id: str | None = None) -> dict:
        with self._lock:
            state = self.store.snapshot()
            if recipient not in state["agents"] and recipient != "all":
                matches = [m["id"] for m in state["agents"].values()
                           if m["name"].casefold() == recipient.casefold()]
                if len(matches) != 1:
                    raise ValueError("Unknown recipient; consult team_read for the roster")
                recipient = matches[0]
            if recipient == actor:
                raise ValueError("Send the message to another team member")
            if reply_to is not None:
                original = self.store.event(reply_to)
                if (original["kind"] != "message" or original["author"] != recipient
                        or original["recipient"] not in {actor, "all"}):
                    raise ValueError("Reply must address the sender of a message delivered to you")

            def append(current, emit):
                if ticket_id:
                    self._ticket(current, ticket_id)
                return emit("message", actor, content, recipient=recipient,
                            ticket_id=ticket_id, reply_to=reply_to)

            event_id = self.store.update(append)
            # Replies are delivered to active loops but do not start an endless
            # chain of idle workers acknowledging each other's acknowledgements.
            if reply_to is None and not self._stopped.is_set():
                targets = state["agents"] if recipient == "all" else [recipient]
                for target in targets:
                    if target not in {ROOT, actor} and target not in self._active:
                        self._schedule(target, assignment=False)
            self._changed.set()
            return self.store.event(event_id)

    def poll(self, actor: str) -> str:
        """Deliver new attributed events without upgrading them to user authority."""
        with self._lock:
            page = self.store.events(after=self._cursors.get(actor, 0),
                                     recipient=actor, limit=20)
            if not page:
                return ""
            self._cursors[actor] = page[-1]["id"]
            self.store.update(lambda state, emit: state["agents"][actor].update(
                cursor=page[-1]["id"]))
            visible = [e for e in page if e["author"] != actor]
            return json.dumps(visible, ensure_ascii=False) if visible else ""

    def forward_user_guidance(self, content: str, attachments: list | None = None) -> None:
        """Called only by the root loop's user-input path, never by an agent tool."""
        with self._lock:
            event_id = self.store.update(lambda state, emit: emit(
                "user_guidance", "user", content, recipient="all"))
            if attachments:
                self._guidance_attachments[event_id] = list(attachments)
                self.attachments.extend(attachments)
        self._changed.set()

    def guidance_attachments(self, event_id: int) -> list:
        """Live image payloads stay in memory, separate from the durable event log."""
        with self._lock:
            return list(self._guidance_attachments.get(event_id, []))

    def read(self, *, view: str = "board", after: int = 0, limit: int = 50,
             ticket_id: str | None = None) -> dict:
        if view == "board":
            state = self.store.snapshot()
            return {
                "team_id": self.store.team_id,
                "agents": [{k: v for k, v in m.items() if k != "system_prompt"}
                           for m in state["agents"].values()],
                "tickets": [dict(t, assignee_label=member_label(state["agents"][t["assignee"]]))
                            if t.get("assignee") in state["agents"] else t
                            for t in state["tickets"].values()],
            }
        kind = {"notes": "note", "messages": "message", "events": None}[view]
        page = self.store.events(after=after, limit=limit, kind=kind, ticket_id=ticket_id)
        return {"events": page, "next_after": page[-1]["id"] if page else after,
                "page_full": len(page) == min(100, limit)}

    def wait(self, seconds: float = 5) -> dict:
        self._changed.wait(timeout=max(0, min(10, seconds)))
        self._changed.clear()
        return self.read()

    @property
    def has_active_workers(self) -> bool:
        with self._lock:
            return bool(self._active)

    def observed_worker_metrics(self) -> dict[str, int]:
        """Counters for all worker invocations in this turn, including follow-ups."""
        with self._lock:
            return dict(self._worker_metrics)

    def completion_blocker(self) -> str:
        with self._lock:
            if self._active:
                return "Workers are still running. Use team_wait and review their reports."
            pending = [t["id"] for t in self.store.snapshot()["tickets"].values()
                       if t["status"] not in {"accepted", "cancelled"}]
            return ("Unresolved tickets: " + ", ".join(pending)
                    + ". Review reports, arrange rework, or cancel obsolete work with a reason."
                    if pending else "")

    def _schedule(self, member_id: str, *, assignment: bool) -> None:
        if self._stopped.is_set() or member_id in self._active:
            return
        if not assignment:
            count = self._followups.get(member_id, 0)
            if count >= self.max_followups:
                self.store.update(lambda state, emit: emit(
                    "lifecycle", ROOT, f"{member_id}: automatic follow-up limit reached; "
                    "message retained for explicit orchestration", recipient=ROOT))
                return
            self._followups[member_id] = count + 1
        self._active.add(member_id)
        self._pool.submit(self._work, member_id, assignment)

    def _work(self, member_id: str, assignment: bool) -> None:
        from infinidev.engine.engines.base import loop_observed_metrics
        from infinidev.engine.team.worker import run_worker

        result, status = "Worker cancelled before starting", "cancelled"
        writer_acquired = False
        try:
            member = self.store.snapshot()["agents"][member_id]
            missing = set(member["tools"]) - self.catalog.keys()
            if missing:
                raise ValueError(f"Previously granted tools no longer available: {sorted(missing)}")
            writes = any(not getattr(self.catalog[n], "is_read_only", False)
                         for n in member["tools"])
            # Shared workspace baselines and edits cannot be owned concurrently.
            # Unknown/MCP effects are conservatively treated as writes.
            if writes:
                while not self._stopped.is_set():
                    if self._writers.acquire(timeout=0.1):
                        writer_acquired = True
                        break
            if not self._stopped.is_set():
                ticket = self.store.snapshot()["tickets"][member["ticket_id"]]
                if ticket["status"] != "cancelled":
                    self.store.update(lambda state, emit: state["agents"][member_id].update(
                        status="running"))
                    runner = self._runner or run_worker
                    result, status = runner(self, member, ticket, assignment)
        except Exception as exc:
            logger.exception("Team worker %s failed", member_id)
            result, status = f"Worker failed: {type(exc).__name__}: {exc}", "failed"
        finally:
            if writer_acquired:
                self._writers.release()
            with self._lock:
                metrics = loop_observed_metrics(self._engines.get(member_id))
                for key, value in metrics.items():
                    self._worker_metrics[key] = self._worker_metrics.get(key, 0) + value

                def finish(state, emit):
                    member = state["agents"][member_id]
                    member.update(status="idle", last_result=result, last_metrics=metrics)
                    ticket = state["tickets"][member["ticket_id"]]
                    if (assignment and ticket["assignee"] == member_id
                            and ticket["status"] != "cancelled"):
                        ticket.update(result=result,
                                      status="review" if status == "completed" else status,
                                      updated_at=now())
                    return emit("report", member_id, result, recipient=ROOT,
                                ticket_id=member["ticket_id"])

                try:
                    self.store.update(finish)
                finally:
                    self._engines.pop(member_id, None)
                    self._active.discard(member_id)
                    self._changed.set()
                # A request can arrive after the last model call but before
                # this transition to idle. Never strand it in that race.
                unread = self.store.events(after=self._cursors.get(member_id, 0),
                                           kind="message", recipient=member_id, limit=100)
                if any(e["author"] != member_id and e["reply_to"] is None for e in unread):
                    self._schedule(member_id, assignment=False)
            self._notice(f"{member_id}: {status}; report saved")

    def attach_engine(self, member_id: str, engine: Any) -> None:
        with self._lock:
            self._engines[member_id] = engine
            if self._stopped.is_set():
                engine.cancel()

    def cancel(self) -> None:
        self._stopped.set()
        with self._lock:
            for engine in self._engines.values():
                engine.cancel()
        self._changed.set()

    def close(self) -> None:
        """Wait for cooperative cancellation before releasing the durable owner."""
        self.cancel()
        self._pool.shutdown(wait=True)

        def release(state, emit):
            state["running"] = False
            state["agents"][ROOT]["status"] = "idle"
            emit("lifecycle", ROOT, "Team turn closed")

        self.store.update(release)

    def _notice(self, message: str) -> None:
        if self._on_status:
            try:
                self._on_status("info", message)
            except Exception:
                logger.debug("Team status callback failed", exc_info=True)
