"""Durable browser conversations sharing one workspace execution lane."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from infinidev.config.secrets import redact
from infinidev.db.service import (
    get_recent_turns_full,
    get_session_messages,
    register_session,
    rename_session,
    store_conversation_turn,
    store_session_message,
)
from infinidev.tools.base.db import execute_with_retry

logger = logging.getLogger(__name__)


def public(value: Any) -> Any:
    """Remove runtime handles and apply the same credential redaction as the TUI."""
    if isinstance(value, dict):
        value = {k: public(v) for k, v in value.items() if not k.startswith("_")}
    elif isinstance(value, list):
        value = [public(v) for v in value]
    return json.loads(redact(json.dumps(value, default=str)))


class WebRuntime:
    """One project per process; global engine settings require serialized turns."""

    def __init__(self, root: Path, loop: asyncio.AbstractEventLoop) -> None:
        self.root = root.resolve()
        self.loop = loop
        self.lock = threading.RLock()
        self.file_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-engine")
        self.sessions: dict[str, WebSession] = {}
        self.active: WebSession | None = None

    def session(self, session_id: str) -> WebSession:
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id]
            row = execute_with_retry(lambda conn: conn.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND workspace_path = ?",
                (session_id, str(self.root)),
            ).fetchone())
            if row is None:
                raise KeyError(session_id)
            session = WebSession(self, session_id, row["title"] or "New session")
            self.sessions[session_id] = session
            return session

    def create_session(self) -> WebSession:
        session_id = uuid.uuid4().hex
        register_session(session_id, str(self.root))
        return self.session(session_id)

    def permission(self, tool: str, description: str, details: str) -> bool:
        session = self.active
        return bool(session and session.ask(
            description, "permission", details=details, tool=tool,
        ) == "allow")

    def close(self) -> None:
        for session in list(self.sessions.values()):
            session.closed = True
            session.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)


class WebSession:
    """A session belongs to the server, never to a WebSocket connection."""

    def __init__(self, runtime: WebRuntime, session_id: str, title: str) -> None:
        self.runtime = runtime
        self.session_id = session_id
        self.title = title
        self.status = "idle"
        self.phase = "idle"
        self.closed = False
        self.engine: Any = None
        self.future: Any = None
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.pending: dict[str, dict] = {}
        self.listeners: set[asyncio.Queue] = set()
        self.sequence = 0
        self.steps: dict = {}
        self.context: dict = {}
        self.streams: dict[tuple[str, str], dict] = {}
        self.deferred: list[str] = []
        self.messages = get_session_messages(session_id)[-500:]
        if not self.messages:
            self.messages = [
                {"speaker": "You" if role == "user" else "Infinidev",
                 "kind": "user" if role == "user" else "agent", "text": text}
                for role, text in get_recent_turns_full(session_id, limit=100,
                                                       max_chars_per_turn=100000)
            ]
        self.needs_resume = bool(self.messages)
        for message in self.messages:
            message.setdefault("id", uuid.uuid4().hex)
            message.setdefault("speaker", message.get("sender", "Infinidev"))
            message.setdefault("kind", message.get("type", "agent"))
            message.setdefault("text", message.get("content", ""))
            if message["kind"] == "think":
                message["kind"] = "reasoning"
            if message["kind"] == "tool_call":
                message.update(kind="tool", text=message.get("tool_name", message["text"]),
                               state="interrupted" if message.get("running") else
                               "error" if message.get("error") else "completed",
                               data={"tool_arguments": message.get("args", {}),
                                     "tool_result_full": message.get("result", ""),
                                     "tool_error": message.get("error", "")})
            if message.get("streaming"):
                message.update(streaming=False, interrupted=True)
            if message["kind"] == "tool" and message.get("state") == "running":
                message.update(state="interrupted", running=False)

    @property
    def busy(self) -> bool:
        return self.status in {"queued", "running", "waiting", "cancelling"}

    def summary(self) -> dict:
        return {"session_id": self.session_id, "title": self.title,
                "status": self.status, "phase": self.phase}

    def snapshot(self) -> dict:
        with self.lock:
            return public({**self.summary(), "sequence": self.sequence,
                           "messages": self.messages, "pending": list(self.pending.values()),
                           "steps": self.steps, "context": self.context})

    def emit(self, envelope: dict) -> None:
        with self.lock:
            self.sequence += 1
            payload = public({**envelope, "sequence": self.sequence})
        if not self.runtime.loop.is_closed():
            self.runtime.loop.call_soon_threadsafe(self._broadcast, payload)

    def _broadcast(self, envelope: dict) -> None:
        for queue in list(self.listeners):
            if queue.full():
                # Slow clients reconnect to a full snapshot instead of losing deltas.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "resync"})
            else:
                queue.put_nowait(envelope)

    def state(self, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)
            self.emit({"type": "state", **self.summary(), "steps": self.steps,
                       "context": self.context})

    def _save(self, message: dict) -> None:
        # Preserve the terminal renderer's field names in the shared ledger.
        message["sender"] = message["speaker"]
        message["type"] = message["kind"]
        if message["kind"] == "reasoning":
            message["type"] = "think"
        if message["kind"] == "tool":
            data = message.get("data", {})
            message.update(type="tool_call", tool_name=message["text"],
                           args=data.get("tool_arguments", {}),
                           result=data.get("tool_result_full", ""),
                           error=data.get("tool_error", ""),
                           running=message.get("state") == "running")
        message["_resume_message_id"] = store_session_message(
            self.session_id, message, message_id=message.get("_resume_message_id"),
        )
        message["_saved_at"] = time.monotonic()

    def add_message(self, speaker: str, text: str, kind: str = "agent", **data: Any) -> dict:
        with self.lock:
            message = {"id": uuid.uuid4().hex, "speaker": speaker, "text": text[:512000],
                       "kind": kind, "created_at": time.time(), **data}
            self.messages.append(message)
            self.messages = self.messages[-500:]
            self._save(message)
            self.emit({"type": "message", "message": message})
            return message

    def stream(self, speaker: str, chunk: str, kind: str = "agent") -> None:
        with self.lock:
            key = (speaker, kind)
            if key not in self.streams:
                self.streams[key] = self.add_message(speaker, "", kind, streaming=True)
            message = self.streams[key]
            chunk = chunk[:max(0, 512000 - len(message["text"]))]
            message["text"] += chunk
            self.emit({"type": "delta", "id": message["id"], "chunk": chunk})
            if time.monotonic() - message.get("_saved_at", 0) > 1:
                self._save(message)

    def end_stream(self, speaker: str, kind: str = "agent") -> None:
        with self.lock:
            message = self.streams.pop((speaker, kind), None)
            if message:
                message["streaming"] = False
                self._save(message)
                self.emit({"type": "message", "message": message})

    def ask(self, prompt: str, kind: str = "text", **details: Any) -> str | None:
        with self.lock:
            if self.closed or self.cancelled.is_set():
                return None
            request_id = uuid.uuid4().hex
            pending = {"request_id": request_id, "prompt": prompt, "kind": kind,
                       **details, "_event": threading.Event(), "_answer": None}
            self.pending[request_id] = pending
            self.state(status="waiting")
            self.emit({"type": "pending", "pending": list(self.pending.values())})
        pending["_event"].wait()
        return pending["_answer"]

    def resolve(self, request_id: str, text: str | None) -> bool:
        with self.lock:
            pending = self.pending.pop(request_id, None)
            if pending is None:
                return False
            pending["_answer"] = text
            pending["_event"].set()
            self.emit({"type": "pending", "pending": list(self.pending.values())})
            if not self.pending:
                self.state(status="running" if self.future and not self.future.done() else "idle")
            return True

    def cancel(self) -> None:
        with self.lock:
            self.cancelled.set()
            self.deferred.clear()
            for request_id in list(self.pending):
                self.resolve(request_id, None)
            if self.engine:
                self.engine.cancel()
            if self.future and self.future.cancel():
                self.state(status="idle", phase="idle")
            elif self.future and not self.future.done():
                self.state(status="cancelling")

    def submit(self, text: str, client_id: str) -> dict:
        with self.runtime.lock, self.lock:
            if self.closed:
                raise ValueError("The server is shutting down.")
            duplicate = next((m for m in self.messages if m.get("client_id") == client_id), None)
            if duplicate is None:
                duplicate = execute_with_retry(lambda conn: conn.execute(
                    "SELECT id FROM session_messages WHERE session_id = ? "
                    "AND json_extract(message_json, '$.client_id') = ? LIMIT 1",
                    (self.session_id, client_id),
                ).fetchone())
            if duplicate:
                return {"accepted": True, "duplicate": True}
            self.add_message("You", text, "user", client_id=client_id)
            store_conversation_turn(self.session_id, "user", text)
            if self.title == "New session":
                self.title = " ".join(text.split())[:80]
                rename_session(self.session_id, self.title)
            if self.busy:
                if self.engine and self.status == "running" and self.phase == "execute":
                    self.engine.inject_message(text)
                else:
                    self.deferred.append(text)
                return {"accepted": True, "steering": True}
            self.cancelled.clear()
            self.state(status="queued")
            self.future = self.runtime.executor.submit(self._run_turn, text)
            return {"accepted": True}

    def _run_turn(self, text: str) -> None:
        from infinidev.flows.event_listeners import event_bus
        from infinidev.server.hooks import ServerHooks

        hooks = ServerHooks(self)
        callback = self.on_engine_event
        failed = False
        self.runtime.active = self
        try:
            if self.cancelled.is_set():
                return
            from infinidev.agents.base import InfinidevAgent
            from infinidev.config.settings import reload_all
            from infinidev.db.service import get_recent_summaries
            from infinidev.engine.analysis.review_engine import ReviewEngine
            from infinidev.engine.loop import LoopEngine
            from infinidev.engine.orchestration import run_task
            from infinidev.engine.orchestration.chat_agent import request_resume_context_once
            import infinidev.prompts.flows  # noqa: F401 — registers flows

            reload_all()
            if self.needs_resume:
                # The browser already restored its transcript; avoid loading all turns
                # again just to enqueue the model's compact continuity checkpoint.
                request_resume_context_once(self.session_id)
                self.needs_resume = False
            agent = InfinidevAgent(agent_id=f"web-{self.session_id[:8]}")
            self.engine = LoopEngine()
            reviewer = ReviewEngine()
            agent._session_summaries = get_recent_summaries(self.session_id, limit=10)
            if self.cancelled.is_set():
                return
            self.state(status="running")
            event_bus.subscribe(callback)
            agent.activate_context(session_id=self.session_id)
            try:
                result = run_task(agent=agent, user_input=text, session_id=self.session_id,
                                  engine=self.engine, reviewer=reviewer, hooks=hooks)
            finally:
                agent.deactivate()
            if result and not hooks.reply_already_shown:
                self.add_message("Infinidev", result)
            if result:
                store_conversation_turn(self.session_id, "assistant", result, result[:200])
        except Exception as exc:
            # The engine boundary surfaces unexpected failures to the user and log.
            failed = True
            logger.exception("Web turn failed (session=%s)", self.session_id)
            self.add_message("Infinidev", str(exc), "error")
        finally:
            event_bus.unsubscribe(callback)
            with self.lock:
                for speaker, kind in list(self.streams):
                    self.end_stream(speaker, kind)
                self.runtime.active = None
                self.engine = None
                self.state(status="error" if failed else "idle", phase="idle")
                self.emit({"type": "refresh"})
                if self.deferred and not self.cancelled.is_set() and not self.closed:
                    next_text = "\n\n".join(self.deferred)
                    self.deferred.clear()
                    self.state(status="queued")
                    self.future = self.runtime.executor.submit(self._run_turn, next_text)

    def on_engine_event(self, event_type: str, project_id: int,
                        agent_id: str, data: dict) -> None:
        if self.cancelled.is_set() and self.engine:
            self.engine.cancel()
        speaker = data.get("agent_name") or "Infinidev"
        team = getattr(self.engine, "_team_runtime", None)
        if team and agent_id.startswith("w_"):
            speaker = team.store.snapshot().get("agents", {}).get(agent_id, {}).get(
                "name", agent_id,
            )
        if event_type == "loop_thinking_chunk":
            self.stream(speaker, data.get("text", ""), "reasoning")
        elif event_type == "loop_stream_status" and data.get("phase") == "done":
            self.end_stream(speaker, "reasoning")
        elif event_type == "loop_user_message":
            self.add_message(speaker, data.get("message", ""), agent_id=agent_id)
        elif event_type in {"loop_tool_start", "loop_tool_call"}:
            run_id = data.get("tool_run_id")
            with self.lock:
                message = next((m for m in reversed(self.messages)
                                if run_id and m.get("tool_run_id") == run_id), None)
                fields = {"tool_run_id": run_id, "agent_id": agent_id, "data": copy.deepcopy(data),
                          "state": "running" if event_type == "loop_tool_start" else
                          "error" if data.get("tool_error") else "completed"}
                if message:
                    message.update(fields)
                    self._save(message)
                    self.emit({"type": "message", "message": message})
                else:
                    self.add_message(speaker, data.get("tool_name", "Tool"), "tool", **fields)
        elif event_type == "loop_tool_output":
            with self.lock:
                run_id = data.get("tool_run_id")
                message = next((m for m in reversed(self.messages) if run_id and
                                m.get("tool_run_id") == run_id), None)
                if message and message.get("state") == "running":
                    detail = message.setdefault("data", {})
                    detail["tool_result_full"] = (detail.get("tool_result_full", "")
                                                  + data.get("chunk", ""))[-512000:]
                    self.emit({"type": "message", "message": message})
                    if time.monotonic() - message.get("_saved_at", 0) > 1:
                        self._save(message)
        elif event_type == "loop_step_update" and not agent_id.startswith("w_"):
            self.state(steps=data)
        elif event_type in {"loop_file_changed", "loop_step_complete", "loop_task_complete"}:
            self.emit({"type": "refresh"})
        if data.get("prompt_tokens"):
            self.state(context={**self.context, "prompt_tokens": data["prompt_tokens"]})
