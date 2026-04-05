"""Agent context management — process-global storage with thread-local fallback.

CrewAI agents with ``max_execution_time`` set run tool execution inside a
``concurrent.futures.ThreadPoolExecutor``, meaning tools execute in a
**different thread** from where ``set_context()`` was called.  Both
``threading.local()`` and ``ContextVar`` are inherently thread-scoped and
cannot propagate across this boundary.

**Primary storage**: a process-global dict (``_agent_contexts``) keyed by
``agent_id``.  Tools are stamped with their owning agent's ID at construction
time (via ``bind_tools_to_agent()``), so they can look up the correct context
from any thread.

**Fallback**: thread-local → ContextVar → environment variables (for backwards
compatibility and non-tool callers).
"""

import logging
import os
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

_ctx_logger = logging.getLogger(__name__)

# ── Process-global storage (primary — survives thread boundaries) ────────────

_agent_contexts: dict[str, "ToolContext"] = {}
_agent_contexts_lock = threading.Lock()

# ── Thread-local storage (fallback for non-tool callers) ─────────────────────

_tls = threading.local()

# ── ContextVar storage (fallback for asyncio-aware code paths) ───────────────

_project_id_var: ContextVar[Optional[int]] = ContextVar("project_id", default=None)
_agent_id_var: ContextVar[Optional[str]] = ContextVar("agent_id", default=None)
_agent_run_id_var: ContextVar[Optional[str]] = ContextVar("agent_run_id", default=None)
_session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_workspace_path_var: ContextVar[Optional[str]] = ContextVar("workspace_path", default=None)


@dataclass
class ToolContext:
    """Snapshot of current agent context."""
    project_id: Optional[int] = None
    agent_id: Optional[str] = None
    agent_run_id: Optional[str] = None
    session_id: Optional[str] = None
    workspace_path: Optional[str] = None
    event_id: Optional[int] = None
    resume_state: Optional[dict] = None
    loop_state: Optional[Any] = None  # LoopState reference for plan tools


# ── Setting context ──────────────────────────────────────────────────────────


def set_context(
    project_id: int | None = None,
    agent_id: str | None = None,
    agent_run_id: str | None = None,
    session_id: str | None = None,
    workspace_path: str | None = None,
    event_id: int | None = None,
    resume_state: dict | None = None,
) -> ToolContext:
    """Set context variables for the current execution scope.

    Writes to:
    1. Process-global dict (keyed by agent_id) — accessible from any thread
    2. Thread-local storage — for callers in the same thread
    3. ContextVar — for asyncio-aware code
    """
    # Robustly cast IDs to int if possible (helps when reading from JSON/ENV)
    if project_id is not None:
        try:
            project_id = int(project_id)
        except (ValueError, TypeError):
            pass

    # ── Write to thread-local + ContextVar (backwards compat) ────────────
    if project_id is not None:
        _tls.project_id = project_id
        _project_id_var.set(project_id)
    if agent_id is not None:
        _tls.agent_id = agent_id
        _agent_id_var.set(agent_id)
    if agent_run_id is not None:
        _tls.agent_run_id = agent_run_id
        _agent_run_id_var.set(agent_run_id)
    if session_id is not None:
        _tls.session_id = session_id
        _session_id_var.set(session_id)
    if workspace_path is not None:
        _tls.workspace_path = workspace_path
        _workspace_path_var.set(workspace_path)

    # ── Write to process-global dict ─────────────────────────────────────
    key = agent_id or getattr(_tls, "agent_id", None) or _agent_id_var.get()
    if key:
        with _agent_contexts_lock:
            existing = _agent_contexts.get(key, ToolContext())
            _agent_contexts[key] = ToolContext(
                project_id=project_id if project_id is not None else existing.project_id,
                agent_id=key,
                agent_run_id=agent_run_id if agent_run_id is not None else existing.agent_run_id,
                session_id=session_id if session_id is not None else existing.session_id,
                workspace_path=workspace_path if workspace_path is not None else existing.workspace_path,
                event_id=event_id if event_id is not None else existing.event_id,
                resume_state=resume_state if resume_state is not None else existing.resume_state,
            )

    if agent_id or project_id or session_id:
        _ctx_logger.debug(
            "set_context: thread=%d(%s) agent_id=%s project_id=%s session_id=%s",
            threading.get_ident(), threading.current_thread().name,
            agent_id, project_id, session_id,
        )
    return get_context()


def get_context_for_agent(agent_id: str) -> ToolContext:
    """Get context for a specific agent from process-global storage."""
    with _agent_contexts_lock:
        return _agent_contexts.get(agent_id, ToolContext())


def set_loop_state(agent_id: str, loop_state: Any) -> None:
    """Attach the LoopState to an agent's context so plan tools can access it."""
    with _agent_contexts_lock:
        ctx = _agent_contexts.get(agent_id)
        if ctx:
            ctx.loop_state = loop_state


def clear_agent_context(agent_id: str) -> None:
    """Remove an agent's context entry (call when agent finishes)."""
    with _agent_contexts_lock:
        _agent_contexts.pop(agent_id, None)


def bind_tools_to_agent(tools: list, agent_id: str) -> None:
    """Stamp tool instances with their owning agent's ID."""
    for tool in tools:
        try:
            object.__setattr__(tool, "_bound_agent_id", agent_id)
        except Exception:
            _ctx_logger.debug(
                "bind_tools_to_agent: could not stamp tool %s",
                getattr(tool, "name", type(tool).__name__),
            )


# ── Getting context ──────────────────────────────────────────────────────────


def get_context() -> ToolContext:
    """Get current context using all available sources."""
    return ToolContext(
        project_id=_get_project_id(),
        agent_id=_get_agent_id(),
        agent_run_id=_get_agent_run_id(),
        session_id=_get_session_id(),
        workspace_path=_get_workspace_path(),
    )


def _get_project_id() -> int | None:
    return (
        getattr(_tls, "project_id", None)
        or _project_id_var.get()
        or _env_int("INFINIDEV_PROJECT_ID")
    )


def _get_agent_id() -> str | None:
    return (
        getattr(_tls, "agent_id", None)
        or _agent_id_var.get()
        or os.environ.get("INFINIDEV_AGENT_ID")
    )


def _get_agent_run_id() -> str | None:
    return (
        getattr(_tls, "agent_run_id", None)
        or _agent_run_id_var.get()
        or os.environ.get("INFINIDEV_AGENT_RUN_ID")
    )


def _get_session_id() -> str | None:
    return (
        getattr(_tls, "session_id", None)
        or _session_id_var.get()
        or os.environ.get("INFINIDEV_SESSION_ID")
    )


def _get_workspace_path() -> str | None:
    return (
        getattr(_tls, "workspace_path", None)
        or _workspace_path_var.get()
        or os.environ.get("INFINIDEV_WORKSPACE_PATH")
    )


# ── Public convenience getters ───────────────────────────────────────────────


def get_current_project_id() -> int | None:
    return _get_project_id()


def get_current_agent_id() -> str | None:
    return _get_agent_id()


def get_current_agent_run_id() -> str | None:
    return _get_agent_run_id()


def get_current_session_id() -> str | None:
    return _get_session_id()


def get_current_workspace_path() -> str | None:
    return _get_workspace_path()


def _env_int(name: str) -> int | None:
    val = os.environ.get(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
