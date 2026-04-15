"""Base tool class for all INFINIBAY tools."""

import inspect
import json
import logging
import os
import sqlite3
from abc import ABC
from typing import Any

from crewai.tools import BaseTool

from infinidev.config.settings import settings
from infinidev.tools.base.context import (
    get_context,
    get_context_for_agent,
    get_current_agent_id,
    get_current_agent_run_id,
    get_current_project_id,
    get_current_session_id,
    get_current_workspace_path,
)
from infinidev.tools.base.db import DBConnection, execute_with_retry, get_db_path

logger = logging.getLogger(__name__)


class InfinibayBaseTool(BaseTool, ABC):
    """Abstract base class for all INFINIBAY tools.

    Provides:
    - Database access with retry logic
    - Context awareness (project_id, agent_id, etc.)
    - Audit logging of tool usage
    - Standard error handling
    - Kwargs filtering: LLMs sometimes hallucinate extra parameters
      (e.g. ``project_id``, ``agent_id``) that are not in the tool's
      schema.  ``run()`` strips them before calling ``_run()``.

    Context resolution order:
    1. Agent-bound context (process-global dict, works across threads)
    2. Thread-local / ContextVar / environment variables (fallback)
    """

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Override CrewAI's run() to strip kwargs not accepted by _run().

        LLMs (especially Gemini) sometimes pass context fields like
        ``project_id`` or ``agent_id`` as tool arguments even though
        they're not in the schema.  This causes TypeError in _run().
        """
        sig = inspect.signature(self._run)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not accepts_var_kw:
            allowed = set(sig.parameters.keys())
            extra = set(kwargs.keys()) - allowed
            if extra:
                logger.debug(
                    "Tool %s: stripping unexpected kwargs %s",
                    self.name, extra,
                )
                kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return super().run(*args, **kwargs)

    def _get_bound_context(self) -> "ToolContext | None":
        """Get context from process-global storage using bound agent ID."""
        bound_id = getattr(self, "_bound_agent_id", None)
        if bound_id:
            ctx = get_context_for_agent(bound_id)
            if ctx.agent_id is not None:
                return ctx
        return None

    @property
    def project_id(self) -> int | None:
        ctx = self._get_bound_context()
        if ctx and ctx.project_id is not None:
            return ctx.project_id
        return get_current_project_id()

    @property
    def agent_id(self) -> str | None:
        ctx = self._get_bound_context()
        if ctx and ctx.agent_id is not None:
            return ctx.agent_id
        return get_current_agent_id()

    @property
    def agent_run_id(self) -> str | None:
        ctx = self._get_bound_context()
        if ctx and ctx.agent_run_id is not None:
            return ctx.agent_run_id
        return get_current_agent_run_id()

    @property
    def session_id(self) -> str | None:
        ctx = self._get_bound_context()
        if ctx and ctx.session_id is not None:
            return ctx.session_id
        return get_current_session_id()

    @property
    def workspace_path(self) -> str | None:
        ctx = self._get_bound_context()
        if ctx and ctx.workspace_path is not None:
            return ctx.workspace_path
        return get_current_workspace_path()

    def _bind_delegate(self, tool: "InfinibayBaseTool") -> None:
        """Propagate agent binding to a tool created at runtime.

        Use this when one tool internally creates another tool instance
        so the delegate can access the same agent context::

            delegate = SomeOtherTool()
            self._bind_delegate(delegate)
            return delegate._run(...)
        """
        bound_id = getattr(self, "_bound_agent_id", None) or self.agent_id
        if bound_id:
            object.__setattr__(tool, "_bound_agent_id", bound_id)

    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths against workspace_path.

        In pod mode, returns paths relative to the pod's working directory
        so that the pod's CWD handles final resolution.  Absolute host
        paths that fall under the workspace are converted to relative.
        """
        if self._is_pod_mode():
            if os.path.isabs(path):
                ws = self.workspace_path
                if ws and (path == ws or path.startswith(ws + os.sep)):
                    return os.path.relpath(path, ws)
                return path
            return os.path.normpath(path)

        if os.path.isabs(path):
            return path
        ws = self.workspace_path or os.getcwd()
        return os.path.normpath(os.path.join(ws, path))

    @property
    def _git_cwd(self) -> str | None:
        """Return the directory to use as ``cwd`` for git subprocess calls.

        Returns ``workspace_path`` if set and the directory exists, otherwise
        ``None`` (subprocess will use the process CWD as fallback).
        """
        ws = self.workspace_path
        if ws and os.path.isdir(ws):
            return ws
        return None

    def _is_pod_mode(self) -> bool:
        """Check if sandbox (pod) mode is active."""
        return settings.SANDBOX_ENABLED

    def _exec_in_pod(
        self,
        command: list[str],
        cwd: str | None = None,
        timeout: int = 300,
        stdin_data: str | None = None,
    ) -> "SandboxResult":
        """Execute a command in the agent's persistent pod."""
        from infinidev.security.pod_manager import pod_manager  # lazy import

        agent_id = self._validate_agent_context()
        return pod_manager.exec_in_pod(
            agent_id=agent_id,
            command=command,
            cwd=cwd or pod_manager.get_workdir(agent_id),
            timeout=timeout,
            stdin_data=stdin_data,
        )

    def _execute_db(self, fn, db_path: str | None = None):
        """Execute a function with database retry logic."""
        return execute_with_retry(fn, db_path=db_path)

    def _log_tool_usage(self, message: str, progress: int | None = None):
        """Record tool usage in status_updates for audit."""
        project_id = self.project_id
        agent_id = self.agent_id or "unknown"
        agent_run_id = self.agent_run_id

        def _insert(conn: sqlite3.Connection):
            conn.execute(
                """INSERT INTO status_updates (project_id, agent_id, agent_run_id, message, progress)
                   VALUES (?, ?, ?, ?, ?)""",
                (project_id, agent_id, agent_run_id, f"[{self.name}] {message}", progress),
            )
            conn.commit()

        try:
            execute_with_retry(_insert)
        except Exception:
            logger.debug("Failed to log tool usage for %s", self.name, exc_info=True)

    def _validate_project_context(self) -> int:
        """Validate that project_id is available, raise if not."""
        pid = self.project_id
        if pid is None:
            tool_name = getattr(self, "name", type(self).__name__)
            bound = getattr(self, "_bound_agent_id", None)
            raise ValueError(
                f"No project_id in context for tool '{tool_name}' "
                f"(bound_agent_id={bound}). "
                "Ensure bind_tools_to_agent() and activate_context() were called."
            )
        return pid

    def _validate_agent_context(self) -> str:
        """Validate that agent_id is available, raise if not."""
        aid = self.agent_id
        if aid is None:
            tool_name = getattr(self, "name", type(self).__name__)
            bound = getattr(self, "_bound_agent_id", None)
            raise ValueError(
                f"No agent_id in context for tool '{tool_name}' "
                f"(bound_agent_id={bound}). "
                "Ensure bind_tools_to_agent() and activate_context() were called."
            )
        return aid

    @staticmethod
    def _validate_sandbox_path(path: str) -> str | None:
        """Validate that *path* is inside the sandbox.

        Resolves symlinks via ``os.path.realpath`` and checks proper
        directory boundaries (not just prefix matching).

        Returns ``None`` if valid, or an error message string if denied.
        """
        if not settings.SANDBOX_ENABLED:
            return None

        real = os.path.realpath(path)
        for allowed in settings.ALLOWED_BASE_DIRS:
            allowed_real = os.path.realpath(allowed)
            # Exact match or proper subdirectory (boundary on os.sep)
            if real == allowed_real or real.startswith(allowed_real + os.sep):
                return None

        return (
            f"Access denied: path '{path}' is outside allowed directories "
            f"({settings.ALLOWED_BASE_DIRS})"
        )

    def _error(self, message: str) -> str:
        """Return a JSON error string for consistent error formatting."""
        return json.dumps({"error": message})

    def _warning(self, message: str) -> str:
        """Return a JSON warning string for advisory messages.

        Unlike _error, this does not block operations — it informs the LLM
        of potential issues (e.g., syntax errors that might be false positives).
        """
        return json.dumps({"warning": message})

    def _success(self, data: Any) -> str:
        """Return a JSON success string."""
        if isinstance(data, str):
            return data
        return json.dumps(data, default=str)
