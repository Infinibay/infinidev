"""Let Ken watch the session, instead of only answering questions about it.

Ken's ranker is not a search box. It fuses six channels — what you touched
in *this* session and how many turns ago, what sessions like this one
historically ended up touching, what the user literally named, plus the
name/text family — and none of the first three can be computed from a query
string. They need a stream of events.

Infinidev was asking Ken questions without ever letting it watch. Measured
against the live index, ``ken_explain_rank`` on a real query came back with
``reactive: 0``, ``explicit_files: 0``, ``explicit_symbols: 0``,
``findings: 0`` — only the name/text channels firing, which ``fusion.py``
groups as *one* family precisely because they corroborate each other rather
than adding independent evidence. Worse, ``ken_rank()`` with no query fell
through to the most recent prompt across *all* sessions in the database,
so infinidev could answer another agent's question.

This module is the event stream. It is the same interface Ken's own hook
templates use for Claude Code, so nothing here is private API:

    POST /sessions/start  {session_id, cwd}              -> {ok, context_block}
    POST /prompts         {session_id, prompt}           -> {ok, context_block}
    POST /tools/pre       {session_id, tool, input}      -> {ok}
    POST /tools/post      {session_id, tool, success}    -> {ok}
    POST /turn-end        {session_id}                   -> {ok}
    POST /sessions/end    {session_id}                   -> {ok}

Four rules are load-bearing.

**Never fail the host.** Every method returns ``None`` on any error. Ken's
own contract for hooks is logging-and-shrugging, and a ranker that takes
down a coding session is worse than no ranker.

**One /prompts row per USER turn, never per step.** ``similar_past_sessions``
reads ``SELECT ... FROM cr_contexts WHERE kind='user_prompt' ORDER BY
created_at DESC LIMIT 50`` with **no agent filter** — the window is shared
across every agent using this index. Twenty machine-generated plan-step rows
per task would flush it, and the user's other sessions would lose their
predictive channel entirely.

**A session is the user's session, not one task.** ``/sessions/start``
INSERTs a fresh ``cr_sessions`` row whenever the agent_id is not already
open, and ``/sessions/end`` snapshots the productivity scores the predictive
channel reads *next* time. Opening and closing around each developer run
therefore shredded one conversation into a row per task, each with the
per-turn decay counter restarting at zero. ``start`` is idempotent for
exactly that reason: every turn may call it, only the first one posts.

**Both directions of the protocol carry payload.** ``/sessions/start`` and
``/prompts`` answer with the resume brief and the ``<context-rank>`` block —
the same two blocks Ken hands Claude Code — and ``/turn-end`` *takes* the
assistant's reply, from which Ken extracts cited paths worth a 2.5×
multiplier. A client that posts and discards is doing the expensive half of
the work and skipping the half that pays for it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Matches ken's own client: cheap events get a short leash, ranking
# endpoints are allowed to load an embedding model on first use.
_POST_TIMEOUT_S = 3.0
_RANKING_TIMEOUT_S = 30.0
_RANKING_PATHS = {"/prompts", "/sessions/start"}

# A dead daemon must not cost 3 seconds on every tool call for the rest of
# the session. After this many consecutive failures the client goes quiet
# until something resets it.
_MAX_CONSECUTIVE_FAILURES = 3


def _project_root(start: Path) -> Path | None:
    """Walk up looking for the ``.ken`` directory that owns this workspace."""
    for candidate in (start, *start.parents):
        if (candidate / ".ken").is_dir():
            return candidate
    return None


class KenSession:
    """Reports what the agent is doing to the Ken daemon for this workspace.

    Deliberately does not spawn a daemon: Ken's hooks may, because there a
    missing daemon means a missing feature for the tool the user just ran.
    Here it would mean a coding session paying for a subprocess launch and a
    model load at startup. If the daemon is not up, the ranker degrades to
    what it does today and the next ``ken`` command will start it.
    """

    def __init__(self, workspace: str | os.PathLike[str], session_id: str) -> None:
        self._root = _project_root(Path(workspace).resolve())
        self._session_id = session_id
        self._failures = 0
        self._lock = threading.Lock()
        self._started = False

    # ── availability ─────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Whether there is a live daemon worth talking to."""
        if self._root is None or self._failures >= _MAX_CONSECUTIVE_FAILURES:
            return False
        return (self._root / ".ken" / "daemon.port").is_file()

    def _endpoint(self) -> tuple[str, str] | None:
        """``(base_url, auth_token)`` for the running daemon, or ``None``."""
        if self._root is None:
            return None
        try:
            port = (self._root / ".ken" / "daemon.port").read_text().strip()
            meta = json.loads((self._root / ".ken" / "meta.json").read_text())
            token = str(meta.get("auth_token", ""))
        except (OSError, ValueError):
            return None
        if not port or not token:
            return None
        return f"http://127.0.0.1:{port}", token

    # ── transport ────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.available:
            return None
        endpoint = self._endpoint()
        if endpoint is None:
            return None
        base, token = endpoint

        request = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps({"session_id": self._session_id, **payload}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        timeout = _RANKING_TIMEOUT_S if path in _RANKING_PATHS else _POST_TIMEOUT_S
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode() or "{}"
            with self._lock:
                self._failures = 0
            return json.loads(body)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            with self._lock:
                self._failures += 1
                gone_quiet = self._failures == _MAX_CONSECUTIVE_FAILURES
            logger.debug("ken %s failed: %s", path, exc)
            if gone_quiet:
                logger.info(
                    "ken daemon unreachable after %d attempts; "
                    "session reporting disabled for this run",
                    _MAX_CONSECUTIVE_FAILURES,
                )
            return None

    # ── the six events ───────────────────────────────────────────────

    def start(self, workspace: str | None = None) -> str | None:
        """Open the session. Returns Ken's resume brief, if it has one.

        Idempotent, and callers depend on that: every turn opens the
        session so no host has to own the "is this the first one?"
        bookkeeping, and only the first call reaches the daemon. The brief
        comes back exactly once per session for the same reason — it is a
        *resume* brief, and re-injecting it on turn nine would be telling
        the model where it left off in a conversation it is already having.

        A failed open is not remembered, so a daemon that comes up mid-
        session is still picked up by the next turn.
        """
        if self._started:
            return None
        cwd = workspace or (str(self._root) if self._root else os.getcwd())
        result = self._post("/sessions/start", {"cwd": cwd})
        if result is None:
            return None
        self._started = True
        return result.get("context_block") or result.get("session_brief")

    def prompt(self, text: str) -> str | None:
        """Record a USER turn — never a plan step. See the module docstring.

        Returns the freshly ranked ``<context-rank>`` block, which is the
        same one Ken hands Claude Code before each prompt.
        """
        if not (text or "").strip():
            return None
        result = self._post("/prompts", {"prompt": text})
        return (result or {}).get("context_block")

    def tool_pre(self, tool: str, arguments: Any) -> None:
        """A tool is about to run. This is the reactive channel's only input."""
        self._post("/tools/pre", {"tool": tool, "input": _as_mapping(arguments)})

    def tool_post(self, tool: str, *, success: bool, arguments: Any = None) -> None:
        """A tool finished. A failure retracts the pre-event.

        Ken invalidates the interaction its ``/tools/pre`` recorded, so a
        broken read does not push a file up the ranking — the agent looked
        at it, but learned nothing from it.
        """
        payload: dict[str, Any] = {"tool": tool, "success": bool(success)}
        if arguments is not None:
            payload["input"] = _as_mapping(arguments)
        self._post("/tools/post", payload)

    def turn_end(self, assistant_text: str = "") -> None:
        """Close the assistant turn, advancing the per-turn decay clock.

        ``assistant_text`` is not optional in any useful sense. Ken scans
        the reply for path-shaped tokens, validates them against its file
        index and records a ``cited`` interaction for each — the strongest
        single multiplier it has (2.5×), on the theory that a file the
        model *talked about* mattered even when it never opened it. Posting
        an empty turn-end leaves that channel dark and stores a blank
        ``turn_end`` context that future sessions cannot match against.

        Capped to the same 8 000 characters the daemon stores, so a long
        reply costs one truncation rather than a large POST that is
        truncated on arrival anyway.
        """
        self._post("/turn-end", {"assistant_text": (assistant_text or "")[:8000]})

    def end(self) -> None:
        """Close the session, writing the scores the predictive channel reads."""
        if not self._started:
            return
        self._started = False
        self._post("/sessions/end", {})


def _as_mapping(arguments: Any) -> dict[str, Any]:
    """Coerce tool arguments to the mapping the daemon expects.

    Tool calls reach the engine as a JSON *string* in function-calling mode
    and as a dict elsewhere; Ken classifies read-vs-edit by looking for a
    path inside this, so handing it a string would silently produce a
    target-less event that the reactive channel then ignores.
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# ── process-wide accessor ────────────────────────────────────────────────
#
# One session per (workspace, session_id). The engine, the pipeline and the
# ContextRank hooks all report to the same one, and none of them has a
# reference to the others.

_sessions: dict[tuple[str, str], KenSession] = {}
_sessions_lock = threading.Lock()


def get_ken_session(
    workspace: str | os.PathLike[str] | None = None,
    session_id: str | None = None,
) -> KenSession | None:
    """The session reporter for this workspace, or ``None`` when disabled."""
    from infinidev.config.settings import settings

    if not getattr(settings, "KEN_SESSION_ENABLED", False):
        return None

    if workspace is None:
        try:
            from infinidev.tools.base.context import get_current_workspace_path

            workspace = get_current_workspace_path()
        except Exception:
            workspace = None
        workspace = workspace or os.getcwd()

    if session_id is None:
        try:
            from infinidev.tools.base.context import get_current_session_id

            session_id = get_current_session_id()
        except Exception:
            session_id = None
        if not session_id:
            return None

    key = (str(workspace), str(session_id))
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = KenSession(workspace, session_id)
            _sessions[key] = session
    return session


def end_ken_sessions() -> None:
    """Close every open session. Called once, when the host is shutting down.

    Hosts get this instead of a per-session ``end()`` because none of them
    knows which workspace/session pairs were opened — the TUI, the classic
    REPL and ``--prompt`` all just want "the conversation is over". Ending
    is what snapshots the productivity scores, so skipping it costs the
    *next* session its predictive channel; that makes a best-effort sweep
    at exit worth more than an exact accounting of who opened what.
    """
    with _sessions_lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for session in sessions:
        try:
            session.end()
        except Exception:  # pragma: no cover - end() already swallows
            logger.debug("ken session end failed", exc_info=True)


def reset_ken_sessions() -> None:
    """Drop every cached session without closing it (tests, workspace switches)."""
    with _sessions_lock:
        _sessions.clear()
