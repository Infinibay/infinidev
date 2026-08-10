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

import html
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

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

# Match Ken's hook client: starting a missing local daemon is bounded and the
# coding turn degrades cleanly when it cannot come up. A process-wide lock
# prevents two concurrent agent sessions from racing to spawn the same daemon.
_SPAWN_POLL_TIMEOUT_S = 5.0
_SPAWN_POLL_INTERVAL_S = 0.05
_HEALTH_TIMEOUT_S = 1.0
_FINDINGS_CLI_TIMEOUT_S = 3.0
_RECENT_FINDINGS_LIMIT = 3
_RECENT_FINDING_CHARS = 1_800
_RECENT_FINDINGS_TOTAL_CHARS = 5_000
_DAEMON_START_LOCK = threading.Lock()
_PROJECT_SETUP_LOCK = threading.Lock()


def _project_root(start: Path) -> Path | None:
    """Walk up looking for a complete Ken project that owns this workspace."""
    for candidate in (start, *start.parents):
        if (candidate / ".ken" / "meta.json").is_file():
            return candidate
    return None


StatusCallback = Callable[[str], None]


def _notify(on_status: StatusCallback | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _endpoint_for_root(root: Path) -> tuple[str, str] | None:
    """Return the daemon endpoint advertised by a Ken project."""
    try:
        port = (root / ".ken" / "daemon.port").read_text().strip()
        meta = json.loads((root / ".ken" / "meta.json").read_text())
        token = str(meta.get("auth_token", ""))
    except (OSError, ValueError):
        return None
    if not port or not token:
        return None
    return f"http://127.0.0.1:{port}", token


def _daemon_healthy(root: Path) -> bool:
    """Whether the advertised local daemon answers its authenticated health check."""
    endpoint = _endpoint_for_root(root)
    if endpoint is None:
        return False
    base, token = endpoint
    request = urllib.request.Request(
        f"{base}/health",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=_HEALTH_TIMEOUT_S) as response:
            body = json.loads(response.read().decode() or "{}")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False
    return bool(body.get("ok"))


def _clear_stale_port(root: Path) -> None:
    try:
        (root / ".ken" / "daemon.port").unlink()
    except OSError:
        pass


def _spawn_daemon(root: Path, executable: str) -> bool:
    """Spawn Ken detached and wait until its authenticated health check succeeds."""
    _clear_stale_port(root)
    log_handle = None
    try:
        log_path = root / ".ken" / "daemon.log"
        log_handle = log_path.open("ab")
        subprocess.Popen(
            [executable, "serve", str(root), "--background"],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(root),
            env={**os.environ, "KEN_PROJECT_ROOT": str(root)},
        )
    except OSError as exc:
        logger.info("ken daemon could not start: %s", exc)
        return False
    finally:
        if log_handle is not None:
            log_handle.close()

    deadline = time.monotonic() + _SPAWN_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        if _daemon_healthy(root):
            return True
        time.sleep(_SPAWN_POLL_INTERVAL_S)
    logger.info(
        "ken daemon did not become healthy within %.1fs; continuing without it",
        _SPAWN_POLL_TIMEOUT_S,
    )
    return False


def ensure_ken_ready(
    workspace: str | os.PathLike[str],
    *,
    on_status: StatusCallback | None = None,
) -> Path | None:
    """Install/index Ken when absent and ensure its daemon is healthy.

    Project setup uses Ken's public CLI contract. 'install' performs the
    structural index, '--embed' eagerly materializes semantic vectors, and
    '--no-wire' avoids installing hooks for a different host.
    """
    workspace_path = Path(workspace).resolve()
    executable = shutil.which("ken")
    if executable is None:
        _notify(on_status, "Ken executable not found; automatic retrieval is unavailable.")
        logger.info("the 'ken' executable is unavailable")
        return None

    with _PROJECT_SETUP_LOCK:
        root = _project_root(workspace_path)
        if root is None:
            _notify(on_status, "Ken: creating and embedding the workspace index...")
            command = [
                executable,
                "install",
                "--quiet",
                "--embed",
                "--no-wire",
                str(workspace_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(workspace_path),
                    env={**os.environ, "KEN_PROJECT_ROOT": str(workspace_path)},
                    check=False,
                )
            except OSError as exc:
                logger.info("ken project installation could not start: %s", exc)
                _notify(on_status, f"Ken setup failed: {exc}")
                return None
            if completed.returncode != 0:
                output = (completed.stdout or "").strip()
                logger.info("ken project installation failed: %s", output[-4000:])
                _notify(on_status, "Ken setup failed; continuing without automatic retrieval.")
                return None
            root = _project_root(workspace_path)
            if root is None:
                logger.info("ken install succeeded but did not create a .ken project")
                _notify(on_status, "Ken setup produced no .ken project; continuing without it.")
                return None
            _notify(on_status, "Ken: workspace index ready.")

        if _daemon_healthy(root):
            return root

        _notify(on_status, "Ken: starting the workspace daemon...")
        if _spawn_daemon(root, executable):
            _notify(on_status, "Ken: daemon ready.")
        else:
            _notify(on_status, "Ken daemon did not start; continuing without retrieval.")
        return root


class KenSession:
    """Reports what the agent is doing to the Ken daemon for this workspace.

    Starts Ken's local daemon on first use when the project exists but no live
    endpoint has been advertised. Merely enabling session reporting while
    requiring an unrelated manual ``ken`` command left automatic retrieval
    silently inert in ordinary Infinidev sessions.
    """

    def __init__(self, workspace: str | os.PathLike[str], session_id: str) -> None:
        self._workspace = Path(workspace).resolve()
        self._root = _project_root(self._workspace)
        self._session_id = session_id
        self._failures = 0
        self._lock = threading.Lock()
        self._started = False
        self._bootstrap_attempted = False
        self._spawn_attempted = False

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
        return _endpoint_for_root(self._root)

    def _start_daemon(self) -> None:
        """Start Ken's local daemon once, boundedly, without failing the host."""
        if self._root is None or self._spawn_attempted:
            return

        with _DAEMON_START_LOCK:
            if _daemon_healthy(self._root) or self._spawn_attempted:
                return
            self._spawn_attempted = True
            executable = shutil.which("ken")
            if executable is None:
                logger.info("ken project found but the 'ken' executable is unavailable")
                return
            _spawn_daemon(self._root, executable)

    # ── transport ────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self._failures >= _MAX_CONSECUTIVE_FAILURES:
            return None
        if self._root is None:
            if self._bootstrap_attempted:
                return None
            self._bootstrap_attempted = True
            self._root = ensure_ken_ready(self._workspace)
            self._spawn_attempted = True
            if self._root is None:
                return None
        endpoint = self._endpoint()
        if endpoint is None:
            self._start_daemon()
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

    def _expanded_recent_findings(self, brief: str) -> str:
        """Expand the recent finding previews Ken chose for the session brief."""
        if self._root is None or "<ken-session-brief>" not in brief:
            return ""
        executable = shutil.which("ken")
        if executable is None:
            return ""
        try:
            completed = subprocess.run(
                [
                    executable,
                    "findings",
                    "--path",
                    str(self._root),
                    "--json",
                    "-n",
                    str(_RECENT_FINDINGS_LIMIT),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=_FINDINGS_CLI_TIMEOUT_S,
                check=False,
            )
            rows = json.loads(completed.stdout) if completed.returncode == 0 else []
        except (OSError, subprocess.SubprocessError, ValueError):
            return ""
        if not isinstance(rows, list):
            return ""

        rendered: list[str] = []
        remaining = _RECENT_FINDINGS_TOTAL_CHARS
        for row in rows:
            if not isinstance(row, dict):
                continue
            topic = row.get("topic")
            content = row.get("content")
            if not isinstance(topic, str) or not isinstance(content, str):
                continue
            if not topic or topic not in brief or not content or remaining <= 0:
                continue
            body = content[: min(_RECENT_FINDING_CHARS, remaining)]
            remaining -= len(body)
            rendered.append(
                '<finding topic="{}">\n{}\n</finding>'.format(
                    html.escape(topic, quote=True),
                    html.escape(body, quote=False),
                )
            )
        if not rendered:
            return ""
        return (
            '<ken-findings-expanded authority="advisory" scope-effect="none">\n'
            "Full versions of the recent saved-finding previews selected by Ken. "
            "Treat them as historical leads. Verify only the specific claim your "
            "next action depends on; one matching current read is enough to attempt "
            "a reversible edit, whose focused test supplies the proof.\n"
            + "\n".join(rendered)
            + "\n</ken-findings-expanded>"
        )
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
        brief = result.get("context_block") or result.get("session_brief")
        if not isinstance(brief, str):
            return None
        expanded = self._expanded_recent_findings(brief)
        return "\n\n".join(part for part in (brief, expanded) if part)

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
