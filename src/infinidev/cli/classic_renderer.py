"""Classic-mode terminal renderer.

Subscribes to ``event_bus`` and paints engine events to stdout in
colour, while also tracking a ``SessionStatus`` snapshot that the
prompt's ``bottom_toolbar`` reads on every refresh.

The renderer replaces the old logger-only bridge in ``cli/main.py``
that registered itself as a no-op subscriber and silently disabled
the inline stderr fallbacks in ``engine/hooks/ui_hooks.py``. The
new contract: events are the single source of output for classic
mode; ``ui_hooks`` no longer prints directly.

Concurrency: event handlers fire from the engine's worker threads.
We only mutate ``SessionStatus`` and call ``print()``/``sys.stdout``,
both of which are safe under Python's GIL. We deliberately avoid
``prompt_toolkit.print_formatted_text`` here because it is not
thread-safe.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from infinidev.flows.event_listeners import event_bus

# ── ANSI colour helpers ──────────────────────────────────────────────────
# Plain ANSI escapes — no Rich/click here so we stay light and
# thread-safe. Disable colour automatically when stdout isn't a TTY.

_USE_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""

def _c(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _USE_COLOUR else text

def _dim(t: str) -> str:    return _c("2", t)
def _bold(t: str) -> str:   return _c("1", t)
def _red(t: str) -> str:    return _c("31", t)
def _green(t: str) -> str:  return _c("32", t)
# Sober palette: secondary roles all collapse to dim. Errors stay red,
# completion stays green. No yellow/blue/cyan/magenta in the scrollback.
_yellow = _dim
_blue = _dim
_magenta = _dim
_cyan = _dim


def _truncate(text: str | None, limit: int = 220) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _fmt_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


# ── SessionStatus dataclass ──────────────────────────────────────────────


@dataclass
class SessionStatus:
    """Mutable snapshot of session state, read by the bottom toolbar."""

    provider: str = ""
    model: str = ""
    critic_model: str = ""
    critic_enabled: bool = False
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    total_tokens: int = 0
    cache_read: int = 0
    cache_create: int = 0
    iteration: int = 0
    step_title: str = ""
    tool_calls_total: int = 0
    last_verdict_action: str = ""
    run_started_at: float | None = None
    git_branch: str = ""
    git_dirty: bool = False
    _git_checked_at: float = 0.0

    def refresh_git(self) -> None:
        now = time.monotonic()
        if now - self._git_checked_at < 5.0:
            return
        self._git_checked_at = now
        try:
            br = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=0.5,
            )
            if br.returncode == 0:
                self.git_branch = br.stdout.strip()
                st = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True, timeout=0.5,
                )
                self.git_dirty = bool(st.stdout.strip())
        except Exception:
            self.git_branch = ""
            self.git_dirty = False


# ── Renderer ─────────────────────────────────────────────────────────────


class ClassicRenderer:
    """Subscribes to event_bus and paints classic-mode terminal output."""

    # Event types we deliberately ignore — too noisy or already covered.
    _IGNORE: frozenset[str] = frozenset({
        "loop_state", "loop_stream_status",
        "loop_llm_call_start", "loop_file_changed", "loop_behavior_update",
    })

    def __init__(self, status: SessionStatus) -> None:
        self.status = status
        # Per-agent buffer for streaming reasoning chunks. Flushed when
        # a newline arrives or the next non-thinking event fires.
        self._think_buf: dict[str, list[str]] = {}
        self._think_lock = threading.Lock()
        self._subscribed = False
        # Cache the bound-method handle. Bound methods are created
        # fresh on every attribute access, so EventBus.unsubscribe
        # (which compares by identity) would never match a freshly
        # produced ``self._on_event`` against the one passed to
        # ``subscribe()``. Cache once at construction.
        self._handler = self._on_event

    # ── Subscription ──────────────────────────────────────────────────

    def subscribe(self) -> None:
        if self._subscribed:
            return
        event_bus.subscribe(self._handler)
        self._subscribed = True

    def unsubscribe(self) -> None:
        if not self._subscribed:
            return
        event_bus.unsubscribe(self._handler)
        self._subscribed = False

    # ── Event dispatch ────────────────────────────────────────────────

    def _on_event(
        self, event_type: str, project_id: int, agent_id: str, data: dict[str, Any],
    ) -> None:
        try:
            if event_type in self._IGNORE:
                return

            # Any non-thinking event flushes pending reasoning.
            if event_type != "loop_thinking_chunk":
                self._flush_think(agent_id)

            handler = getattr(self, f"_on_{event_type}", None)
            if handler:
                handler(agent_id, data)
        except Exception:
            # Renderer must NEVER break the engine. Swallow.
            pass

    # ── Handlers ──────────────────────────────────────────────────────

    def _on_loop_start(self, agent_id: str, data: dict) -> None:
        self.status.run_started_at = time.monotonic()
        self.status.tool_calls_total = 0
        self.status.iteration = 0
        prompt = _truncate(data.get("prompt"), 200)
        if prompt:
            self._println(_dim(f"▸ {prompt}"))

    def _on_loop_step_update(self, agent_id: str, data: dict) -> None:
        self._absorb_tokens(data)
        iteration = data.get("iteration", 0) or 0
        if iteration:
            self.status.iteration = int(iteration)
        title = _truncate(data.get("step_title"), 100)
        status_str = data.get("status", "")
        if title:
            self.status.step_title = title
        if status_str == "active" and title:
            self._println(_dim(f"step {iteration}  {title}"))
        elif status_str == "done" and title:
            self._println(_dim(f"step {iteration} ✓ {title}"))
        elif status_str == "blocked" and title:
            self._println(_dim(f"step {iteration} ⊘ {title}"))

    def _on_loop_tool_call(self, agent_id: str, data: dict) -> None:
        self._absorb_tokens(data)
        self.status.tool_calls_total = int(data.get("total_calls", self.status.tool_calls_total) or self.status.tool_calls_total)
        name = data.get("tool_name", "?")
        detail = _truncate(data.get("tool_detail"), 100)
        err = data.get("tool_error") or ""
        tk = data.get("tokens_total", 0)
        head = f"▸ {_bold(name)} {_dim(detail)}" if detail else f"▸ {_bold(name)}"
        meta = _dim(f"{_fmt_tokens(tk)}tk")
        if err:
            self._println(f"{head}  {meta}  {_red('✗ ' + _truncate(err, 160))}")
        else:
            self._println(f"{head}  {meta}")
            preview = data.get("tool_output_preview") or ""
            if preview:
                for line in str(preview).splitlines()[:6]:
                    self._println(_dim(f"  {line}"))

    def _on_loop_think(self, agent_id: str, data: dict) -> None:
        reasoning = (data.get("reasoning") or "").strip()
        if reasoning:
            self._println(f"{_dim('💭 ' + _truncate(reasoning, 320))}")

    def _on_loop_thinking_chunk(self, agent_id: str, data: dict) -> None:
        text = data.get("text") or ""
        if not text:
            return
        with self._think_lock:
            buf = self._think_buf.setdefault(agent_id, [])
            buf.append(text)
            joined = "".join(buf)
            # Flush per-line so the user sees reasoning stream in.
            if "\n" in joined:
                lines = joined.split("\n")
                full_lines, tail = lines[:-1], lines[-1]
                for line in full_lines:
                    line = line.strip()
                    if line:
                        self._println(_dim("💭 " + line))
                self._think_buf[agent_id] = [tail] if tail else []

    def _flush_think(self, agent_id: str) -> None:
        with self._think_lock:
            buf = self._think_buf.pop(agent_id, None)
        if buf:
            text = "".join(buf).strip()
            if text:
                self._println(_dim("💭 " + _truncate(text, 320)))

    def _on_loop_user_message(self, agent_id: str, data: dict) -> None:
        msg = _truncate(data.get("message"), 320)
        if msg:
            self._println(f"→ {msg}")

    def _on_loop_assistant_message(self, agent_id: str, data: dict) -> None:
        action = data.get("action", "?")
        msg = _truncate(data.get("message"), 320)
        model = data.get("model", "")
        blocked = data.get("blocked", False)
        self.status.last_verdict_action = action
        tag = f"critic [{action}]" if not model else f"critic [{action}] {model}"
        prefix = _dim(tag) if not blocked else _red(f"{tag} BLOCKING")
        self._println(f"{prefix}  {msg}")

    def _on_loop_log(self, agent_id: str, data: dict) -> None:
        level = (data.get("level") or "info").upper()
        msg = _truncate(data.get("message"), 280)
        colour = _red if level in ("ERROR", "CRITICAL") else _dim
        self._println(colour(f"[{level}] {msg}"))

    def _on_loop_finished(self, agent_id: str, data: dict) -> None:
        self._on_loop_end(agent_id, data)

    def _on_loop_end(self, agent_id: str, data: dict) -> None:
        self.status.run_started_at = None
        summary = _truncate(data.get("summary") or data.get("reason"), 160)
        if summary:
            self._println(f"{_green('✓ done')} {_dim(summary)}")

    # ── Internals ─────────────────────────────────────────────────────

    def _absorb_tokens(self, data: dict) -> None:
        if "prompt_tokens" in data:
            self.status.last_prompt_tokens = int(data.get("prompt_tokens") or 0)
        if "completion_tokens" in data:
            self.status.last_completion_tokens = int(data.get("completion_tokens") or 0)
        if "tokens_total" in data:
            self.status.total_tokens = int(data.get("tokens_total") or 0)

    def _println(self, line: str) -> None:
        # Plain print with implicit newline; Python's stdout is GIL-protected.
        try:
            print(line, flush=True)
        except Exception:
            pass


# ── Permission queue (replaces click.confirm in classic) ─────────────────


@dataclass
class _PermissionRequest:
    tool_name: str
    description: str
    details: str
    result: bool = False
    done: threading.Event = field(default_factory=threading.Event)


class PermissionQueue:
    """Thread-safe queue for tool-permission round-trips.

    The engine's worker thread submits a request via ``request()`` and
    blocks on its ``done`` event. The CLI main thread drains the queue
    between renders, prompts the user, and resolves the event.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[_PermissionRequest] = queue.Queue()

    def request(self, tool_name: str, description: str, details: str) -> bool:
        req = _PermissionRequest(tool_name, description, details)
        self._q.put(req)
        # Block worker thread until main thread resolves.
        req.done.wait()
        return req.result

    def pending(self) -> _PermissionRequest | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None


def make_permission_handler(pq: PermissionQueue):
    """Build a callback compatible with ``set_permission_handler``."""
    def _handler(tool_name: str, description: str, details: str) -> bool:
        return pq.request(tool_name, description, details)
    return _handler


# ── Bottom-toolbar callable ──────────────────────────────────────────────


def make_status_renderer(status: SessionStatus):
    """Return a callable suitable for ``PromptSession(bottom_toolbar=...)``.

    prompt_toolkit invokes this on every keystroke and on the configured
    refresh interval. Keep it allocation-light and never raise.
    """
    from prompt_toolkit.formatted_text import FormattedText

    def _render():
        try:
            status.refresh_git()
            parts: list[tuple[str, str]] = []
            # Provider + model
            if status.provider:
                parts.append(("class:tb.provider", f" {status.provider} "))
            if status.model:
                parts.append(("class:tb.model", f"{status.model} "))
            parts.append(("class:tb.sep", "│ "))
            # Critic
            if status.critic_enabled:
                dot = "●"
                parts.append(("class:tb.critic_on", f"critic{dot} "))
                if status.critic_model:
                    parts.append(("class:tb.model", f"{status.critic_model} "))
            else:
                parts.append(("class:tb.critic_off", "critic○ "))
            parts.append(("class:tb.sep", "│ "))
            # Iteration + tools
            parts.append(("class:tb.it", f"it {status.iteration} "))
            parts.append(("class:tb.sep", "▸ "))
            parts.append(("class:tb.tools", f"tools {status.tool_calls_total} "))
            parts.append(("class:tb.sep", "│ "))
            # Tokens
            last = status.last_prompt_tokens + status.last_completion_tokens
            cache_total = status.cache_read + status.cache_create
            cache_pct = (
                int(100 * status.cache_read / cache_total)
                if cache_total > 0 else 0
            )
            tok = f"tok {_fmt_tokens(last)}/{_fmt_tokens(status.total_tokens)}"
            if cache_total > 0:
                tok += f" (cache {cache_pct}%)"
            parts.append(("class:tb.tokens", tok + " "))
            parts.append(("class:tb.sep", "│ "))
            # Git
            if status.git_branch:
                br = status.git_branch + ("✱" if status.git_dirty else "")
                parts.append(("class:tb.git", f"{br} "))
                parts.append(("class:tb.sep", "│ "))
            # Elapsed
            if status.run_started_at is not None:
                elapsed = int(time.monotonic() - status.run_started_at)
                parts.append(("class:tb.elapsed", f"{elapsed}s"))
            else:
                parts.append(("class:tb.idle", "idle"))
            return FormattedText(parts)
        except Exception:
            return FormattedText([("", "")])

    return _render


def status_bar_style():
    """prompt_toolkit Style for the bottom toolbar.

    Sober palette: one foreground colour over a subtle background. The
    ``model`` and ``elapsed`` segments get bold so the eye lands on them
    first; everything else is the same dim text.
    """
    from prompt_toolkit.styles import Style
    base = "bg:#1c1c1c #999999"
    return Style.from_dict({
        "bottom-toolbar": base,
        "tb.provider": base,
        "tb.model": f"{base} bold",
        "tb.sep": "bg:#1c1c1c #444444",
        "tb.critic_on": base,
        "tb.critic_off": "bg:#1c1c1c #555555",
        "tb.it": base,
        "tb.tools": base,
        "tb.tokens": base,
        "tb.git": base,
        "tb.elapsed": f"{base} bold",
        "tb.idle": "bg:#1c1c1c #555555",
        "tb.working": f"{base} bold",
    })


# ── /status slash command ────────────────────────────────────────────────


def render_status_table(status: SessionStatus) -> str:
    """Render the SessionStatus as a multi-line text block.

    Used by the ``/status`` slash command. Plain text (no Rich) so it
    works under all terminals and stays thread-light.
    """
    lines = [
        _bold(_cyan("── Session Status ──────────────────────────────────")),
        f"  provider          : {status.provider or _dim('(unset)')}",
        f"  model             : {status.model or _dim('(unset)')}",
        f"  critic enabled    : {_green('yes') if status.critic_enabled else _dim('no')}",
        f"  critic model      : {status.critic_model or _dim('(n/a)')}",
        f"  iteration         : {status.iteration}",
        f"  step title        : {status.step_title or _dim('(idle)')}",
        f"  total tool calls  : {status.tool_calls_total}",
        f"  last prompt tk    : {status.last_prompt_tokens}",
        f"  last completion tk: {status.last_completion_tokens}",
        f"  total tokens      : {status.total_tokens}",
        f"  cache read / cre  : {status.cache_read} / {status.cache_create}",
        f"  last verdict      : {status.last_verdict_action or _dim('(none)')}",
        f"  git branch        : {status.git_branch}{' ✱' if status.git_dirty else ''}",
        f"  running           : {_green('yes') if status.run_started_at else _dim('no')}",
        _bold(_cyan("────────────────────────────────────────────────────")),
    ]
    return "\n".join(lines)


# ── SIGINT-driven cancellation helper ────────────────────────────────────


class CancellationGuard:
    """Context manager that translates SIGINT into ``engine.cancel()``.

    During a run, instead of letting Ctrl+C raise KeyboardInterrupt
    (which kills run_task mid-tool), we install a signal handler that
    asks the engine to cancel cooperatively. The engine's ``_cancel_event``
    is checked in 4 places (engine.py:505, 1084, 1474, 1491), so the
    next checkpoint exits cleanly and run_task returns.

    Doubles as the user's "are you sure?" gate: a second Ctrl+C within
    1.5s falls back to raising KeyboardInterrupt, which aborts the
    main loop's turn entirely.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._old: Any = None
        self._last_press: float = 0.0

    def __enter__(self):
        import signal
        self._old = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *exc):
        import signal
        signal.signal(signal.SIGINT, self._old)
        return False

    def _handle(self, signum, frame):
        now = time.monotonic()
        if now - self._last_press < 1.5:
            # Second press within 1.5s — let it raise.
            import signal
            signal.signal(signal.SIGINT, self._old)
            print(_red("\n[force interrupt]"), flush=True)
            raise KeyboardInterrupt
        self._last_press = now
        try:
            self._engine.cancel()
            print(_yellow("\n[cancelling — press Ctrl+C again to force]"), flush=True)
        except Exception:
            pass


# ── Terminal width helper for separators ─────────────────────────────────


def hr() -> str:
    width = shutil.get_terminal_size((80, 24)).columns
    return _dim("─" * min(width, 80))


# ── Live-status runner: keeps toolbar visible during a run ───────────────


def run_with_live_status(
    engine: Any,
    task_callable,
    status: SessionStatus,
):
    """Run ``task_callable()`` in a worker thread while a tiny
    ``prompt_toolkit.Application`` keeps the bottom toolbar live.

    The classic mode used to block the main thread on ``run_task()``,
    which meant the status bar disappeared the moment the user hit
    Enter — exactly what the user complained about ("el input no
    debería desaparecer cuando el modelo está pensando").

    Behaviour:
    - Worker thread executes ``task_callable``; the engine's events
      print *above* the app via ``print()`` + ``patch_stdout``.
    - One-line "working …" indicator + bottom toolbar stay anchored.
    - Ctrl+C: first press calls ``engine.cancel()`` cooperatively;
      a second press within 1.5s aborts the app with ``KeyboardInterrupt``.
    - When the worker finishes (success, failure, or cancellation),
      the app is closed from the worker thread via
      ``loop.call_soon_threadsafe`` and ``app.run()`` returns.

    Returns the worker's return value, or re-raises its exception.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout

    done = threading.Event()
    result_holder: list = []
    error_holder: list[BaseException] = []
    cancel_state = {"presses": 0, "last_press": 0.0}

    def _worker() -> None:
        try:
            result_holder.append(task_callable())
        except BaseException as exc:
            error_holder.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    kb = KeyBindings()

    @kb.add("c-c")
    def _(event):
        now = time.monotonic()
        if now - cancel_state["last_press"] < 1.5 and cancel_state["presses"] >= 1:
            event.app.exit(exception=KeyboardInterrupt())
            return
        cancel_state["presses"] += 1
        cancel_state["last_press"] = now
        try:
            engine.cancel()
        except Exception:
            pass

    def _working_line():
        if status.run_started_at is not None:
            elapsed = int(time.monotonic() - status.run_started_at)
        else:
            elapsed = 0
        suffix = " — Ctrl+C to cancel"
        if cancel_state["presses"] >= 1:
            label = f"cancelling… ({elapsed}s)"
        else:
            label = f"working… ({elapsed}s)"
        return FormattedText([("class:tb.working", f"  {label}{suffix}")])

    layout = Layout(HSplit([
        Window(content=FormattedTextControl(text=_working_line), height=1),
        Window(
            content=FormattedTextControl(text=make_status_renderer(status)),
            height=1,
            style="class:bottom-toolbar",
        ),
    ]))

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=status_bar_style(),
        full_screen=False,
        refresh_interval=0.5,
        mouse_support=False,
    )

    def _waiter() -> None:
        done.wait()
        try:
            loop = app.loop
            if loop is not None:
                loop.call_soon_threadsafe(lambda: app.exit())
        except Exception:
            pass

    threading.Thread(target=_waiter, daemon=True).start()

    try:
        with patch_stdout(raw=True):
            app.run()
    except KeyboardInterrupt:
        # Force-exit path: tell engine to cancel and wait for the
        # worker to wind down so we don't leak partial state.
        try:
            engine.cancel()
        except Exception:
            pass
        done.wait(timeout=10.0)
        raise

    if error_holder:
        raise error_holder[0]
    return result_holder[0] if result_holder else None
