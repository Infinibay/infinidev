"""Main entry point for Infinidev CLI."""

# Pre-load dotenv before crewai imports to avoid find_dotenv() stack frame assertion
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import sys
import os

# Auto-detect NVIDIA GPU libraries installed via pip (nvidia-cudnn-cu12, etc.)
# and add them to LD_LIBRARY_PATH so ONNX Runtime can use CUDAExecutionProvider.
# This runs before any onnxruntime import to ensure GPU is available.
def _setup_gpu_library_path():
    try:
        import importlib.util
        for pkg in ("nvidia.cudnn", "nvidia.cublas"):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                for loc in spec.submodule_search_locations:
                    lib_dir = os.path.join(loc, "lib")
                    if os.path.isdir(lib_dir):
                        current = os.environ.get("LD_LIBRARY_PATH", "")
                        if lib_dir not in current:
                            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir
    except Exception:
        pass

_setup_gpu_library_path()
import logging
import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from infinidev.config.settings import settings, get_base_dir
from infinidev.config.llm import get_litellm_params
import uuid
from infinidev.db.service import init_db, get_recent_summaries
from infinidev.agents.base import InfinidevAgent
from infinidev.engine.loop import LoopEngine
from infinidev.engine.analysis.review_engine import ReviewEngine
import infinidev.prompts.flows  # noqa: F401 — registers flows

# ── Logging setup ────────────────────────────────────────────────────────
#
# The default is SILENCE. Real users should never see log spam on stderr
# or find log files accumulating on disk they didn't ask for. Logging is
# opt-in through two env vars:
#
#   INFINIDEV_LOG_FILE=<path>   → write logs to this file (any non-empty value)
#   INFINIDEV_LOG_LEVEL=<lvl>   → DEBUG | INFO | WARNING | ERROR (default: WARNING)
#   INFINIDEV_LOG_STDERR=1      → also stream logs to stderr (off by default)
#
# Without any of these set, the root logger has a single NullHandler and
# nothing is emitted anywhere. Third-party libraries that log at INFO
# (httpx, litellm, openai, ...) are still clamped to ERROR as a safety
# net in case something slips through.
get_base_dir().mkdir(parents=True, exist_ok=True)

_log_level_name = os.environ.get("INFINIDEV_LOG_LEVEL", "").strip().upper()
_log_level = getattr(logging, _log_level_name, None) if _log_level_name else logging.WARNING
if not isinstance(_log_level, int):
    _log_level = logging.WARNING

_log_file_path = os.environ.get("INFINIDEV_LOG_FILE", "").strip()
_log_stderr = os.environ.get("INFINIDEV_LOG_STDERR", "").strip() not in ("", "0", "false", "False")

_handlers: list[logging.Handler] = []
if _log_file_path:
    try:
        _handlers.append(logging.FileHandler(_log_file_path))
    except Exception:
        # If the file can't be opened, fall through to NullHandler —
        # we must never crash a user's session because of logging.
        pass
if _log_stderr:
    _handlers.append(logging.StreamHandler(sys.stderr))

_root = logging.getLogger()
# Wipe any handlers an imported library may have attached during its
# own module-level side effects, then install ours (or a NullHandler).
_root.handlers.clear()
if _handlers:
    # Keep root at WARNING so DEBUG/INFO from stdlib (asyncio's
    # "Using selector: EpollSelector", urllib3 connection pool
    # chatter, etc.) doesn't flood the file when a user asks for
    # DEBUG. Apply the user's chosen level only to the `infinidev`
    # tree — that's the part they actually want to inspect.
    _root.setLevel(logging.WARNING)
    for _h in _handlers:
        _h.setFormatter(logging.Formatter("%(message)s"))
        _root.addHandler(_h)
    logging.getLogger("infinidev").setLevel(_log_level)
else:
    # Silent default — absorb every log record.
    _root.setLevel(logging.CRITICAL)
    _root.addHandler(logging.NullHandler())

# Clamp noisy third-party loggers to ERROR unconditionally, as a
# safety net even when the user does enable logging. If someone wants
# the full firehose they can override per-logger.
for _noisy in ("httpx", "httpcore", "litellm", "LiteLLM", "litellm.utils",
               "litellm.llms", "litellm.main", "litellm.cost_calculator",
               "litellm.litellm_core_utils", "litellm.router",
               "openai", "openai._base_client"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# Suppress litellm's own verbose/debug output regardless of log
# settings — its internal prints bypass the logging module.
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

# Slash-command handlers live in their own module so this file can
# focus on bootstrap + REPL loop. Re-exported for back-compat with any
# code that imports ``handle_command`` / ``handle_settings_command``
# directly from ``cli.main``.
from infinidev.cli.commands import handle_command, handle_settings_command  # noqa: E402

_IMPERATIVE_PREFIXES = (
    "create ", "add ", "fix ", "implement ", "write ", "build ",
    "refactor ", "rename ", "delete ", "remove ", "update ", "change ",
    "make ", "generate ", "convert ",
)


def _bootstrap_single_prompt_runtime() -> None:
    """Side effects shared by every classic entry point: DB init, ui/behavior
    hook registration, initial index, background indexer queue. Pulled out so
    both ``--prompt`` and the interactive ``_run_main`` setup do this exactly
    once and never drift apart."""
    init_db()
    from infinidev.engine.hooks.ui_hooks import register_ui_hooks
    register_ui_hooks()
    from infinidev.engine.behavior.hook import register_behavior_hooks
    register_behavior_hooks()

    from infinidev.cli.initial_index import run_initial_index
    run_initial_index(
        project_id=1,
        on_progress=lambda msg: click.echo(click.style(f"  {msg}", dim=True)),
    )

    from infinidev.cli.index_queue import IndexQueue
    from infinidev.code_intel.background_indexer import set_global_queue
    _q = IndexQueue(project_id=1)
    _q.start()
    set_global_queue(_q)

    # Warm Pydantic schema introspection so the first
    # ``LoopEngine._build_context()`` call doesn't pay ~500 ms of
    # tool-schema build time on the analysis → develop transition.
    # The schemas themselves are recomputed each time (the result is
    # not cached), but Pydantic's per-model introspection cache is
    # populated by this call, which is what actually dominates the
    # cost. Best-effort: any failure here just leaves the warm-up
    # for the first real call, no functional impact.
    try:
        from infinidev.tools import get_tools_for_role
        from infinidev.engine.loop.tools import build_tool_schemas
        _warm_tools = get_tools_for_role("developer", small_model=True)
        build_tool_schemas(_warm_tools, small_model=True)
    except Exception:
        pass

    # File watcher — catches every modification to the workspace,
    # including shell commands (``sed ... > file.ts``), external
    # editors, IDE saves, and ``git checkout``. Routes changes to the
    # background index queue; the indexer's file-integrity hook then
    # pushes a notification to the engine if the new content is broken.
    # This is the single-source-of-truth for "something changed on disk"
    # — direct file tool writes ALSO trigger the queue explicitly so
    # the 500 ms debounce of ``watchfiles`` doesn't leave the index
    # stale between a replace_lines call and the next get_symbol_code.
    # The dual trigger is safe because ``ensure_indexed`` short-circuits
    # on unchanged content hashes.
    try:
        from infinidev.cli.file_watcher import FileWatcher, WATCHFILES_AVAILABLE
        if WATCHFILES_AVAILABLE:
            import os as _os
            workspace = _os.getcwd()

            def _index_on_change(changed_path: str) -> None:
                try:
                    from infinidev.code_intel.background_indexer import enqueue_or_sync
                    enqueue_or_sync(1, changed_path)
                except Exception:
                    pass

            _watcher = FileWatcher(
                workspace=workspace,
                callback=lambda _p: None,  # no visual callback in classic
                index_callback=_index_on_change,
            )
            _watcher.start()
            # Stash on the IndexQueue so _run_main can stop it on exit.
            setattr(_q, "_file_watcher", _watcher)
    except Exception:
        # File watcher is best-effort. If watchfiles isn't installed
        # or the start fails, classic mode still works — it just
        # misses shell-bypass detection.
        pass


# ── Classic-mode play-by-play log bridge ────────────────────────────────
#
# In TUI mode the event bus drives the chat panel: every tool call,
# critic verdict, think block, etc. shows up live on screen. In classic
# / ``--no-tui`` mode (which the bench uses) there is NO subscriber, so
# all those events evaporate — the user only sees raw stderr from
# ``log()`` calls, which makes runs feel like they hang silently.
# The bridge below subscribes once and re-emits each event as a
# Python ``logging`` record, so anything routed to ``INFINIDEV_LOG_FILE``
# captures the full play-by-play. Registration is gated on the absence
# of any other subscriber to avoid double-logging when both the bridge
# and a TUI are active.

_EVENT_LOG = logging.getLogger("infinidev.events")


def _truncate(text: str | None, limit: int = 220) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _format_event(event_type: str, data: dict) -> str | None:
    """Render an EventBus event as a single human-readable log line.

    Returns ``None`` for events that should be silently dropped (e.g.
    ``loop_thinking_chunk`` which fires per-token and would flood the
    log). The other handlers favour brevity over completeness — the
    raw event still exists on the bus for any other consumer.
    """
    if event_type in ("loop_thinking_chunk", "loop_stream_status",
                      "loop_llm_call_start", "loop_state",
                      "loop_file_changed", "loop_behavior_update"):
        return None
    if event_type == "loop_start":
        return f"▶ start: {_truncate(data.get('prompt'), 120)}"
    if event_type == "loop_step_update":
        i = data.get("iteration", "?")
        title = _truncate(data.get("step_title"), 80)
        status = data.get("status", "")
        return f"── step {i} [{status}]: {title}"
    if event_type == "loop_think":
        return f"💭 think: {_truncate(data.get('reasoning'), 280)}"
    if event_type == "loop_tool_call":
        name = data.get("tool_name", "?")
        detail = _truncate(data.get("tool_detail"), 80)
        err = data.get("tool_error") or ""
        call = data.get("call_num", 0)
        tokens = data.get("tokens_total", 0)
        suffix = f"  ❌ {_truncate(err, 120)}" if err else ""
        return f"🔧 {name}({detail}) [#{call} · {tokens}tk]{suffix}"
    if event_type == "loop_user_message":
        return f"💬 → user: {_truncate(data.get('message'), 220)}"
    if event_type == "loop_assistant_message":
        action = data.get("action", "?")
        msg = _truncate(data.get("message"), 320)
        model = data.get("model", "")
        blocked = " [BLOCKING]" if data.get("blocked") else ""
        return f"🤝 critic ({model}) [{action}]{blocked}: {msg}"
    if event_type == "loop_log":
        lvl = data.get("level", "info").upper()
        return f"[{lvl}] {_truncate(data.get('message'), 320)}"
    if event_type in ("loop_finished", "loop_end"):
        summary = _truncate(data.get("summary") or data.get("reason"), 160)
        return f"✓ done: {summary}" if summary else "✓ done"
    return f"· {event_type}: {_truncate(str(data), 200)}"


def _install_classic_event_bridge() -> None:
    """Register a logger-only EventBus subscriber for ``INFINIDEV_LOG_FILE``.

    Previously this ran unconditionally at module import time and
    secondary handlers in ``ui_hooks.py`` keyed off
    ``event_bus.has_subscribers`` — which made the inline classic-mode
    output disappear the moment this bridge attached. We now gate the
    bridge on ``INFINIDEV_LOG_FILE`` so it only attaches when the user
    actually wants a file log; classic-mode terminal output is the
    job of :class:`infinidev.cli.classic_renderer.ClassicRenderer`,
    which is registered explicitly from ``_run_main``.
    """
    if not _log_file_path:
        return
    try:
        from infinidev.flows.event_listeners import event_bus
    except Exception:
        return

    def _bridge(event_type: str, project_id: int, agent_id: str, data: dict) -> None:
        try:
            line = _format_event(event_type, data)
            if line:
                _EVENT_LOG.info(line)
        except Exception:
            # Logging must never break the engine — swallow everything.
            pass

    try:
        event_bus.subscribe(_bridge)
    except Exception:
        pass


# Only attaches when INFINIDEV_LOG_FILE is set. Coexists with the
# classic renderer (both are independent subscribers).
_install_classic_event_bridge()


def _run_single_prompt(prompt_text: str, use_phase_engine: bool = False) -> None:
    """Run a single prompt non-interactively and exit.

    Thin adapter over :func:`engine.orchestration.run_task` /
    :func:`run_flow_task`. Uses :class:`NonInteractiveHooks` so the
    pipeline never blocks waiting for user input. Imperative tasks
    (action-verb prompts like "refactor X" or "create Y") skip the
    analyst, which historically wrapped them in an "analyze only" envelope
    and made the loop run in circles."""
    from infinidev.engine.orchestration import (
        run_task, run_flow_task, NonInteractiveHooks,
    )

    _bootstrap_single_prompt_runtime()

    agent = InfinidevAgent(agent_id="cli_agent")
    session_id = str(uuid.uuid4())
    hooks = NonInteractiveHooks()
    engine = LoopEngine()

    # /explore and /brainstorm prefixes bypass the full pipeline and run
    # the TreeEngine directly with no analysis or review.
    if prompt_text.startswith("/explore "):
        problem = prompt_text[len("/explore "):]
        result = run_flow_task(
            agent=agent, flow="explore",
            task_prompt=(problem, ""),  # expected_output picked up from flow_config
            session_id=session_id, engine=engine, hooks=hooks,
            use_tree_engine=True,
        )
        click.echo(result or "Done.")
        return
    if prompt_text.startswith("/brainstorm "):
        problem = prompt_text[len("/brainstorm "):]
        result = run_flow_task(
            agent=agent, flow="brainstorm",
            task_prompt=(problem, ""),
            session_id=session_id, engine=engine, hooks=hooks,
            use_tree_engine=True,
        )
        click.echo(result or "Done.")
        return

    # Every turn runs through chat agent → (maybe escalate) → planner →
    # developer. The chat agent itself detects action-verb requests and
    # escalates immediately, so the legacy imperative bypass is gone.
    result = run_task(
        agent=agent,
        user_input=prompt_text,
        session_id=session_id,
        engine=engine,
        reviewer=ReviewEngine(),
        hooks=hooks,
        use_phase_engine=use_phase_engine,
    )
    click.echo(result or "Done.")


@click.command()
@click.option("--no-tui", is_flag=True, help="Run in classic CLI mode instead of TUI.")
@click.option("--classic", is_flag=True, hidden=True, help="Alias for --no-tui.")
@click.option("--prompt", "-p", default=None, help="Run a single prompt non-interactively and exit.")
@click.option("--model", "-m", default=None, help="Override LLM model for this run (e.g., ollama_chat/qwen3:32b).")
@click.option("--provider", default=None, help="Override LLM provider (ollama, openai, anthropic, gemini, etc.).")
@click.option("--think", is_flag=True, help="Use phase engine (ANALYZE → PLAN → EXECUTE) for deeper reasoning.")
@click.option("--profile", is_flag=True, help="Enable session profiling (saves to ~/.infinidev/profiles/).")
def main(no_tui: bool, classic: bool, prompt: str | None, model: str | None, provider: str | None, think: bool, profile: bool):
    """Main entry point for Infinidev CLI."""
    from infinidev.cli.profiler import SessionProfiler

    # Apply provider override in-memory (does NOT persist to settings.json)
    if provider:
        settings.LLM_PROVIDER = provider

    # Apply model override in-memory (does NOT persist to settings.json)
    if model:
        if "/" not in model:
            # Use provider prefix if available, otherwise default to ollama_chat/
            from infinidev.config.providers import get_provider
            prov = get_provider(settings.LLM_PROVIDER)
            model = f"{prov.prefix}{model}"
        settings.LLM_MODEL = model

    # Reset capability cache so it re-detects for the new model/provider
    if model or provider:
        from infinidev.config.model_capabilities import _reset_capabilities
        _reset_capabilities()

    with SessionProfiler(enabled=profile) as profiler:
        _run_main(no_tui, classic, prompt, think, profile)

    if profile and profiler.report_path:
        click.echo(click.style(f"\nProfile saved to: {profiler.report_path}", fg="cyan"))

    # Bypass interpreter finalisation to dodge a gRPC SIGSEGV. See
    # _fast_exit_workaround() below for the full diagnosis. By the
    # time we reach this point, the session is durable on disk
    # (DB committed, settings flushed, logs written) so cutting
    # straight to os._exit is safe.
    _fast_exit_workaround()


def _run_main(no_tui: bool, classic: bool, prompt: str | None, think: bool, profile: bool):
    """Inner dispatch — runs inside the profiler context manager."""
    # Non-interactive --prompt mode
    if prompt:
        _run_single_prompt(prompt, use_phase_engine=think)
        return

    if not (no_tui or classic):
        # TUI mode. The stderr handler scrub that used to live here is
        # gone — stderr logging is now opt-in only (INFINIDEV_LOG_STDERR=1)
        # so the TUI is safe by default. If a user deliberately enabled
        # stderr logging AND then started the TUI, that's on them.
        if profile:
            click.echo(click.style("Profiling enabled — profile will be saved on exit.", fg="yellow"), err=True)
        from infinidev.ui.app import run_tui
        run_tui()
        return

    # Classic interactive mode — thin adapter around the unified pipeline.
    _bootstrap_single_prompt_runtime()

    from infinidev.engine.orchestration import (
        run_task, run_flow_task, ClickHooks,
    )
    from infinidev.db.service import store_conversation_turn
    from infinidev.cli.classic_renderer import (
        ClassicRenderer, SessionStatus, PermissionQueue,
        make_permission_handler, make_status_renderer, status_bar_style,
        render_status_table, hr, run_with_live_status,
    )
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.key_binding import KeyBindings

    # ── Session state + renderer ─────────────────────────────────────
    status = SessionStatus(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL.split("/", 1)[-1] if "/" in settings.LLM_MODEL else settings.LLM_MODEL,
        critic_enabled=bool(getattr(settings, "ASSISTANT_LLM_ENABLED", False)),
        critic_model=(
            (getattr(settings, "ASSISTANT_LLM_MODEL", "") or settings.LLM_MODEL).split("/", 1)[-1]
            if (getattr(settings, "ASSISTANT_LLM_MODEL", "") or settings.LLM_MODEL)
            else ""
        ),
    )
    renderer = ClassicRenderer(status)
    renderer.subscribe()

    # ── Banner ───────────────────────────────────────────────────────
    click.echo(click.style("Infinidev — Classic Mode", bold=True))
    click.echo(click.style(
        f"  model: {status.model}   provider: {status.provider}"
        + (f"   critic: {status.critic_model}" if status.critic_enabled else "   critic: off"),
        dim=True,
    ))
    click.echo(click.style("  /help · /status · Ctrl+C cancel · Ctrl+D quit", dim=True))
    click.echo(hr())

    # Final-result renderer: tries Rich Markdown, falls back to plain.
    def _render_final(text: str | None) -> None:
        if not text:
            click.echo("Done.")
            return
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            Console(file=sys.stdout, force_terminal=sys.stdout.isatty()).print(Markdown(text))
        except Exception:
            click.echo(text)

    # ── prompt_toolkit session with status bar ───────────────────────
    kb = KeyBindings()

    @kb.add("c-l")
    def _(event):
        # Clear screen; redraw prompt.
        click.clear()
        event.app.renderer.reset()

    @kb.add("escape", "enter")
    def _(event):
        # Alt+Enter inserts newline (multi-line mode opt-in).
        event.current_buffer.insert_text("\n")

    session = PromptSession(
        history=FileHistory(str(get_base_dir() / "history")),
        bottom_toolbar=make_status_renderer(status),
        style=status_bar_style(),
        refresh_interval=0.5,
        key_bindings=kb,
        multiline=False,
    )

    agent = InfinidevAgent(agent_id="cli_agent")
    session_id = str(uuid.uuid4())
    engine = LoopEngine()
    reviewer = ReviewEngine()
    hooks = ClickHooks(session=session)
    _gather_next_task = False
    _use_phase_engine = False

    # Inline permission handler — bridges engine worker thread → main.
    pq = PermissionQueue()
    from infinidev.tools.permission import set_permission_handler
    set_permission_handler(make_permission_handler(pq))

    def _drain_permission_requests() -> None:
        """Pop any pending permission requests and prompt the user."""
        while True:
            req = pq.pending()
            if req is None:
                return
            click.echo(click.style(
                f"\n⚠ Permission required: {req.description}",
                fg="yellow", bold=True,
            ))
            if req.details:
                click.echo(click.style(f"  {req.details}", fg="yellow"))
            try:
                ans = session.prompt("  Allow? [y/N] ").strip().lower()
                req.result = ans in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                req.result = False
            finally:
                req.done.set()

    # ── REPL loop ────────────────────────────────────────────────────
    while True:
        try:
            with patch_stdout(raw=True):
                user_input = session.prompt("infinidev ▸ ")
            if not user_input.strip():
                continue

            # /status: render the SessionStatus dump and continue.
            if user_input.strip() == "/status":
                click.echo(render_status_table(status))
                continue

            _fallthrough_to_pipeline = False
            if user_input.startswith("/"):
                cmd_result = handle_command(user_input)

                if isinstance(cmd_result, tuple) and cmd_result[0] == "prompt":
                    user_input = cmd_result[1]
                    _fallthrough_to_pipeline = True

                elif isinstance(cmd_result, tuple) and cmd_result[0] in ("explore", "brainstorm"):
                    flow_name, problem = cmd_result
                    click.echo(click.style(f"[{flow_name}] {problem}", dim=True))
                    result = run_with_live_status(
                        engine,
                        lambda: run_flow_task(
                            agent=agent, flow=flow_name,
                            task_prompt=(problem, ""),
                            session_id=session_id, engine=engine, hooks=hooks,
                            use_tree_engine=True,
                        ),
                        status,
                    )
                    _drain_permission_requests()
                    click.echo(click.style(f"\n{flow_name.title()} Result:", bold=True))
                    _render_final(result)
                    continue

                elif cmd_result == "init":
                    from infinidev.prompts.init_project import (
                        INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT,
                    )
                    click.echo(click.style("[init] Exploring and documenting project...", dim=True))
                    result = run_with_live_status(
                        engine,
                        lambda: run_flow_task(
                            agent=agent, flow="document",
                            task_prompt=(INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT),
                            session_id=session_id, engine=engine, hooks=hooks,
                        ),
                        status,
                    )
                    _drain_permission_requests()
                    click.echo(click.style("\nInit Result:", bold=True))
                    _render_final(result)
                    continue

                elif cmd_result == "think":
                    _gather_next_task = True
                    _use_phase_engine = True
                    click.echo(click.style(
                        "Phase mode: ANALYZE → PLAN → EXECUTE. Send your task.",
                        dim=True,
                    ))

                if not _fallthrough_to_pipeline:
                    continue

            # Reload settings between turns so /settings changes apply
            # on the next task. The pipeline itself does NOT reload.
            from infinidev.config.settings import reload_all
            reload_all()
            # Refresh status with potentially-new model/provider/critic.
            status.provider = settings.LLM_PROVIDER
            status.model = (
                settings.LLM_MODEL.split("/", 1)[-1]
                if "/" in settings.LLM_MODEL else settings.LLM_MODEL
            )
            status.critic_enabled = bool(getattr(settings, "ASSISTANT_LLM_ENABLED", False))
            critic_full = getattr(settings, "ASSISTANT_LLM_MODEL", "") or settings.LLM_MODEL
            status.critic_model = critic_full.split("/", 1)[-1] if "/" in critic_full else critic_full

            store_conversation_turn(session_id, "user", user_input)

            result = run_with_live_status(
                engine,
                lambda: run_task(
                    agent=agent,
                    user_input=user_input,
                    session_id=session_id,
                    engine=engine,
                    reviewer=reviewer,
                    hooks=hooks,
                    use_phase_engine=_use_phase_engine,
                    force_gather=_gather_next_task,
                ),
                status,
            )
            _drain_permission_requests()
            _gather_next_task = False
            _use_phase_engine = False

            store_conversation_turn(
                session_id, "assistant",
                result or "",
                (result or "")[:200],
            )

            if result:
                click.echo(click.style("\nFinal Result:", bold=True))
                _render_final(result)

        except KeyboardInterrupt:
            # Force-interrupt path: a second Ctrl+C inside the live-status
            # app surfaces here. Drop the pending turn and prompt again.
            click.echo(click.style("\n[interrupted]", dim=True))
            continue
        except EOFError:
            break
        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg="red"))
            logging.exception("Error in main loop")

    # Graceful renderer detach on exit.
    try:
        renderer.unsubscribe()
    except Exception:
        pass

    # Cleanup background services — the IndexQueue was started inside
    # _bootstrap_single_prompt_runtime() and registered globally; fetch
    # it through the public accessor and stop it on exit.
    from infinidev.code_intel.background_indexer import get_global_queue
    _q = get_global_queue()
    if _q is not None:
        try:
            _q.stop()
        except Exception:
            pass

def _fast_exit_workaround() -> None:
    """Bypass Python interpreter finalisation to dodge a gRPC SIGSEGV.

    Diagnosed 2026-04-07: every CLI run that touched a real workspace
    crashed with SIGSEGV during shutdown (exit code 139 / -11). The
    fault handler dump pointed to a non-Python thread (no Python frame)
    while the loaded extension list included ``grpc._cython.cygrpc``.

    Root cause: ``opentelemetry.exporter.otlp.proto.grpc`` is pulled
    in transitively (via litellm/chromadb/etc), and importing that
    module spawns a background C++ thread inside grpc to keep its
    channels alive. When Python's interpreter starts ``Py_Finalize``
    on a clean exit, that thread is still running and races with
    the freeing of Python objects it referenced — classic
    grpc/grpc#28632 territory.

    Setting ``OTEL_*`` env vars does NOT fix it because the thread
    is created at IMPORT time, not at first export. The standard
    community workaround is to skip ``Py_Finalize`` entirely by
    calling ``os._exit(0)`` after our own cleanup runs. Atexit
    handlers and __del__ methods are NOT executed by ``os._exit``,
    so we have to make sure the CLI's own teardown (saving session
    state, flushing logs, closing DB connections) is already
    complete before this fires. The CLI does that synchronously
    inside ``main()`` already, so by the time we reach this point
    everything we care about is durable on disk.
    """
    import os as _os
    import sys as _sys
    # Belt-and-suspenders: ensure the background IndexQueue worker is
    # joined before we bypass finalisation. The normal path in _run_main
    # already calls stop(), but alternate exit paths (errors, TUI crash)
    # may land here without having done so — and IndexQueue.stop() is
    # now idempotent, so a second call is cheap and safe.
    try:
        from infinidev.code_intel.background_indexer import get_global_queue
        _q = get_global_queue()
        if _q is not None:
            _q.stop()
    except Exception:
        pass
    try:
        _sys.stdout.flush()
        _sys.stderr.flush()
    except Exception:
        pass
    _os._exit(0)


if __name__ == "__main__":
    main()
