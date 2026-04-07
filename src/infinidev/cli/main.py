"""Main entry point for Infinidev CLI."""

# Pre-load dotenv before crewai imports to avoid find_dotenv() stack frame assertion
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import sys
import os
import logging
import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from infinidev.config.settings import settings, DEFAULT_BASE_DIR
from infinidev.config.llm import get_litellm_params
import uuid
from infinidev.db.service import init_db, get_recent_summaries
from infinidev.agents.base import InfinidevAgent
from infinidev.engine.loop import LoopEngine
from infinidev.engine.analysis.analysis_engine import AnalysisEngine
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
DEFAULT_BASE_DIR.mkdir(parents=True, exist_ok=True)

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
    _root.setLevel(_log_level)
    for _h in _handlers:
        _h.setFormatter(logging.Formatter("%(message)s"))
        _root.addHandler(_h)
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

def handle_command(cmd_text: str):
    """Handle / commands."""
    parts = cmd_text.split()
    cmd = parts[0].lower()
    
    if cmd == "/models":
        subcmd = parts[1].lower() if len(parts) > 1 else "info"
        
        if subcmd == "set" and len(parts) > 2:
            new_model = parts[2]
            # Ensure it has a provider prefix. If no '/' is found, assume ollama_chat/
            if "/" not in new_model:
                new_model = f"ollama_chat/{new_model}"
            
            settings.save_user_settings({"LLM_MODEL": new_model})
            # Reload settings globally for all modules (llm.py, engine, etc.)
            from infinidev.config.settings import reload_all
            reload_all()
            # Re-detect capabilities for the new model
            from infinidev.config.model_capabilities import _reset_capabilities
            _reset_capabilities()

            click.echo(click.style(f"Model updated to: {settings.LLM_MODEL}", fg="green"))
        elif subcmd == "list":
            import httpx
            click.echo(click.style("Fetching models from Ollama...", dim=True))
            try:
                # Use base URL from settings
                base_url = settings.LLM_BASE_URL.rstrip("/")
                resp = httpx.get(f"{base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("models", [])
                    if not models:
                        click.echo("No models found in Ollama.")
                    else:
                        click.echo(click.style("\nAvailable Ollama models:", bold=True))
                        for m in models:
                            name = m.get("name")
                            size_gb = m.get("size", 0) / (1024**3)
                            # Highlight the current model
                            current_tag = settings.LLM_MODEL.split("/", 1)[-1]
                            if name == current_tag:
                                click.echo(f"  * {click.style(name, fg='green')} ({size_gb:.1f} GB)")
                            else:
                                click.echo(f"  - {name} ({size_gb:.1f} GB)")
                else:
                    click.echo(click.style(f"Error fetching models: {resp.status_code}", fg="red"))
            except Exception as e:
                click.echo(click.style(f"Could not connect to Ollama: {e}", fg="red"))
        else:
            click.echo(click.style("Model configuration:", bold=True))
            click.echo(f"  Current model: {settings.LLM_MODEL}")
            click.echo(f"  Base URL:      {settings.LLM_BASE_URL}")
            click.echo("\nCommands:")
            click.echo("  /models list       - List available Ollama models")
            click.echo("  /models set <name> - Change current model")
        return True
    elif cmd in ("/exit", "/quit"):
        click.echo("Goodbye!")
        sys.exit(0)
    elif cmd == "/reindex":
        # Reindex the current workspace's code intelligence index.
        # `--full` (or `-f`) clears the existing index for this project
        # first, forcing every file to be re-parsed from scratch. Useful
        # after upgrading the indexer / fixing a parser bug — incremental
        # indexing skips files whose content hash hasn't changed, so
        # those files would never benefit from the new logic without a
        # full clear.
        full = any(a in ("--full", "-f") for a in parts[1:])
        from infinidev.code_intel.indexer import index_directory
        from infinidev.tools.base.context import get_current_workspace_path
        from infinidev.tools.base.db import execute_with_retry
        workdir = get_current_workspace_path() or os.getcwd()
        if full:
            click.echo(click.style(f"Clearing existing index for project 1...", dim=True))
            def _clear(conn):
                conn.execute("DELETE FROM ci_files WHERE project_id = 1")
                conn.execute("DELETE FROM ci_symbols WHERE project_id = 1")
                conn.execute("DELETE FROM ci_references WHERE project_id = 1")
                conn.execute("DELETE FROM ci_imports WHERE project_id = 1")
                conn.execute("DELETE FROM ci_method_bodies WHERE project_id = 1")
                conn.commit()
            try:
                execute_with_retry(_clear)
            except Exception as exc:
                click.echo(click.style(f"Clear failed: {exc}", fg="red"))
                return True
        click.echo(click.style(f"Indexing {workdir} ...", dim=True))
        try:
            stats = index_directory(1, workdir)
            click.echo(click.style(
                f"Done: {stats['files_indexed']} files indexed, "
                f"{stats['symbols_total']} symbols, "
                f"{stats['files_skipped']} skipped, "
                f"{stats['elapsed_ms']}ms",
                fg="green",
            ))
        except Exception as exc:
            click.echo(click.style(f"Index failed: {exc}", fg="red"))
        return True
    elif cmd == "/help":
        click.echo(click.style("Available commands:", bold=True))
        click.echo("  /models            - Show current model configuration")
        click.echo("  /models set <name> - Change Ollama model (e.g., /models set llama3)")
        click.echo("  /settings          - Show current settings")
        click.echo("  /settings <key>    - Show specific setting")
        click.echo("  /settings <key> <val> - Change setting")
        click.echo("  /settings reset    - Reset to defaults")
        click.echo("  /settings export   - Export settings to file")
        click.echo("  /settings import   - Import settings from file")
        click.echo("  /reindex [--full]  - Re-index the workspace (--full clears DB first)")
        click.echo("  /think             - Enable deep analysis for the next task")
        click.echo("  /explore <problem> - Decompose and explore a complex problem")
        click.echo("  /brainstorm <problem> - Creative ideation with forced perspectives")
        click.echo("  /refactor [scope]  - Refactor code (modularize, clean, restructure)")
        click.echo("  /init              - Explore and document the current project")
        click.echo("  /exit, /quit       - Exit the CLI")
        click.echo("  /help              - Show this help")
        return True
    
    elif cmd == "/settings":
        handle_settings_command(parts)
        return True

    elif cmd == "/think":
        return "think"

    elif cmd == "/init":
        return "init"  # Signal to main loop to run init

    elif cmd == "/explore":
        problem = " ".join(parts[1:]) if len(parts) > 1 else ""
        if not problem:
            click.echo("Usage: /explore <problem description>")
            return True
        return ("explore", problem)

    elif cmd == "/brainstorm":
        problem = " ".join(parts[1:]) if len(parts) > 1 else ""
        if not problem:
            click.echo("Usage: /brainstorm <problem description>")
            return True
        return ("brainstorm", problem)

    elif cmd == "/refactor":
        user_scope = " ".join(parts[1:]) if len(parts) > 1 else ""
        from infinidev.ui.handlers.commands import _build_refactor_prompt
        prompt = _build_refactor_prompt(user_scope)
        label = f"[refactor] {user_scope}" if user_scope else "[refactor] auto-scoped"
        click.echo(click.style(label, fg="cyan"))
        return ("prompt", prompt)

    click.echo(f"Unknown command: {cmd}")
    return True


def handle_settings_command(parts: list[str]):
    """Handle /settings command in classic CLI mode."""
    from infinidev.config.settings import settings, SETTINGS_FILE, reload_all
    import shutil
    from pathlib import Path

    subcmd = parts[1].lower() if len(parts) > 1 else "info"

    if subcmd == "reset":
        if SETTINGS_FILE.exists():
            SETTINGS_FILE.unlink()
        reload_all()
        click.echo(click.style("Settings reset to defaults. Reloaded.", fg="green"))
    elif subcmd == "export" and len(parts) > 2:
        export_path = parts[2]
        try:
            shutil.copy(SETTINGS_FILE, export_path)
            click.echo(click.style(f"Settings exported to: {export_path}", fg="green"))
        except Exception as e:
            click.echo(click.style(f"Export failed: {e}", fg="red"))
    elif subcmd == "import" and len(parts) > 2:
        import_path = parts[2]
        try:
            settings_file_path = SETTINGS_FILE if isinstance(SETTINGS_FILE, Path) else Path(SETTINGS_FILE)
            if not settings_file_path.exists():
                settings_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(import_path, settings_file_path)
            reload_all()
            click.echo(click.style(f"Settings imported from: {import_path}. Reloaded.", fg="green"))
        except FileNotFoundError:
            click.echo(click.style(f"Import failed: File not found: {import_path}", fg="red"))
        except Exception as e:
            click.echo(click.style(f"Import failed: {e}", fg="red"))
    elif subcmd == "info" or subcmd == "" or len(parts) == 1:
        # Show all settings in formatted table
        click.echo(click.style("Infinidev Settings", bold=True))
        click.echo(click.style(f"(from {SETTINGS_FILE})", dim=True))
        click.echo("")
        click.echo(click.style("LLM", bold=True))
        click.echo(f"  {settings.LLM_MODEL:<50} (LLM_MODEL)")
        click.echo(f"  {settings.LLM_BASE_URL:<50} (LLM_BASE_URL)")
        click.echo("")
        click.echo(click.style("Loop Engine", bold=True))
        click.echo(f"  {settings.LOOP_MAX_ITERATIONS:<50} (LOOP_MAX_ITERATIONS)")
        click.echo(f"  {settings.LOOP_MAX_TOTAL_TOOL_CALLS:<50} (LOOP_MAX_TOTAL_TOOL_CALLS)")
        click.echo("")
        click.echo(click.style("Code Interpreter", bold=True))
        click.echo(f"  {settings.CODE_INTERPRETER_TIMEOUT:<50} (CODE_INTERPRETER_TIMEOUT)")
        click.echo("")
        click.echo(click.style("Phases", bold=True))
        click.echo(f"  {str(settings.ANALYSIS_ENABLED):<50} (ANALYSIS_ENABLED)")
        click.echo(f"  {str(settings.REVIEW_ENABLED):<50} (REVIEW_ENABLED)")
        click.echo("")
        click.echo(click.style("UI", bold=True))
        log_level = settings.model_dump().get("LOG_LEVEL", "warning")
        click.echo(f"  {log_level:<50} (LOG_LEVEL)")
    else:
        # Show or set specific setting
        setting_key = subcmd.upper()
        value_to_set = parts[2] if len(parts) > 2 else None

        type_map = {
            "LLM_MODEL": str,
            "LLM_BASE_URL": str,
            "DB_PATH": str,
            "WORKSPACE_BASE_DIR": str,
            "EMBEDDING_PROVIDER": str,
            "EMBEDDING_MODEL": str,
            "EMBEDDING_BASE_URL": str,
            "LOOP_MAX_ITERATIONS": int,
            "LOOP_MAX_TOOL_CALLS_PER_ACTION": int,
            "LOOP_MAX_TOTAL_TOOL_CALLS": int,
            "LOOP_HISTORY_WINDOW": int,
            "MAX_RETRIES": int,
            "RETRY_BASE_DELAY": float,
            "COMMAND_TIMEOUT": int,
            "WEB_TIMEOUT": int,
            "GIT_PUSH_TIMEOUT": int,
            "MAX_FILE_SIZE_BYTES": int,
            "MAX_DIR_LISTING": int,
            "WEB_CACHE_TTL_SECONDS": int,
            "WEB_RPM_LIMIT": int,
            "WEB_ROBOTS_CACHE_TTL": int,
            "DEDUP_SIMILARITY_THRESHOLD": float,
            "CODE_INTERPRETER_TIMEOUT": int,
            "CODE_INTERPRETER_MAX_OUTPUT": int,
            "SANDBOX_ENABLED": bool,
            "ANALYSIS_ENABLED": bool,
            "REVIEW_ENABLED": bool,
            "EXECUTE_COMMANDS_PERMISSION": str,
            "ALLOWED_COMMANDS_LIST": list,
            "FILE_OPERATIONS_PERMISSION": str,
            "ALLOWED_FILE_PATHS": list,
        }

        if value_to_set:
            # Convert to appropriate type
            type_class = type_map.get(setting_key, str)
            try:
                if type_class == bool:
                    converted = value_to_set.lower() in ("true", "1", "yes")
                elif type_class == list:
                    converted = [item.strip() for item in value_to_set.split(",") if item.strip()]
                else:
                    converted = type_class(value_to_set)
                settings.save_user_settings({setting_key: converted})
                reload_all()
                click.echo(click.style(f"Updated {setting_key} to: {converted}", fg="green"))
            except ValueError as e:
                click.echo(click.style(f"Error: {e}", fg="red"))
        else:
            # Show current value
            value = getattr(settings, setting_key, None)
            if value is not None:
                click.echo(f"{setting_key}: {value}")
            else:
                click.echo(click.style(f"Unknown setting: {setting_key}", fg="yellow"))


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

    # Develop flow — full pipeline. Imperative bypass: skip the analyst
    # for obvious action-verb tasks (see _IMPERATIVE_PREFIXES). The
    # gather phase still runs if globally enabled.
    _imperative = prompt_text.lstrip().lower().startswith(_IMPERATIVE_PREFIXES)
    result = run_task(
        agent=agent,
        user_input=prompt_text,
        session_id=session_id,
        engine=engine,
        analyst=AnalysisEngine(),
        reviewer=ReviewEngine(),
        hooks=hooks,
        use_phase_engine=use_phase_engine,
        skip_analysis=_imperative,
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

    click.echo(click.style("Welcome to Infinidev CLI (Classic Mode)!", fg="cyan", bold=True))
    click.echo("Type your instructions or /help for commands.")

    session = PromptSession(history=FileHistory(str(DEFAULT_BASE_DIR / "history")))

    agent = InfinidevAgent(agent_id="cli_agent")
    session_id = str(uuid.uuid4())
    engine = LoopEngine()
    analyst = AnalysisEngine()
    reviewer = ReviewEngine()
    hooks = ClickHooks(session=session)
    _gather_next_task = False
    _use_phase_engine = False

    # Permission handler — kept inline because click.confirm() needs the
    # caller's stdin. Could be moved to ClickHooks later if more permission
    # surfaces appear, but one callsite doesn't earn an abstraction yet.
    def _classic_permission_handler(tool_name: str, description: str, details: str) -> bool:
        click.echo(click.style(f"\n⚠ Permission required: {description}", fg="yellow", bold=True))
        click.echo(click.style(f"  {details}", fg="yellow"))
        return click.confirm("  Allow?", default=False)

    from infinidev.tools.permission import set_permission_handler
    set_permission_handler(_classic_permission_handler)

    while True:
        try:
            user_input = session.prompt("infinidev> ")
            if not user_input.strip():
                continue

            _fallthrough_to_pipeline = False
            if user_input.startswith("/"):
                cmd_result = handle_command(user_input)

                if isinstance(cmd_result, tuple) and cmd_result[0] == "prompt":
                    # /refactor and friends — rewrite user_input and fall through.
                    user_input = cmd_result[1]
                    _fallthrough_to_pipeline = True

                elif isinstance(cmd_result, tuple) and cmd_result[0] in ("explore", "brainstorm"):
                    flow_name, problem = cmd_result
                    click.echo(click.style(f"[{flow_name}] {problem}", fg="yellow"))
                    result = run_flow_task(
                        agent=agent, flow=flow_name,
                        task_prompt=(problem, ""),
                        session_id=session_id, engine=engine, hooks=hooks,
                        use_tree_engine=True,
                    )
                    click.echo(click.style(f"\n{flow_name.title()} Result:", fg="green", bold=True))
                    click.echo(result)
                    continue

                elif cmd_result == "init":
                    from infinidev.prompts.init_project import (
                        INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT,
                    )
                    click.echo(click.style("[init] Exploring and documenting project...", fg="yellow"))
                    result = run_flow_task(
                        agent=agent, flow="document",
                        task_prompt=(INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT),
                        session_id=session_id, engine=engine, hooks=hooks,
                    )
                    click.echo(click.style("\nInit Result:", fg="green", bold=True))
                    click.echo(result)
                    continue

                elif cmd_result == "think":
                    _gather_next_task = True
                    _use_phase_engine = True
                    click.echo(click.style(
                        "Phase mode enabled: ANALYZE → PLAN → EXECUTE. Send your task.",
                        fg="cyan",
                    ))

                if not _fallthrough_to_pipeline:
                    continue

            # Reload settings between turns so /settings changes apply
            # on the next task. The pipeline itself does NOT reload.
            from infinidev.config.settings import reload_all
            reload_all()

            store_conversation_turn(session_id, "user", user_input)

            result = run_task(
                agent=agent,
                user_input=user_input,
                session_id=session_id,
                engine=engine,
                analyst=analyst,
                reviewer=reviewer,
                hooks=hooks,
                use_phase_engine=_use_phase_engine,
                force_gather=_gather_next_task,
            )
            _gather_next_task = False
            _use_phase_engine = False

            store_conversation_turn(
                session_id, "assistant",
                result or "",
                (result or "")[:200],
            )

            if result:
                click.echo(click.style("\nFinal Result:", fg="green", bold=True))
                click.echo(result)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg="red"))
            logging.exception("Error in main loop")

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

if __name__ == "__main__":
    main()
