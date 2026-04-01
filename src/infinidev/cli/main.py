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
from infinidev.engine.loop_engine import LoopEngine
from infinidev.engine.analysis_engine import AnalysisEngine
from infinidev.engine.review_engine import ReviewEngine
# InfinidevTUI (Textual) replaced by infinidev.ui.app.run_tui (prompt_toolkit)
import infinidev.prompts.flows  # noqa: F401 — registers flows

# Configure logging (ensure base dir exists before creating file handler)
DEFAULT_BASE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(DEFAULT_BASE_DIR / "infinidev.log"), logging.StreamHandler(sys.stderr)]
)
# Silence noisy third-party loggers
for _noisy in ("httpx", "httpcore", "litellm", "LiteLLM", "litellm.utils",
               "litellm.llms", "litellm.main", "litellm.cost_calculator",
               "litellm.litellm_core_utils", "litellm.router",
               "openai", "openai._base_client"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# Suppress litellm's own verbose/debug output
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
        click.echo("  /think             - Enable deep analysis for the next task")
        click.echo("  /explore <problem> - Decompose and explore a complex problem")
        click.echo("  /brainstorm <problem> - Creative ideation with forced perspectives")
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


def _run_single_prompt(prompt_text: str, use_phase_engine: bool = False) -> None:
    """Run a single prompt non-interactively and exit.

    Supports /explore and /brainstorm prefixes, otherwise runs as develop flow.
    """
    init_db()
    from infinidev.engine.ui_hooks import register_ui_hooks
    register_ui_hooks()

    # Index workspace before LLM starts so code intelligence is available
    from infinidev.cli.initial_index import run_initial_index
    run_initial_index(
        project_id=1,
        on_progress=lambda msg: click.echo(click.style(f"  {msg}", dim=True)),
    )

    agent = InfinidevAgent(agent_id="cli_agent")
    session_id = str(uuid.uuid4())

    # Determine mode from prompt prefix
    if prompt_text.startswith("/explore "):
        mode = "explore"
        problem = prompt_text[len("/explore "):]
    elif prompt_text.startswith("/brainstorm "):
        mode = "brainstorm"
        problem = prompt_text[len("/brainstorm "):]
    else:
        mode = "develop"
        problem = prompt_text

    from infinidev.engine.flows import get_flow_config
    from infinidev.engine.tree_engine import TreeEngine

    if mode in ("explore", "brainstorm"):
        flow_config = get_flow_config(mode)
        agent._system_prompt_identity = flow_config.identity_prompt
        agent.backstory = flow_config.backstory
        agent.activate_context(session_id=session_id)
        try:
            tree_engine = TreeEngine()
            result = tree_engine.execute(
                agent=agent,
                task_prompt=(problem, flow_config.expected_output),
                mode=mode,
            )
        finally:
            agent.deactivate()
    else:
        # Analysis phase for single-prompt mode (non-interactive)
        detected_flow = "develop"
        task_type = "feature"
        analysis_prompt = None
        if settings.ANALYSIS_ENABLED:
            try:
                from infinidev.engine.analysis_engine import AnalysisEngine
                analyst_sp = AnalysisEngine()
                click.echo(click.style("Analyzing request...", fg="cyan", dim=True))
                analysis = analyst_sp.analyze(problem)
                if analysis.flow and analysis.flow != "done":
                    detected_flow = analysis.flow
                if hasattr(analysis, 'specification'):
                    task_type = analysis.specification.get("task_type", "feature")
                analysis_prompt = analysis.build_flow_prompt()
                click.echo(click.style(f"  Flow: {detected_flow}, Type: {task_type}", fg="cyan", dim=True))
            except Exception as exc:
                click.echo(click.style(f"  Analysis failed: {exc}", fg="yellow", dim=True))

        flow_config = get_flow_config(detected_flow)
        agent._system_prompt_identity = flow_config.identity_prompt
        agent.backstory = flow_config.backstory
        agent.activate_context(session_id=session_id)

        # Use analysis-enhanced prompt if available
        if analysis_prompt:
            task_prompt_sp = analysis_prompt
        else:
            task_prompt_sp = (problem, flow_config.expected_output)

        # Gather phase for single-prompt mode
        if settings.GATHER_ENABLED:
            try:
                from infinidev.gather import run_gather
                click.echo(click.style("Gathering context...", fg="cyan", dim=True))
                brief = run_gather(problem, [], None, agent)
                desc, expected = task_prompt_sp
                task_prompt_sp = (brief.render() + "\n\n" + desc, expected)
                click.echo(click.style(f"  {brief.summary()}", fg="cyan", dim=True))
            except Exception as exc:
                click.echo(click.style(f"  Gather failed: {exc}", fg="yellow", dim=True))

        try:
            if use_phase_engine:
                from infinidev.engine.phase_engine import PhaseEngine
                phase_eng = PhaseEngine()
                result = phase_eng.execute(
                    agent=agent,
                    task_prompt=task_prompt_sp,
                    task_type=task_type,
                    verbose=True,
                )
                engine = phase_eng  # for has_file_changes() check below
            else:
                engine = LoopEngine()
                result = engine.execute(
                    agent=agent,
                    task_prompt=task_prompt_sp,
                    verbose=True,
                )
        finally:
            agent.deactivate()

        # Code review phase for single-prompt mode (with rework loop)
        if settings.REVIEW_ENABLED and engine.has_file_changes():
            click.echo(click.style("\nRunning code review...", fg="magenta", dim=True))
            from infinidev.engine.review_engine import run_review_rework_loop

            def _sp_review_status(level: str, msg: str) -> None:
                if level == "verification_pass":
                    click.echo(click.style(f"Verification: PASS. {msg}", fg="green", dim=True))
                elif level == "verification_fail":
                    click.echo(click.style(f"Verification: FAIL. {msg}", fg="red"))
                    click.echo(click.style("Re-running developer to fix test failures...", fg="magenta", dim=True))
                elif level == "approved":
                    click.echo(click.style(f"Code review: APPROVED. {msg}", fg="green", dim=True))
                elif level == "rejected":
                    click.echo(click.style(msg, fg="red"))
                    click.echo(click.style("Re-running developer to fix review issues...", fg="magenta", dim=True))
                elif level == "max_reviews":
                    click.echo(click.style("Max review rounds reached — stopping.", fg="yellow", dim=True))

            result, _ = run_review_rework_loop(
                engine=engine,
                agent=agent,
                session_id=session_id,
                task_prompt=task_prompt_sp,
                initial_result=result or "",
                reviewer=ReviewEngine(),
                recent_messages=get_recent_summaries(session_id, limit=5),
                on_status=_sp_review_status,
            )

    click.echo(result or "Done.")


@click.command()
@click.option("--no-tui", is_flag=True, help="Run in classic CLI mode instead of TUI.")
@click.option("--classic", is_flag=True, hidden=True, help="Alias for --no-tui.")
@click.option("--prompt", "-p", default=None, help="Run a single prompt non-interactively and exit.")
@click.option("--model", "-m", default=None, help="Override LLM model for this run (e.g., ollama_chat/qwen3:32b).")
@click.option("--think", is_flag=True, help="Use phase engine (ANALYZE → PLAN → EXECUTE) for deeper reasoning.")
@click.option("--profile", is_flag=True, help="Enable session profiling (saves to ~/.infinidev/profiles/).")
def main(no_tui: bool, classic: bool, prompt: str | None, model: str | None, think: bool, profile: bool):
    """Main entry point for Infinidev CLI."""
    from infinidev.cli.profiler import SessionProfiler

    # Apply model override in-memory (does NOT persist to settings.json)
    if model:
        if "/" not in model:
            model = f"ollama_chat/{model}"
        settings.LLM_MODEL = model

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
        # TUI mode: remove the stderr handler so log output doesn't corrupt
        # the terminal.  File logging is preserved.
        root = logging.getLogger()
        root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)
                         or getattr(h, 'stream', None) is not sys.stderr]
        if profile:
            click.echo(click.style("Profiling enabled — profile will be saved on exit.", fg="yellow"), err=True)
        from infinidev.ui.app import run_tui
        run_tui()
        return

    # Classic mode (the original while True loop)
    init_db()

    # Index workspace before LLM starts so code intelligence is available
    from infinidev.cli.initial_index import run_initial_index
    run_initial_index(
        project_id=1,
        on_progress=lambda msg: click.echo(click.style(f"  {msg}", dim=True)),
    )

    # Start background file watcher + indexing queue
    from infinidev.cli.index_queue import IndexQueue
    from infinidev.cli.file_watcher import FileWatcher as _FileWatcher
    _index_queue = IndexQueue(project_id=1)
    _index_queue.start()
    _classic_watcher = _FileWatcher(
        workspace=os.getcwd(),
        callback=lambda p: None,  # no visual callback in classic mode
        index_callback=_index_queue.enqueue,
    )
    _classic_watcher.start()

    from infinidev.engine.ui_hooks import register_ui_hooks
    register_ui_hooks()

    click.echo(click.style("Welcome to Infinidev CLI (Classic Mode)!", fg="cyan", bold=True))
    click.echo("Type your instructions or /help for commands.")
    
    session = PromptSession(history=FileHistory(str(DEFAULT_BASE_DIR / "history")))
    
    # Create the agent
    agent = InfinidevAgent(agent_id="cli_agent")
    session_id = str(uuid.uuid4())
    engine = LoopEngine()
    analyst = AnalysisEngine()
    reviewer = ReviewEngine()
    _gather_next_task = False
    _use_phase_engine = False

    # Register permission handler for classic CLI
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

            if user_input.startswith("/"):
                cmd_result = handle_command(user_input)
                if isinstance(cmd_result, tuple) and cmd_result[0] == "explore":
                    # /explore command — run exploration tree engine
                    problem = cmd_result[1]
                    from infinidev.config.settings import reload_all
                    reload_all()

                    from infinidev.engine.tree_engine import TreeEngine
                    from infinidev.engine.flows import get_flow_config

                    click.echo(click.style(f"[explore] Exploring: {problem}", fg="yellow"))
                    flow_config = get_flow_config("explore")
                    agent._system_prompt_identity = flow_config.identity_prompt
                    agent.backstory = flow_config.backstory
                    agent.activate_context(session_id=session_id)
                    try:
                        tree_engine = TreeEngine()
                        result = tree_engine.execute(
                            agent=agent,
                            task_prompt=(problem, flow_config.expected_output),
                            verbose=True,
                        )
                        if not result or not result.strip():
                            result = "Exploration complete (no synthesis produced)."
                    finally:
                        agent.deactivate()
                    click.echo(click.style("\nExploration Result:", fg="green", bold=True))
                    click.echo(result)
                    continue
                elif isinstance(cmd_result, tuple) and cmd_result[0] == "brainstorm":
                    # /brainstorm command — run brainstorm tree engine
                    problem = cmd_result[1]
                    from infinidev.config.settings import reload_all
                    reload_all()

                    from infinidev.engine.tree_engine import TreeEngine
                    from infinidev.engine.flows import get_flow_config

                    click.echo(click.style(f"[brainstorm] Brainstorming: {problem}", fg="magenta"))
                    flow_config = get_flow_config("brainstorm")
                    agent._system_prompt_identity = flow_config.identity_prompt
                    agent.backstory = flow_config.backstory
                    agent.activate_context(session_id=session_id)
                    try:
                        tree_engine = TreeEngine()
                        result = tree_engine.execute(
                            agent=agent,
                            task_prompt=(problem, flow_config.expected_output),
                            mode="brainstorm",
                        )
                        if not result or not result.strip():
                            result = "Brainstorm complete (no synthesis produced)."
                    finally:
                        agent.deactivate()
                    click.echo(click.style("\nBrainstorm Result:", fg="magenta", bold=True))
                    click.echo(result)
                    continue
                elif cmd_result == "init":
                    # /init command — run project exploration
                    from infinidev.prompts.init_project import INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT
                    from infinidev.engine.flows import get_flow_config

                    from infinidev.config.settings import reload_all
                    reload_all()

                    click.echo(click.style("[init] Exploring and documenting project...", fg="yellow"))
                    flow_config = get_flow_config("document")
                    agent._system_prompt_identity = flow_config.identity_prompt
                    agent.backstory = flow_config.backstory
                    agent.activate_context(session_id=session_id)
                    try:
                        result = engine.execute(
                            agent=agent,
                            task_prompt=(INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT),
                            verbose=True,
                        )
                        if not result or not result.strip():
                            result = "Project initialization complete."
                    finally:
                        agent.deactivate()
                    click.echo(click.style("\nInit Result:", fg="green", bold=True))
                    click.echo(result)
                elif cmd_result == "think":
                    _gather_next_task = True
                    _use_phase_engine = True
                    click.echo(click.style("Phase mode enabled: ANALYZE → PLAN → EXECUTE. Send your task.", fg="cyan"))
                continue

            from infinidev.config.settings import reload_all
            reload_all()

            # --- Analysis phase ---
            if settings.ANALYSIS_ENABLED:
                analyst.reset()
                click.echo(click.style("Analyzing request...", fg="cyan", dim=True))

                analysis = analyst.analyze(user_input)

                # Handle question loop
                while analysis.action == "ask" and analyst.can_ask_more:
                    questions_text = analysis.format_questions_for_user()
                    click.echo(click.style("\n" + questions_text, fg="cyan"))

                    answer = session.prompt("Your answer> ")
                    if not answer.strip():
                        # User skipped — force proceed with assumptions
                        break
                    analyst.add_answer(questions_text, answer)
                    analysis = analyst.analyze(
                        user_input + "\n\nUser clarification: " + answer
                    )

                # Build the task prompt from analysis result
                task_prompt = analysis.build_flow_prompt()

                # Handle "done" pseudo-flow (greetings, simple questions answered by analyst)
                if analysis.flow == "done":
                    click.echo(click.style("\n" + (analysis.reason or analysis.original_input), fg="green"))
                    continue

                if analysis.action == "proceed":
                    # Show spec and wait for user confirmation
                    spec = analysis.specification
                    click.echo(click.style("\n── Analysis Result ──", fg="cyan", bold=True))
                    if spec.get("summary"):
                        click.echo(click.style(f"Summary: {spec['summary']}", fg="cyan"))
                    for key in ("requirements", "hidden_requirements", "assumptions", "out_of_scope"):
                        items = spec.get(key, [])
                        if items:
                            click.echo(click.style(f"\n{key.replace('_', ' ').title()}:", fg="cyan", bold=True))
                            for item in items:
                                click.echo(f"  • {item}")
                    if spec.get("technical_notes"):
                        click.echo(click.style(f"\nTechnical Notes:", fg="cyan", bold=True))
                        click.echo(f"  {spec['technical_notes']}")
                    click.echo(click.style("─" * 40, fg="cyan"))

                    confirm = session.prompt(
                        "Proceed with implementation? [Y/n/feedback] "
                    ).strip()
                    if confirm.lower() in ("n", "no", "cancel"):
                        click.echo(click.style("Skipped development.", dim=True))
                        continue
                    if confirm and confirm.lower() not in ("y", "yes", ""):
                        # User gave extra feedback — append to the task prompt
                        desc, expected = task_prompt
                        desc += f"\n\n## Additional User Feedback\n{confirm}"
                        task_prompt = (desc, expected)

                # Get flow config and configure agent
                from infinidev.engine.flows import get_flow_config
                flow_config = get_flow_config(analysis.flow)
                agent._system_prompt_identity = flow_config.identity_prompt
                agent.backstory = flow_config.backstory

                # Override expected output with flow-specific template
                desc, _ = task_prompt
                task_prompt = (desc, flow_config.expected_output)
            else:
                task_prompt = (user_input, "Complete the task and report findings.")
                flow_config = None
                analysis = None
            # --- End analysis phase ---

            current_flow = analysis.flow if analysis is not None else "develop"

            # --- Gather phase ---
            _do_gather = settings.GATHER_ENABLED or _gather_next_task
            _gather_next_task = False  # Reset after use
            if _do_gather and current_flow == "develop":
                try:
                    from infinidev.gather import run_gather
                    agent.activate_context(session_id=session_id)
                    click.echo(click.style("Gathering context...", fg="cyan", dim=True))
                    chat_history = [
                        {"role": "user" if "[user]" in s.lower() else "assistant", "content": s}
                        for s in get_recent_summaries(session_id, limit=10)
                    ]
                    brief = run_gather(user_input, chat_history, analysis, agent)
                    desc, expected = task_prompt
                    desc = brief.render() + "\n\n" + desc
                    task_prompt = (desc, expected)
                    click.echo(click.style(f"  {brief.summary()}", fg="cyan", dim=True))
                except Exception as exc:
                    click.echo(click.style(f"  Gather failed (proceeding without): {exc}", fg="yellow", dim=True))
            # --- End gather phase ---

            click.echo(click.style(f"[{current_flow}] Working on: {user_input}", fg="yellow"))

            # --- Development/Exploration phase ---
            agent.activate_context(session_id=session_id)
            try:
                if current_flow in ("explore", "brainstorm"):
                    from infinidev.engine.tree_engine import TreeEngine
                    tree_engine = TreeEngine()
                    result = tree_engine.execute(
                        agent=agent,
                        task_prompt=task_prompt,
                        mode=current_flow,
                    )
                elif _use_phase_engine:
                    _use_phase_engine = False  # Reset after use
                    from infinidev.engine.phase_engine import PhaseEngine
                    # Determine task type from analysis or default to feature
                    _task_type = "feature"
                    if analysis and hasattr(analysis, 'specification'):
                        _task_type = analysis.specification.get("task_type", "feature")
                    # Extract depth config from gather brief if available
                    _depth_config = None
                    if hasattr(agent, '_gather_brief') and agent._gather_brief:
                        try:
                            from infinidev.gather.models import DEPTH_CONFIGS
                            _depth_config = DEPTH_CONFIGS.get(agent._gather_brief.classification.depth)
                        except Exception:
                            pass
                    phase_eng = PhaseEngine()
                    result = phase_eng.execute(
                        agent=agent,
                        task_prompt=task_prompt,
                        task_type=_task_type,
                        verbose=True,
                        depth_config=_depth_config,
                    )
                else:
                    result = engine.execute(
                        agent=agent,
                        task_prompt=task_prompt,
                        verbose=True
                    )
                if not result or not result.strip():
                    result = "Done. (no additional output)"
            finally:
                agent.deactivate()

            # --- Code review phase (with review-rework loop) ---
            run_review = flow_config.run_review if flow_config else True
            if settings.REVIEW_ENABLED and run_review and current_flow != "explore" and engine.has_file_changes():
                click.echo(click.style("\nRunning code review...", fg="magenta", dim=True))
                from infinidev.engine.review_engine import run_review_rework_loop

                def _cli_review_status(level: str, msg: str) -> None:
                    if level == "verification_pass":
                        click.echo(click.style(f"Verification: PASS. {msg}", fg="green", dim=True))
                    elif level == "verification_fail":
                        click.echo(click.style(f"Verification: FAIL. {msg}", fg="red"))
                        click.echo(click.style("Re-running developer to fix test failures...", fg="magenta", dim=True))
                    elif level == "approved":
                        click.echo(click.style(f"Code review: APPROVED. {msg}", fg="green", dim=True))
                    elif level == "rejected":
                        click.echo(click.style(msg, fg="red"))
                        click.echo(click.style("Re-running developer to fix review issues...", fg="magenta", dim=True))
                    elif level == "max_reviews":
                        click.echo(click.style("Max review rounds reached — stopping.", fg="yellow", dim=True))

                result, _ = run_review_rework_loop(
                    engine=engine,
                    agent=agent,
                    session_id=session_id,
                    task_prompt=task_prompt,
                    initial_result=result,
                    reviewer=reviewer,
                    recent_messages=get_recent_summaries(session_id, limit=5),
                    on_status=_cli_review_status,
                )
            # --- End code review phase ---

            click.echo(click.style("\nFinal Result:", fg="green", bold=True))
            click.echo(result)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg="red"))
            logging.exception("Error in main loop")

    # Cleanup background services
    _classic_watcher.stop()
    _index_queue.stop()

if __name__ == "__main__":
    main()
