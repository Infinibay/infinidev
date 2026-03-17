"""Main entry point for Infinidev CLI."""

import sys
import os
import logging
import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from infinidev.config.settings import settings, DEFAULT_BASE_DIR
from infinidev.config.llm import get_litellm_params
import uuid
from infinidev.db.service import init_db
from infinidev.agents.base import InfinidevAgent
from infinidev.engine.loop_engine import LoopEngine
from infinidev.cli.tui import InfinidevTUI

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
        click.echo("  /exit, /quit       - Exit the CLI")
        click.echo("  /help              - Show this help")
        return True
    
    elif cmd == "/settings":
        handle_settings_command(parts)
        return True

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
            "FORGEJO_API_URL": str,
            "FORGEJO_OWNER": str,
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
        }

        if value_to_set:
            # Convert to appropriate type
            type_class = type_map.get(setting_key, str)
            try:
                if type_class == bool:
                    converted = value_to_set.lower() in ("true", "1", "yes")
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


@click.command()
@click.option("--no-tui", is_flag=True, help="Run in classic CLI mode instead of TUI.")
@click.option("--classic", is_flag=True, hidden=True, help="Alias for --no-tui.")
def main(no_tui: bool, classic: bool):
    """Main entry point for Infinidev CLI."""
    if not (no_tui or classic):
        # TUI mode: remove the stderr handler so log output doesn't corrupt
        # the Textual terminal.  File logging is preserved.
        root = logging.getLogger()
        root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)
                         or getattr(h, 'stream', None) is not sys.stderr]
        app = InfinidevTUI()
        app.run()
        return

    # Classic mode (the original while True loop)
    init_db()
    
    click.echo(click.style("Welcome to Infinidev CLI (Classic Mode)!", fg="cyan", bold=True))
    click.echo("Type your instructions or /help for commands.")
    
    session = PromptSession(history=FileHistory(str(DEFAULT_BASE_DIR / "history")))
    
    # Create the agent
    agent = InfinidevAgent(agent_id="cli_agent")
    session_id = str(uuid.uuid4())
    engine = LoopEngine()

    while True:
        try:
            user_input = session.prompt("infinidev> ")
            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                handle_command(user_input)
                continue

            from infinidev.config.settings import reload_all
            reload_all()

            click.echo(click.style(f"Working on: {user_input}", fg="yellow"))

            agent.activate_context(session_id=session_id)
            try:
                result = engine.execute(
                    agent=agent,
                    task_prompt=(user_input, "Complete the task and report findings."),
                    verbose=True
                )
                if not result or not result.strip():
                    result = "Done. (no additional output)"
                click.echo(click.style("\nFinal Result:", fg="green", bold=True))
                click.echo(result)
            finally:
                agent.deactivate()
                
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg="red"))
            logging.exception("Error in main loop")

if __name__ == "__main__":
    main()
