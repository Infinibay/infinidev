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
        click.echo("  /exit, /quit       - Exit the CLI")
        click.echo("  /help              - Show this help")
        return True
    
    click.echo(f"Unknown command: {cmd}")
    return True

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
