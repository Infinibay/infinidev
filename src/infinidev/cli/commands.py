"""Slash-command handlers for the classic CLI REPL.

Extracted from ``cli/main.py`` so command dispatch lives in one file
and the REPL loop in main.py can stay focused on bootstrap/runtime
concerns. All functions here take plain arguments and use ``click``
for output — no app state, no engine references.

Every handler either returns a sentinel recognized by the REPL
(``"think"``, ``"init"``, a ``(tag, payload)`` tuple) or ``True`` to
mean "command handled, keep the REPL running".
"""

from __future__ import annotations

import os
import sys
import click

from infinidev.config.settings import settings


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
                resp = httpx.get(f"{base_url}/api/tags", timeout=8)
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
