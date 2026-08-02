"""Command routing for /commands in the TUI.

Uses the Command pattern: each /command maps to a handler function.
The router dispatches based on the command name, keeping the app free
from command-specific logic.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp


def handle_command(app: InfinidevApp, cmd_text: str) -> None:
    """Dispatch a /command to the appropriate handler."""
    import logging as _log
    _logger = _log.getLogger("infinidev.tui.cmd")
    _logger.warning("[CMD] dispatch %r engine_running=%s",
                    cmd_text, getattr(app, "_engine_running", None))

    parts = cmd_text.split()
    cmd = parts[0].lower()

    handler = _COMMAND_TABLE.get(cmd)
    if handler:
        _logger.warning("[CMD] calling handler for %s", cmd)
        handler(app, parts)
        _logger.warning("[CMD] handler for %s returned", cmd)
    else:
        app.add_message("System", f"Unknown command: {cmd}", "system")


# ── Individual command handlers ─────────────────────────────────────────


def _cmd_exit(app: InfinidevApp, parts: list[str]) -> None:
    app.app.exit()


def _cmd_clear(app: InfinidevApp, parts: list[str]) -> None:
    app.chat_messages.clear()
    app._chat_history_control.invalidate_cache()
    app._log_entries.clear()
    app._plan_text = ""
    app._thinking_text = ""
    app._steps_text = ""
    app._actions_text = ""
    app._streaming_tool_name = None
    app._streaming_token_count = 0
    app.invalidate()


def _cmd_help(app: InfinidevApp, parts: list[str]) -> None:
    app.add_message(
        "System",
        "PANELS\n"
        "  Alt+.  ·  F4  ·  /sidebar    Session panel (steps, files, context)\n"
        "  Ctrl+E        ·  /files      File tree\n"
        "  Ctrl+B        ·  /tasks      Background tasks\n"
        "\n"
        "KEYS\n"
        "  ?                            This help (on an empty prompt)\n"
        "  Esc Esc                      Stop the current tool only\n"
        "  Hold Esc                     Stop the running task\n"
        "  Esc                          Close a panel or dialog\n"
        "  Ctrl+C                       Clear input, twice to quit\n"
        "  \\ + Enter                    New line without sending\n"
        "  PgUp / PgDn                  Scroll the transcript\n"
        "  Ctrl+O                       Open a file    ·  Ctrl+W close tab\n"
        "  Ctrl+G                       Search the project\n"
        "\n"
        "SHELL\n"
        "  !ls -la                      Run a shell command directly\n"
        "\n"
        "COMMANDS\n"
        "  /models [list|set|manage]    Show or change the model\n"
        "  /effort [level]              Reasoning depth this model accepts\n"
        "  /settings [key] [value]      Show or change settings\n"
        "  /mcp [restart <name>]        Index server health (Ken and others)\n"
        "  /plan <task>                 Plan, review, then execute\n"
        "  /refactor [scope]            Modularize, clean, restructure\n"
        "  /explore <problem>           Decompose a complex problem\n"
        "  /init                        Explore and document this project\n"
        "  /debug                       Agent internals: notes, plan, state\n"
        "  /findings   /knowledge       Browse what the agent learned\n"
        "  /documentation               Browse cached library docs\n"
        "  /reindex [--full]            Rebuild the local symbol index\n"
        "  /clear                       Clear the transcript\n"
        "  /exit                        Quit",
        "system",
    )


def _cmd_settings(app: InfinidevApp, parts: list[str]) -> None:
    handle_settings(app, parts)


def _cmd_models(app: InfinidevApp, parts: list[str]) -> None:
    handle_models(app, parts)


def _cmd_effort(app: InfinidevApp, parts: list[str]) -> None:
    handle_effort(app, parts)


def _cmd_findings(app: InfinidevApp, parts: list[str]) -> None:
    app.dialog_manager.open_findings(filter_type=None)


def _cmd_tasks(app: InfinidevApp, parts: list[str]) -> None:
    """Open the background-tasks explorer (same as Ctrl+B)."""
    app.dialog_manager.open_background_tasks()


def _cmd_knowledge(app: InfinidevApp, parts: list[str]) -> None:
    app.dialog_manager.open_findings(filter_type="project_context")


def _cmd_docs(app: InfinidevApp, parts: list[str]) -> None:
    app.add_message("System", "[Docs browser — coming soon]", "system")


def _cmd_notes(app: InfinidevApp, parts: list[str]) -> None:
    """Show agent notes — alias for /debug."""
    app.dialog_manager.open_debug()


def _cmd_debug(app: InfinidevApp, parts: list[str]) -> None:
    """Open the debug panel with notes, history, plan, and state."""
    app.dialog_manager.open_debug()


def _cmd_think(app: InfinidevApp, parts: list[str]) -> None:
    app._gather_next_task = True
    app.add_message(
        "System",
        "Gather mode enabled for the next task. Send your prompt and "
        "infinidev will deeply analyze the codebase before acting.",
        "system",
    )


def _cmd_engine_task(app: InfinidevApp, parts: list[str], cmd: str, task_runner_name: str, label: str) -> None:
    """Generic handler for /explore, /brainstorm, /plan, /init."""
    if cmd != "/init":
        problem = " ".join(parts[1:]) if len(parts) > 1 else ""
        if not problem:
            app.add_message("System", f"Usage: {cmd} <description>", "system")
            return
    else:
        problem = None

    if app._engine_running:
        app.add_message("System", f"Cannot run {cmd} while a task is running.", "system")
        return

    app._engine_running = True
    if problem:
        app.add_message("System", f"{label}: {problem}", "system")
    else:
        app.add_message("System", f"{label}...", "system")
    app._chat_history_control.show_thinking = True
    app.invalidate()
    app._ensure_engine()

    from infinidev.ui import workers as _workers
    from infinidev.ui.workers import run_in_background
    task_fn = getattr(_workers, task_runner_name)
    if problem is not None:
        run_in_background(app, task_fn, app, problem, exclusive=True)
    else:
        run_in_background(app, task_fn, app, exclusive=True)


def _cmd_explore(app: InfinidevApp, parts: list[str]) -> None:
    _cmd_engine_task(app, parts, "/explore", "run_explore_task", "Exploring")


def _cmd_brainstorm(app: InfinidevApp, parts: list[str]) -> None:
    _cmd_engine_task(app, parts, "/brainstorm", "run_brainstorm_task", "Brainstorming")


def _cmd_plan(app: InfinidevApp, parts: list[str]) -> None:
    _cmd_engine_task(app, parts, "/plan", "run_plan_task", "Planning")


def _cmd_init(app: InfinidevApp, parts: list[str]) -> None:
    _cmd_engine_task(app, parts, "/init", "run_init_task", "Exploring and documenting project")


# ── /refactor ───────────────────────────────────────────────────────────

REFACTOR_PROMPT = (
    "Refactor code to improve its quality. Focus on:\n"
    "- Modularize: split large or monolithic units into smaller, cohesive pieces.\n"
    "- Clean: remove dead code, redundant logic, and unused imports or variables.\n"
    "- Order: consistent naming, grouping of related code, clearer organization.\n"
    "- Structure: better separation of concerns and clearer abstractions.\n"
    "\n"
    "Rules:\n"
    "- Preserve existing behavior — this is a refactor, not a rewrite or a new feature.\n"
    "- Do not change public APIs unless explicitly requested.\n"
    "- Do not add new features, options, or speculative abstractions.\n"
    "- Read files before editing them and keep changes focused and minimal.\n"
)


def _build_refactor_prompt(user_scope: str) -> str:
    scope = user_scope.strip()
    if scope:
        return REFACTOR_PROMPT + f"\nUser-specified scope and instructions:\n{scope}\n"
    return (
        REFACTOR_PROMPT
        + "\nNo specific scope was provided. Identify a small, well-scoped area of the "
        "codebase that would most benefit from refactoring and limit the work to that area. "
        "Do not attempt to refactor the whole codebase in one pass.\n"
    )


def _cmd_reindex(app: InfinidevApp, parts: list[str]) -> None:
    """Re-index the current workspace's code intelligence index.

    `--full` (or `-f`) clears the existing index for project 1 first,
    forcing every file to be re-parsed from scratch. Without it the
    reindex is incremental — files whose content hash hasn't changed
    are skipped. Mirrors the same logic as the classic CLI's /reindex
    handler in cli/main.py so the TUI and classic mode behave the same.
    """
    if app._engine_running:
        app.add_message("System", "Cannot reindex while a task is running.", "system")
        return

    full = any(a in ("--full", "-f") for a in parts[1:])
    import os as _os
    from infinidev.code_intel.indexer import index_directory
    from infinidev.tools.base.context import get_current_workspace_path
    from infinidev.tools.base.db import execute_with_retry

    workdir = get_current_workspace_path() or _os.getcwd()

    if full:
        app.add_message("System", "Clearing existing index for project 1...", "system")

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
            app.add_message("System", f"Clear failed: {exc}", "system")
            return

    app.add_message("System", f"Indexing {workdir} ...", "system")
    try:
        stats = index_directory(1, workdir)
        app.add_message(
            "System",
            f"Done: {stats['files_indexed']} files indexed, "
            f"{stats['symbols_total']} symbols, "
            f"{stats['files_skipped']} skipped, "
            f"{stats['elapsed_ms']}ms",
            "system",
        )
    except Exception as exc:
        app.add_message("System", f"Index failed: {exc}", "system")


def _cmd_refactor(app: InfinidevApp, parts: list[str]) -> None:
    """Run a refactor task through the normal engine pipeline."""
    user_scope = " ".join(parts[1:]) if len(parts) > 1 else ""
    prompt = _build_refactor_prompt(user_scope)

    if app._engine_running:
        app.add_message("System", "Cannot run /refactor while a task is running.", "system")
        return

    label = f"Refactoring: {user_scope}" if user_scope else "Refactoring (auto-scoped)"
    app.add_message("System", label, "system")

    app._engine_running = True
    app._chat_history_control.show_thinking = True
    app.invalidate()
    app._ensure_engine()

    from infinidev.ui.workers import run_in_background, run_engine_task
    run_in_background(app, run_engine_task, app, prompt, exclusive=True)


def _cmd_sidebar(app: InfinidevApp, parts: list[str]) -> None:
    """Toggle the right panel — the fallback when Alt+. / F4 are eaten."""
    app.toggle_sidebar()


def _cmd_explorer(app: InfinidevApp, parts: list[str]) -> None:
    """Toggle the file tree."""
    app.toggle_explorer()


def _cmd_mcp(app: InfinidevApp, parts: list[str]) -> None:
    """Show MCP server health, or start/stop/restart one.

    Usage: ``/mcp`` · ``/mcp restart ken`` · ``/mcp stop ken``
    """
    from infinidev.engine.mcp_client import get_default_mcp_manager

    manager = get_default_mcp_manager()
    action = parts[1].lower() if len(parts) > 1 else "status"

    if action in {"start", "stop", "restart"}:
        if len(parts) < 3:
            app.add_message("System", f"Usage: /mcp {action} <server>", "system")
            return
        name = parts[2]
        ok = getattr(manager, action)(name)
        verb = f"{action}ed" if action != "stop" else "stopped"
        app.add_message(
            "System",
            f"MCP server {name!r} {verb}." if ok else f"MCP server {name!r} not found.",
            "system",
        )
        return

    status = manager.status()
    if not status:
        app.add_message(
            "System",
            "No MCP servers configured. Add one to .mcp.json — Ken is the "
            "default when the file is absent.",
            "system",
        )
        return

    lines = ["MCP servers:"]
    for name, info in status.items():
        if info["running"] and info["initialized"]:
            state = f"● running · {info['tools_loaded']} tools"
        elif not info["available"]:
            state = f"✗ unavailable — {info['reason'] or 'unknown'}"
        elif info["failure_count"]:
            # Installed but it did not come up. Reporting "idle" here would
            # send the user hunting for a problem the manager already knows
            # about, so lead with the actual reason.
            state = f"✗ failed — {info['reason'] or 'unknown error'}"
        else:
            state = "○ idle (starts on first use)"
        lines.append(f"  {name}: {state}")
        lines.append(f"      {info['command']}")
        if info["failure_count"]:
            lines.append(f"      {info['failure_count']} recent failure(s)")
        for stderr_line in info["stderr"]:
            lines.append(f"      ! {stderr_line}")
    lines.append("")
    lines.append("/mcp restart <server> · /mcp stop <server>")
    app.add_message("System", "\n".join(lines), "system")


# ── Command dispatch table ──────────────────────────────────────────────

_COMMAND_TABLE: dict[str, Any] = {
    "/mcp": _cmd_mcp,
    "/sidebar": _cmd_sidebar,
    "/panel": _cmd_sidebar,
    "/files": _cmd_explorer,
    "/exit": _cmd_exit,
    "/quit": _cmd_exit,
    "/clear": _cmd_clear,
    "/help": _cmd_help,
    "/settings": _cmd_settings,
    "/models": _cmd_models,
    "/effort": _cmd_effort,
    "/debug": _cmd_debug,
    "/notes": _cmd_notes,
    "/findings": _cmd_findings,
    "/tasks": _cmd_tasks,
    "/knowledge": _cmd_knowledge,
    "/documentation": _cmd_docs,
    "/docs": _cmd_docs,
    "/explore": _cmd_explore,
    "/brainstorm": _cmd_brainstorm,
    "/plan": _cmd_plan,
    "/think": _cmd_think,
    "/init": _cmd_init,
    "/refactor": _cmd_refactor,
    "/reindex": _cmd_reindex,
}


# ── Settings subcommand handler ─────────────────────────────────────────


def handle_settings(app: InfinidevApp, parts: list[str]) -> None:
    """Handle /settings subcommands."""
    from infinidev.config.secrets import mask_if_secret
    from infinidev.config.settings import settings, reload_all

    if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "browse"):
        app.dialog_manager.open_settings()
        return

    subcmd = parts[1].lower()

    if subcmd == "reset":
        settings.reset_to_defaults()
        reload_all()
        app.add_message("System", "Settings reset to defaults.", "system")

    elif subcmd == "export" and len(parts) > 2:
        path = parts[2]
        settings.export_to_file(path)
        app.add_message("System", f"Settings exported to {path}", "system")

    elif subcmd == "import" and len(parts) > 2:
        path = parts[2]
        settings.import_from_file(path)
        reload_all()
        app.add_message("System", f"Settings imported from {path}", "system")

    elif len(parts) == 2:
        key = parts[1].upper()
        val = getattr(settings, key, None)
        if val is not None:
            # Credentials are echoed as a shape: the transcript is scrollback,
            # and scrollback outlives the session.
            app.add_message("System", f"{key}: {mask_if_secret(key, val)}", "system")
        else:
            app.add_message("System", f"Unknown setting: {key}", "system")

    elif len(parts) >= 3:
        key = parts[1].upper()
        value = " ".join(parts[2:])
        try:
            settings.save_user_settings({key: value})
            reload_all()
            app.add_message(
                "System", f"{key} = {mask_if_secret(key, value)}", "system"
            )
            app._update_status_bar()
        except Exception as e:
            app.add_message("System", f"Error setting {key}: {e}", "system")


# ── Models subcommand handler ───────────────────────────────────────────


def _effort_choices() -> tuple[list[str], bool]:
    """The reasoning levels worth offering, and whether they came from a model.

    On the ChatGPT subscription the levels are per-model and published by the
    Codex catalog, so a frontier model offers ``max`` and ``ultra`` while a
    smaller one stops at ``xhigh``. Offering a level the model does not have
    earns a 400 the user cannot predict, so the catalog decides. Every other
    provider gets the generic presets, which is what THINKING_BUDGET has
    always meant there.
    """
    from infinidev.config.llm import CHATGPT_SUBSCRIPTION_PROVIDER
    from infinidev.config.settings import settings
    from infinidev.config.thinking_budget import subscription_efforts

    if settings.LLM_PROVIDER == CHATGPT_SUBSCRIPTION_PROVIDER:
        levels = subscription_efforts()
        if levels:
            return levels, True
    return ["low", "medium", "high", "ultra"], False


def _effort_in_effect() -> str:
    """What the next request will actually carry, described in one line.

    Built by running the real ``apply_thinking_budget`` over an empty kwargs
    dict rather than by re-deriving it here. A description that can disagree
    with the code is worse than no description.
    """
    from infinidev.config.settings import settings
    from infinidev.config.thinking_budget import apply_thinking_budget

    probe: dict[str, Any] = {}
    try:
        apply_thinking_budget(probe, settings.LLM_PROVIDER, settings.LLM_MODEL)
    except Exception as exc:
        return f"could not resolve ({exc})"

    if "reasoning_effort" in probe:
        return f"reasoning_effort={probe['reasoning_effort']}"
    if "thinking" in probe:
        return f"thinking={probe['thinking']}"
    if "max_tokens" in probe:
        return f"max_tokens={probe['max_tokens']}"
    return "no thinking parameter sent"


def handle_effort(app: InfinidevApp, parts: list[str]) -> None:
    """Show or set the reasoning effort, bounded by what the model accepts.

    Usage: ``/effort`` · ``/effort high`` · ``/effort max``
    """
    from infinidev.config.settings import settings, reload_all

    choices, from_catalog = _effort_choices()
    current = (settings.THINKING_BUDGET or "").lower().strip()

    if len(parts) == 1:
        source = (
            f"published by {settings.LLM_MODEL}"
            if from_catalog
            else "generic presets for this provider"
        )
        listed = "\n".join(
            f"  {'>' if level == current else ' '} {level}" for level in choices
        )
        extra = "" if current in choices else f"\n  (current: {current})"
        app.add_message(
            "System",
            f"Reasoning effort — {source}\n{listed}{extra}\n\n"
            f"In effect now: {_effort_in_effect()}\n"
            f"Change it with /effort <level>",
            "system",
        )
        return

    wanted = parts[1].lower().strip()
    if wanted not in choices:
        app.add_message(
            "System",
            f"'{wanted}' is not available for {settings.LLM_MODEL}.\n"
            f"Choose one of: {', '.join(choices)}",
            "system",
        )
        return

    settings.save_user_settings({"THINKING_BUDGET": wanted})
    reload_all()
    app.add_message(
        "System",
        f"Reasoning effort set to {wanted}. In effect: {_effort_in_effect()}",
        "system",
    )
    app._update_status_bar()


def handle_models(app: InfinidevApp, parts: list[str]) -> None:
    """Handle /models subcommands."""
    from infinidev.config.settings import settings, reload_all
    from infinidev.config.providers import get_provider, fetch_models

    subcmd = parts[1].lower() if len(parts) > 1 else "info"

    if subcmd == "set" and len(parts) > 2:
        new_model = parts[2]
        if "/" not in new_model:
            provider = get_provider(settings.LLM_PROVIDER)
            new_model = f"{provider.prefix}{new_model}"
        settings.save_user_settings({"LLM_MODEL": new_model})
        reload_all()
        from infinidev.config.model_capabilities import _reset_capabilities
        _reset_capabilities()
        # Re-fetch the new model's context window (async, best effort)
        if app.context_calculator:
            import asyncio
            try:
                asyncio.run(app.context_calculator.update_model_context())
                app._context_status = app.context_calculator.get_context_status()
            except Exception:
                pass
        app.add_message("System", f"Model updated to: {settings.LLM_MODEL}", "system")
        app._update_status_bar()

    elif subcmd == "list":
        provider = get_provider(settings.LLM_PROVIDER)
        app.add_message("System", f"Fetching models for {provider.display_name}...", "system")
        try:
            models = fetch_models(
                settings.LLM_PROVIDER,
                settings.LLM_API_KEY,
                settings.LLM_BASE_URL,
                raise_on_error=True,
            )
            if models:
                model_list = "\n".join(f"  {m}" for m in models)
                app.add_message("System", f"Available models:\n{model_list}", "system")
            else:
                app.add_message("System", "No models found. Check API key and connection.", "system")
        except Exception as e:
            app.add_message("System", f"Error fetching models: {e}", "system")

    elif subcmd == "manage":
        app.add_message("System", "[Model picker — coming soon]", "system")

    else:
        provider = get_provider(settings.LLM_PROVIDER)
        app.add_message(
            "System",
            f"Provider: {provider.display_name}\n"
            f"Model: {settings.LLM_MODEL}\n"
            f"Base URL: {settings.LLM_BASE_URL}",
            "system",
        )


# ── Shell command executor ──────────────────────────────────────────────


def execute_shell_command(app: InfinidevApp, cmd: str) -> None:
    """Execute a shell command and display results in chat."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
        )
        output_parts = []
        if result.stdout.strip():
            output_parts.append(result.stdout.strip())
        if result.stderr.strip():
            output_parts.append(f"stderr:\n{result.stderr.strip()}")
        if result.returncode != 0:
            output_parts.append(f"Exit code: {result.returncode}")
        output = "\n".join(output_parts) or "(no output)"
    except subprocess.TimeoutExpired:
        output = "Command timed out after 30 seconds."
    except Exception as e:
        output = f"Error: {e}"
    app.add_message("System", output, "system")
