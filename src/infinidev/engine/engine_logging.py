"""Logging and event emission for the loop engine.

Provides:
- Pretty ANSI-colored stdout logging for classic CLI mode
- Event emission through EventBus for TUI mode
- Tool detail/error extraction for UI display
"""

from __future__ import annotations

import json
import sys
from typing import Any

from infinidev.flows.event_listeners import event_bus

# ── ANSI color codes ──────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"

STATUS_ICON = {
    "continue": f"{CYAN}→{RESET}",
    "done": f"{GREEN}✓{RESET}",
    "blocked": f"{RED}✗{RESET}",
}


# ── Tool detail extraction for UI visibility ─────────────────────────────

# Maps tool name → list of arg keys to show in UI (in priority order).
TOOL_DETAIL_KEYS: dict[str, list[str]] = {
    "read_file": ["path", "file_path"],
    "write_file": ["path", "file_path"],
    "edit_file": ["path", "file_path"],
    "multi_edit_file": ["path", "file_path"],
    "apply_patch": ["patch"],
    "list_directory": ["path", "directory"],
    "code_search": ["query", "pattern", "search_query"],
    "glob": ["pattern", "glob_pattern"],
    "execute_command": ["command", "cmd"],
    "git_branch": ["branch_name", "name"],
    "git_commit": ["message"],
    "git_push": ["branch"],
    "git_diff": ["branch", "file_path"],
    "git_status": [],

    "web_search": ["query"],
    "web_fetch": ["url"],
    "code_search_web": ["query"],
    "find_definition": ["name"],
    "find_references": ["name"],
    "list_symbols": ["file_path", "path"],
    "search_symbols": ["query"],
    "get_symbol_code": ["name"],
    "project_structure": ["path", "directory", "dir", "folder", "subdir"],
    "search_knowledge": ["query"],
    "record_finding": ["title"],
    "search_findings": ["query"],
    "read_findings": ["query"],
    "update_finding": ["finding_id"],
    "delete_finding": ["finding_id"],
}


# ── Core log / event helpers ─────────────────────────────────────────────

def emit_loop_event(
    event_type: str,
    project_id: int,
    agent_id: str,
    data: dict[str, Any],
) -> None:
    """Emit event to all subscribers via the EventBus."""
    event_bus.emit(event_type, project_id, agent_id, data)


def log(msg: str) -> None:
    """Print to stderr in classic CLI mode.  Silent when a TUI/event callback
    is registered (the TUI owns the terminal, so raw prints would corrupt it).
    """
    if not event_bus.has_subscribers:
        print(msg, file=sys.stderr, flush=True)


def emit_log(level: str, text: str, *, project_id: int = 0, agent_id: str = "") -> None:
    """Emit a log entry through the event system for TUI display.

    *level* is ``"warning"`` or ``"error"``.  When no event callback is
    registered, the message is also printed via ``log`` as a fallback.
    """
    import re
    clean = re.sub(r"\033\[[0-9;]*m", "", text)  # strip ANSI for the TUI
    emit_loop_event("loop_log", project_id, agent_id, {
        "level": level,
        "message": clean,
    })
    log(text)  # no-op in TUI mode, shows in classic mode


# ── Tool detail extraction ───────────────────────────────────────────────

def extract_tool_detail(tool_name: str, arguments: str) -> str:
    """Extract a short human-readable detail from tool call arguments.

    Returns e.g. "src/auth.py" for read_file, "gradient optimizer" for code_search.
    Returns empty string if no useful detail can be extracted.
    """
    keys = TOOL_DETAIL_KEYS.get(tool_name)
    if keys is None:
        keys = ["path", "file_path", "query", "title", "name"]
    if not keys:
        return ""

    try:
        args = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else {}
    except (json.JSONDecodeError, TypeError):
        return ""

    if not isinstance(args, dict):
        return ""

    for key in keys:
        val = args.get(key)
        if val is not None:
            s = str(val).strip()
            if len(s) > 80:
                s = s[:77] + "..."
            return s
    return ""


def extract_tool_output_preview(tool_name: str, result: str, max_lines: int = 4, max_width: int = 100) -> str:
    """Extract a short preview of tool output for display in the CLI.

    Shows the first few meaningful lines of the result, truncated.
    Returns empty string if there's nothing useful to show (errors are
    handled separately by extract_tool_error).
    """
    if not result or not result.strip():
        return ""

    stripped = result.strip()

    # Skip if it's an error (handled by extract_tool_error)
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                return ""
        except (json.JSONDecodeError, TypeError):
            pass

    # For execute_command: show the actual output (most valuable)
    if tool_name == "execute_command":
        lines = stripped.splitlines()
        # Take last N lines (most relevant for test output, build output)
        if len(lines) > max_lines:
            preview_lines = lines[-max_lines:]
        else:
            preview_lines = lines
        preview = "\n".join(
            line[:max_width] + ("…" if len(line) > max_width else "")
            for line in preview_lines
        )
        if len(lines) > max_lines:
            preview = f"… ({len(lines) - max_lines} lines above)\n{preview}"
        return preview

    # For file tools: show line count or brief summary
    if tool_name in ("read_file", "partial_read"):
        lines = stripped.splitlines()
        if len(lines) > 3:
            return f"({len(lines)} lines)"
        return ""

    # For create_file, replace_lines: just confirm
    if tool_name in ("create_file",):
        if "Created file" in stripped or "created" in stripped.lower():
            return stripped[:max_width]
        return ""

    if tool_name in ("replace_lines",):
        if len(stripped) < max_width:
            return stripped
        return ""

    # For search/list tools: show count
    if tool_name in ("code_search", "glob", "search_symbols", "find_references"):
        lines = stripped.splitlines()
        if len(lines) > 3:
            return f"({len(lines)} results)"
        elif lines:
            return "\n".join(line[:max_width] for line in lines[:3])
        return ""

    return ""


def extract_tool_error(result: str) -> str:
    """Extract error message from a tool result, if any.

    Returns a short error string for display, or empty string if no error.
    """
    if not result:
        return ""
    stripped = result.strip()
    if not stripped.startswith("{"):
        return ""
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "error" in parsed:
            err = str(parsed["error"])
            if "Unknown tool:" in err:
                tool_name = err.split("Unknown tool:", 1)[1].strip()
                return f"hallucinated tool '{tool_name}'"
            if len(err) > 120:
                err = err[:117] + "..."
            return err
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


# ── Step-level logging functions ─────────────────────────────────────────

def log_start(agent_id: str, agent_name: str, role: str, desc: str, tool_count: int) -> None:
    log(f"\n{BOLD}{CYAN}✨ Infinidev{RESET}  {DIM}•{RESET}  {BOLD}{agent_name}{RESET} {DIM}({role}){RESET}")
    log(f"{DIM}   {desc[:120]}{'…' if len(desc) > 120 else ''}{RESET}")
    log(f"{DIM}   {tool_count} tools ready{RESET}")
    log(f"{DIM}{'─' * 60}{RESET}")


def log_step_start(iteration: int, step_desc: str | None) -> None:
    # Suppress the "Planning..." pseudo-step from the chat. It fires
    # on the very first iteration of every task before the model has
    # produced a real plan, adds no information (the user can see
    # the assistant is working from the streaming indicator), and
    # creates visual noise on simple tasks. Real steps with a title
    # still print normally.
    if not step_desc or step_desc == "Planning...":
        return
    log(f"\n{BOLD}{BLUE}󰄵 Step {iteration}{RESET} {step_desc}")


def log_tool(agent_name: str, iteration: int, tool_name: str, call_num: int, total: int) -> None:
    log(f"  {MAGENTA}⚙️  {tool_name}{RESET}")


def log_step_done(iteration: int, status: str, summary: str, tool_calls: int, tokens: int) -> None:
    icon = "✔" if status == "done" else "➜"
    color = GREEN if status == "done" else YELLOW
    log(f"  {color}{icon} {status.title()}{RESET}  {DIM}({tool_calls} calls · {tokens} tokens){RESET}")
    if summary:
        log(f"    {DIM}{summary[:150]}{RESET}")


def log_plan(plan: Any) -> None:
    """Log plan with status icons. Accepts a LoopPlan instance."""
    if not plan.steps:
        return
    log(f"\n  {DIM}Proposed plan:{RESET}")
    for s in plan.steps:
        if s.status == "done":
            icon, color = "●", GREEN
        elif s.status == "active":
            icon, color = "○", CYAN
        else:
            icon, color = "◌", DIM
        log(f"    {color}{icon} {s.title[:80]}{RESET}")


def log_prompt(user_prompt: str, max_section: int = 300) -> None:
    """Log the XML-structured prompt sent to the LLM, truncating each section."""
    import re
    sections = re.findall(r"<(\w[\w-]*)>\n?(.*?)\n?</\1>", user_prompt, re.DOTALL)
    if not sections:
        log(f"{DIM}   Prompt: {user_prompt[:max_section]}{RESET}")
        return
    log(f"{DIM}   Prompt:{RESET}")
    for tag, content in sections:
        preview = content.strip().replace("\n", " ↵ ")
        if len(preview) > max_section:
            preview = preview[:max_section] + "…"
        log(f"   {DIM}<{tag}>{RESET} {preview}")


def log_finish(agent_name: str, status: str, iterations: int, total_tools: int, total_tokens: int) -> None:
    icon = "✅" if status == "done" else "🏁"
    log(f"\n{DIM}{'─' * 60}{RESET}")
    log(
        f"{icon} {BOLD}Completed{RESET}  "
        f"{DIM}{iterations} steps · {total_tools} tools · {total_tokens} tokens{RESET}\n"
    )
