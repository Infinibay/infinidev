"""Plan-execute-summarize loop engine.

Replaces the opaque CrewAI ReAct loop with a controlled cycle:
each iteration rebuilds the prompt from scratch with only system prompt +
task + plan + compact summaries of previous actions + current step.
Raw tool output is discarded after each step; only ~50-token summaries survive.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from infinidev.engine.base import AgentEngine
from infinidev.engine.loop_context import (
    build_iteration_prompt,
    build_system_prompt,
    build_tools_prompt_section,
)
from infinidev.engine.loop_models import (
    ActionRecord,
    LoopPlan,
    LoopState,
    PlanStep,
    StepOperation,
    StepResult,
)
from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.loop_tools import (
    STEP_COMPLETE_SCHEMA,
    build_tool_dispatch,
    build_tool_schemas,
    execute_tool_call,
)

# Max consecutive calls to the same tool before forcing a step_complete nudge
_MAX_SAME_TOOL_CONSECUTIVE = 3


def _get_model_max_context(llm_params: dict[str, Any]) -> int:
    """Fetch the model's max context window from Ollama /api/show.

    Returns 0 if unknown (disables context budget in the prompt).
    """
    import httpx

    model = llm_params.get("model", "")
    base_url = llm_params.get("base_url", "http://localhost:11434")

    bare_model = model
    for prefix in ("ollama_chat/", "ollama/"):
        if bare_model.startswith(prefix):
            bare_model = bare_model[len(prefix):]
            break

    try:
        resp = httpx.post(
            f"{base_url}/api/show",
            json={"name": bare_model},
            timeout=5.0,
        )
        if resp.status_code == 200:
            model_info = resp.json().get("model_info", {})
            for key, val in model_info.items():
                if key.endswith(".context_length") and isinstance(val, int):
                    return val
    except Exception:
        pass
    return 0

# Max retries when LLM returns text instead of tool calls
_MAX_TEXT_RETRIES = 3

logger = logging.getLogger(__name__)


# -- Event handling for TUI/WebSocket integration --
_event_callback = None

def set_event_callback(callback):
    """Set a callback function to receive loop events."""
    global _event_callback
    _event_callback = callback

def _emit_loop_event(
    event_type: str,
    project_id: int,
    agent_id: str,
    data: dict[str, Any],
) -> None:
    """Emit event to registered callback or WebSocket."""
    if _event_callback:
        _event_callback(event_type, project_id, agent_id, data)


# ── Pretty stdout logging ────────────────────────────────────────────────────

_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"

_STATUS_ICON = {
    "continue": f"{_CYAN}→{_RESET}",
    "done": f"{_GREEN}✓{_RESET}",
    "blocked": f"{_RED}✗{_RESET}",
}


def _log(msg: str) -> None:
    """Print to stderr in classic CLI mode.  Silent when a TUI/event callback
    is registered (the TUI owns the terminal, so raw prints would corrupt it).
    """
    if _event_callback is None:
        print(msg, file=sys.stderr, flush=True)


def _emit_log(level: str, text: str, *, project_id: int = 0, agent_id: str = "") -> None:
    """Emit a log entry through the event system for TUI display.

    *level* is ``"warning"`` or ``"error"``.  When no event callback is
    registered, the message is also printed via ``_log`` as a fallback.
    """
    import re
    clean = re.sub(r"\033\[[0-9;]*m", "", text)  # strip ANSI for the TUI
    _emit_loop_event("loop_log", project_id, agent_id, {
        "level": level,
        "message": clean,
    })
    _log(text)  # no-op in TUI mode, shows in classic mode


# ── Tool detail extraction for UI visibility ─────────────────────────────────

# Maps tool name → list of arg keys to show in UI (in priority order).
# Only the first matching key is shown, truncated to keep it short.
_TOOL_DETAIL_KEYS: dict[str, list[str]] = {
    "read_file": ["file_path", "path"],
    "write_file": ["file_path", "path"],
    "edit_file": ["file_path", "path"],
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
    "search_knowledge": ["query"],
    "record_finding": ["title"],
    "search_findings": ["query"],
    "read_findings": ["query"],
    "update_finding": ["finding_id"],
    "delete_finding": ["finding_id"],
}


def _extract_tool_detail(tool_name: str, arguments: str) -> str:
    """Extract a short human-readable detail from tool call arguments.

    Returns e.g. "src/auth.py" for read_file, "gradient optimizer" for code_search.
    Returns empty string if no useful detail can be extracted.
    """
    keys = _TOOL_DETAIL_KEYS.get(tool_name)
    if keys is None:
        # Unknown tool — try common keys
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
            # Truncate long values (file contents, long commands)
            if len(s) > 80:
                s = s[:77] + "..."
            return s
    return ""


def _extract_tool_error(result: str) -> str:
    """Extract error message from a tool result, if any.

    Returns a short error string for display, or empty string if no error.
    Detects both JSON {"error": "..."} and "Unknown tool:" patterns.
    """
    if not result:
        return ""
    # Fast path: most results don't start with {"error
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
            # Truncate long errors
            if len(err) > 120:
                err = err[:117] + "..."
            return err
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


# ── File change tracking helpers ────────────────────────────────────────────

_FILE_CHANGE_TOOLS = {"edit_file", "write_file"}
_MAX_TRACK_FILE_SIZE = 1_000_000  # 1 MB — skip tracking larger files


def _extract_file_path_from_args(tool_name: str, arguments: str | dict) -> str | None:
    """Extract the file path from edit_file / write_file arguments."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(args, dict):
        return None
    return args.get("path")


def _capture_pre_content(
    tool_name: str,
    arguments: str | dict,
    tracker: FileChangeTracker,
) -> str | None:
    """Read file content before a write/edit tool mutates it."""
    import os as _os
    if tool_name not in _FILE_CHANGE_TOOLS or not tracker.active:
        return None
    file_path = _extract_file_path_from_args(tool_name, arguments)
    if not file_path:
        return None
    file_path = _os.path.abspath(_os.path.expanduser(file_path))
    if not _os.path.isfile(file_path):
        return None  # new file — original is empty
    try:
        size = _os.path.getsize(file_path)
        if size > _MAX_TRACK_FILE_SIZE:
            return None
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def _maybe_emit_file_change(
    tool_name: str,
    arguments: str | dict,
    result: str,
    pre_content: str | None,
    tracker: FileChangeTracker,
    project_id: int,
    agent_id: str,
) -> None:
    """After a write/edit tool call, record the change and emit a TUI event."""
    import os as _os
    if tool_name not in _FILE_CHANGE_TOOLS or not tracker.active:
        return

    # Skip if the tool returned an error
    if _extract_tool_error(result):
        return

    file_path = _extract_file_path_from_args(tool_name, arguments)
    if not file_path:
        return
    file_path = _os.path.abspath(_os.path.expanduser(file_path))

    # Read current content after the mutation
    try:
        if not _os.path.isfile(file_path):
            return
        size = _os.path.getsize(file_path)
        if size > _MAX_TRACK_FILE_SIZE:
            return
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            after_content = f.read()
    except Exception:
        return

    before = pre_content if pre_content is not None else ""
    diff_text = tracker.record(file_path, before, after_content)
    if not diff_text:
        return

    _emit_loop_event("loop_file_changed", project_id, agent_id, {
        "path": file_path,
        "diff": diff_text,
        "action": tracker.get_action(file_path),
        "num_changes": tracker.get_change_count(file_path),
    })


def _log_start(agent_id: str, agent_name: str, role: str, desc: str, tool_count: int) -> None:
    _log(f"\n{_BOLD}{_CYAN}✨ Infinidev{_RESET}  {_DIM}•{_RESET}  {_BOLD}{agent_name}{_RESET} {_DIM}({role}){_RESET}")
    _log(f"{_DIM}   {desc[:120]}{'…' if len(desc) > 120 else ''}{_RESET}")
    _log(f"{_DIM}   {tool_count} tools ready{_RESET}")
    _log(f"{_DIM}{'─' * 60}{_RESET}")


def _log_step_start(iteration: int, step_desc: str | None) -> None:
    label = step_desc or "Planning..."
    _log(f"\n{_BOLD}{_BLUE}󰄵 Step {iteration}{_RESET} {label}")


def _log_tool(agent_name: str, iteration: int, tool_name: str, call_num: int, total: int) -> None:
    _log(f"  {_MAGENTA}⚙️  {tool_name}{_RESET}")


def _log_step_done(iteration: int, status: str, summary: str, tool_calls: int, tokens: int) -> None:
    # Use generic icon if specific one not found
    icon = "✔" if status == "done" else "➜"
    color = _GREEN if status == "done" else _YELLOW
    _log(f"  {color}{icon} {status.title()}{_RESET}  {_DIM}({tool_calls} calls · {tokens} tokens){_RESET}")
    if summary:
        _log(f"    {_DIM}{summary[:150]}{_RESET}")


def _log_plan(plan: LoopPlan) -> None:
    if not plan.steps:
        return
    _log(f"\n  {_DIM}Proposed plan:{_RESET}")
    for s in plan.steps:
        if s.status == "done":
            icon, color = "●", _GREEN
        elif s.status == "active":
            icon, color = "○", _CYAN
        elif s.status == "skipped":
            icon, color = "◌", _DIM
        else:
            icon, color = "◌", _DIM
        _log(f"    {color}{icon} {s.description[:80]}{_RESET}")


def _log_prompt(user_prompt: str, max_section: int = 300) -> None:
    """Log the XML-structured prompt sent to the LLM, truncating each section."""
    import re
    sections = re.findall(r"<(\w[\w-]*)>\n?(.*?)\n?</\1>", user_prompt, re.DOTALL)
    if not sections:
        _log(f"{_DIM}   Prompt: {user_prompt[:max_section]}{_RESET}")
        return
    _log(f"{_DIM}   Prompt:{_RESET}")
    for tag, content in sections:
        preview = content.strip().replace("\n", " ↵ ")
        if len(preview) > max_section:
            preview = preview[:max_section] + "…"
        _log(f"   {_DIM}<{tag}>{_RESET} {preview}")


def _log_finish(agent_name: str, status: str, iterations: int, total_tools: int, total_tokens: int) -> None:
    icon = "✅" if status == "done" else "🏁"
    _log(f"\n{_DIM}{'─' * 60}{_RESET}")
    _log(
        f"{icon} {_BOLD}Completed{_RESET}  "
        f"{_DIM}{iterations} steps · {total_tools} tools · {total_tokens} tokens{_RESET}\n"
    )

# Transient LLM errors that should be retried
_TRANSIENT_ERRORS = (
    "connection error",
    "connectionerror",
    "disconnected",
    "rate limit",
    "timeout",
    "503",
    "502",
    "429",
    "overloaded",
    "internal server error",
)

# Permanent errors that look transient but aren't (substrings that
# override a _TRANSIENT_ERRORS match when present)
_PERMANENT_ERRORS = (
    "does not support tools",
    "does not support function",
    "tool_choice is not supported",
    "tools is not supported",
    "not found",        # Ollama: {"error":"tool 'X' not found"}
)

_LLM_RETRIES = 3
_LLM_RETRY_DELAY = 5.0


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    # Check permanent exclusions first — these are capability errors
    # wrapped in APIConnectionError, not actual network issues
    if any(p in msg for p in _PERMANENT_ERRORS):
        return False
    return any(p in msg for p in _TRANSIENT_ERRORS)


# Patterns that indicate the LLM produced a malformed tool call
# (e.g. Ollama mixing natural language text with JSON arguments)
_MALFORMED_TOOL_PATTERNS = (
    "error parsing tool call",
    "invalid character",
    "looking for beginning of value",
)


def _is_malformed_tool_call(exc: Exception) -> bool:
    """Check if an LLM error is due to a malformed tool call response."""
    msg = str(exc).lower()
    return any(p in msg for p in _MALFORMED_TOOL_PATTERNS)


class _ManualToolCall:
    """Lightweight stand-in for native tool call objects in manual TC mode.

    Mirrors the attribute structure of litellm/OpenAI tool call objects
    so the rest of the pipeline (dispatch, logging) works unchanged.
    """

    __slots__ = ("id", "function")

    class _Function:
        __slots__ = ("name", "arguments")

        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = self._Function(name, arguments)


def _parse_text_tool_calls(content: str) -> list[dict[str, Any]] | None:
    """Parse tool calls from model text when native FC is unavailable.

    Supports multiple formats that models use to express tool calls:

    1. Our manual-mode JSON: ``{"tool_calls": [{"name": ..., "arguments": ...}]}``
    2. Qwen/GLM ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
    3. Qwen pipe-delimited ``<|tool_call|>...<|/tool_call|>``
    4. Mistral ``[TOOL_CALLS] [{"name": ..., "arguments": ...}]``
    5. Llama ``<|python_tag|>`` function calls
    6. ``<function_call>`` / ``<functioncall>`` wrappers
    7. Markdown code blocks with JSON
    8. Bare JSON objects

    Returns a list of dicts with "name" and "arguments" keys,
    or None if no valid tool calls found.
    """
    import re

    if not content or not content.strip():
        return None

    # Strip thinking sections (various model formats)
    cleaned = re.sub(
        r"<(?:thinking|think|\|thinking\|)>.*?</(?:thinking|think|\|thinking\|)>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # ── 1. Native model tool-call tokens ─────────────────────────────
    # Try these first — they're unambiguous signals of tool use intent.

    # Qwen / GLM: <tool_call>{...}</tool_call>  (one or more)
    tc_tag_matches = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        cleaned, re.DOTALL,
    )
    if tc_tag_matches:
        calls = _extract_calls_from_fragments(tc_tag_matches)
        if calls:
            return calls

    # Qwen pipe-delimited: <|tool_call|>{...}<|/tool_call|>
    tc_pipe_matches = re.findall(
        r"<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>",
        cleaned, re.DOTALL,
    )
    if tc_pipe_matches:
        calls = _extract_calls_from_fragments(tc_pipe_matches)
        if calls:
            return calls

    # Mistral: [TOOL_CALLS] [{...}, ...]
    mistral_match = re.search(
        r"\[TOOL_CALLS\]\s*(\[.*?\])",
        cleaned, re.DOTALL,
    )
    if mistral_match:
        calls = _extract_calls_from_array(mistral_match.group(1))
        if calls:
            return calls

    # Llama: <|python_tag|> followed by JSON (function call format)
    python_tag_match = re.search(
        r"<\|python_tag\|>\s*(.*)",
        cleaned, re.DOTALL,
    )
    if python_tag_match:
        calls = _extract_calls_from_fragments([python_tag_match.group(1)])
        if calls:
            return calls

    # Generic: <function_call>{...}</function_call> or <functioncall>{...}</functioncall>
    fc_matches = re.findall(
        r"<function_?call>\s*(.*?)\s*</function_?call>",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if fc_matches:
        calls = _extract_calls_from_fragments(fc_matches)
        if calls:
            return calls

    # ── 2. Our manual-mode JSON: {"tool_calls": [...]} ───────────────
    # Check markdown code blocks first, then bare text.
    json_candidates: list[str] = []

    # Match ```json ... ``` or ``` ... ```
    code_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    json_candidates.extend(code_blocks)

    # Also try the raw cleaned text (model might output bare JSON)
    json_candidates.append(cleaned.strip())

    for candidate in json_candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        # Try to find a JSON object in the candidate
        brace_start = candidate.find("{")
        if brace_start == -1:
            continue

        # Find the matching closing brace
        depth = 0
        for i, ch in enumerate(candidate[brace_start:], start=brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = candidate[brace_start : i + 1]
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            # {"tool_calls": [...]} wrapper
                            if "tool_calls" in parsed:
                                calls = _normalize_call_list(parsed["tool_calls"])
                                if calls:
                                    return calls
                            # Bare tool call object: {"name": "...", "arguments": {...}}
                            if "name" in parsed:
                                calls = _normalize_call_list([parsed])
                                if calls:
                                    return calls
                    except (json.JSONDecodeError, TypeError):
                        pass
                    continue  # try next brace pair in the candidate

    return None


def _extract_calls_from_fragments(fragments: list[str]) -> list[dict[str, Any]] | None:
    """Parse JSON tool call objects from text fragments.

    Each fragment may contain a single JSON object with "name" + "arguments",
    or a "function" key wrapping them (some models use this nesting).
    """
    calls: list[dict[str, Any]] = []
    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue
        parsed = None
        try:
            parsed = json.loads(frag)
        except (json.JSONDecodeError, TypeError):
            # Try to extract first JSON object from fragment
            brace = frag.find("{")
            if brace == -1:
                continue
            depth = 0
            for i, ch in enumerate(frag[brace:], start=brace):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(frag[brace : i + 1])
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break
            if parsed is None:
                continue

        if not isinstance(parsed, dict):
            continue

        call = _normalize_single_call(parsed)
        if call:
            calls.append(call)

    return calls if calls else None


def _extract_calls_from_array(text: str) -> list[dict[str, Any]] | None:
    """Parse a JSON array of tool call objects."""
    try:
        arr = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arr, list):
        return None
    return _normalize_call_list(arr)


def _normalize_call_list(raw: list) -> list[dict[str, Any]] | None:
    """Normalize a list of raw tool call dicts into [{name, arguments}, ...]."""
    calls: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            call = _normalize_single_call(item)
            if call:
                calls.append(call)
    return calls if calls else None


def _normalize_single_call(obj: dict) -> dict[str, Any] | None:
    """Normalize a single tool call dict.

    Handles variants:
    - {"name": "x", "arguments": {...}}
    - {"function": {"name": "x", "arguments": {...}}}
    - {"function": "x", "arguments": {...}}  (Llama-style)
    - {"name": "x", "parameters": {...}}
    """
    name = obj.get("name")
    arguments = obj.get("arguments") or obj.get("parameters") or {}

    # Nested "function" key (OpenAI-style wrapper)
    if not name and "function" in obj:
        func = obj["function"]
        if isinstance(func, dict):
            name = func.get("name")
            arguments = func.get("arguments") or func.get("parameters") or {}
        elif isinstance(func, str):
            # {"function": "read_file", "arguments": {...}}
            name = func

    if not name or not isinstance(name, str):
        return None

    return {"name": name, "arguments": arguments}


def _parse_step_complete_args(arguments: str | dict[str, Any]) -> StepResult:
    """Parse step_complete tool call arguments into a StepResult."""
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            args = {}
    else:
        args = arguments or {}

    # Parse next_steps into StepOperation objects
    raw_next_steps = args.get("next_steps", [])
    next_steps: list[StepOperation] = []
    if isinstance(raw_next_steps, list):
        for item in raw_next_steps:
            if isinstance(item, dict) and "op" in item and "index" in item:
                try:
                    next_steps.append(StepOperation(
                        op=item["op"],
                        index=item["index"],
                        description=item.get("description", ""),
                    ))
                except Exception:
                    pass

    return StepResult(
        summary=args.get("summary", "Step completed (no summary provided)"),
        status=args.get("status", "continue"),
        next_steps=next_steps,
        final_answer=args.get("final_answer"),
    )


def _call_llm(
    params: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] = "auto",
) -> Any:
    """Call litellm.completion with retry for transient errors.

    Adapts request parameters based on probed model capabilities:
    - Downgrades tool_choice="required" → "auto" if unsupported
    - Skips response_format=json_object if unsupported
    """
    import litellm
    from infinidev.config.model_capabilities import get_model_capabilities

    caps = get_model_capabilities()

    kwargs: dict[str, Any] = {**params, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        # Downgrade tool_choice if model doesn't support "required"
        if tool_choice == "required" and not caps.supports_tool_choice_required:
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tool_choice"] = tool_choice
    # Force JSON output — prevents models (especially Ollama) from mixing
    # natural language text into tool call arguments.
    # NOTE: Do NOT set response_format when tools are present — for llama-server
    # and similar backends, the JSON grammar constraint conflicts with the
    # function calling grammar, causing the model to intermittently return
    # JSON text instead of tool calls.
    if caps.supports_json_mode and not tools:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            return litellm.completion(**kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == _LLM_RETRIES:
                raise
            delay = _LLM_RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Transient LLM error (attempt %d/%d), retrying in %.1fs: %s",
                attempt, _LLM_RETRIES, delay, str(exc)[:200],
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _synthesize_final(state: LoopState) -> str:
    """Synthesize a final answer from accumulated history when loop exhausts iterations."""
    if not state.history:
        return "No actions were completed."

    parts = ["Task execution summary (iteration limit reached):"]
    for record in state.history:
        parts.append(f"- Step {record.step_index}: {record.summary}")
    return "\n".join(parts)


class LoopEngine(AgentEngine):
    """Plan-execute-summarize loop engine.

    Each iteration rebuilds the prompt from scratch with only:
    system prompt + task + plan + compact summaries + current step.
    Raw tool output exists only temporarily during a step, then is
    discarded and replaced with a ~50-token summary.
    """

    def execute(
        self,
        agent: Any,
        task_prompt: tuple[str, str],
        *,
        verbose: bool = True,
        guardrail: Any | None = None,
        guardrail_max_retries: int = 5,
        output_pydantic: type | None = None,
        task_tools: list | None = None,
        event_id: int | None = None,
        resume_state: dict | None = None,
    ) -> str:
        from infinidev.config.llm import get_litellm_params
        from infinidev.config.settings import settings

        llm_params = get_litellm_params()
        if llm_params is None:
            raise RuntimeError(
                "LoopEngine requires LiteLLM parameters. "
                "Ensure INFINIDEV_LLM_MODEL is set."
            )

        max_iterations = settings.LOOP_MAX_ITERATIONS
        max_total_calls = settings.LOOP_MAX_TOTAL_TOOL_CALLS
        max_per_action = settings.LOOP_MAX_TOOL_CALLS_PER_ACTION or max_total_calls
        history_window = settings.LOOP_HISTORY_WINDOW

        # Fetch model's context window for budget awareness
        max_context_tokens = _get_model_max_context(llm_params)

        # Resolve tools
        tools = task_tools if task_tools is not None else getattr(agent, "tools", [])
        if task_tools is not None:
            from infinidev.tools.base.context import bind_tools_to_agent
            bind_tools_to_agent(task_tools, agent.agent_id)

        tool_schemas = build_tool_schemas(tools) if tools else [STEP_COMPLETE_SCHEMA]
        tool_dispatch = build_tool_dispatch(tools) if tools else {}

        # File change tracker for this task
        file_tracker = FileChangeTracker()

        # Check model capabilities for manual tool calling mode
        from infinidev.config.model_capabilities import get_model_capabilities
        caps = get_model_capabilities()
        manual_tc = not caps.supports_function_calling

        # Build system prompt
        system_prompt = build_system_prompt(
            agent.backstory,
            tech_hints=getattr(agent, '_tech_hints', None),
            session_summaries=getattr(agent, '_session_summaries', None),
        )

        # For non-FC models, embed tool descriptions in the system prompt
        if manual_tc:
            tools_section = build_tools_prompt_section(tool_schemas)
            system_prompt = f"{system_prompt}\n\n{tools_section}"
            logger.info(
                "LoopEngine [%s]: manual tool calling mode (model lacks FC support)",
                getattr(agent, "agent_id", "?"),
            )

        desc, expected = task_prompt
        agent_name = getattr(agent, "name", agent.agent_id)
        agent_role = getattr(agent, "role", "agent")

        # Read event_id / resume_state from tool context if not passed directly
        if event_id is None or resume_state is None:
            from infinidev.tools.base.context import get_context_for_agent

            ctx = get_context_for_agent(agent.agent_id)
            if ctx:
                event_id = event_id or ctx.event_id
                resume_state = resume_state or ctx.resume_state

        # Resume from checkpoint or start fresh
        if resume_state:
            state = LoopState.model_validate(resume_state)
            if state.plan.steps and not state.plan.active_step:
                for s in state.plan.steps:
                    if s.status == "pending":
                        s.status = "active"
                        break
            logger.info("LoopEngine: resuming from iteration %d", state.iteration_count)
        else:
            state = LoopState()

        start_iteration = state.iteration_count

        if verbose:
            _log_start(agent.agent_id, agent_name, agent_role, desc, len(tools))

        consecutive_all_done = 0  # Safety: terminate after 2 consecutive all-done iterations

        # --- Outer loop (plan-level) ---
        for iteration in range(start_iteration, max_iterations):
            state.iteration_count = iteration + 1

            # Apply history window
            effective_state = state
            if history_window > 0 and len(state.history) > history_window:
                effective_state = state.model_copy(deep=True)
                effective_state.history = state.history[-history_window:]

            # Fetch project knowledge (only on first iteration to save DB calls)
            if iteration == start_iteration:
                try:
                    from infinidev.db.service import get_project_knowledge
                    _project_knowledge = get_project_knowledge(
                        project_id=agent.project_id,
                    )
                except Exception:
                    _project_knowledge = []

            user_prompt = build_iteration_prompt(
                desc, expected, effective_state,
                project_knowledge=_project_knowledge if iteration == start_iteration else None,
                max_context_tokens=max_context_tokens,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Log step start
            active = state.plan.active_step
            if active:
                active_desc = active.description
            elif not state.plan.steps:
                active_desc = "Planning..."
            else:
                # All plan steps done but loop continues — show last done step
                done_steps = [s for s in state.plan.steps if s.status == "done"]
                active_desc = f"Continuing ({done_steps[-1].description})" if done_steps else "Working..."
            if verbose:
                _log_step_start(iteration + 1, active_desc)
                # _log_prompt(user_prompt)  # Silenced for CLI

            # Emit step-start event so UI shows the current step immediately
            _emit_loop_event("loop_step_update", agent.project_id, agent.agent_id, {
                "agent_id": agent.agent_id,
                "agent_name": agent_name,
                "iteration": iteration + 1,
                "step_description": active_desc,
                "status": "active",
                "summary": "",
                "plan_steps": [
                    {"index": s.index, "description": s.description, "status": s.status}
                    for s in state.plan.steps
                ],
                "tool_calls_step": 0,
                "tool_calls_total": state.total_tool_calls,
                "tokens_total": state.total_tokens,
                "prompt_tokens": state.last_prompt_tokens,
                "completion_tokens": state.last_completion_tokens,
            })

            # --- Inner loop (function calling within one step) ---
            step_result: StepResult | None = None
            action_tool_calls = 0
            last_tool_sig: str | None = None  # Track consecutive identical calls (name:args)
            same_tool_streak = 0  # How many times in a row the same call was made
            repetition_nudged = False  # Whether we already nudged for repetition

            # Planning phase: only send step_complete schema (no agent tools)
            # so the LLM creates a plan instead of jumping into tool calls.
            is_planning = not state.plan.steps
            planning_schemas = [STEP_COMPLETE_SCHEMA]

            _malformed_retries = 0
            _MAX_MALFORMED_RETRIES = 2
            _text_retries = 0

            while action_tool_calls < max_per_action and state.total_tool_calls < max_total_calls:
                # ── LLM call: FC mode vs manual mode ─────────────────
                if manual_tc:
                    # Manual mode: no tools param, parse from text
                    response = _call_llm(llm_params, messages)

                    usage = getattr(response, "usage", None)
                    if usage:
                        state.total_tokens += getattr(usage, "total_tokens", 0)
                        state.last_prompt_tokens = getattr(usage, "prompt_tokens", 0)
                        state.last_completion_tokens = getattr(usage, "completion_tokens", 0)

                    choice = response.choices[0]
                    message = choice.message
                    raw_content = (message.content or "").strip()

                    # Parse tool calls from text
                    parsed_calls = _parse_text_tool_calls(raw_content)
                    if parsed_calls:
                        _malformed_retries = 0
                        _text_retries = 0
                        # Convert parsed dicts to a lightweight namespace for
                        # uniform handling below (same attrs as native TC objects)
                        tool_calls = []
                        for i, pc in enumerate(parsed_calls):
                            tc_obj = _ManualToolCall(
                                id=f"manual_{action_tool_calls + i}",
                                name=pc["name"],
                                arguments=(
                                    json.dumps(pc["arguments"])
                                    if isinstance(pc["arguments"], dict)
                                    else str(pc["arguments"])
                                ),
                            )
                            tool_calls.append(tc_obj)
                    else:
                        tool_calls = None
                        # No parsable tool calls — will be handled by text retry below
                else:
                    # FC mode: pass tools to litellm, read tool_calls from response
                    iter_tools = planning_schemas if is_planning else tool_schemas
                    try:
                        response = _call_llm(
                            llm_params,
                            messages,
                            iter_tools,
                            tool_choice="required",
                        )
                    except Exception as exc:
                        if _is_malformed_tool_call(exc):
                            _malformed_retries += 1
                            _emit_log(
                                "warning",
                                f"{_YELLOW}⚠ Malformed tool call from provider "
                                f"(attempt {_malformed_retries}/{_MAX_MALFORMED_RETRIES}): "
                                f"{str(exc)[:120]}{_RESET}",
                                project_id=agent.project_id, agent_id=agent.agent_id,
                            )
                            if _malformed_retries < _MAX_MALFORMED_RETRIES:
                                continue  # Retry — model output is stochastic
                            # Exhausted retries — degrade gracefully
                            _emit_log(
                                "error",
                                f"{_RED}⚠ Malformed tool calls persisted — forcing step completion{_RESET}",
                                project_id=agent.project_id, agent_id=agent.agent_id,
                            )
                            step_result = StepResult(
                                summary=(
                                    f"Step interrupted: LLM produced malformed tool calls "
                                    f"({_malformed_retries} attempts). Will retry on next step."
                                ),
                                status="continue",
                            )
                            break
                        # Check if this is a permanent tool/FC error from
                        # the provider (e.g. Ollama "tool 'X' not found").
                        # Fall back to manual tool calling instead of crashing.
                        exc_msg = str(exc).lower()
                        if any(p in exc_msg for p in _PERMANENT_ERRORS):
                            _emit_log(
                                "warning",
                                f"{_YELLOW}⚠ Provider rejected function calling: "
                                f"{str(exc)[:120]} — switching to manual tool calling{_RESET}",
                                project_id=agent.project_id, agent_id=agent.agent_id,
                            )
                            manual_tc = True
                            # Rebuild system prompt with embedded tool descriptions
                            tools_section = build_tools_prompt_section(tool_schemas)
                            system_prompt = build_system_prompt(
                                agent.backstory,
                                tech_hints=getattr(agent, '_tech_hints', None),
                                session_summaries=getattr(agent, '_session_summaries', None),
                            )
                            system_prompt = f"{system_prompt}\n\n{tools_section}"
                            messages[0] = {"role": "system", "content": system_prompt}
                            continue  # Retry this iteration in manual mode
                        raise  # Non-recoverable errors propagate normally

                    _malformed_retries = 0  # Reset on success

                    # Track token usage
                    usage = getattr(response, "usage", None)
                    if usage:
                        state.total_tokens += getattr(usage, "total_tokens", 0)
                        state.last_prompt_tokens = getattr(usage, "prompt_tokens", 0)
                        state.last_completion_tokens = getattr(usage, "completion_tokens", 0)

                    choice = response.choices[0]
                    message = choice.message
                    tool_calls = getattr(message, "tool_calls", None)

                # ── Process tool calls (unified for both modes) ──────
                if tool_calls:
                    _text_retries = 0  # Reset only when tool calls are present
                    # Separate step_complete from regular tool calls
                    regular_calls = []
                    sc_call = None
                    for tc in tool_calls:
                        if tc.function.name == "step_complete":
                            sc_call = tc
                        else:
                            regular_calls.append(tc)

                    # Execute regular tool calls first
                    if regular_calls:
                        if manual_tc:
                            # Manual mode: model doesn't understand tool_calls format.
                            # Append the raw assistant text, then tool results as user msg.
                            messages.append({
                                "role": "assistant",
                                "content": getattr(message, "content", "") or raw_content,
                            })
                        else:
                            # FC mode: structured tool_calls in assistant message
                            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
                            assistant_msg["tool_calls"] = [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in regular_calls
                            ]
                            # Include step_complete in the message if present (needed for API)
                            if sc_call:
                                assistant_msg["tool_calls"].append({
                                    "id": sc_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": sc_call.function.name,
                                        "arguments": sc_call.function.arguments,
                                    },
                                })
                            messages.append(assistant_msg)

                        # Collect tool results for both modes
                        tool_results_text: list[str] = []  # For manual mode
                        for tc in regular_calls:
                            # Pre-hook: capture file content before edit/write
                            _pre_content = _capture_pre_content(
                                tc.function.name, tc.function.arguments, file_tracker,
                            )

                            result = execute_tool_call(
                                tool_dispatch,
                                tc.function.name,
                                tc.function.arguments,
                            )

                            # Post-hook: record file change and emit diff event
                            _maybe_emit_file_change(
                                tc.function.name, tc.function.arguments, result,
                                _pre_content, file_tracker,
                                agent.project_id, agent.agent_id,
                            )

                            # Intercept send_message: emit to TUI chat
                            if tc.function.name == "send_message":
                                try:
                                    _msg_args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                                    _emit_loop_event("loop_user_message", agent.project_id, agent.agent_id, {
                                        "message": _msg_args.get("message", ""),
                                    })
                                except Exception:
                                    pass

                            # Detect errors / hallucinated tools and log visibly
                            _tool_error = _extract_tool_error(result)

                            if manual_tc:
                                tool_results_text.append(
                                    f"[Tool: {tc.function.name}] Result:\n{result}"
                                )
                            else:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": result,
                                })
                            action_tool_calls += 1
                            state.total_tool_calls += 1

                            tool_detail = _extract_tool_detail(tc.function.name, tc.function.arguments)

                            if verbose:
                                _log_tool(agent_name, iteration + 1, tc.function.name, action_tool_calls, state.total_tool_calls)
                                if tool_detail:
                                    _log(f"{_BLUE}│{_RESET}     {_DIM}{tool_detail}{_RESET}")
                                if _tool_error:
                                    _log(f"{_BLUE}│{_RESET}     {_RED}✗ {_tool_error}{_RESET}")

                            _emit_loop_event("loop_tool_call", agent.project_id, agent.agent_id, {
                                "agent_id": agent.agent_id,
                                "agent_name": agent_name,
                                "tool_name": tc.function.name,
                                "tool_detail": tool_detail,
                                "tool_error": _tool_error,
                                "call_num": action_tool_calls,
                                "total_calls": state.total_tool_calls,
                                "iteration": iteration + 1,
                                "tokens_total": state.total_tokens,
                                "prompt_tokens": state.last_prompt_tokens,
                                "completion_tokens": state.last_completion_tokens,
                            })

                        # Manual mode: send all tool results as a single user message
                        if manual_tc and tool_results_text:
                            messages.append({
                                "role": "user",
                                "content": "\n\n".join(tool_results_text),
                            })

                        # Track consecutive identical tool calls (same name + args)
                        # to detect loops. Different args = legitimate usage.
                        batch_tool = regular_calls[-1].function.name
                        batch_args = regular_calls[-1].function.arguments
                        batch_sig = f"{batch_tool}:{batch_args}"
                        if batch_sig == last_tool_sig:
                            same_tool_streak += 1
                        else:
                            last_tool_sig = batch_sig
                            same_tool_streak = 1
                            repetition_nudged = False

                        # Provide step_complete tool result if it was in this batch
                        if sc_call and not manual_tc:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": sc_call.id,
                                "content": '{"status": "acknowledged"}',
                            })

                    elif sc_call:
                        # Only step_complete, no regular tools
                        if manual_tc:
                            messages.append({
                                "role": "assistant",
                                "content": getattr(message, "content", "") or raw_content,
                            })
                        else:
                            assistant_msg = {"role": "assistant", "content": message.content or ""}
                            assistant_msg["tool_calls"] = [{
                                "id": sc_call.id,
                                "type": "function",
                                "function": {
                                    "name": sc_call.function.name,
                                    "arguments": sc_call.function.arguments,
                                },
                            }]
                            messages.append(assistant_msg)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": sc_call.id,
                                "content": '{"status": "acknowledged"}',
                            })

                    # If step_complete was called, parse it and break
                    if sc_call:
                        step_result = _parse_step_complete_args(sc_call.function.arguments)
                        break

                    # Detect identical tool call repetition — force step completion
                    # Extract tool name from signature for readable messages
                    _loop_tool = (last_tool_sig or "").split(":", 1)[0]
                    if same_tool_streak >= _MAX_SAME_TOOL_CONSECUTIVE and not repetition_nudged:
                        repetition_nudged = True
                        _emit_log(
                            "warning",
                            f"{_YELLOW}⚠ Identical '{_loop_tool}' call repeated "
                            f"{same_tool_streak}x — nudging step_complete{_RESET}",
                            project_id=agent.project_id, agent_id=agent.agent_id,
                        )
                        messages.append({
                            "role": "user",
                            "content": (
                                f"STOP: You have made the exact same '{_loop_tool}' call "
                                f"{same_tool_streak} times in a row with identical arguments. "
                                f"This is a loop. You MUST now call the step_complete "
                                f"tool to summarize what you've accomplished and move on."
                            ),
                        })
                        continue
                    if same_tool_streak >= _MAX_SAME_TOOL_CONSECUTIVE + 2:
                        # Nudge failed — force break with synthesized result
                        _emit_log(
                            "error",
                            f"{_RED}⚠ Tool loop detected: identical '{_loop_tool}' call "
                            f"{same_tool_streak}x — forcing step completion{_RESET}",
                            project_id=agent.project_id, agent_id=agent.agent_id,
                        )
                        step_result = StepResult(
                            summary=(
                                f"Step interrupted: identical {_loop_tool} calls "
                                f"({same_tool_streak}x) without progress."
                            ),
                            status="continue",
                        )
                        break

                else:
                    # LLM responded with text despite tool_choice="required".
                    # Retry with a corrective nudge before giving up.
                    content = (message.content or "").strip()
                    _text_retries += 1
                    logger.warning(
                        "LoopEngine [%s]: LLM returned text instead of tool call "
                        "(attempt %d/%d): %s",
                        agent.agent_id, _text_retries, _MAX_TEXT_RETRIES,
                        content[:200],
                    )

                    if _text_retries < _MAX_TEXT_RETRIES:
                        _emit_log(
                            "warning",
                            f"{_YELLOW}⚠ LLM returned text instead of tool call "
                            f"(retry {_text_retries}/{_MAX_TEXT_RETRIES}){_RESET}",
                            project_id=agent.project_id, agent_id=agent.agent_id,
                        )
                        # Append assistant text + corrective nudge so the model
                        # sees what it did wrong and tries again with a tool call.
                        truncated = content[:500] + ("..." if len(content) > 500 else "")
                        messages.append({"role": "assistant", "content": content})
                        if manual_tc:
                            nudge = (
                                "Your response could not be parsed as a tool call. "
                                "You MUST respond with a JSON object containing a "
                                '"tool_calls" array. Example:\n'
                                '{"tool_calls": [{"name": "step_complete", '
                                '"arguments": {"summary": "...", "status": "continue"}}]}\n\n'
                                f"Your response was: {truncated}"
                            )
                        else:
                            nudge = (
                                "You responded with text but you MUST call a tool. "
                                f"Your response was: {truncated}\n\n"
                                "Call the appropriate tool to proceed, or call "
                                "step_complete if you are done with this step."
                            )
                        messages.append({"role": "user", "content": nudge})
                        continue  # Retry the inner loop

                    # Exhausted retries — fall back to StepResult
                    _emit_log(
                        "error",
                        f"{_RED}⚠ LLM returned text {_text_retries}x despite "
                        f"tool_choice=required — forcing step completion{_RESET}",
                        project_id=agent.project_id, agent_id=agent.agent_id,
                    )
                    if len(content) > 200:
                        content = content[:197] + "..."
                    step_result = StepResult(
                        summary=content or "Step completed (provider ignored tool_choice=required).",
                        status="continue",
                    )
                    break
            else:
                # Inner loop exhausted — force step completion
                if step_result is None:
                    if state.total_tool_calls >= max_total_calls:
                        limit_msg = f"global tool call limit reached ({state.total_tool_calls}/{max_total_calls} total calls)"
                    else:
                        limit_msg = f"per-step tool call limit reached ({action_tool_calls}/{max_per_action} calls)"
                    step_result = StepResult(
                        summary=f"Step interrupted: {limit_msg}.",
                        status="continue",
                    )
                    _emit_log(
                        "error",
                        f"{_RED}⚠ Inner loop exhausted: {limit_msg}{_RESET}",
                        project_id=agent.project_id, agent_id=agent.agent_id,
                    )

            # Fallback if step_result is still None (shouldn't happen but be safe)
            if step_result is None:
                step_result = StepResult(summary="Step completed.", status="continue")

            # --- Plan management ---
            # If we don't have a plan yet, use next_steps from step_result to create one
            if not state.plan.steps:
                if step_result.next_steps:
                    state.plan.apply_operations(step_result.next_steps)
                # Activate the first step if we got a plan
                if state.plan.steps:
                    for s in state.plan.steps:
                        if s.status == "pending":
                            s.status = "active"
                            break
            else:
                # Existing plan: mark current step done, apply changes, activate next
                state.plan.mark_active_done()
                if step_result.next_steps:
                    state.plan.apply_operations(step_result.next_steps)
                state.plan.activate_next()

            step_index = state.plan.active_step.index if state.plan.active_step else iteration + 1
            # If we just marked active done and activated next, use the previous active step
            done_steps = [s for s in state.plan.steps if s.status == "done"]
            if done_steps:
                step_index = done_steps[-1].index

            state.history.append(ActionRecord(
                step_index=step_index,
                summary=step_result.summary,
                tool_calls_count=action_tool_calls,
            ))

            state.current_step_index = step_index

            if verbose:
                _log_step_done(iteration + 1, step_result.status, step_result.summary, action_tool_calls, state.total_tokens)
                _log_plan(state.plan)

            # Emit step-done update to UI
            done_steps = [s for s in state.plan.steps if s.status == "done"]
            if done_steps:
                done_desc = done_steps[-1].description
            elif step_result.summary:
                done_desc = step_result.summary[:120]
            else:
                done_desc = active_desc
            _emit_loop_event("loop_step_update", agent.project_id, agent.agent_id, {
                "agent_id": agent.agent_id,
                "agent_name": agent_name,
                "iteration": iteration + 1,
                "step_description": done_desc,
                "status": step_result.status,
                "summary": step_result.summary[:200],
                "plan_steps": [
                    {"index": s.index, "description": s.description, "status": s.status}
                    for s in state.plan.steps
                ],
                "tool_calls_step": action_tool_calls,
                "tool_calls_total": state.total_tool_calls,
                "tokens_total": state.total_tokens,
                "prompt_tokens": state.last_prompt_tokens,
                "completion_tokens": state.last_completion_tokens,
            })

            # Checkpoint for crash recovery
            if event_id:
                self._checkpoint(event_id, state)

            # --- Check termination ---
            if step_result.status == "done":
                # Guard: if the LLM said "done" but gave no final_answer
                # (only a short summary), it likely finished too early.
                # Force it to continue so it produces a real answer.
                if not step_result.final_answer and iteration == start_iteration:
                    _emit_log(
                        "warning",
                        f"{_YELLOW}⚠ LLM declared done on first step without final_answer "
                        f"— forcing continue{_RESET}",
                        project_id=agent.project_id, agent_id=agent.agent_id,
                    )
                    step_result = StepResult(
                        summary=step_result.summary,
                        status="continue",
                        next_steps=step_result.next_steps,
                    )
                else:
                    file_tracker.deactivate()
                    if verbose:
                        _log_finish(agent_name, "done", iteration + 1, state.total_tool_calls, state.total_tokens)
                    _emit_loop_event("loop_finished", agent.project_id, agent.agent_id, {
                        "agent_id": agent.agent_id, "agent_name": agent_name,
                        "status": "done", "iterations": iteration + 1,
                        "tool_calls_total": state.total_tool_calls, "tokens_total": state.total_tokens,
                    })
                    result = step_result.final_answer or step_result.summary
                    return self._apply_guardrail(
                        result, guardrail, guardrail_max_retries,
                        llm_params, system_prompt, desc, expected, state, tool_schemas, tool_dispatch,
                    )

            if step_result.status == "blocked":
                file_tracker.deactivate()
                if verbose:
                    _log_finish(agent_name, "blocked", iteration + 1, state.total_tool_calls, state.total_tokens)
                _emit_loop_event("loop_finished", agent.project_id, agent.agent_id, {
                    "agent_id": agent.agent_id, "agent_name": agent_name,
                    "status": "blocked", "iterations": iteration + 1,
                    "tool_calls_total": state.total_tool_calls, "tokens_total": state.total_tokens,
                })
                return step_result.summary

            # Safety: if all steps done and LLM didn't add new ones, allow
            # one more "planning" iteration to add steps or declare done.
            # After 2 consecutive all-done iterations, force terminate.
            if state.plan.steps and not state.plan.has_pending:
                consecutive_all_done += 1
                if consecutive_all_done >= 2:
                    file_tracker.deactivate()
                    if verbose:
                        _log_finish(agent_name, "done", iteration + 1, state.total_tool_calls, state.total_tokens)
                    _emit_loop_event("loop_finished", agent.project_id, agent.agent_id, {
                        "agent_id": agent.agent_id, "agent_name": agent_name,
                        "status": "done", "iterations": iteration + 1,
                        "tool_calls_total": state.total_tool_calls, "tokens_total": state.total_tokens,
                    })
                    result = step_result.summary
                    return self._apply_guardrail(
                        result, guardrail, guardrail_max_retries,
                        llm_params, system_prompt, desc, expected, state, tool_schemas, tool_dispatch,
                    )
            else:
                consecutive_all_done = 0

        # Outer loop exhausted
        file_tracker.deactivate()
        if verbose:
            _log_finish(agent_name, "exhausted", max_iterations, state.total_tool_calls, state.total_tokens)
        _emit_loop_event("loop_finished", agent.project_id, agent.agent_id, {
            "agent_id": agent.agent_id, "agent_name": agent_name,
            "status": "exhausted", "iterations": max_iterations,
            "tool_calls_total": state.total_tool_calls, "tokens_total": state.total_tokens,
        })
        return _synthesize_final(state)

    def _checkpoint(self, event_id: int, state: LoopState) -> None:
        """No-op in CLI mode."""
        pass

    def _apply_guardrail(
        self,
        result: str,
        guardrail: Any | None,
        max_retries: int,
        llm_params: dict[str, Any],
        system_prompt: str,
        desc: str,
        expected: str,
        state: LoopState,
        tool_schemas: list[dict[str, Any]],
        tool_dispatch: dict[str, Any],
    ) -> str:
        """Validate result with guardrail; retry with feedback if it fails."""
        if guardrail is None:
            return result

        from infinidev.config.settings import settings

        max_per_action = settings.LOOP_MAX_TOOL_CALLS_PER_ACTION

        for attempt in range(max_retries):
            try:
                validation = guardrail(result)
                # CrewAI guardrail convention: returns (success, result_or_feedback)
                if isinstance(validation, tuple):
                    success, feedback = validation
                    if success:
                        return result
                    # Retry with feedback
                    logger.info(
                        "Guardrail failed (attempt %d/%d): %s",
                        attempt + 1, max_retries, str(feedback)[:200],
                    )
                    feedback_prompt = (
                        f"Your previous output was rejected by validation.\n"
                        f"Feedback: {feedback}\n\n"
                        f"Please fix your output and try again.\n\n"
                        f"Previous output:\n{result}"
                    )
                    messages: list[dict[str, Any]] = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": feedback_prompt},
                    ]

                    # Run inner loop for retry
                    step_text = ""
                    action_tool_calls = 0
                    while action_tool_calls < max_per_action:
                        response = _call_llm(
                            llm_params, messages,
                            tool_schemas if tool_schemas else None,
                        )
                        choice = response.choices[0]
                        msg = choice.message
                        tc_list = getattr(msg, "tool_calls", None)
                        if tc_list:
                            assistant_msg: dict[str, Any] = {
                                "role": "assistant",
                                "content": msg.content or "",
                            }
                            assistant_msg["tool_calls"] = [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tc_list
                            ]
                            messages.append(assistant_msg)
                            for tc in tc_list:
                                if tc.function.name == "step_complete":
                                    # Parse final answer from step_complete
                                    sr = _parse_step_complete_args(tc.function.arguments)
                                    step_text = sr.final_answer or sr.summary
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": '{"status": "acknowledged"}',
                                    })
                                    break
                                _pre_content_g = _capture_pre_content(
                                    tc.function.name, tc.function.arguments, file_tracker,
                                )
                                tc_result = execute_tool_call(
                                    tool_dispatch,
                                    tc.function.name,
                                    tc.function.arguments,
                                )
                                _maybe_emit_file_change(
                                    tc.function.name, tc.function.arguments, tc_result,
                                    _pre_content_g, file_tracker,
                                    agent.project_id, agent.agent_id,
                                )
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tc_result,
                                })
                                action_tool_calls += 1
                            if step_text:
                                break
                        else:
                            step_text = msg.content or ""
                            break

                    result = step_text or result
                else:
                    # Simple bool guardrail
                    if validation:
                        return result
            except Exception as exc:
                logger.warning("Guardrail raised exception: %s", exc)

        return result
