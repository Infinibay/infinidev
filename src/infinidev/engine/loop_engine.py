"""Plan-execute-summarize loop engine.

Replaces the opaque CrewAI ReAct loop with a controlled cycle:
each iteration rebuilds the prompt from scratch with only system prompt +
task + plan + compact summaries of previous actions + current step.
Raw tool output is discarded after each step; only ~50-token summaries survive.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from infinidev.engine.base import AgentEngine
from infinidev.engine.hooks import hook_manager as _hook_manager, HookContext as _HookContext, HookEvent as _HookEvent
from infinidev.engine.llm_client import (
    call_llm as _call_llm,
    is_malformed_tool_call as _is_malformed_tool_call,
    is_transient as _is_transient,
    LLM_RETRIES as _LLM_RETRIES,
    LLM_RETRY_DELAY as _LLM_RETRY_DELAY,
    MALFORMED_TOOL_PATTERNS as _MALFORMED_TOOL_PATTERNS,
    PERMANENT_ERRORS as _PERMANENT_ERRORS,
)
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
    ADD_NOTE_SCHEMA,
    STEP_COMPLETE_SCHEMA,
    build_tool_dispatch,
    build_tool_schemas,
    execute_tool_call,
)
from infinidev.engine.tool_call_parser import (
    safe_json_loads as _safe_json_loads,
    ManualToolCall as _ManualToolCall,
    parse_text_tool_calls as _parse_text_tool_calls,
    parse_step_complete_args as _parse_step_complete_args,
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

# Max times the LLM can respond with text instead of tool calls before
# forcing a step_complete.  Text responses are kept as context (the model
# may be reasoning), so a higher limit is fine.
_MAX_TEXT_RETRIES = 5

logger = logging.getLogger(__name__)



# Logging helpers — imported from canonical module
from infinidev.engine.engine_logging import (
    emit_loop_event as _emit_loop_event,
    log as _log,
    emit_log as _emit_log,
    extract_tool_detail as _extract_tool_detail,
    extract_tool_error as _extract_tool_error,
    extract_tool_output_preview as _extract_tool_output_preview,
    log_start as _log_start,
    log_step_start as _log_step_start,
    log_tool as _log_tool,
    log_step_done as _log_step_done,
    log_plan as _log_plan,
    log_prompt as _log_prompt,
    log_finish as _log_finish,
    DIM as _DIM,
    BOLD as _BOLD,
    RESET as _RESET,
    CYAN as _CYAN,
    GREEN as _GREEN,
    YELLOW as _YELLOW,
    RED as _RED,
    MAGENTA as _MAGENTA,
    BLUE as _BLUE,
    STATUS_ICON as _STATUS_ICON,
    TOOL_DETAIL_KEYS as _TOOL_DETAIL_KEYS,
)

def _reindex_if_enabled(file_path: str) -> None:
    """Trigger incremental reindex of a file after it's been modified."""
    try:
        from infinidev.config.settings import settings
        if settings.CODE_INTEL_ENABLED and settings.CODE_INTEL_AUTO_INDEX:
            from infinidev.code_intel.indexer import reindex_file
            reindex_file(1, file_path)  # project_id=1 (default)
    except Exception:
        pass  # Never block the main loop for indexing


def _update_opened_files_cache(
    state: LoopState,
    tool_name: str,
    arguments: str | dict,
    result: str,
) -> None:
    """Update the opened files cache based on tool calls.

    - read_file: cache the returned content
    - write_file: cache the written content (from arguments)
    - edit_file: re-read the file and update cache
    """
    import os as _os

    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(args, dict):
        return

    path = args.get("path")
    if not path:
        return

    # Resolve to absolute path for consistent cache keys
    from infinidev.tools.base.context import get_current_workspace_path
    ws = get_current_workspace_path() or _os.getcwd()
    if not _os.path.isabs(path):
        path = _os.path.normpath(_os.path.join(ws, path))

    if tool_name == "read_file":
        # result is the file content (or JSON error)
        if result and not result.strip().startswith('{"error'):
            state.cache_file(path, result)

    elif tool_name == "write_file":
        # Content was in the arguments
        content = args.get("content", "")
        if content:
            state.refresh_file(path, content)
        _reindex_if_enabled(path)

    elif tool_name in ("edit_file", "multi_edit_file"):
        # After a successful edit, re-read the file to get updated content
        try:
            if _os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                state.refresh_file(path, content)
        except Exception:
            pass
        _reindex_if_enabled(path)

    elif tool_name == "apply_patch":
        # After applying a patch, re-read all modified files into cache
        try:
            res = json.loads(result) if isinstance(result, str) else result
            if isinstance(res, dict) and "files_modified" in res:
                ws = get_current_workspace_path() or _os.getcwd()
                for fpath in res["files_modified"]:
                    abs_path = _os.path.join(ws, fpath) if not _os.path.isabs(fpath) else fpath
                    if _os.path.isfile(abs_path):
                        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                            state.refresh_file(abs_path, f.read())
        except Exception:
            pass

    elif tool_name == "list_directory":
        # Cache directory listing so the model doesn't re-list
        if result and not result.strip().startswith('{"error'):
            state.cache_file(f"[dir] {path}", result)

    elif tool_name == "glob":
        pattern = args.get("pattern", "")
        if pattern and result and not result.strip().startswith('{"error'):
            state.cache_file(f"[glob] {pattern}", result)

    elif tool_name == "code_search":
        query = args.get("query") or args.get("pattern") or args.get("search_query", "")
        if query and result and not result.strip().startswith('{"error'):
            state.cache_file(f"[search] {query}", result)


# ── File change tracking helpers ────────────────────────────────────────────

_FILE_CHANGE_TOOLS = {
    "edit_file", "write_file", "multi_edit_file", "apply_patch",
    "create_file", "replace_lines",
    "edit_symbol", "add_symbol", "remove_symbol",
}

# Tools that modify state — these act as barriers in parallel execution.
# All read-only tools before a write are executed in parallel, then the write runs alone.
_WRITE_TOOLS = {
    "edit_file", "write_file", "multi_edit_file", "apply_patch",
    "create_file", "replace_lines",
    "edit_symbol", "add_symbol", "remove_symbol",
    "git_commit", "git_branch", "git_push",
    "execute_command",  # Commands can have side effects
    "record_finding", "update_finding", "delete_finding",
    "write_report", "delete_report",
    "update_documentation", "delete_documentation",
    "send_message",
}


def _batch_tool_calls(calls: list) -> list[list]:
    """Group tool calls into batches for parallel/sequential execution.

    Consecutive read-only tools are grouped together (run in parallel).
    Write tools each get their own single-item batch (run sequentially).

    Example: [r, r, r, w, r, r, w, w, r, r]
    → [[r, r, r], [w], [r, r], [w], [w], [r, r]]
    """
    batches: list[list] = []
    current_reads: list = []

    for tc in calls:
        name = tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", "")
        if name in _WRITE_TOOLS:
            # Flush accumulated reads as a parallel batch
            if current_reads:
                batches.append(current_reads)
                current_reads = []
            # Write is its own batch (sequential)
            batches.append([tc])
        else:
            current_reads.append(tc)

    # Flush remaining reads
    if current_reads:
        batches.append(current_reads)

    return batches


def _execute_tool_calls_parallel(
    batch: list,
    tool_dispatch: dict,
    hook_metadata: dict[str, Any] | None = None,
) -> list[tuple]:
    """Execute a batch of read-only tool calls in parallel.

    Returns list of (tc, result) tuples in original order.
    """
    if len(batch) <= 1:
        # No parallelism needed
        results = []
        for tc in batch:
            result = execute_tool_call(
                tool_dispatch, tc.function.name, tc.function.arguments,
                hook_metadata=hook_metadata,
            )
            results.append((tc, result))
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _exec(tc):
        result = execute_tool_call(
            tool_dispatch, tc.function.name, tc.function.arguments,
            hook_metadata=hook_metadata,
        )
        return (tc, result)

    results = []
    with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as pool:
        futures = {pool.submit(_exec, tc): i for i, tc in enumerate(batch)}
        indexed_results = [None] * len(batch)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                indexed_results[idx] = future.result()
            except Exception as exc:
                tc = batch[idx]
                indexed_results[idx] = (tc, json.dumps({"error": f"Parallel execution failed: {exc}"}))
        results = [r for r in indexed_results if r is not None]

    return results
_MAX_TRACK_FILE_SIZE = 1_000_000  # 1 MB — skip tracking larger files


def _extract_file_path_from_args(tool_name: str, arguments: str | dict) -> str | None:
    """Extract the file path from tool arguments."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(args, dict):
        return None
    return args.get("file_path") or args.get("path")


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


def _extract_reason_from_args(arguments: str | dict) -> str:
    """Extract the reason/description from tool call arguments."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(args, dict):
        return ""
    return args.get("reason") or args.get("description") or ""


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

    # Record reason if provided
    reason = _extract_reason_from_args(arguments)
    if reason:
        tracker.record_reason(file_path, reason)

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


# Error classification constants and functions are in llm_client.py
# Imported at top: _is_transient, _is_malformed_tool_call, _PERMANENT_ERRORS, etc.



# _ManualToolCall, _parse_text_tool_calls, _parse_step_complete_args:
# imported from tool_call_parser at top of file


# _call_llm is imported from llm_client.py at top of file


_SUMMARIZER_SYSTEM_PROMPT = """\
You are a step summarizer for a coding agent. Analyze the raw step data and produce a structured JSON summary.
The summary helps the agent remember what happened and plan the next step effectively.

Output EXACTLY this JSON format (no markdown, no code fences, just JSON):
{
  "files_to_preload": ["path1", "path2"],
  "changes_made": "Files modified: what changed and why. Include brief diffs if possible.",
  "discovered": "Relevant classes, files, function signatures, architecture patterns, web content, command results found.",
  "pending": "What still needs doing: code to fix/implement, problems found, things to investigate.",
  "anti_patterns": "What went wrong or was wasteful. Failed approaches, dead ends, repeated errors that should NOT be repeated.",
  "summary": "1-2 sentences: what was done, how, and why."
}

Rules:
- files_to_preload: ONLY files the NEXT step will need to read/edit. Max 5 paths.
- Keep each text field under 150 tokens. Focus on FACTS, not narration.
- anti_patterns: Look for repeated failed tool calls, re-reading same files, loops without progress.
- If no anti-patterns were observed, set it to empty string.
"""


def _summarize_step(
    messages: list[dict],
    task_description: str,
    state: LoopState,
    step_result: "StepResult",
    llm_params: dict,
) -> dict:
    """Make a dedicated LLM call to produce a structured step summary.

    Returns a dict with keys: summary, files_to_preload, changes_made,
    discovered, pending, anti_patterns. Falls back to step_result.summary
    on any error.
    """
    fallback = {
        "summary": step_result.summary,
        "files_to_preload": [],
        "changes_made": "",
        "discovered": "",
        "pending": "",
        "anti_patterns": "",
    }

    # Build the user prompt with raw step data
    parts = [f"<task>\n{task_description}\n</task>"]

    # Current plan state
    plan_text = state.plan.render() if state.plan.steps else "No plan yet."
    parts.append(f"<plan>\n{plan_text}\n</plan>")

    # Next pending steps
    next_pending = [s for s in state.plan.steps if s.status == "pending"]
    if next_pending:
        next_lines = [f"- {s.description}" for s in next_pending[:5]]
        parts.append(f"<next-steps>\n{chr(10).join(next_lines)}\n</next-steps>")

    # Previous summaries for context
    if state.history:
        prev = [f"- Step {r.step_index}: {r.summary}" for r in state.history[-3:]]
        parts.append(f"<previous-summaries>\n{chr(10).join(prev)}\n</previous-summaries>")

    # Raw step messages (truncated)
    max_input = settings.LOOP_SUMMARIZER_MAX_INPUT_TOKENS
    step_msgs_text = _truncate_step_messages(messages, max_input)
    parts.append(f"<step-messages>\n{step_msgs_text}\n</step-messages>")

    user_prompt = "\n\n".join(parts)

    # Make the summarizer LLM call (no tools)
    summarizer_messages = [
        {"role": "system", "content": _SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        import litellm
        summarizer_params = {k: v for k, v in llm_params.items() if k != "tool_choice"}
        summarizer_params.pop("tools", None)
        response = litellm.completion(
            **summarizer_params,
            messages=summarizer_messages,
            max_tokens=500,
        )
        content = response.choices[0].message.content or ""

        # Try to parse as JSON
        import json as _json
        # Strip markdown code fences if present
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()

        parsed = _safe_json_loads(clean)
        if isinstance(parsed, dict):
            return {
                "summary": str(parsed.get("summary", step_result.summary))[:500],
                "files_to_preload": list(parsed.get("files_to_preload", []))[:5],
                "changes_made": str(parsed.get("changes_made", ""))[:500],
                "discovered": str(parsed.get("discovered", ""))[:500],
                "pending": str(parsed.get("pending", ""))[:500],
                "anti_patterns": str(parsed.get("anti_patterns", ""))[:500],
            }
    except Exception as exc:
        logger.debug("Summarizer call failed, using fallback: %s", str(exc)[:200])

    return fallback


def _truncate_step_messages(messages: list[dict], max_tokens: int) -> str:
    """Truncate step messages to fit within a token budget.

    Keeps tool call names and arguments, truncates tool results.
    Rough estimate: 1 token ≈ 4 chars.
    """
    max_chars = max_tokens * 4
    parts = []
    total_chars = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue  # Skip system prompt (already in summarizer context)

        if role == "tool":
            # Truncate tool results to 500 chars each
            tool_id = msg.get("tool_call_id", "")
            truncated = content[:500] + ("..." if len(content) > 500 else "")
            line = f"[Tool result {tool_id}]: {truncated}"
        elif role == "assistant":
            # Keep tool call info
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_lines = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = str(fn.get("arguments", ""))[:300]
                    tc_lines.append(f"  → {name}({args})")
                line = "Assistant tool calls:\n" + "\n".join(tc_lines)
            else:
                line = f"Assistant: {content[:500]}"
        elif role == "user":
            line = f"User: {content[:500]}"
        else:
            line = f"{role}: {content[:300]}"

        line_len = len(line)
        if total_chars + line_len > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                parts.append(line[:remaining] + "...[truncated]")
            break
        parts.append(line)
        total_chars += line_len

    return "\n".join(parts)


def _synthesize_final(state: LoopState) -> str:
    """Synthesize a final answer from accumulated history when loop exhausts iterations."""
    if not state.history:
        return "No actions were completed."

    parts = ["Task execution summary (iteration limit reached):"]
    for record in state.history:
        parts.append(f"- Step {record.step_index}: {record.summary}")
    return "\n".join(parts)


# ── Composition components for execute() ────────────────────────────────────


@dataclass
class ExecutionContext:
    """All shared state for a single execute() invocation.

    Replaces ~20 local variables that were threaded through the old
    monolithic execute() method. Components read config fields and
    mutate ``state`` / ``file_tracker`` as needed.
    """

    # Config (immutable after setup)
    llm_params: dict[str, Any]
    manual_tc: bool
    is_small: bool
    system_prompt: str
    tool_schemas: list[dict[str, Any]]
    tool_dispatch: dict[str, Any]
    planning_schemas: list[dict[str, Any]]
    tools: list[Any]
    max_iterations: int
    max_per_action: int
    max_total_calls: int
    history_window: int
    max_context_tokens: int
    verbose: bool
    guardrail: Any | None
    guardrail_max_retries: int
    output_pydantic: type | None

    # Agent identity
    agent: Any
    agent_name: str
    agent_role: str
    desc: str
    expected: str
    event_id: int | None

    # Mutable state
    state: LoopState
    file_tracker: FileChangeTracker
    start_iteration: int

    @property
    def project_id(self) -> int:
        return self.agent.project_id

    @property
    def agent_id(self) -> str:
        return self.agent.agent_id


@dataclass
class LLMCallResult:
    """Result of a single LLM call from :class:`LLMCaller`."""

    tool_calls: list[Any] | None = None
    message: Any = None            # Raw LLM message object
    raw_content: str = ""
    reasoning_content: str = ""
    forced_step_result: StepResult | None = None   # Set when retries exhausted
    should_retry: bool = False                      # True on FC→manual switch


@dataclass
class ClassifiedCalls:
    """Tool calls separated by category."""

    regular: list[Any] = field(default_factory=list)
    step_complete: Any | None = None
    notes: list[Any] = field(default_factory=list)
    session_notes: list[Any] = field(default_factory=list)
    thinks: list[Any] = field(default_factory=list)


class LLMCaller:
    """Encapsulates LLM calling with manual-TC / FC-mode branching and retry."""

    def __init__(self) -> None:
        self._malformed_retries = 0
        self._MAX_MALFORMED_RETRIES = 4

    def reset(self) -> None:
        """Reset per-inner-loop counters."""
        self._malformed_retries = 0

    def call(
        self,
        ctx: ExecutionContext,
        messages: list[dict[str, Any]],
        is_planning: bool,
        action_tool_calls: int = 0,
    ) -> LLMCallResult:
        """Make one LLM call and return a parsed result.

        Handles manual-TC vs FC mode, parse-error retries, malformed
        tool-call retries, and FC→manual fallback on permanent errors.
        """
        if ctx.manual_tc:
            return self._call_manual(ctx, messages, action_tool_calls)
        return self._call_fc(ctx, messages, is_planning, action_tool_calls)

    # ── Manual TC mode ──────────────────────────────────────────────

    def _call_manual(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        action_tool_calls: int,
    ) -> LLMCallResult:
        _MANUAL_PARSE_RETRIES = 3
        response = None
        for attempt in range(1, _MANUAL_PARSE_RETRIES + 1):
            try:
                response = _call_llm(ctx.llm_params, messages)
                break
            except Exception as exc:
                msg = str(exc).lower()
                is_parse_error = (
                    "failed to parse" in msg or "internal server error" in msg
                )
                if is_parse_error and attempt < _MANUAL_PARSE_RETRIES:
                    _emit_log(
                        "warning",
                        f"{_YELLOW}⚠ Server parse error (attempt "
                        f"{attempt}/{_MANUAL_PARSE_RETRIES}), retrying...{_RESET}",
                        project_id=ctx.project_id, agent_id=ctx.agent_id,
                    )
                    time.sleep(1.0 * attempt)
                    continue
                raise

        self._track_usage(ctx, response)
        choice = response.choices[0]
        message = choice.message
        raw_content = (message.content or "").strip()
        reasoning_content = (getattr(message, "reasoning_content", None) or "").strip()

        # Parse tool calls from text
        parsed = _parse_text_tool_calls(raw_content)
        if not parsed and reasoning_content:
            parsed = _parse_text_tool_calls(reasoning_content)
        if not parsed and raw_content and reasoning_content:
            parsed = _parse_text_tool_calls(reasoning_content + "\n" + raw_content)

        if parsed:
            self._malformed_retries = 0
            tool_calls = [
                _ManualToolCall(
                    id=f"manual_{action_tool_calls + i}",
                    name=pc["name"],
                    arguments=(
                        json.dumps(pc["arguments"])
                        if isinstance(pc["arguments"], dict)
                        else str(pc["arguments"])
                    ),
                )
                for i, pc in enumerate(parsed)
            ]
            return LLMCallResult(
                tool_calls=tool_calls, message=message,
                raw_content=raw_content, reasoning_content=reasoning_content,
            )

        return LLMCallResult(
            message=message, raw_content=raw_content,
            reasoning_content=reasoning_content,
        )

    # ── FC mode ─────────────────────────────────────────────────────

    def _call_fc(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        is_planning: bool, action_tool_calls: int,
    ) -> LLMCallResult:
        iter_tools = ctx.planning_schemas if is_planning else ctx.tool_schemas
        try:
            response = _call_llm(ctx.llm_params, messages, iter_tools, tool_choice="required")
        except Exception as exc:
            return self._handle_fc_error(ctx, exc, messages)

        self._malformed_retries = 0
        self._track_usage(ctx, response)
        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        # FC mode fallback: some models return tool calls as tags in content
        if not tool_calls:
            raw_content = (getattr(message, "content", None) or "").strip()
            if not raw_content:
                raw_content = (getattr(message, "reasoning_content", None) or "").strip()
            if raw_content:
                parsed = _parse_text_tool_calls(raw_content)
                if parsed:
                    tool_calls = [
                        _ManualToolCall(
                            id=f"fc_fallback_{action_tool_calls + i}",
                            name=pc["name"],
                            arguments=(
                                json.dumps(pc["arguments"])
                                if isinstance(pc["arguments"], dict)
                                else str(pc["arguments"])
                            ),
                        )
                        for i, pc in enumerate(parsed)
                    ]

        raw = (getattr(message, "content", None) or "").strip()
        return LLMCallResult(
            tool_calls=tool_calls, message=message, raw_content=raw,
            reasoning_content=(getattr(message, "reasoning_content", None) or "").strip(),
        )

    def _handle_fc_error(
        self, ctx: ExecutionContext, exc: Exception, messages: list[dict[str, Any]],
    ) -> LLMCallResult:
        """Handle FC mode exceptions: malformed retries, permanent error fallback."""
        if _is_malformed_tool_call(exc):
            self._malformed_retries += 1
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ Malformed tool call from provider "
                f"(attempt {self._malformed_retries}/{self._MAX_MALFORMED_RETRIES}): "
                f"{str(exc)[:120]}{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            if self._malformed_retries < self._MAX_MALFORMED_RETRIES:
                return LLMCallResult(should_retry=True)
            _emit_log(
                "error",
                f"{_RED}⚠ Malformed tool calls persisted — forcing step completion{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            return LLMCallResult(forced_step_result=StepResult(
                summary=(
                    f"Step interrupted: LLM produced malformed tool calls "
                    f"({self._malformed_retries} attempts). Will retry on next step."
                ),
                status="continue",
            ))

        exc_msg = str(exc).lower()
        if any(p in exc_msg for p in _PERMANENT_ERRORS):
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ Provider rejected function calling: "
                f"{str(exc)[:120]} — switching to manual tool calling{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            ctx.manual_tc = True
            tools_section = build_tools_prompt_section(ctx.tool_schemas)
            ctx.system_prompt = build_system_prompt(
                ctx.agent.backstory,
                tech_hints=getattr(ctx.agent, '_tech_hints', None),
                session_summaries=getattr(ctx.agent, '_session_summaries', None),
                identity_override=getattr(ctx.agent, '_system_prompt_identity', None),
            )
            ctx.system_prompt = f"{ctx.system_prompt}\n\n{tools_section}"
            messages[0] = {"role": "system", "content": ctx.system_prompt}
            return LLMCallResult(should_retry=True)

        raise exc  # Non-recoverable

    @staticmethod
    def _track_usage(ctx: ExecutionContext, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            ctx.state.total_tokens += getattr(usage, "total_tokens", 0)
            ctx.state.last_prompt_tokens = getattr(usage, "prompt_tokens", 0)
            ctx.state.last_completion_tokens = getattr(usage, "completion_tokens", 0)


class ToolProcessor:
    """Classifies tool calls and orchestrates execution + message building."""

    @staticmethod
    def classify(tool_calls: list[Any]) -> ClassifiedCalls:
        """Separate tool calls into categories."""
        result = ClassifiedCalls()
        for tc in tool_calls:
            name = tc.function.name
            if name == "step_complete":
                result.step_complete = tc
            elif name == "add_note":
                result.notes.append(tc)
            elif name == "add_session_note":
                result.session_notes.append(tc)
            elif name == "think":
                result.thinks.append(tc)
            else:
                result.regular.append(tc)
        return result

    @staticmethod
    def process_pseudo_tools(
        ctx: ExecutionContext, classified: ClassifiedCalls,
        engine: "LoopEngine",
    ) -> None:
        """Handle think, add_note, add_session_note calls."""
        _MAX_NOTES = 20
        _MAX_SESSION_NOTES = 10

        for tk in classified.thinks:
            try:
                tk_args = _safe_json_loads(tk.function.arguments) if isinstance(tk.function.arguments, str) else (tk.function.arguments or {})
                reasoning = tk_args.get("reasoning", "").strip()
                if reasoning:
                    _hook_manager.dispatch(_HookContext(
                        event=_HookEvent.POST_TOOL,
                        tool_name="think",
                        arguments=tk_args,
                        result=reasoning,
                        project_id=ctx.project_id,
                        agent_id=ctx.agent_id,
                    ))
            except (json.JSONDecodeError, AttributeError):
                pass

        for nc in classified.notes:
            try:
                nc_args = _safe_json_loads(nc.function.arguments) if isinstance(nc.function.arguments, str) else (nc.function.arguments or {})
                note_text = nc_args.get("note", "").strip()
                if note_text and len(ctx.state.notes) < _MAX_NOTES:
                    ctx.state.notes.append(note_text)
                    ctx.state.tool_calls_since_last_note = 0
            except (json.JSONDecodeError, AttributeError):
                pass

        for snc in classified.session_notes:
            try:
                snc_args = _safe_json_loads(snc.function.arguments) if isinstance(snc.function.arguments, str) else (snc.function.arguments or {})
                note_text = snc_args.get("note", "").strip()
                if note_text and len(engine.session_notes) < _MAX_SESSION_NOTES:
                    engine.session_notes.append(note_text)
            except (json.JSONDecodeError, AttributeError):
                pass


class LoopGuard:
    """Detects repetition loops, error cascades, and budget exhaustion."""

    def __init__(self, is_small: bool = False) -> None:
        self._is_small = is_small
        self.reset()

    def reset(self) -> None:
        self.text_retries = 0
        self.consecutive_tool_errors = 0
        self.last_tool_sig: str | None = None
        self.same_tool_streak = 0
        self.repetition_nudged = False

    def on_tool_result(self, tool_name: str, args: str, had_error: bool) -> None:
        """Track a tool call for repetition/error detection."""
        if had_error:
            self.consecutive_tool_errors += 1
        else:
            self.consecutive_tool_errors = 0

        sig = f"{tool_name}:{args}"
        if sig == self.last_tool_sig:
            self.same_tool_streak += 1
        else:
            self.last_tool_sig = sig
            self.same_tool_streak = 1
            self.repetition_nudged = False

    def check_repetition(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
    ) -> StepResult | None:
        """Returns StepResult if loop detected and must force-break, else None."""
        threshold = 2 if self._is_small else _MAX_SAME_TOOL_CONSECUTIVE
        tool_name = (self.last_tool_sig or "").split(":", 1)[0]

        if self.same_tool_streak >= threshold and not self.repetition_nudged:
            self.repetition_nudged = True
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ Identical '{tool_name}' call repeated "
                f"{self.same_tool_streak}x — nudging step_complete{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            messages.append({
                "role": "user",
                "content": (
                    f"STOP: You have made the exact same '{tool_name}' call "
                    f"{self.same_tool_streak} times in a row with identical arguments. "
                    f"This is a loop. You MUST now call the step_complete "
                    f"tool to summarize what you've accomplished and move on."
                ),
            })
            return None  # nudged, not forced — caller should continue

        if self.same_tool_streak >= threshold + 2:
            _emit_log(
                "error",
                f"{_RED}⚠ Tool loop detected: identical '{tool_name}' call "
                f"{self.same_tool_streak}x — forcing step completion{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            return StepResult(
                summary=f"Step interrupted: identical {tool_name} calls ({self.same_tool_streak}x) without progress.",
                status="continue",
            )
        return None

    def check_error_circuit_breaker(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
    ) -> None:
        """Append nudge if too many consecutive tool errors."""
        _MAX = 4
        if self.consecutive_tool_errors >= _MAX:
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ {self.consecutive_tool_errors} consecutive tool errors "
                f"— nudging model to try a different approach{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            self.consecutive_tool_errors = 0
            messages.append({
                "role": "user",
                "content": (
                    f"WARNING: Your last {_MAX} tool calls all failed. "
                    "You are stuck in a failing pattern. Change your approach:\n"
                    "- If edit_file keeps failing, use write_file to rewrite the entire file.\n"
                    "- If read_file keeps failing on a path, use list_directory to find the correct path.\n"
                    "- If nothing works, call step_complete to move on and revisit later."
                ),
            })

    def handle_text_only(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        content: str,
    ) -> StepResult | None:
        """Handle LLM text response without tool calls.

        Returns StepResult if retries exhausted, None to continue inner loop.
        """
        self.text_retries += 1

        if self.text_retries < _MAX_TEXT_RETRIES:
            if content:
                _hook_manager.dispatch(_HookContext(
                    event=_HookEvent.POST_TOOL,
                    tool_name="think",
                    arguments={"reasoning": content},
                    result=content,
                    project_id=ctx.project_id, agent_id=ctx.agent_id,
                ))
            messages.append({"role": "assistant", "content": content})
            if ctx.manual_tc:
                nudge = (
                    "Good reasoning. Now execute it by responding with a "
                    "JSON tool call. Example:\n"
                    '{"tool_calls": [{"name": "tool_name", '
                    '"arguments": {"param": "value"}}]}'
                )
            else:
                nudge = (
                    "Good reasoning. Now call the appropriate tool to "
                    "execute your plan, or call step_complete if done."
                )
            messages.append({"role": "user", "content": nudge})
            return None  # continue inner loop

        _emit_log(
            "warning",
            f"{_YELLOW}⚠ LLM returned text {self.text_retries}x without "
            f"calling a tool — moving to next step{_RESET}",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        summary = content[:197] + "..." if len(content) > 200 else content
        return StepResult(
            summary=summary or "Step completed (model reasoned but did not call tools).",
            status="continue",
        )


class StepManager:
    """Post-step processing: plan management, summarization, termination."""

    def __init__(self, engine: "LoopEngine") -> None:
        self._engine = engine

    def auto_split(self, ctx: ExecutionContext, step_result: StepResult) -> StepResult:
        """Prevent premature 'done' when plan steps are still pending."""
        if step_result.status == "done" and not step_result.final_answer:
            pending = sum(1 for s in ctx.state.plan.steps if s.status == "pending")
            if pending > 0:
                step_result.status = "continue"
                _emit_log(
                    "warning",
                    f"{_YELLOW}⚠ Override: status='done' but {pending} steps pending → continue{_RESET}",
                    project_id=ctx.project_id, agent_id=ctx.agent_id,
                )
        return step_result

    def advance_plan(self, ctx: ExecutionContext, step_result: StepResult) -> None:
        """Create or update plan from step_result, activate next step."""
        if not ctx.state.plan.steps:
            if step_result.next_steps:
                ctx.state.plan.apply_operations(step_result.next_steps)
            if ctx.state.plan.steps:
                for s in ctx.state.plan.steps:
                    if s.status == "pending":
                        s.status = "active"
                        break
        else:
            ctx.state.plan.mark_active_done()
            if step_result.next_steps:
                ctx.state.plan.apply_operations(step_result.next_steps)
            ctx.state.plan.activate_next()

    def summarize_and_record(
        self, ctx: ExecutionContext, step_result: StepResult,
        messages: list[dict[str, Any]], action_tool_calls: int,
        iteration: int,
    ) -> None:
        """Run summarizer, build ActionRecord, append to history, preload files."""
        step_index = ctx.state.plan.active_step.index if ctx.state.plan.active_step else iteration + 1
        done_steps = [s for s in ctx.state.plan.steps if s.status == "done"]
        if done_steps:
            step_index = done_steps[-1].index

        _summarizer_on = (
            self._engine._summarizer_override
            if self._engine._summarizer_override is not None
            else _get_settings().LOOP_SUMMARIZER_ENABLED
        )
        if _summarizer_on:
            try:
                structured = _summarize_step(messages, ctx.desc, ctx.state, step_result, ctx.llm_params)
                record = ActionRecord(
                    step_index=step_index,
                    summary=structured.get("summary", step_result.summary),
                    tool_calls_count=action_tool_calls,
                    files_to_preload=structured.get("files_to_preload", []),
                    changes_made=structured.get("changes_made", ""),
                    discovered_context=structured.get("discovered", ""),
                    pending_items=structured.get("pending", ""),
                    anti_patterns=structured.get("anti_patterns", ""),
                )
            except Exception:
                record = ActionRecord(step_index=step_index, summary=step_result.summary, tool_calls_count=action_tool_calls)
        else:
            record = ActionRecord(step_index=step_index, summary=step_result.summary, tool_calls_count=action_tool_calls)

        ctx.state.history.append(record)

        # Pre-load files recommended by summarizer
        for fpath in record.files_to_preload:
            if fpath not in ctx.state.opened_files and os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        ctx.state.cache_file(fpath, f.read())
                except Exception:
                    pass

        ctx.state.current_step_index = step_index

    def finish(
        self, ctx: ExecutionContext, status: str,
        iteration: int, result: str | None = None,
    ) -> str:
        """Common finish logic: deactivate tracker, log, emit events, store stats."""
        ctx.file_tracker.deactivate()
        if ctx.verbose:
            _log_finish(ctx.agent_name, status, iteration + 1, ctx.state.total_tool_calls, ctx.state.total_tokens)
        _emit_loop_event("loop_finished", ctx.project_id, ctx.agent_id, {
            "agent_id": ctx.agent_id, "agent_name": ctx.agent_name,
            "status": status, "iterations": iteration + 1,
            "tool_calls_total": ctx.state.total_tool_calls,
            "tokens_total": ctx.state.total_tokens,
        })
        _hook_manager.dispatch(_HookContext(
            event=_HookEvent.LOOP_END,
            metadata={"state": ctx.state, "result": result, "status": status},
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        ))
        self._engine._store_stats(ctx.state)
        if result is None:
            return _synthesize_final(ctx.state)
        return result


def _get_settings():
    """Lazy import to avoid circular import at module load time."""
    from infinidev.config.settings import settings
    return settings


class LoopEngine(AgentEngine):
    """Plan-execute-summarize loop engine.

    Each iteration rebuilds the prompt from scratch with only:
    system prompt + task + plan + compact summaries + current step.
    Raw tool output exists only temporarily during a step, then is
    discarded and replaced with a ~50-token summary.
    """

    def __init__(self) -> None:
        self._last_file_tracker: FileChangeTracker | None = None
        self._nudge_threshold_override: int | None = None
        self._summarizer_override: bool | None = None
        self._cancel_event: __import__('threading').Event = __import__('threading').Event()
        self.session_notes: list[str] = []  # Persist across tasks within a session

    def cancel(self) -> None:
        """Signal the engine to stop after the current tool call."""
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def get_changed_files_summary(self) -> str:
        """Return a summary of files changed in the last execution.

        Used by the code review engine to review changes.
        Returns empty string if no files were changed.
        """
        if self._last_file_tracker is None:
            return ""

        paths = self._last_file_tracker.get_all_paths()
        if not paths:
            return ""

        parts = []
        for path in paths:
            action = self._last_file_tracker.get_action(path)
            diff = self._last_file_tracker.get_diff(path)
            if diff:
                parts.append(f"### {path} ({action})\n```diff\n{diff}\n```")
            else:
                parts.append(f"### {path} ({action}, no diff)")

        return "\n\n".join(parts)

    def has_file_changes(self) -> bool:
        """Whether the last execution modified any files."""
        if self._last_file_tracker is None:
            return False
        return bool(self._last_file_tracker.get_all_paths())

    def get_file_change_reasons(self) -> dict[str, list[str]]:
        """Return path → list of reasons for each changed file."""
        if self._last_file_tracker is None:
            return {}
        result = {}
        for path in self._last_file_tracker.get_all_paths():
            reasons = self._last_file_tracker.get_reasons(path)
            if reasons:
                result[path] = reasons
        return result

    def get_file_contents(self) -> dict[str, str]:
        """Return path → current content for each changed file."""
        import os as _os
        if self._last_file_tracker is None:
            return {}
        result = {}
        for path in self._last_file_tracker.get_all_paths():
            try:
                if _os.path.isfile(path) and _os.path.getsize(path) <= _MAX_TRACK_FILE_SIZE:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        result[path] = f.read()
            except Exception:
                pass
        return result

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
        max_iterations: int | None = None,
        max_total_tool_calls: int | None = None,
        max_tool_calls_per_action: int | None = None,
        nudge_threshold: int | None = None,
        summarizer_enabled: bool | None = None,
    ) -> str:
        """Plan-execute-summarize loop.

        Delegates to composition components: LLMCaller, ToolProcessor,
        LoopGuard, StepManager. See class docstrings for details.
        """
        ctx = self._build_context(
            agent, task_prompt,
            verbose=verbose, guardrail=guardrail,
            guardrail_max_retries=guardrail_max_retries,
            output_pydantic=output_pydantic, task_tools=task_tools,
            event_id=event_id, resume_state=resume_state,
            max_iterations=max_iterations,
            max_total_tool_calls=max_total_tool_calls,
            max_tool_calls_per_action=max_tool_calls_per_action,
            nudge_threshold=nudge_threshold,
            summarizer_enabled=summarizer_enabled,
        )
        llm_caller = LLMCaller()
        tool_proc = ToolProcessor()
        guard = LoopGuard(is_small=ctx.is_small)
        step_mgr = StepManager(self)

        self._cancel_event.clear()
        _hook_manager.dispatch(_HookContext(
            event=_HookEvent.LOOP_START,
            metadata={"task_prompt": task_prompt, "tools": ctx.tools, "state": ctx.state},
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        ))

        consecutive_all_done = 0

        for iteration in range(ctx.start_iteration, ctx.max_iterations):
            if self._cancel_event.is_set():
                logger.info("LoopEngine: cancelled by user")
                _emit_log("info", f"{_YELLOW}⚠ Task cancelled by user{_RESET}",
                          project_id=ctx.project_id, agent_id=ctx.agent_id)
                break

            ctx.state.iteration_count = iteration + 1
            messages = self._build_iteration_messages(ctx, iteration)

            # Log step start
            active = ctx.state.plan.active_step
            if active:
                active_desc = active.description
            elif not ctx.state.plan.steps:
                active_desc = "Planning..."
            else:
                done_steps = [s for s in ctx.state.plan.steps if s.status == "done"]
                active_desc = f"Continuing ({done_steps[-1].description})" if done_steps else "Working..."
            if ctx.verbose:
                _log_step_start(iteration + 1, active_desc)

            _hook_manager.dispatch(_HookContext(
                event=_HookEvent.PRE_STEP,
                metadata={"iteration": iteration, "state": ctx.state, "plan": ctx.state.plan, "agent_name": ctx.agent_name},
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            ))

            # ── Inner loop ──────────────────────────────────────────
            step_result = self._run_inner_loop(ctx, messages, iteration, llm_caller, tool_proc, guard)

            # ── Post-step processing ────────────────────────────────
            step_result = step_mgr.auto_split(ctx, step_result)

            _hook_manager.dispatch(_HookContext(
                event=_HookEvent.STEP_TRANSITION,
                metadata={"step_result": step_result, "plan": ctx.state.plan, "iteration": iteration},
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            ))

            step_mgr.advance_plan(ctx, step_result)

            action_tool_calls = step_result._action_tool_calls if hasattr(step_result, '_action_tool_calls') else 0
            step_mgr.summarize_and_record(ctx, step_result, messages, action_tool_calls, iteration)

            if ctx.verbose:
                _log_step_done(iteration + 1, step_result.status, step_result.summary, action_tool_calls, ctx.state.total_tokens)
                _log_plan(ctx.state.plan)

            _hook_manager.dispatch(_HookContext(
                event=_HookEvent.POST_STEP,
                metadata={
                    "iteration": iteration, "step_result": step_result,
                    "record": ctx.state.history[-1] if ctx.state.history else None,
                    "state": ctx.state, "agent_name": ctx.agent_name,
                    "action_tool_calls": action_tool_calls,
                },
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            ))

            if ctx.event_id:
                self._checkpoint(ctx.event_id, ctx.state)

            # ── Check termination ───────────────────────────────────
            if step_result.status == "explore":
                self._handle_explore(ctx, step_result, iteration)
                consecutive_all_done = 0
                continue

            if step_result.status == "done":
                if not step_result.final_answer and iteration == ctx.start_iteration:
                    _emit_log("warning",
                              f"{_YELLOW}⚠ LLM declared done on first step without final_answer — forcing continue{_RESET}",
                              project_id=ctx.project_id, agent_id=ctx.agent_id)
                    step_result = StepResult(summary=step_result.summary, status="continue", next_steps=step_result.next_steps)
                else:
                    result = step_result.final_answer or step_result.summary
                    result = step_mgr.finish(ctx, "done", iteration, result)
                    return self._apply_guardrail(
                        result, ctx.guardrail, ctx.guardrail_max_retries,
                        ctx.llm_params, ctx.system_prompt, ctx.desc, ctx.expected,
                        ctx.state, ctx.tool_schemas, ctx.tool_dispatch,
                        max_per_action=ctx.max_per_action,
                    )

            if step_result.status == "blocked":
                return step_mgr.finish(ctx, "blocked", iteration, step_result.summary)

            # Safety: consecutive all-done detection
            if ctx.state.plan.steps and not ctx.state.plan.has_pending:
                consecutive_all_done += 1
                if consecutive_all_done >= 2:
                    result = step_mgr.finish(ctx, "done", iteration, step_result.summary)
                    return self._apply_guardrail(
                        result, ctx.guardrail, ctx.guardrail_max_retries,
                        ctx.llm_params, ctx.system_prompt, ctx.desc, ctx.expected,
                        ctx.state, ctx.tool_schemas, ctx.tool_dispatch,
                        max_per_action=ctx.max_per_action,
                    )
            else:
                consecutive_all_done = 0

        # Outer loop exhausted
        return step_mgr.finish(ctx, "exhausted", ctx.max_iterations - 1)

    # ── Private helpers for execute() ───────────────────────────────────

    def _build_context(
        self, agent: Any, task_prompt: tuple[str, str], **kwargs: Any,
    ) -> ExecutionContext:
        """Build ExecutionContext from agent, task_prompt, and overrides."""
        from infinidev.config.llm import get_litellm_params, _is_small_model
        from infinidev.config.settings import settings
        from infinidev.config.model_capabilities import get_model_capabilities

        llm_params = get_litellm_params()
        if llm_params is None:
            raise RuntimeError("LoopEngine requires LiteLLM parameters. Ensure INFINIDEV_LLM_MODEL is set.")

        max_iterations = kwargs.get('max_iterations') or settings.LOOP_MAX_ITERATIONS
        max_total_calls = kwargs.get('max_total_tool_calls') or settings.LOOP_MAX_TOTAL_TOOL_CALLS
        max_per_action = (kwargs.get('max_tool_calls_per_action') or settings.LOOP_MAX_TOOL_CALLS_PER_ACTION) or max_total_calls
        self._nudge_threshold_override = kwargs.get('nudge_threshold')
        self._summarizer_override = kwargs.get('summarizer_enabled')

        max_context_tokens = _get_model_max_context(llm_params)

        # Resolve tools
        task_tools = kwargs.get('task_tools')
        tools = task_tools if task_tools is not None else getattr(agent, "tools", [])
        if task_tools is not None:
            from infinidev.tools.base.context import bind_tools_to_agent
            bind_tools_to_agent(task_tools, agent.agent_id)

        tool_schemas = build_tool_schemas(tools) if tools else [STEP_COMPLETE_SCHEMA]
        tool_dispatch = build_tool_dispatch(tools) if tools else {}

        file_tracker = FileChangeTracker()
        self._last_file_tracker = file_tracker
        self._last_total_tool_calls = 0
        self._last_state = None

        caps = get_model_capabilities()
        manual_tc = not caps.supports_function_calling
        is_small = _is_small_model()

        if is_small:
            logger.info("LoopEngine: small model detected — using simplified prompts and reduced tools")

        if is_small and task_tools is None:
            from infinidev.tools import get_tools_for_role
            tools = get_tools_for_role("developer", small_model=True)
            tool_schemas = build_tool_schemas(tools)
            tool_dispatch = build_tool_dispatch(tools)

        system_prompt = build_system_prompt(
            agent.backstory,
            tech_hints=getattr(agent, '_tech_hints', None),
            session_summaries=getattr(agent, '_session_summaries', None),
            identity_override=getattr(agent, '_system_prompt_identity', None),
            small_model=is_small,
        )
        if manual_tc:
            tools_section = build_tools_prompt_section(tool_schemas)
            system_prompt = f"{system_prompt}\n\n{tools_section}"
            logger.info("LoopEngine [%s]: manual tool calling mode", getattr(agent, "agent_id", "?"))

        desc, expected = task_prompt

        # Read event_id / resume_state from tool context if not passed
        event_id = kwargs.get('event_id')
        resume_state = kwargs.get('resume_state')
        if event_id is None or resume_state is None:
            from infinidev.tools.base.context import get_context_for_agent
            tool_ctx = get_context_for_agent(agent.agent_id)
            if tool_ctx:
                event_id = event_id or tool_ctx.event_id
                resume_state = resume_state or tool_ctx.resume_state

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

        if kwargs.get('verbose', True):
            _log_start(agent.agent_id, getattr(agent, "name", agent.agent_id),
                       getattr(agent, "role", "agent"), desc, len(tools))

        return ExecutionContext(
            llm_params=llm_params, manual_tc=manual_tc, is_small=is_small,
            system_prompt=system_prompt, tool_schemas=tool_schemas,
            tool_dispatch=tool_dispatch, planning_schemas=[STEP_COMPLETE_SCHEMA],
            tools=tools, max_iterations=max_iterations, max_per_action=max_per_action,
            max_total_calls=max_total_calls, history_window=settings.LOOP_HISTORY_WINDOW,
            max_context_tokens=max_context_tokens,
            verbose=kwargs.get('verbose', True),
            guardrail=kwargs.get('guardrail'), guardrail_max_retries=kwargs.get('guardrail_max_retries', 5),
            output_pydantic=kwargs.get('output_pydantic'),
            agent=agent, agent_name=getattr(agent, "name", agent.agent_id),
            agent_role=getattr(agent, "role", "agent"),
            desc=desc, expected=expected, event_id=event_id,
            state=state, file_tracker=file_tracker,
            start_iteration=state.iteration_count,
        )

    def _build_iteration_messages(
        self, ctx: ExecutionContext, iteration: int,
    ) -> list[dict[str, Any]]:
        """Build the messages list for one iteration."""
        effective_state = ctx.state
        if ctx.history_window > 0 and len(ctx.state.history) > ctx.history_window:
            effective_state = ctx.state.model_copy(deep=True)
            effective_state.history = ctx.state.history[-ctx.history_window:]

        if iteration == ctx.start_iteration:
            try:
                from infinidev.db.service import get_project_knowledge
                self._project_knowledge = get_project_knowledge(project_id=ctx.project_id)
            except Exception:
                self._project_knowledge = []

        user_prompt = build_iteration_prompt(
            ctx.desc, ctx.expected, effective_state,
            project_knowledge=self._project_knowledge if iteration == ctx.start_iteration else None,
            max_context_tokens=ctx.max_context_tokens,
            session_notes=self.session_notes if self.session_notes else None,
        )
        return [
            {"role": "system", "content": ctx.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _run_inner_loop(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        iteration: int,
        llm_caller: LLMCaller, tool_proc: ToolProcessor, guard: LoopGuard,
    ) -> StepResult:
        """Run the inner tool-calling loop for one step.

        Returns the StepResult for this step.
        """
        step_result: StepResult | None = None
        action_tool_calls = 0
        is_planning = not ctx.state.plan.steps

        llm_caller.reset()
        guard.reset()

        while action_tool_calls < ctx.max_per_action and ctx.state.total_tool_calls < ctx.max_total_calls:
            result = llm_caller.call(ctx, messages, is_planning, action_tool_calls)

            if result.should_retry:
                continue
            if result.forced_step_result:
                step_result = result.forced_step_result
                break

            if result.tool_calls:
                guard.text_retries = 0
                classified = tool_proc.classify(result.tool_calls)
                tool_proc.process_pseudo_tools(ctx, classified, self)

                if classified.regular:
                    action_tool_calls = self._execute_regular_tools(
                        ctx, classified, messages, result, action_tool_calls, iteration, guard,
                    )
                    if self._cancel_event.is_set():
                        break
                    # Check guard conditions
                    forced = guard.check_repetition(ctx, messages)
                    if forced:
                        step_result = forced
                        break
                    guard.check_error_circuit_breaker(ctx, messages)
                elif classified.step_complete or classified.notes or classified.session_notes or classified.thinks:
                    # Only pseudo-tools, no regular tools
                    self._build_pseudo_only_messages(ctx, classified, messages, result)

                if classified.step_complete:
                    step_result = _parse_step_complete_args(classified.step_complete.function.arguments)
                    break
            else:
                # Text-only response
                content = (result.message.content or "").strip() if result.message else result.raw_content
                forced = guard.handle_text_only(ctx, messages, content)
                if forced:
                    step_result = forced
                    break
                continue
        else:
            # Inner loop exhausted (while condition became false)
            if step_result is None:
                if ctx.state.total_tool_calls >= ctx.max_total_calls:
                    limit_msg = f"global tool call limit reached ({ctx.state.total_tool_calls}/{ctx.max_total_calls} total calls)"
                else:
                    limit_msg = f"per-step tool call limit reached ({action_tool_calls}/{ctx.max_per_action} calls)"
                step_result = StepResult(summary=f"Step interrupted: {limit_msg}.", status="continue")
                _emit_log("error", f"{_RED}⚠ Inner loop exhausted: {limit_msg}{_RESET}",
                          project_id=ctx.project_id, agent_id=ctx.agent_id)

        if step_result is None:
            step_result = StepResult(summary="Step completed.", status="continue")

        # Attach tool call count for post-step processing
        step_result._action_tool_calls = action_tool_calls  # type: ignore[attr-defined]
        return step_result

    def _execute_regular_tools(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], llm_result: LLMCallResult,
        action_tool_calls: int, iteration: int, guard: LoopGuard,
    ) -> int:
        """Execute regular tool calls and build messages. Returns updated action_tool_calls."""
        message = llm_result.message
        raw_content = llm_result.raw_content

        if ctx.manual_tc:
            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", "") or raw_content,
            })
        else:
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in classified.regular
            ]
            for pseudo_tc in classified.thinks + classified.notes + ([classified.step_complete] if classified.step_complete else []):
                assistant_msg["tool_calls"].append({
                    "id": pseudo_tc.id, "type": "function",
                    "function": {"name": pseudo_tc.function.name, "arguments": pseudo_tc.function.arguments},
                })
            messages.append(assistant_msg)

        tool_results_text: list[str] = []
        _tool_hook_meta = {
            "agent_name": ctx.agent_name, "iteration": iteration,
            "verbose": ctx.verbose, "tokens_total": ctx.state.total_tokens,
            "prompt_tokens": ctx.state.last_prompt_tokens,
            "completion_tokens": ctx.state.last_completion_tokens,
        }
        batches = _batch_tool_calls(classified.regular)

        for batch in batches:
            is_parallel = len(batch) > 1 and batch[0].function.name not in _WRITE_TOOLS

            if is_parallel:
                _tool_hook_meta["call_num"] = action_tool_calls + 1
                _tool_hook_meta["total_calls"] = ctx.state.total_tool_calls + 1
                _tool_hook_meta["project_id"] = ctx.project_id
                _tool_hook_meta["agent_id"] = ctx.agent_id
                batch_results = _execute_tool_calls_parallel(batch, ctx.tool_dispatch, hook_metadata=_tool_hook_meta)
            else:
                batch_results = []
                for _bi, tc in enumerate(batch):
                    _pre = _capture_pre_content(tc.function.name, tc.function.arguments, ctx.file_tracker)
                    _tool_hook_meta["call_num"] = action_tool_calls + _bi + 1
                    _tool_hook_meta["total_calls"] = ctx.state.total_tool_calls + _bi + 1
                    _tool_hook_meta["project_id"] = ctx.project_id
                    _tool_hook_meta["agent_id"] = ctx.agent_id
                    result = execute_tool_call(ctx.tool_dispatch, tc.function.name, tc.function.arguments, hook_metadata=_tool_hook_meta)
                    _maybe_emit_file_change(tc.function.name, tc.function.arguments, result, _pre, ctx.file_tracker, ctx.project_id, ctx.agent_id)
                    batch_results.append((tc, result))

            if self._cancel_event.is_set():
                break
            for tc, result in batch_results:
                _tool_error = _extract_tool_error(result)
                guard.on_tool_result(tc.function.name, tc.function.arguments, bool(_tool_error))

                if not _tool_error:
                    _update_opened_files_cache(ctx.state, tc.function.name, tc.function.arguments, result)

                counter_tag = f"\n[Tool call {action_tool_calls + 1}/{ctx.max_per_action} for this step]"
                if is_parallel:
                    counter_tag += " (parallel)"
                result_with_counter = result + counter_tag

                if ctx.manual_tc:
                    tool_results_text.append(f"[Tool: {tc.function.name}] Result:\n{result_with_counter}")
                else:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_with_counter})

                action_tool_calls += 1
                ctx.state.total_tool_calls += 1
                ctx.state.tool_calls_since_last_note += 1

                # Budget nudge
                _default_nudge = 4 if ctx.is_small else _get_settings().LOOP_STEP_NUDGE_THRESHOLD
                _nudge_threshold = self._nudge_threshold_override if self._nudge_threshold_override is not None else _default_nudge
                if _nudge_threshold > 0 and action_tool_calls == _nudge_threshold:
                    _active_desc = ctx.state.plan.active_step.description if ctx.state.plan.active_step else ""
                    _nudge_msg = (
                        f"You have used {action_tool_calls}/{ctx.max_per_action} tool calls for this step. "
                        f"Step scope: \"{_active_desc}\". "
                        f"Call step_complete now. If the step is not finished, set status=\'continue\' "
                        f"and add/modify next_steps to capture the remaining work."
                    )
                    if ctx.manual_tc:
                        tool_results_text.append(f"\n⚠ STEP BUDGET: {_nudge_msg}")
                    else:
                        messages.append({"role": "user", "content": _nudge_msg})

                ctx.state.tick_opened_files(1)

        # Manual mode: send all results as single user message
        if ctx.manual_tc:
            for nc in classified.notes:
                tool_results_text.append('[Tool: add_note] Result:\n{"status": "noted"}')
            for snc in classified.session_notes:
                tool_results_text.append('[Tool: add_session_note] Result:\n{"status": "noted"}')
            for tk in classified.thinks:
                tool_results_text.append('[Tool: think] Result:\n{"status": "acknowledged"}')
            if tool_results_text:
                messages.append({"role": "user", "content": "\n\n".join(tool_results_text)})

        # FC mode: pseudo-tool results
        if not ctx.manual_tc:
            for tk in classified.thinks:
                messages.append({"role": "tool", "tool_call_id": tk.id, "content": '{"status": "acknowledged"}'})
            for nc in classified.notes:
                messages.append({"role": "tool", "tool_call_id": nc.id, "content": '{"status": "noted"}'})
            for snc in classified.session_notes:
                messages.append({"role": "tool", "tool_call_id": snc.id, "content": '{"status": "noted"}'})
            if classified.step_complete:
                messages.append({"role": "tool", "tool_call_id": classified.step_complete.id, "content": '{"status": "acknowledged"}'})

        return action_tool_calls

    def _build_pseudo_only_messages(
        self, ctx: ExecutionContext, classified: ClassifiedCalls,
        messages: list[dict[str, Any]], llm_result: LLMCallResult,
    ) -> None:
        """Build messages when only pseudo-tools were called (no regular tools)."""
        message = llm_result.message
        raw_content = llm_result.raw_content

        if ctx.manual_tc:
            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", "") or raw_content,
            })
        else:
            assistant_msg = {"role": "assistant", "content": message.content or ""}
            pseudo_calls = classified.thinks + classified.notes + classified.session_notes + ([classified.step_complete] if classified.step_complete else [])
            assistant_msg["tool_calls"] = [
                {"id": pc.id, "type": "function",
                 "function": {"name": pc.function.name, "arguments": pc.function.arguments}}
                for pc in pseudo_calls
            ]
            messages.append(assistant_msg)
            for tk in classified.thinks:
                messages.append({"role": "tool", "tool_call_id": tk.id, "content": '{"status": "acknowledged"}'})
            for nc in classified.notes:
                messages.append({"role": "tool", "tool_call_id": nc.id, "content": '{"status": "noted"}'})
            for snc in classified.session_notes:
                messages.append({"role": "tool", "tool_call_id": snc.id, "content": '{"status": "noted"}'})
            if classified.step_complete:
                messages.append({"role": "tool", "tool_call_id": classified.step_complete.id, "content": '{"status": "acknowledged"}'})

    def _handle_explore(
        self, ctx: ExecutionContext, step_result: StepResult, iteration: int,
    ) -> None:
        """Delegate sub-problem to TreeEngine."""
        step_index = ctx.state.plan.active_step.index if ctx.state.plan.active_step else iteration + 1
        _emit_log("warning",
                   f"{_YELLOW}🌳 Delegating to exploration tree: {step_result.summary[:120]}{_RESET}",
                   project_id=ctx.project_id, agent_id=ctx.agent_id)
        try:
            from infinidev.engine.tree_engine import TreeEngine
            tree_engine = TreeEngine()
            explore_result = tree_engine.explore_subproblem(ctx.agent, step_result.summary)
            if len(ctx.state.notes) < 20:
                ctx.state.notes.append(f"Exploration result: {explore_result[:500]}")
            ctx.state.history.append(ActionRecord(
                step_index=step_index,
                summary=f"Explored via tree: {explore_result[:200]}",
                tool_calls_count=0,
            ))
        except Exception as exc:
            logger.warning("TreeEngine exploration failed: %s", exc)
            if len(ctx.state.notes) < 20:
                ctx.state.notes.append(f"Exploration failed: {exc}")


    def _checkpoint(self, event_id: int, state: LoopState) -> None:
        """No-op in CLI mode."""
        pass

    def _store_stats(self, state: LoopState) -> None:
        """Store execution stats for external access."""
        self._last_total_tool_calls = state.total_tool_calls
        self._last_state = state

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
        max_per_action: int = 0,
    ) -> str:
        """Validate result with guardrail; retry with feedback if it fails."""
        if guardrail is None:
            return result

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
