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

# json_repair is now used via tool_call_parser.safe_json_loads

from infinidev.engine.base import AgentEngine
from infinidev.engine.engine_logging import (
    emit_loop_event as _emit_loop_event,
    log as _log,
    emit_log as _emit_log,
    extract_tool_detail as _extract_tool_detail,
    extract_tool_error as _extract_tool_error,
    log_start as _log_start,
    log_step_start as _log_step_start,
    log_tool as _log_tool,
    log_step_done as _log_step_done,
    log_plan as _log_plan,
    log_prompt as _log_prompt,
    log_finish as _log_finish,
    # ANSI codes used inline in execute()
    DIM as _DIM,
    BOLD as _BOLD,
    RESET as _RESET,
    CYAN as _CYAN,
    GREEN as _GREEN,
    YELLOW as _YELLOW,
    RED as _RED,
    MAGENTA as _MAGENTA,
    BLUE as _BLUE,
    TOOL_DETAIL_KEYS as _TOOL_DETAIL_KEYS,
)
from infinidev.engine.tool_call_parser import (
    safe_json_loads as _safe_json_loads,
    ManualToolCall as _ManualToolCall,
    parse_text_tool_calls as _parse_text_tool_calls,
    parse_step_complete_args as _parse_step_complete_args,
)
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

# Max consecutive calls to the same tool before forcing a step_complete nudge
_MAX_SAME_TOOL_CONSECUTIVE = 3


# _safe_json_loads imported from tool_call_parser at top of file


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


# -- Event handling via centralized EventBus --
# event_bus access is now via engine_logging module

# Logging, event emission, ANSI codes, and tool detail/error extraction
# are all imported from engine_logging.py at top of file


# ── Opened files cache helper ──────────────────────────────────────────────

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

_FILE_CHANGE_TOOLS = {"edit_file", "write_file", "multi_edit_file", "apply_patch"}

# Tools that modify state — these act as barriers in parallel execution.
# All read-only tools before a write are executed in parallel, then the write runs alone.
_WRITE_TOOLS = {
    "edit_file", "write_file", "multi_edit_file", "apply_patch",
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
            )
            results.append((tc, result))
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _exec(tc):
        result = execute_tool_call(
            tool_dispatch, tc.function.name, tc.function.arguments,
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


# All _log_* functions, error classification, and tool call parsers
# are imported from engine_logging.py, llm_client.py, and tool_call_parser.py


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
        # Override loop limits without mutating global settings
        max_iterations: int | None = None,
        max_total_tool_calls: int | None = None,
        max_tool_calls_per_action: int | None = None,
        nudge_threshold: int | None = None,
        summarizer_enabled: bool | None = None,
    ) -> str:
        from infinidev.config.llm import get_litellm_params
        from infinidev.config.settings import settings

        llm_params = get_litellm_params()
        if llm_params is None:
            raise RuntimeError(
                "LoopEngine requires LiteLLM parameters. "
                "Ensure INFINIDEV_LLM_MODEL is set."
            )

        max_iterations = max_iterations if max_iterations is not None else settings.LOOP_MAX_ITERATIONS
        max_total_calls = max_total_tool_calls if max_total_tool_calls is not None else settings.LOOP_MAX_TOTAL_TOOL_CALLS
        max_per_action = (max_tool_calls_per_action if max_tool_calls_per_action is not None else settings.LOOP_MAX_TOOL_CALLS_PER_ACTION) or max_total_calls
        history_window = settings.LOOP_HISTORY_WINDOW
        self._nudge_threshold_override = nudge_threshold
        self._summarizer_override = summarizer_enabled

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
        self._last_file_tracker = file_tracker  # Expose for post-execution review
        self._last_total_tool_calls = 0  # Expose for gather phase
        self._last_state = None  # Expose LoopState for chaining (gather phase)

        # Check model capabilities for manual tool calling mode
        from infinidev.config.model_capabilities import get_model_capabilities
        caps = get_model_capabilities()
        manual_tc = not caps.supports_function_calling

        # Build system prompt
        system_prompt = build_system_prompt(
            agent.backstory,
            tech_hints=getattr(agent, '_tech_hints', None),
            session_summaries=getattr(agent, '_session_summaries', None),
            identity_override=getattr(agent, '_system_prompt_identity', None),
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
            _MAX_MALFORMED_RETRIES = 4
            _text_retries = 0
            _consecutive_tool_errors = 0  # Track consecutive tool failures
            _MAX_CONSECUTIVE_ERRORS = 4  # Force step_complete after this many

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
                                identity_override=getattr(agent, '_system_prompt_identity', None),
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

                    # FC mode fallback: some models (e.g. LFM2) return tool
                    # calls as <tool_call> tags in content instead of native
                    # tool_calls.  Parse text as a last resort before giving up.
                    if not tool_calls:
                        raw_content = (getattr(message, "content", None) or "").strip()
                        if raw_content:
                            parsed_calls = _parse_text_tool_calls(raw_content)
                            if parsed_calls:
                                tool_calls = []
                                for i, pc in enumerate(parsed_calls):
                                    tc_obj = _ManualToolCall(
                                        id=f"fc_fallback_{action_tool_calls + i}",
                                        name=pc["name"],
                                        arguments=(
                                            json.dumps(pc["arguments"])
                                            if isinstance(pc["arguments"], dict)
                                            else str(pc["arguments"])
                                        ),
                                    )
                                    tool_calls.append(tc_obj)

                # ── Process tool calls (unified for both modes) ──────
                if tool_calls:
                    _text_retries = 0  # Reset only when tool calls are present
                    # Separate engine pseudo-tools from regular tool calls
                    regular_calls = []
                    sc_call = None
                    note_calls = []
                    think_calls = []
                    for tc in tool_calls:
                        if tc.function.name == "step_complete":
                            sc_call = tc
                        elif tc.function.name == "add_note":
                            note_calls.append(tc)
                        elif tc.function.name == "think":
                            think_calls.append(tc)
                        else:
                            regular_calls.append(tc)

                    # Process think calls (emit reasoning to chat, don't count as tool call)
                    for tk in think_calls:
                        try:
                            tk_args = _safe_json_loads(tk.function.arguments) if isinstance(tk.function.arguments, str) else (tk.function.arguments or {})
                            reasoning = tk_args.get("reasoning", "").strip()
                            if reasoning:
                                if verbose:
                                    _log(f"  {_DIM}💭 {reasoning[:200]}{_RESET}")
                                _emit_loop_event("loop_think", agent.project_id, agent.agent_id, {
                                    "reasoning": reasoning,
                                })
                        except (json.JSONDecodeError, AttributeError):
                            pass

                    # Process add_note calls (write to state.notes)
                    _MAX_NOTES = 20
                    for nc in note_calls:
                        try:
                            nc_args = _safe_json_loads(nc.function.arguments) if isinstance(nc.function.arguments, str) else (nc.function.arguments or {})
                            note_text = nc_args.get("note", "").strip()
                            if note_text and len(state.notes) < _MAX_NOTES:
                                state.notes.append(note_text)
                                state.tool_calls_since_last_note = 0  # Reset nudge counter
                        except (json.JSONDecodeError, AttributeError):
                            pass

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
                            # Include engine pseudo-tools in the message (needed for API)
                            for pseudo_tc in think_calls + note_calls + ([sc_call] if sc_call else []):
                                assistant_msg["tool_calls"].append({
                                    "id": pseudo_tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": pseudo_tc.function.name,
                                        "arguments": pseudo_tc.function.arguments,
                                    },
                                })
                            messages.append(assistant_msg)

                        # Collect tool results for both modes
                        # Batch tool calls: consecutive reads run in parallel, writes are barriers
                        tool_results_text: list[str] = []  # For manual mode
                        batches = _batch_tool_calls(regular_calls)

                        for batch in batches:
                            is_parallel = len(batch) > 1 and batch[0].function.name not in _WRITE_TOOLS

                            if is_parallel:
                                # Execute read-only batch in parallel
                                batch_results = _execute_tool_calls_parallel(batch, tool_dispatch)
                            else:
                                # Sequential execution (single tool or write)
                                batch_results = []
                                for tc in batch:
                                    # Pre-hook BEFORE execution for file changes
                                    _pre = _capture_pre_content(
                                        tc.function.name, tc.function.arguments, file_tracker,
                                    )
                                    result = execute_tool_call(
                                        tool_dispatch, tc.function.name, tc.function.arguments,
                                    )
                                    # Post-hook AFTER execution
                                    _maybe_emit_file_change(
                                        tc.function.name, tc.function.arguments, result,
                                        _pre, file_tracker,
                                        agent.project_id, agent.agent_id,
                                    )
                                    batch_results.append((tc, result))

                            # Process results from the batch
                            for tc, result in batch_results:
                                # Intercept send_message: emit to TUI chat
                                if tc.function.name == "send_message":
                                    try:
                                        _msg_args = _safe_json_loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                                        _emit_loop_event("loop_user_message", agent.project_id, agent.agent_id, {
                                            "message": _msg_args.get("message", ""),
                                        })
                                    except Exception:
                                        pass

                                # Detect errors / hallucinated tools and log visibly
                                _tool_error = _extract_tool_error(result)

                                # Track consecutive tool errors for circuit breaker
                                if _tool_error:
                                    _consecutive_tool_errors += 1
                                else:
                                    _consecutive_tool_errors = 0

                                # --- Opened files cache ---
                                if not _tool_error:
                                    _update_opened_files_cache(
                                        state, tc.function.name, tc.function.arguments, result,
                                    )

                                # Append tool call counter to result
                                counter_tag = f"\n[Tool call {action_tool_calls + 1}/{max_per_action} for this step]"
                                if is_parallel:
                                    counter_tag += " (parallel)"
                                result_with_counter = result + counter_tag

                                if manual_tc:
                                    tool_results_text.append(
                                        f"[Tool: {tc.function.name}] Result:\n{result_with_counter}"
                                    )
                                else:
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": result_with_counter,
                                    })
                                action_tool_calls += 1
                                state.total_tool_calls += 1
                                state.tool_calls_since_last_note += 1

                                # Nudge at threshold
                                _nudge_threshold = self._nudge_threshold_override if self._nudge_threshold_override is not None else settings.LOOP_STEP_NUDGE_THRESHOLD
                                if _nudge_threshold > 0 and action_tool_calls == _nudge_threshold:
                                    _active_desc = ""
                                    if state.plan.active_step:
                                        _active_desc = state.plan.active_step.description
                                    _nudge_msg = (
                                        f"You have used {action_tool_calls}/{max_per_action} tool calls for this step. "
                                        f"Step scope: \"{_active_desc}\". "
                                        f"Call step_complete now. If the step is not finished, set status='continue' "
                                        f"and add/modify next_steps to capture the remaining work."
                                    )
                                    if manual_tc:
                                        tool_results_text.append(f"\n⚠ STEP BUDGET: {_nudge_msg}")
                                    else:
                                        messages.append({"role": "user", "content": _nudge_msg})

                                # Tick opened files cache TTL
                                state.tick_opened_files(1)

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
                        if manual_tc:
                            for nc in note_calls:
                                tool_results_text.append('[Tool: add_note] Result:\n{"status": "noted"}')
                            for tk in think_calls:
                                tool_results_text.append('[Tool: think] Result:\n{"status": "acknowledged"}')
                            if tool_results_text:
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

                        # Provide think + add_note tool results
                        if not manual_tc:
                            for tk in think_calls:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tk.id,
                                    "content": '{"status": "acknowledged"}',
                                })
                            for nc in note_calls:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": nc.id,
                                    "content": '{"status": "noted"}',
                                })

                        # Provide step_complete tool result if it was in this batch
                        if sc_call and not manual_tc:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": sc_call.id,
                                "content": '{"status": "acknowledged"}',
                            })

                    elif sc_call or note_calls or think_calls:
                        # Only engine pseudo-tools, no regular tools
                        if manual_tc:
                            messages.append({
                                "role": "assistant",
                                "content": getattr(message, "content", "") or raw_content,
                            })
                        else:
                            assistant_msg = {"role": "assistant", "content": message.content or ""}
                            pseudo_calls = think_calls + note_calls + ([sc_call] if sc_call else [])
                            assistant_msg["tool_calls"] = [
                                {
                                    "id": pc.id,
                                    "type": "function",
                                    "function": {
                                        "name": pc.function.name,
                                        "arguments": pc.function.arguments,
                                    },
                                }
                                for pc in pseudo_calls
                            ]
                            messages.append(assistant_msg)
                            for tk in think_calls:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tk.id,
                                    "content": '{"status": "acknowledged"}',
                                })
                            for nc in note_calls:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": nc.id,
                                    "content": '{"status": "noted"}',
                                })
                            if sc_call:
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

                    # Circuit breaker: too many consecutive tool errors
                    # (e.g. edit_file failing repeatedly with different args)
                    if _consecutive_tool_errors >= _MAX_CONSECUTIVE_ERRORS:
                        _emit_log(
                            "warning",
                            f"{_YELLOW}⚠ {_consecutive_tool_errors} consecutive tool errors "
                            f"— nudging model to try a different approach{_RESET}",
                            project_id=agent.project_id, agent_id=agent.agent_id,
                        )
                        _consecutive_tool_errors = 0  # Reset after nudge
                        nudge_msg = (
                            f"WARNING: Your last {_MAX_CONSECUTIVE_ERRORS} tool calls all failed. "
                            "You are stuck in a failing pattern. Change your approach:\n"
                            "- If edit_file keeps failing, use write_file to rewrite the entire file.\n"
                            "- If read_file keeps failing on a path, use list_directory to find the correct path.\n"
                            "- If nothing works, call step_complete to move on and revisit later."
                        )
                        if manual_tc:
                            messages.append({"role": "user", "content": nudge_msg})
                        else:
                            messages.append({"role": "user", "content": nudge_msg})

                else:
                    # LLM responded with text instead of a tool call.
                    # This is often the model reasoning/thinking — treat it as
                    # useful context, keep it in the conversation, and gently
                    # remind it to call a tool next.
                    content = (message.content or "").strip()
                    _text_retries += 1

                    if _text_retries < _MAX_TEXT_RETRIES:
                        # Show the reasoning to the user (like a think call)
                        if content:
                            preview = content[:200] + ("..." if len(content) > 200 else "")
                            if verbose:
                                _log(f"  {_DIM}💭 {preview}{_RESET}")
                            _emit_loop_event("loop_think", agent.project_id, agent.agent_id, {
                                "reasoning": content,
                            })

                        # Keep the full text as assistant message in context
                        # (the model may need its own reasoning for the next call)
                        messages.append({"role": "assistant", "content": content})

                        # Gentle nudge — not aggressive, just a reminder
                        if manual_tc:
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
                        continue  # Retry the inner loop

                    # Exhausted retries — fall back to StepResult
                    _emit_log(
                        "warning",
                        f"{_YELLOW}⚠ LLM returned text {_text_retries}x without "
                        f"calling a tool — moving to next step{_RESET}",
                        project_id=agent.project_id, agent_id=agent.agent_id,
                    )
                    if len(content) > 200:
                        content = content[:197] + "..."
                    step_result = StepResult(
                        summary=content or "Step completed (model reasoned but did not call tools).",
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

            # --- Auto-split: prevent premature "done" ---
            if step_result.status == "done" and not step_result.final_answer:
                pending_count = sum(1 for s in state.plan.steps if s.status == "pending")
                if pending_count > 0:
                    step_result.status = "continue"
                    _emit_log(
                        "warning",
                        f"{_YELLOW}⚠ Override: status='done' but {pending_count} steps pending → continue{_RESET}",
                        project_id=agent.project_id, agent_id=agent.agent_id,
                    )

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

            # --- Step summarization ---
            _summarizer_on = self._summarizer_override if self._summarizer_override is not None else settings.LOOP_SUMMARIZER_ENABLED
            if _summarizer_on:
                try:
                    structured = _summarize_step(
                        messages, desc, state, step_result, llm_params,
                    )
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
                    record = ActionRecord(
                        step_index=step_index,
                        summary=step_result.summary,
                        tool_calls_count=action_tool_calls,
                    )
            else:
                record = ActionRecord(
                    step_index=step_index,
                    summary=step_result.summary,
                    tool_calls_count=action_tool_calls,
                )
            state.history.append(record)

            # Pre-load files recommended by summarizer
            if record.files_to_preload:
                for fpath in record.files_to_preload:
                    if fpath not in state.opened_files and _os.path.isfile(fpath):
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                state.cache_file(fpath, f.read())
                        except Exception:
                            pass

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
            if step_result.status == "explore":
                # Delegate sub-problem to TreeEngine
                _emit_log(
                    "warning",
                    f"{_YELLOW}🌳 Delegating to exploration tree: {step_result.summary[:120]}{_RESET}",
                    project_id=agent.project_id, agent_id=agent.agent_id,
                )
                try:
                    from infinidev.engine.tree_engine import TreeEngine
                    tree_engine = TreeEngine()
                    explore_result = tree_engine.explore_subproblem(agent, step_result.summary)
                    # Add exploration result as a note for context in subsequent steps
                    if len(state.notes) < 20:
                        state.notes.append(f"Exploration result: {explore_result[:500]}")
                    # Record as an action
                    state.history.append(ActionRecord(
                        step_index=step_index,
                        summary=f"Explored via tree: {explore_result[:200]}",
                        tool_calls_count=0,
                    ))
                except Exception as exc:
                    logger.warning("TreeEngine exploration failed: %s", exc)
                    if len(state.notes) < 20:
                        state.notes.append(f"Exploration failed: {exc}")
                # Continue the loop after exploration
                consecutive_all_done = 0
                continue

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
                    self._store_stats(state)
                    return self._apply_guardrail(
                        result, guardrail, guardrail_max_retries,
                        llm_params, system_prompt, desc, expected, state, tool_schemas, tool_dispatch,
                        max_per_action=max_per_action,
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
                self._store_stats(state)
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
                    self._store_stats(state)
                    return self._apply_guardrail(
                        result, guardrail, guardrail_max_retries,
                        llm_params, system_prompt, desc, expected, state, tool_schemas, tool_dispatch,
                        max_per_action=max_per_action,
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
        self._store_stats(state)
        return _synthesize_final(state)

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
