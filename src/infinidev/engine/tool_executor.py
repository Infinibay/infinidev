"""Tool execution pipeline: batching, parallel execution, file tracking, and caching.

Handles:
- Batching tool calls (reads in parallel, writes sequential)
- Parallel execution with ThreadPoolExecutor
- File change tracking (pre/post content, diffs)
- Opened files cache management
- Code intelligence reindexing
"""

from __future__ import annotations

import json
from typing import Any

from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.engine_logging import emit_loop_event, extract_tool_error
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.tools import execute_tool_call


# ── Constants ─────────────────────────────────────────────────────────────

# Tools that modify files — tracked for diff generation
FILE_CHANGE_TOOLS = {
    "edit_file", "write_file", "multi_edit_file", "apply_patch",
    "create_file", "replace_lines",
    "add_content_after_line", "add_content_before_line",
    "edit_symbol", "add_symbol", "remove_symbol",
    "rename_symbol", "move_symbol",
}

# Tools with side effects — act as barriers in parallel execution.
WRITE_TOOLS = {
    "edit_file", "write_file", "multi_edit_file", "apply_patch",
    "create_file", "replace_lines",
    "add_content_after_line", "add_content_before_line",
    "edit_symbol", "add_symbol", "remove_symbol",
    "rename_symbol", "move_symbol",
    "git_commit", "git_branch", "git_push",
    "execute_command",  # Commands can have side effects
    "code_interpreter",  # Executes arbitrary code — must be sequential
    "record_finding", "update_finding", "delete_finding",
    "write_report", "delete_report",
    "update_documentation", "delete_documentation",
    "send_message",
}

MAX_TRACK_FILE_SIZE = 1_000_000  # 1 MB — skip tracking larger files


# ── Code intelligence reindex ─────────────────────────────────────────────

def reindex_if_enabled(file_path: str) -> None:
    """Trigger incremental reindex of a file after it's been modified."""
    try:
        from infinidev.config.settings import settings
        if settings.CODE_INTEL_ENABLED and settings.CODE_INTEL_AUTO_INDEX:
            from infinidev.code_intel.indexer import reindex_file
            reindex_file(1, file_path)  # project_id=1 (default)
    except Exception:
        pass  # Never block the main loop for indexing


# ── Opened files cache management ─────────────────────────────────────────


def _reread_and_cache(state: LoopState, path: str) -> None:
    """Re-read a file from disk and refresh the opened files cache."""
    import os as _os
    try:
        if _os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Format with line numbers like read_file does
            lines = content.split("\n")
            numbered = "\n".join(f"{i+1:>6}\t{line}" for i, line in enumerate(lines))
            state.refresh_file(path, numbered)
    except Exception:
        pass


def _extract_path_from_result(result: str) -> str | None:
    """Extract file path from a tool result JSON."""
    try:
        res = json.loads(result) if isinstance(result, str) else result
        if isinstance(res, dict):
            return res.get("path") or res.get("file_path")
    except Exception:
        pass
    return None

def update_opened_files_cache(
    state: LoopState,
    tool_name: str,
    arguments: str | dict,
    result: str,
) -> None:
    """Update the opened files cache based on tool calls.

    Dispatches to tool-specific handlers via ``_CACHE_HANDLERS``.
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

    from infinidev.tools.base.context import get_current_workspace_path
    ws = get_current_workspace_path() or _os.getcwd()
    if not _os.path.isabs(path):
        path = _os.path.normpath(_os.path.join(ws, path))

    handler = _CACHE_HANDLERS.get(tool_name)
    if handler:
        handler(state, path, args, result, ws)


def _cache_read(state, path, args, result, ws):
    if result and not result.strip().startswith('{"error'):
        state.cache_file(path, result)


def _cache_write(state, path, args, result, ws):
    content = args.get("content", "")
    if content:
        state.refresh_file(path, content)
    reindex_if_enabled(path)


def _cache_edit(state, path, args, result, ws):
    _reread_and_cache(state, path)
    reindex_if_enabled(path)


def _cache_line_edit(state, path, args, result, ws):
    import os as _os
    file_path_arg = args.get("file_path") or path
    if file_path_arg:
        if not _os.path.isabs(file_path_arg):
            file_path_arg = _os.path.normpath(_os.path.join(ws, file_path_arg))
        _reread_and_cache(state, file_path_arg)
        reindex_if_enabled(file_path_arg)


def _cache_symbol_edit(state, path, args, result, ws):
    affected_path = _extract_path_from_result(result)
    if affected_path:
        _reread_and_cache(state, affected_path)
        reindex_if_enabled(affected_path)


def _cache_rename_symbol(state, path, args, result, ws):
    try:
        res = json.loads(result) if isinstance(result, str) else result
        if isinstance(res, dict):
            for fpath in res.get("files_modified", []):
                _reread_and_cache(state, fpath)
                reindex_if_enabled(fpath)
    except Exception:
        pass


def _cache_move_symbol(state, path, args, result, ws):
    try:
        res = json.loads(result) if isinstance(result, str) else result
        if isinstance(res, dict):
            for key in ("source_file", "target_file"):
                fpath = res.get(key)
                if fpath:
                    _reread_and_cache(state, fpath)
                    reindex_if_enabled(fpath)
    except Exception:
        pass


def _cache_apply_patch(state, path, args, result, ws):
    import os as _os
    try:
        res = json.loads(result) if isinstance(result, str) else result
        if isinstance(res, dict) and "files_modified" in res:
            for fpath in res["files_modified"]:
                abs_path = _os.path.join(ws, fpath) if not _os.path.isabs(fpath) else fpath
                _reread_and_cache(state, abs_path)
    except Exception:
        pass


def _cache_list_dir(state, path, args, result, ws):
    if result and not result.strip().startswith('{"error'):
        state.cache_file(f"[dir] {path}", result)


def _cache_glob(state, path, args, result, ws):
    pattern = args.get("pattern", "")
    if pattern and result and not result.strip().startswith('{"error'):
        state.cache_file(f"[glob] {pattern}", result)


def _cache_code_search(state, path, args, result, ws):
    query = args.get("query") or args.get("pattern") or args.get("search_query", "")
    if query and result and not result.strip().startswith('{"error'):
        state.cache_file(f"[search] {query}", result)


# Dispatch table: tool_name → handler function
_CACHE_HANDLERS = {
    "read_file": _cache_read,
    "partial_read": _cache_read,
    "write_file": _cache_write,
    "create_file": _cache_write,
    "edit_file": _cache_edit,
    "multi_edit_file": _cache_edit,
    "replace_lines": _cache_line_edit,
    "add_content_after_line": _cache_line_edit,
    "add_content_before_line": _cache_line_edit,
    "edit_symbol": _cache_symbol_edit,
    "add_symbol": _cache_symbol_edit,
    "remove_symbol": _cache_symbol_edit,
    "rename_symbol": _cache_rename_symbol,
    "move_symbol": _cache_move_symbol,
    "apply_patch": _cache_apply_patch,
    "list_directory": _cache_list_dir,
    "glob": _cache_glob,
    "code_search": _cache_code_search,
}


# ── Batching ──────────────────────────────────────────────────────────────

def batch_tool_calls(calls: list) -> list[list]:
    """Group tool calls into batches for parallel/sequential execution.

    Consecutive read-only tools are grouped together (run in parallel).
    Write tools each get their own single-item batch (run sequentially).
    """
    batches: list[list] = []
    current_reads: list = []

    for tc in calls:
        name = tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", "")
        if name in WRITE_TOOLS:
            if current_reads:
                batches.append(current_reads)
                current_reads = []
            batches.append([tc])
        else:
            current_reads.append(tc)

    if current_reads:
        batches.append(current_reads)

    return batches


def execute_tool_calls_parallel(
    batch: list,
    tool_dispatch: dict,
) -> list[tuple]:
    """Execute a batch of read-only tool calls in parallel.

    Returns list of (tc, result) tuples in original order.
    """
    if len(batch) <= 1:
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


# ── File change tracking helpers ──────────────────────────────────────────

def extract_file_path_from_args(tool_name: str, arguments: str | dict) -> str | None:
    """Extract the file path from tool arguments (checks 'path' and 'file_path')."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(args, dict):
        return None
    return args.get("path") or args.get("file_path")


def capture_pre_content(
    tool_name: str,
    arguments: str | dict,
    tracker: FileChangeTracker,
) -> str | None:
    """Read file content before a write/edit tool mutates it."""
    import os as _os
    if tool_name not in FILE_CHANGE_TOOLS or not tracker.active:
        return None
    file_path = extract_file_path_from_args(tool_name, arguments)
    if not file_path:
        return None
    try:
        file_path = _os.path.abspath(_os.path.expanduser(file_path))
    except (OSError, ValueError):
        return None
    if not _os.path.isfile(file_path):
        return None  # new file — original is empty
    try:
        size = _os.path.getsize(file_path)
        if size > MAX_TRACK_FILE_SIZE:
            return None
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def extract_reason_from_args(arguments: str | dict) -> str:
    """Extract the reason/description from tool call arguments."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(args, dict):
        return ""
    return args.get("reason") or args.get("description") or ""


def maybe_emit_file_change(
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
    if tool_name not in FILE_CHANGE_TOOLS or not tracker.active:
        return

    if extract_tool_error(result):
        return

    file_path = extract_file_path_from_args(tool_name, arguments)
    if not file_path:
        return
    try:
        file_path = _os.path.abspath(_os.path.expanduser(file_path))
    except (OSError, ValueError):
        return

    reason = extract_reason_from_args(arguments)
    if reason:
        tracker.record_reason(file_path, reason)

    try:
        if not _os.path.isfile(file_path):
            return
        size = _os.path.getsize(file_path)
        if size > MAX_TRACK_FILE_SIZE:
            return
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            after_content = f.read()
    except Exception:
        return

    before = pre_content if pre_content is not None else ""
    diff_text = tracker.record(file_path, before, after_content)
    if not diff_text:
        return

    emit_loop_event("loop_file_changed", project_id, agent_id, {
        "path": file_path,
        "diff": diff_text,
        "action": tracker.get_action(file_path),
        "num_changes": tracker.get_change_count(file_path),
    })
