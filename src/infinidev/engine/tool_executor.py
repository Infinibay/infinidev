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

# Tools that modify files — tracked for diff generation.
# ``edit_file`` and ``apply_patch`` are included as the raw names the
# model may emit before the hallucination alias rewrites them to
# ``replace_lines``; keeping them here ensures batching and diff
# capture see the write intent even on the unresolved name.
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


def _evict_symbol_cache(state: LoopState) -> None:
    """Drop every ``[symbol] X`` entry from the opened-files cache.

    Called whenever an edit happens to a file the model previously
    inspected via ``get_symbol_code``. We don't track which symbols
    came from which file in the cache key (that would require
    extending the OpenedFile model), so we evict pessimistically:
    on any edit, all cached symbol bodies are dropped. This is
    correct (no stale code in cache) and cheap in the common case
    (the model rarely has more than 3-5 symbols cached at once).
    """
    stale = [k for k in state.opened_files if k.startswith("[symbol] ")]
    for k in stale:
        state.opened_files.pop(k, None)


def _reread_and_cache(state: LoopState, path: str) -> None:
    """Re-read a file from disk and refresh the opened files cache.

    Also evicts every cached symbol body — those came from a tool
    call BEFORE this edit and may now reflect stale source. The
    next ``get_symbol_code`` call will fetch the fresh version.
    The eviction runs unconditionally (even if the file re-read
    fails) because we KNOW an edit was just attempted; conservatism
    here prevents the model from acting on stale cached source.
    """
    import os as _os
    _evict_symbol_cache(state)
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

    Most handlers key the cache by file path, so we extract ``path``
    upfront for their convenience. But some tools (e.g.
    ``get_symbol_code``) key by symbol name instead — those handlers
    receive ``path=None`` and pull what they need from ``args``.
    """
    import os as _os

    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(args, dict):
        return

    handler = _CACHE_HANDLERS.get(tool_name)
    if not handler:
        return

    from infinidev.tools.base.context import get_current_workspace_path
    ws = get_current_workspace_path() or _os.getcwd()

    # Path is a convenience for handlers that key by file. Symbol-based
    # handlers get None and use args["name"] / args["symbol"] instead.
    path = args.get("path") or args.get("file_path")
    if path and not _os.path.isabs(path):
        path = _os.path.normpath(_os.path.join(ws, path))

    handler(state, path, args, result, ws)


def _is_error_result(result: Any) -> bool:
    """True if a tool result is missing, empty, or encodes an error payload.

    Centralized so every cache handler uses the same guard — previously
    each site did ``result and not result.strip().startswith('{"error')``
    which crashes if a tool returns ``None`` or a non-string. The empty
    check (``not result``) preserves the original truthy semantics: an
    empty string was never cached as a successful read.
    """
    if not isinstance(result, str) or not result:
        return True
    return result.strip().startswith('{"error')


def _cache_read(state, path, args, result, ws):
    if not _is_error_result(result):
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


def _cache_list_dir(state, path, args, result, ws):
    if not _is_error_result(result):
        state.cache_file(f"[dir] {path}", result)


def _cache_glob(state, path, args, result, ws):
    pattern = args.get("pattern", "")
    if pattern and not _is_error_result(result):
        state.cache_file(f"[glob] {pattern}", result)


def _cache_code_search(state, path, args, result, ws):
    query = args.get("query") or args.get("pattern") or args.get("search_query", "")
    if query and not _is_error_result(result):
        state.cache_file(f"[search] {query}", result)


def _cache_get_symbol_code(state, path, args, result, ws):
    """Cache the source body of a symbol so the next step can read it
    from the prompt instead of re-issuing get_symbol_code.

    Diagnosed 2026-04-07: small models routinely re-fetch the same
    symbol across consecutive steps because the raw tool output is
    discarded between steps and step summaries can't fit ~600 tokens
    of source code. Caching the body under a stable key
    (``[symbol] qualified_name``) lets the prompt builder include it
    in the next ``<opened-files>`` block — same TTL/eviction rules as
    file content, same edit-invalidation guarantees (because any
    edit_symbol/replace_lines on the underlying file evicts ALL
    cached entries for that file via _cache_symbol_edit).
    """
    name = args.get("name") or args.get("symbol") or args.get("qualified_name", "")
    if not name:
        return
    if _is_error_result(result):
        return
    # The model expects to find this entry by symbol name, so use a
    # distinctive prefix that won't collide with any real file path.
    state.cache_file(f"[symbol] {name}", result)


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
    "list_directory": _cache_list_dir,
    "glob": _cache_glob,
    "code_search": _cache_code_search,
    "get_symbol_code": _cache_get_symbol_code,
}


# ── Anchored memory injection ─────────────────────────────────────────────
#
# When the agent calls a tool that touches a concrete anchor (a file
# path, a symbol name, a tool name, or produces an error pattern),
# ``annotate_with_memory`` runs a cheap DB lookup against the findings
# table for anything with a matching anchor and appends a compact
# ``[📌 Known lessons for this <anchor>]`` block to the tool result.
# The model sees the lesson next to the data that provoked it — which
# is dramatically more effective than pushing lessons into the system
# prompt and hoping they're still in attention 20 iterations later.
#
# The handlers extract anchors from the tool args (``read_file`` →
# ``path``, ``edit_symbol`` → ``name``, etc.). Tools without a
# meaningful anchor are not in the dispatch table; they pay zero cost.


def _anchor_from_file_arg(args: dict, result: str, ws: str) -> dict:
    import os as _os
    path = args.get("path") or args.get("file_path")
    if not path:
        return {}
    if not _os.path.isabs(path):
        path = _os.path.normpath(_os.path.join(ws, path))
    return {"anchor_file": path}


def _anchor_from_symbol_arg(args: dict, result: str, ws: str) -> dict:
    name = args.get("name") or args.get("symbol") or args.get("qualified_name")
    if not name:
        return {}
    out: dict = {"anchor_symbol": name}
    # A symbol edit / read also has a file anchor derived from the
    # tool result; pick it up so file-anchored lessons fire too.
    path = _extract_path_from_result(result)
    if path:
        out["anchor_file"] = path
    return out


def _anchor_from_command_arg(args: dict, result: str, ws: str) -> dict:
    # Commands don't have a natural anchor, but the command name
    # itself (first token) can match tool-level rules. e.g. a
    # ``pytest ...`` command matches rules anchored to ``pytest``.
    cmd = args.get("command") or args.get("cmd") or ""
    if not cmd or not isinstance(cmd, str):
        return {}
    first = cmd.strip().split(None, 1)[0] if cmd.strip() else ""
    if not first:
        return {}
    return {"anchor_tool": first}


# Per-tool anchor extractors. Keyed by tool name; each returns the
# subset of anchor kwargs to pass to ``get_anchored_findings``.
_MEMORY_HANDLERS: dict = {
    "read_file": _anchor_from_file_arg,
    "partial_read": _anchor_from_file_arg,
    "create_file": _anchor_from_file_arg,
    "edit_file": _anchor_from_file_arg,
    "replace_lines": _anchor_from_file_arg,
    "add_content_after_line": _anchor_from_file_arg,
    "add_content_before_line": _anchor_from_file_arg,
    "list_directory": _anchor_from_file_arg,
    "get_symbol_code": _anchor_from_symbol_arg,
    "edit_symbol": _anchor_from_symbol_arg,
    "add_symbol": _anchor_from_symbol_arg,
    "remove_symbol": _anchor_from_symbol_arg,
    "search_symbols": _anchor_from_symbol_arg,
    "execute_command": _anchor_from_command_arg,
}


def annotate_with_memory(
    tool_name: str,
    arguments: str | dict,
    result: str,
    project_id: int,
) -> str:
    """Append matching anchored memories to a tool result.

    Returns the result unchanged if no anchor can be extracted, no
    memory matches, or the result already encodes an error (errors
    are already noisy; no point adding lessons next to them). All
    exceptions are swallowed — memory injection is best-effort.
    """
    handler = _MEMORY_HANDLERS.get(tool_name)
    if handler is None:
        return result
    if not isinstance(result, str) or _is_error_result_annotation(result):
        return result
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        if not isinstance(args, dict):
            return result
        from infinidev.tools.base.context import get_current_workspace_path
        ws = get_current_workspace_path() or ""
        anchors = handler(args, result, ws)
        if not anchors:
            return result
        from infinidev.db.service import get_anchored_findings
        matches = get_anchored_findings(
            project_id=project_id, limit=3, **anchors,
        )
        if not matches:
            return result
        lines: list[str] = ["", "[📌 Known lessons relevant to this action:]"]
        for m in matches:
            kind = (m.get("finding_type") or "lesson").upper()
            topic = m.get("topic") or ""
            content = (m.get("content") or "").strip()
            if topic:
                lines.append(f"- {kind} — {topic}: {content}")
            else:
                lines.append(f"- {kind}: {content}")
        return result + "\n" + "\n".join(lines)
    except Exception:
        return result


def _is_error_result_annotation(result: str) -> bool:
    """Local guard for annotate_with_memory. Same semantics as
    ``_is_error_result`` but named differently to avoid shadowing."""
    if not isinstance(result, str) or not result:
        return True
    return result.strip().startswith('{"error')


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
    hook_metadata: dict[str, Any] | None = None,
) -> list[tuple]:
    """Execute a batch of read-only tool calls in parallel.

    Returns list of (tc, result) tuples in original order.
    """
    if len(batch) <= 1:
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

    # Forward silent-deletion info from the tool result to the tracker so
    # the post-task verification can catch orphaned references.
    try:
        res_obj = json.loads(result) if isinstance(result, str) else result
    except (ValueError, TypeError):
        return
    if isinstance(res_obj, dict):
        removed = res_obj.get("removed_symbols")
        if isinstance(removed, list) and removed:
            tracker.record_deleted_symbols(file_path, [str(s) for s in removed])
