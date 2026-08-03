"""Tool dispatch + execution for the loop engine.

Schema conversion and the pseudo-tool schema constants now live in
``loop/schema_sanitizer.py``. They are re-exported here so existing
``from infinidev.engine.tool_dispatch import ...`` imports keep working
after the extraction.
"""

from __future__ import annotations

import inspect
import json
import logging
import uuid
from typing import Any

from infinidev.engine.schema_sanitizer import (
    _clean_schema,
    _sanitize_schema_deep,
    _simplify_node,
    _sanitize_tool_schema,
    _simplify_schema_for_small,
    tool_to_openai_schema,
    build_tool_schemas,
    STEP_COMPLETE_SCHEMA,
    ADD_NOTE_SCHEMA,
    ADD_SESSION_NOTE_SCHEMA,
    ADD_STEP_SCHEMA,
    MODIFY_STEP_SCHEMA,
    REMOVE_STEP_SCHEMA,
)

logger = logging.getLogger(__name__)

_USER_STOPPED_TOOL_ERROR = (
    "The user stopped this tool execution. Do not retry the same call unchanged; "
    "choose a narrower approach or ask the user before retrying."
)


def _mark_result_stopped_by_user(result: str) -> str:
    """Preserve partial output while making user cancellation explicit to the agent."""
    try:
        payload = json.loads(result) if result else {}
    except (json.JSONDecodeError, TypeError):
        payload = {"partial_result": result} if result else {}

    if not isinstance(payload, dict):
        payload = {"partial_result": payload}
    previous_error = payload.get("error")
    if previous_error and previous_error != _USER_STOPPED_TOOL_ERROR:
        payload["error_before_cancellation"] = previous_error
    payload["error"] = _USER_STOPPED_TOOL_ERROR
    payload["cancelled_by_user"] = True
    return json.dumps(payload)


def build_tool_dispatch(tools: list[Any]) -> dict[str, Any]:
    """Build a name→tool instance dispatch map."""
    return {t.name: t for t in tools}


# Tool name aliases for backward compatibility
_TOOL_ALIASES: dict[str, str] = {
    "edit_method": "edit_symbol",
    "add_method": "add_symbol",
    "remove_method": "remove_symbol",
    "write_file": "create_file",
    "find_definition": "search_symbols",
    # partial_read was a 6-line wrapper around read_file's offset/limit.
    # The executor still accepts its start_line/end_line arguments for
    # backward compatibility, while the public read_file schema exposes
    # offset/limit as the canonical parameters.
    "partial_read": "read_file",
    # ``help`` collides with Python's builtin ``help()`` which confuses
    # the model — in the bridge experiment, qwen tried to run
    # ``python3 -c "help('code_interpreter')"`` three times instead of
    # calling the help tool. ``explain_tool`` is the unambiguous name
    # we recommend in the new system prompt; the alias keeps the old
    # name working so existing prompts don't break.
    "explain_tool": "help",
    # read_findings and search_knowledge were the same algorithm (FTS over
    # findings) behind two names. search_knowledge took over, including the
    # no-query browse mode and the session/type filters, so the old name is
    # a pure rename — every read_findings parameter exists there unchanged.
    "read_findings": "search_knowledge",
}


# Common hallucinations from small models — names that aren't real
# tools but map 1-to-1 to ones that are. Lives at module level so
# ``execute_tool_call`` doesn't rebuild this dict on every invocation.
_HALLUCINATION_MAP: dict[str, str] = {
    "write_file": "create_file",
    "apply_patch": "edit_file",
    "str_replace": "edit_file",
    "read": "read_file",
    "search": "code_search",
    "run": "execute_command",
    "run_command": "execute_command",
    "ls": "list_directory",
    "find": "glob",
    "grep": "code_search",
    "cat": "read_file",
    "vim": "edit_file",
}


# Retired tools: the name is gone and its arguments do not survive a rename,
# so there is nothing to alias it to. Saying "unknown tool" would send the
# model hunting; naming the replacement and the shape it wants turns a dead
# end into one wasted call.
_RETIRED_TOOLS: dict[str, str] = {
    "replace_lines": (
        "replace_lines was retired. Use edit_file(file_path, old_string, "
        "new_string) — paste the exact text to replace instead of line "
        "numbers, which shift as soon as an earlier edit lands."
    ),
    "add_content_after_line": (
        "add_content_after_line was retired. Use edit_file(file_path, "
        "old_string, new_string) with the line you are inserting after as "
        "old_string, and that same line plus your new content as new_string."
    ),
    "add_content_before_line": (
        "add_content_before_line was retired. Use edit_file(file_path, "
        "old_string, new_string) with the line you are inserting before as "
        "old_string, and your new content plus that line as new_string."
    ),
    "edit_symbol": (
        "edit_symbol was retired. Read the symbol (get_symbol_code) and "
        "replace its body with edit_file(file_path, old_string, new_string)."
    ),
    "add_symbol": (
        "add_symbol was retired. Use edit_file(file_path, old_string, "
        "new_string) anchored on the line you want to insert after."
    ),
    "remove_symbol": (
        "remove_symbol was retired. Use edit_file(file_path, old_string, "
        "new_string='') with the symbol's source as old_string."
    ),
    "multi_edit_file": (
        "multi_edit_file was retired. Call edit_file once per change; each "
        "one validates its own match."
    ),
}


def _resolve_tool(
    dispatch: dict[str, Any], name: str,
) -> tuple[Any | None, str]:
    """Resolve a tool name to ``(tool, canonical_name)`` using the
    alias → case-insensitive → hallucination cascade.

    Returns ``(None, name)`` if nothing matches. Kept as a single helper
    so ``execute_tool_call`` doesn't have to interleave three lookup
    tables with the rest of its dispatch logic. Logs each correction
    once at INFO so misbehaving models show up in the logs.
    """
    # 1. Back-compat aliases (deprecated names that still resolve)
    if name in _TOOL_ALIASES:
        canonical = _TOOL_ALIASES[name]
        logger.info("Tool alias: '%s' -> '%s'", name, canonical)
        name = canonical

    tool = dispatch.get(name)
    if tool is not None:
        return tool, name

    # 2. Case-insensitive match
    lower = name.lower()
    for rname, rtool in dispatch.items():
        if rname.lower() == lower:
            logger.info("Tool case-corrected: '%s' → '%s'", name, rname)
            return rtool, rname

    # 3. Hallucinations from small models
    canonical = _HALLUCINATION_MAP.get(name) or _TOOL_ALIASES.get(name)
    if canonical:
        tool = dispatch.get(canonical)
        if tool is not None:
            logger.info("Tool hallucination recovered: '%s' → '%s'", name, canonical)
            return tool, canonical

    return None, name


# How many alternatives an unknown-tool error offers. Enough to contain the
# right one, few enough that the model reads them all.
_SUGGESTION_LIMIT = 8
# Below this score a candidate is not a near-miss, it is padding. Without a
# floor the list fills to the limit with whatever ranked next, which puts
# unrelated tools beside the right answer and dilutes it.
_SUGGESTION_FLOOR = 0.5
# …but never answer with nothing: a wrong guess the model can reject still
# beats an error that offers no way forward.
_MIN_SUGGESTIONS = 3


def _unknown_tool_message(dispatch: dict[str, Any], name: str) -> str:
    """Name the tools the model probably meant.

    This used to answer with ``sorted(dispatch)[:15]`` — the alphabetically
    first fifteen of ninety-odd, which for this toolset is everything from
    ``add_content_after_line`` to ``delete_report`` and never once the tool
    that was actually wanted. Ranking by similarity to what the model typed
    turns a dead end into a correction it can make on the next call.
    """
    from difflib import SequenceMatcher

    # A retired tool is not a typo — the model is remembering a real tool that
    # used to exist. Guessing at neighbours would be noise; say what replaced it.
    if (retired := _RETIRED_TOOLS.get(name)):
        return retired

    typed = name.lower()

    def closeness(candidate: str) -> float:
        low = candidate.lower()
        # A shared prefix or suffix ("ken_search_file" → "ken_search_files")
        # is a far stronger signal than raw edit distance, which would rank
        # every ken_* tool identically.
        bonus = 0.3 if low.startswith(typed[:6]) or typed.startswith(low[:6]) else 0.0
        return SequenceMatcher(None, typed, low).ratio() + bonus

    scored = sorted(
        ((closeness(candidate), candidate) for candidate in dispatch), reverse=True,
    )
    close = [n for score, n in scored[:_SUGGESTION_LIMIT] if score >= _SUGGESTION_FLOOR]
    ranked = close or [n for _, n in scored[:_MIN_SUGGESTIONS]]
    return (
        f"Unknown tool: {name}. Did you mean one of: {', '.join(ranked)}? "
        f"Call help() to list every tool you have."
    )


def execute_tool_call(
    dispatch: dict[str, Any],
    name: str,
    arguments: str | dict[str, Any],
    hook_metadata: dict[str, Any] | None = None,
    attachments_out: list | None = None,
) -> str:
    """Execute a tool call and return the result as a string.

    Calls ``tool._run()`` directly with kwargs filtering to strip hallucinated
    parameters.

    If ``attachments_out`` is provided and the tool returned a
    ``ToolResult`` with image attachments, those ``ImageAttachment`` objects
    are appended to it. The returned string is always plain text, safe to
    embed in a ``role=tool`` message.
    """
    tool, name = _resolve_tool(dispatch, name)

    if tool is None:
        return json.dumps({"error": _unknown_tool_message(dispatch, name)})

    # Parse arguments
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid JSON arguments: {arguments[:200]}"})
    else:
        args = arguments or {}

    if not isinstance(args, dict):
        return json.dumps({"error": f"Expected dict arguments, got {type(args).__name__}"})

    # Auto-correct common parameter name aliases that LLMs frequently use.
    # Maps wrong_param -> correct_param. Applied globally to all tools
    # because the wrong names listed here never collide with any real
    # parameter.
    _PARAM_ALIASES = {
        "old_str": "old_string",
        "new_str": "new_string",
        # All tools now use file_path — alias common LLM variants
        "path": "file_path",
        "filepath": "file_path",
        "file": "file_path",
        "filename": "file_path",
        "name": "file_path",  # safe: only applies when tool has "file_path" but not "name"
        "directory": "file_path",
        "dir": "file_path",
        "dir_path": "file_path",
        # "content" is a valid param in create_file, replace_lines — no longer alias to new_string
        "query": "pattern",
        "search_query": "pattern",
        # Line range aliases (gpt-oss uses line_start/line_end)
        "line_start": "start_line",
        "line_end": "end_line",
        # Command aliases
        "cmd": "command",
        # Replace aliases
        "replacement": "content",
        "new_body": "new_code",
    }

    # Per-tool aliases — aplied BEFORE the global ones. Used when a
    # wrong param name would collide with a real param somewhere else
    # (e.g. ``command`` is the correct param for execute_command but
    # the wrong one for code_interpreter, where the model meant ``code``).
    # Only the listed tool gets the rewrite.
    _TOOL_SPECIFIC_PARAM_ALIASES = {
        "code_interpreter": {
            "command": "code",
            "script": "code",
            "python": "code",
            "source": "code",
        },
    }

    # Validate kwargs against _run() signature — reject unknown parameters
    # so the LLM learns the correct schema instead of silently losing data.
    try:
        sig = inspect.signature(tool._run)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not accepts_var_kw:
            allowed = set(sig.parameters.keys())
            # Apply tool-specific aliases first (they take priority
            # over the global ones because they exist precisely for
            # cases where a global alias would be wrong).
            tool_aliases = _TOOL_SPECIFIC_PARAM_ALIASES.get(name, {})
            fixed = {}
            for key, value in list(args.items()):
                if key not in allowed and key in tool_aliases:
                    correct = tool_aliases[key]
                    if correct in allowed and correct not in args:
                        logger.info(
                            "Tool %s: auto-corrected param '%s' -> '%s' (per-tool alias)",
                            name, key, correct,
                        )
                        fixed[correct] = value
                        del args[key]
            # Then global aliases
            for key, value in list(args.items()):
                if key not in allowed and key in _PARAM_ALIASES:
                    correct = _PARAM_ALIASES[key]
                    if correct in allowed and correct not in args:
                        logger.info("Tool %s: auto-corrected param '%s' -> '%s'", name, key, correct)
                        fixed[correct] = value
                        del args[key]
            args.update(fixed)
            # Silently strip metadata params that LLMs commonly add
            _METADATA_PARAMS = {"description", "reason", "explanation", "language"}
            for meta in _METADATA_PARAMS:
                if meta in args and meta not in allowed:
                    logger.debug("Tool %s: stripping metadata param '%s'", name, meta)
                    del args[meta]
            extra = set(args.keys()) - allowed
            # Zero-arg tools: the rejection message "valid params are: ."
            # is incoherent to the LLM and it concludes the tool is broken.
            # Silently drop extras instead — the tool takes no args so there
            # is nothing to validate, and the hallucinated kwargs are safe
            # to ignore.
            if extra and not allowed:
                logger.debug(
                    "Tool %s: zero-arg tool, dropping hallucinated kwargs %s",
                    name, extra,
                )
                for k in extra:
                    del args[k]
                extra = set()
            if extra:
                logger.warning("Tool %s: unexpected kwargs %s", name, extra)
                # Stronger error message — small models that see
                # "does not accept parameter" tend to conclude "tool
                # doesn't exist". The phrasing below makes it
                # IMPOSSIBLE to misread: the tool exists, the call
                # was almost right, fix the param and retry.
                return json.dumps({
                    "error": (
                        f"Tool '{name}' EXISTS and is callable — your "
                        f"call was rejected only because of wrong "
                        f"parameter name(s): {', '.join(sorted(extra))}. "
                        f"The valid parameter names for this tool are: "
                        f"{', '.join(sorted(allowed))}. Re-call the same "
                        f"tool with the corrected parameter name(s) — "
                        f"do NOT switch to a different tool, do NOT "
                        f"conclude the tool is unavailable."
                    ),
                })
    except (ValueError, TypeError):
        pass  # Can't inspect, pass all args

    # Coerce argument types based on _run() annotations.
    # LLMs frequently send ints as strings (e.g. "300" instead of 300),
    # or dicts/lists for params that expect simple types.
    try:
        import typing, types
        sig = inspect.signature(tool._run)
        for p_name, p in sig.parameters.items():
            if p_name not in args:
                continue
            ann = p.annotation
            if ann is inspect.Parameter.empty:
                continue
            # Unwrap Optional[X] / X | None to get the inner type
            _target = ann
            origin = getattr(ann, "__origin__", None)
            if origin is types.UnionType or origin is typing.Union:
                _inner = [a for a in typing.get_args(ann) if a is not type(None)]
                if _inner:
                    _target = _inner[0]
            val = args[p_name]
            # Skip if already correct type or None
            if val is None:
                continue
            if _target is int and not isinstance(val, int):
                try:
                    args[p_name] = int(str(val) if not isinstance(val, str) else val)
                except (ValueError, TypeError):
                    pass
            elif _target is float and not isinstance(val, (int, float)):
                try:
                    args[p_name] = float(str(val) if not isinstance(val, str) else val)
                except (ValueError, TypeError):
                    pass
            elif _target is bool and isinstance(val, str):
                args[p_name] = val.lower() in ("true", "1", "yes")
            elif _target is str and not isinstance(val, str):
                if isinstance(val, dict):
                    # LLM wrapped a simple value in a dict — try to extract it.
                    # e.g. {"command": "ls", "cwd": "."} for param "command"
                    # → extract "ls" and promote extra keys (cwd, timeout, env)
                    #   to top-level args if the tool accepts them.
                    if p_name in val:
                        # {"command": "ls"} → extract "ls"
                        extracted = val.pop(p_name)
                        args[p_name] = str(extracted)
                        # Promote remaining keys as extra args
                        for ek, ev in val.items():
                            if ek not in args:
                                args[ek] = ev
                    elif len(val) == 1:
                        # Single key dict — use the value
                        args[p_name] = str(next(iter(val.values())))
                    else:
                        # Try common aliases: cmd, value, text, code, query
                        for alias in ("cmd", "value", "text", "code", "query", "content"):
                            if alias in val:
                                args[p_name] = str(val[alias])
                                break
                        else:
                            args[p_name] = str(val)
                elif isinstance(val, list):
                    args[p_name] = " ".join(str(v) for v in val)
                else:
                    args[p_name] = str(val)
    except (ValueError, TypeError):
        pass

    # Check for missing required parameters before calling
    try:
        sig = inspect.signature(tool._run)
        required_params = {
            p_name for p_name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        missing = required_params - set(args.keys())
        if missing:
            return json.dumps({
                "error": (
                    f"Tool '{name}' is missing required parameter(s): "
                    f"{', '.join(sorted(missing))}. "
                    f"Valid parameters are: {', '.join(sorted(sig.parameters.keys()))}. "
                    f"Re-call the tool with all required parameters."
                ),
            })
    except (ValueError, TypeError):
        pass

    # --- Pre-tool hook ---
    from infinidev.engine.hooks.hooks import hook_manager, HookContext, HookEvent

    _meta = dict(hook_metadata) if hook_metadata else {}
    _meta["tool_run_id"] = str(_meta.get("tool_run_id") or uuid.uuid4().hex)
    ctx = HookContext(
        event=HookEvent.PRE_TOOL,
        tool_name=name,
        arguments=dict(args),
        metadata=_meta,
        project_id=_meta.pop("project_id", 0),
        agent_id=_meta.pop("agent_id", ""),
    )
    hook_manager.dispatch(ctx)
    if ctx.skip:
        return ctx.result or json.dumps({"skipped": True, "tool": name})
    args = ctx.arguments

    # Execute
    cancelled_by_user = False
    try:
        from infinidev.engine.tool_progress import (
            is_tool_cancelled,
            tool_progress_context,
        )

        with tool_progress_context(
            ctx.metadata["tool_run_id"],
            ctx.project_id,
            ctx.agent_id,
            cancel_event=ctx.metadata.get("cancel_event"),
        ):
            result = None if is_tool_cancelled() else tool._run(**args)
            cancelled_by_user = is_tool_cancelled()
        # Unwrap ToolResult (text + optional image attachments). The text
        # goes into the role=tool message; attachments are surfaced via
        # attachments_out so the engine can push them as a follow-up
        # multimodal user message.
        from infinidev.tools.base.base_tool import ToolResult, normalize_tool_result
        if isinstance(result, ToolResult):
            text, atts = normalize_tool_result(result)
            result_str = text
            if attachments_out is not None and atts:
                attachments_out.extend(atts)
        else:
            result_str = str(result) if result is not None else ""
    except Exception as exc:
        logger.warning("Tool %s raised %s: %s", name, type(exc).__name__, exc)
        suggestion = _suggest_alternative(name, str(exc))
        error_msg = f"Tool '{name}' failed: {exc}"
        if suggestion:
            error_msg += f"\n\nSuggestion: {suggestion}"
        result_str = json.dumps({"error": error_msg})

        cancel_event = ctx.metadata.get("cancel_event")
        cancelled_by_user = bool(
            cancel_event is not None and cancel_event.is_set()
        )

    if cancelled_by_user:
        result_str = _mark_result_stopped_by_user(result_str)

    # --- Post-tool hook ---
    ctx.event = HookEvent.POST_TOOL
    ctx.result = result_str
    hook_manager.dispatch(ctx)
    return ctx.result


# Tool failure → alternative suggestion mapping
_TOOL_ALTERNATIVES: dict[str, str] = {
    "edit_symbol": (
        "Read the symbol with get_symbol_code, then replace its exact source "
        "with edit_file."
    ),
    "add_symbol": "Use edit_file to insert into an existing file, or create_file for a new one.",
    "remove_symbol": "Use edit_file with the symbol source and new_string=''.",
    "partial_read": "Use read_file with file_path, offset, and limit.",
    "web_fetch": "Try web_search to find the information instead.",
    "web_search": "Try execute_command with 'curl' as a fallback.",
    "code_search": "Try glob to find the file, then read_file to search its contents.",
    "create_file": "If the file already exists, use edit_file to modify it.",
}


def _suggest_alternative(tool_name: str, error_msg: str) -> str:
    """Suggest an alternative tool when one fails."""
    # Direct mapping
    if tool_name in _TOOL_ALTERNATIVES:
        return _TOOL_ALTERNATIVES[tool_name]
    # File not found → suggest glob
    if "not found" in error_msg.lower() or "no such file" in error_msg.lower():
        return "File not found. Use glob or list_directory to find the correct path."
    # Permission denied
    if "permission" in error_msg.lower():
        return "Permission denied. Check the file path and try a different approach."
    return ""
