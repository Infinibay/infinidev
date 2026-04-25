"""Tool dispatch + execution for the loop engine.

Schema conversion and the pseudo-tool schema constants now live in
``loop/schema_sanitizer.py``. They are re-exported here so existing
``from infinidev.engine.loop.tools import ...`` imports keep working
after the extraction.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from infinidev.engine.loop.schema_sanitizer import (
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
    THINK_SCHEMA,
    ADD_STEP_SCHEMA,
    MODIFY_STEP_SCHEMA,
    REMOVE_STEP_SCHEMA,
)

logger = logging.getLogger(__name__)


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
    # partial_read was a 6-line wrapper that just delegated to read_file
    # with offset/limit. read_file now accepts start_line/end_line as
    # native parameters, so the wrapper added zero value. Aliased here
    # for any model that learned the old name; the parameter mapping
    # below converts (start_line, end_line) to read_file's signature
    # without the model needing to know.
    "partial_read": "read_file",
    # ``help`` collides with Python's builtin ``help()`` which confuses
    # the model — in the bridge experiment, qwen tried to run
    # ``python3 -c "help('code_interpreter')"`` three times instead of
    # calling the help tool. ``explain_tool`` is the unambiguous name
    # we recommend in the new system prompt; the alias keeps the old
    # name working so existing prompts don't break.
    "explain_tool": "help",
}


# Common hallucinations from small models — names that aren't real
# tools but map 1-to-1 to ones that are. Lives at module level so
# ``execute_tool_call`` doesn't rebuild this dict on every invocation.
_HALLUCINATION_MAP: dict[str, str] = {
    "write_file": "create_file",
    "edit_file": "replace_lines",
    "apply_patch": "replace_lines",
    "read": "read_file",
    "search": "code_search",
    "run": "execute_command",
    "run_command": "execute_command",
    "ls": "list_directory",
    "find": "glob",
    "grep": "code_search",
    "cat": "read_file",
    "vim": "replace_lines",
    "search_knowledge": "search_findings",
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


def execute_tool_call(
    dispatch: dict[str, Any],
    name: str,
    arguments: str | dict[str, Any],
    hook_metadata: dict[str, Any] | None = None,
    attachments_out: list | None = None,
) -> str:
    """Execute a tool call and return the result as a string.

    Calls ``tool._run()`` directly (bypassing CrewAI's ``BaseTool.run()``)
    with kwargs filtering to strip hallucinated parameters.

    If ``attachments_out`` is provided and the tool returned a
    ``ToolResult`` with image attachments, those ``ImageAttachment`` objects
    are appended to it. The returned string is always plain text, safe to
    embed in a ``role=tool`` message.
    """
    tool, name = _resolve_tool(dispatch, name)

    if tool is None:
        available = sorted(dispatch.keys())[:15]
        return json.dumps({
            "error": f"Unknown tool: {name}. Available tools: {', '.join(available)}",
        })

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
    try:
        result = tool._run(**args)
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

    # --- Post-tool hook ---
    ctx.event = HookEvent.POST_TOOL
    ctx.result = result_str
    hook_manager.dispatch(ctx)
    return ctx.result


# Tool failure → alternative suggestion mapping
_TOOL_ALTERNATIVES: dict[str, str] = {
    "edit_symbol": "Try replace_lines instead — read the file first to get line numbers.",
    "add_symbol": "Try replace_lines or create_file instead.",
    "remove_symbol": "Try replace_lines to delete the line range instead.",
    "partial_read": "Try read_file with the full path instead.",
    "web_fetch": "Try web_search to find the information instead.",
    "web_search": "Try execute_command with 'curl' as a fallback.",
    "code_search": "Try glob to find the file, then read_file to search its contents.",
    "create_file": "If the file already exists, use replace_lines or edit_symbol to modify it.",
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
