"""Tool schema conversion and execution for the loop engine.

Converts InfinibayBaseTool instances to OpenAI function-calling format
and dispatches tool calls by name.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove Pydantic v2 artifacts that confuse LLM providers."""
    schema.pop("title", None)
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    # Recurse into properties
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return schema


def tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    """Convert a InfinibayBaseTool to an OpenAI function-calling tool schema."""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    if hasattr(tool, "args_schema") and tool.args_schema is not None:
        try:
            parameters = tool.args_schema.model_json_schema()
        except Exception:
            try:
                parameters = tool.args_schema.schema()
            except Exception:
                pass
        parameters = _clean_schema(parameters)

    # Ensure required fields
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "")[:1024],
            "parameters": parameters,
        },
    }


STEP_COMPLETE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "step_complete",
        "description": (
            "Signal that the current step is complete. "
            "You MUST call this after finishing each step."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Structured summary (~150 tokens): Read: files+findings | Changed: files+edits | Remaining: next work | Decisions: key choices. Skip empty categories.",
                },
                "status": {
                    "type": "string",
                    "enum": ["continue", "done", "blocked", "explore"],
                    "description": "continue = more work to do, done = task complete, blocked = cannot proceed, explore = delegate sub-problem to exploration tree",
                },
                "next_steps": {
                    "type": "array",
                    "description": "Operations to update the plan (add/modify/remove steps)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["add", "modify", "remove"],
                            },
                            "index": {"type": "integer"},
                            "description": {"type": "string"},
                        },
                        "required": ["op", "index"],
                    },
                },
                "final_answer": {
                    "type": "string",
                    "description": "When status=done, the final result to return",
                },
            },
            "required": ["summary", "status"],
        },
    },
}


ADD_NOTE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_note",
        "description": (
            "Save a note to the task scratchpad. Notes persist across all steps "
            "and are always visible in the <notes> block. Use for: key decisions, "
            "file paths found, things to remember, warnings to yourself. "
            "Notes are short (1-2 sentences each). Max 20 notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to save (1-2 sentences)",
                },
            },
            "required": ["note"],
        },
    },
}


ADD_SESSION_NOTE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_session_note",
        "description": (
            "Save a note that persists across tasks in this session. Unlike add_note "
            "(which resets each task), session notes survive until the session ends. "
            "Use for: project-wide context, user preferences discovered during work, "
            "cross-task decisions, and anything the next task will need. "
            "Max 10 session notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The session note to save (1-2 sentences)",
                },
            },
            "required": ["note"],
        },
    },
}


THINK_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "think",
        "description": (
            "Think through a problem before acting. Use this when you need to "
            "reason about what to do next, analyze an error, or plan your approach. "
            "Your reasoning is shown to the user as a progress update. "
            "This does NOT count as a tool call — use it freely."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your reasoning, analysis, or plan",
                },
            },
            "required": ["reasoning"],
        },
    },
}


GENERATE_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_question",
        "description": (
            "Generate one investigation question about the task. "
            "Call this once per question. When you have generated enough "
            "questions, call step_complete with status='done'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A specific investigation question answerable by reading "
                        "code, running tests, or searching the project."
                    ),
                },
                "intent": {
                    "type": "string",
                    "description": (
                        "What you hope to learn: find_code, find_patterns, "
                        "check_tests, find_config, find_dependents, reproduce, "
                        "baseline, understand_spec"
                    ),
                },
            },
            "required": ["question", "intent"],
        },
    },
}


def build_tool_schemas(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of tools to OpenAI function-calling schemas.

    Always appends the engine pseudo-tools (step_complete, add_note, think)
    so the LLM can signal step completion, take notes, and reason.
    """
    schemas = [tool_to_openai_schema(t) for t in tools]
    schemas.append(STEP_COMPLETE_SCHEMA)
    schemas.append(ADD_NOTE_SCHEMA)
    schemas.append(ADD_SESSION_NOTE_SCHEMA)
    schemas.append(THINK_SCHEMA)
    return schemas


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
}


def execute_tool_call(
    dispatch: dict[str, Any],
    name: str,
    arguments: str | dict[str, Any],
    hook_metadata: dict[str, Any] | None = None,
) -> str:
    """Execute a tool call and return the result as a string.

    Calls ``tool._run()`` directly (bypassing CrewAI's ``BaseTool.run()``)
    with kwargs filtering to strip hallucinated parameters.
    """
    # Resolve tool name aliases
    if name in _TOOL_ALIASES:
        canonical = _TOOL_ALIASES[name]
        logger.info("Tool alias: '%s' -> '%s'", name, canonical)
        name = canonical

    tool = dispatch.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

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
    # Maps (tool_name, wrong_param) -> correct_param.
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
            # Try to fix unknown params via aliases before rejecting
            fixed = {}
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
            if extra:
                logger.warning("Tool %s: unexpected kwargs %s", name, extra)
                return json.dumps({
                    "error": (
                        f"Tool '{name}' does not accept parameter(s): "
                        f"{', '.join(sorted(extra))}. "
                        f"Valid parameters are: {', '.join(sorted(allowed))}. "
                        f"Re-call the tool with the correct parameter names."
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
    from infinidev.engine.hooks import hook_manager, HookContext, HookEvent

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
