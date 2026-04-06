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


def _sanitize_schema_deep(schema: dict[str, Any]) -> dict[str, Any]:
    """Aggressively simplify a JSON schema for providers that reject anyOf/oneOf.

    Qwen/DashScope and some other providers reject complex schema constructs.
    This flattens anyOf/oneOf to the first non-null type and recurses into
    nested properties and array items.
    """
    import copy
    schema = copy.deepcopy(schema)
    _simplify_node(schema)
    return schema


def _simplify_node(node: dict[str, Any]) -> None:
    """Recursively simplify a schema node in-place."""
    # Resolve anyOf/oneOf → pick first non-null type
    for key in ("anyOf", "oneOf"):
        if key in node:
            variants = node.pop(key)
            chosen = None
            for v in variants:
                if isinstance(v, dict) and v.get("type") != "null":
                    chosen = v
                    break
            if chosen:
                # Merge the chosen variant into the node
                for k, v in chosen.items():
                    if k not in node:
                        node[k] = v

    # Remove unsupported keywords
    for drop in ("$defs", "definitions", "title", "default", "examples"):
        node.pop(drop, None)

    # Recurse into properties
    for prop in node.get("properties", {}).values():
        if isinstance(prop, dict):
            _simplify_node(prop)

    # Recurse into array items
    items = node.get("items")
    if isinstance(items, dict):
        _simplify_node(items)


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
            "You MUST call this after finishing each step. "
            "WARNING: After this call, ALL tool outputs and conversation from this step will be discarded. "
            "Only the summary and your notes (add_note) survive to the next step. "
            "Before calling this, save key facts via add_note (file paths, function names, decisions). "
            "Before status='done', call add_session_note with what you learned. "
            "To modify the plan, use add_step/modify_step/remove_step BEFORE calling this."
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
            "IMPORTANT: Save a fact to your persistent memory. Your context is rebuilt "
            "from scratch each step — anything not saved here is PERMANENTLY LOST. "
            "Call this after every file read, discovery, or decision. "
            "Notes appear in <notes> at every step. Max 20 notes."
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


ADD_STEP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_step",
        "description": (
            "Add a new step to the plan WITHOUT completing the current step. "
            "Use this when you discover new work mid-step. "
            "Does NOT count as a tool call. "
            "If index is omitted, the step is appended at the end of the plan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Step number (position in plan). Omit to append at end.",
                },
                "title": {
                    "type": "string",
                    "description": "Short step title naming FILE, FUNCTION, and CHANGE",
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed explanation of how to approach the step (optional)",
                },
            },
            "required": ["title"],
        },
    },
}


MODIFY_STEP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "modify_step",
        "description": (
            "Modify the title or explanation of an existing pending step "
            "WITHOUT completing the current step. "
            "Does NOT count as a tool call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Step number to modify",
                },
                "title": {
                    "type": "string",
                    "description": "New title (leave empty to keep current)",
                },
                "explanation": {
                    "type": "string",
                    "description": "New explanation (leave empty to keep current)",
                },
            },
            "required": ["index"],
        },
    },
}


REMOVE_STEP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "remove_step",
        "description": (
            "Remove a pending step from the plan WITHOUT completing the current step. "
            "Does NOT count as a tool call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Step number to remove",
                },
            },
            "required": ["index"],
        },
    },
}


def build_tool_schemas(tools: list[Any], *, small_model: bool = False) -> list[dict[str, Any]]:
    """Convert a list of tools to OpenAI function-calling schemas.

    Always appends the engine pseudo-tools (step_complete, add_note)
    so the LLM can signal step completion and take notes.
    The ``think`` pseudo-tool is excluded for small models to prevent
    reasoning bloat (small models waste tokens on think → reason → think loops).
    """
    schemas = [tool_to_openai_schema(t) for t in tools]
    schemas.append(STEP_COMPLETE_SCHEMA)
    schemas.append(ADD_NOTE_SCHEMA)
    schemas.append(ADD_SESSION_NOTE_SCHEMA)
    # think pseudo-tool disabled — models abuse it to loop without acting
    # Plan tools (add_step, modify_step, remove_step) are real tools
    # registered in META_TOOLS — they get their schemas via tool_to_openai_schema().

    # Deep-sanitize schemas for providers that reject anyOf/oneOf/complex constructs
    from infinidev.config.model_capabilities import get_model_capabilities
    if get_model_capabilities().needs_schema_sanitization:
        schemas = [_sanitize_tool_schema(s) for s in schemas]

    # For small models: shorten descriptions and remove "explore" status
    if small_model:
        schemas = [_simplify_schema_for_small(s) for s in schemas]

    return schemas


def _sanitize_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Apply deep sanitization to a single tool schema."""
    import copy
    schema = copy.deepcopy(schema)
    params = schema.get("function", {}).get("parameters")
    if params:
        sanitized = _sanitize_schema_deep(params)
        schema["function"]["parameters"] = sanitized
    return schema


def _simplify_schema_for_small(schema: dict[str, Any]) -> dict[str, Any]:
    """Simplify a tool schema for small models (<40B).

    - Shortens descriptions to ≤120 chars
    - Removes 'explore' from step_complete status enum
    - Strips optional parameter descriptions to save tokens
    """
    import copy
    schema = copy.deepcopy(schema)
    func = schema.get("function", {})

    # Shorten description
    desc = func.get("description", "")
    if len(desc) > 120:
        func["description"] = desc[:117] + "..."

    # Remove 'explore' status from step_complete (confuses small models)
    if func.get("name") == "step_complete":
        props = func.get("parameters", {}).get("properties", {})
        status_prop = props.get("status", {})
        if "enum" in status_prop:
            status_prop["enum"] = [s for s in status_prop["enum"] if s != "explore"]

    return schema


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
        # Case-insensitive match
        for rname, rtool in dispatch.items():
            if rname.lower() == name.lower():
                tool, name = rtool, rname
                logger.info("Tool case-corrected: '%s' → '%s'", name, rname)
                break

    if tool is None:
        # Common hallucinations from small models
        _HALLUCINATION_MAP = {
            "write_file": "create_file",
            "edit_file": "replace_lines",
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
        canonical = _HALLUCINATION_MAP.get(name) or _TOOL_ALIASES.get(name)
        if canonical:
            tool = dispatch.get(canonical)
            if tool:
                logger.info("Tool hallucination recovered: '%s' → '%s'", name, canonical)
                name = canonical

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
