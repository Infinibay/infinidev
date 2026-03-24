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


def build_tool_schemas(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of tools to OpenAI function-calling schemas.

    Always appends the engine pseudo-tools (step_complete, add_note, think)
    so the LLM can signal step completion, take notes, and reason.
    """
    schemas = [tool_to_openai_schema(t) for t in tools]
    schemas.append(STEP_COMPLETE_SCHEMA)
    schemas.append(ADD_NOTE_SCHEMA)
    schemas.append(THINK_SCHEMA)
    return schemas


def build_tool_dispatch(tools: list[Any]) -> dict[str, Any]:
    """Build a name→tool instance dispatch map."""
    return {t.name: t for t in tools}


def execute_tool_call(
    dispatch: dict[str, Any],
    name: str,
    arguments: str | dict[str, Any],
) -> str:
    """Execute a tool call and return the result as a string.

    Calls ``tool._run()`` directly (bypassing CrewAI's ``BaseTool.run()``)
    with kwargs filtering to strip hallucinated parameters.
    """
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
        "file_path": "path",
        "filepath": "path",
        "file": "path",
        "filename": "path",
        "directory": "path",
        "dir": "path",
        "dir_path": "path",
        "content": "new_string",
        "query": "pattern",
        "search_query": "pattern",
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

    # Execute
    try:
        result = tool._run(**args)
        return str(result) if result is not None else ""
    except Exception as exc:
        logger.warning("Tool %s raised %s: %s", name, type(exc).__name__, exc)
        return json.dumps({"error": f"Tool '{name}' failed: {exc}"})
