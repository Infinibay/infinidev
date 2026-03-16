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
                    "description": "1-2 sentence summary of what you did and key facts discovered",
                },
                "status": {
                    "type": "string",
                    "enum": ["continue", "done", "blocked"],
                    "description": "continue = more work to do, done = task complete, blocked = cannot proceed",
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


def build_tool_schemas(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of tools to OpenAI function-calling schemas.

    Always appends the step_complete schema so the LLM can signal
    step completion via tool calling.
    """
    schemas = [tool_to_openai_schema(t) for t in tools]
    schemas.append(STEP_COMPLETE_SCHEMA)
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

    # Execute
    try:
        result = tool._run(**args)
        return str(result) if result is not None else ""
    except Exception as exc:
        logger.warning("Tool %s raised %s: %s", name, type(exc).__name__, exc)
        return json.dumps({"error": f"Tool '{name}' failed: {exc}"})
