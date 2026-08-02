"""Schema-backed help for registered tools and engine pseudo-tools."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.meta.help_content import HELP_CONTENT, _CATEGORY_INDEX
from infinidev.tools.meta.help_input import HelpInput
from infinidev.tools.mcp_bridge import discover_mcp_tool_classes


class HelpTool(InfinibayBaseTool):
    name: str = "help"
    description: str = "Get schema-backed help and examples for any available tool."
    args_schema: Type[BaseModel] = HelpInput

    def _run(self, context: str | None = None) -> str:
        topic = context.strip().lower() if context is not None else None
        registered = _registered_tools()
        pseudo = _pseudo_tool_schemas()

        if topic is None:
            return _render_overview(registered, pseudo)
        if topic in _CATEGORY_INDEX or topic == "mcp":
            return _render_category(topic, registered, pseudo)

        # The code-interpreter bridge has a richer hand-written reference than
        # its outer Pydantic schema can express. Every other real tool is
        # rendered from the live schema so renamed arguments cannot leave help
        # silently stale.
        if topic in HELP_CONTENT:
            return HELP_CONTENT[topic]
        if topic in registered:
            return _render_registered_tool(registered[topic])
        if topic in pseudo:
            return _render_schema_tool(pseudo[topic])

        from infinidev.engine.tool_dispatch import _RETIRED_TOOLS, _TOOL_ALIASES

        if topic in _RETIRED_TOOLS:
            return _RETIRED_TOOLS[topic]
        if topic in _TOOL_ALIASES:
            canonical = _TOOL_ALIASES[topic]
            if canonical in registered:
                return (
                    f"{topic} is a compatibility alias for {canonical}.\n\n"
                    + _render_registered_tool(registered[canonical])
                )
            if canonical in _RETIRED_TOOLS:
                return _RETIRED_TOOLS[canonical]

        available = (
            set(_CATEGORY_INDEX)
            | set(HELP_CONTENT)
            | set(registered)
            | set(pseudo)
        )
        if any(getattr(tool, "mcp_server", None) for tool in registered.values()):
            available.add("mcp")
        matches = sorted(name for name in available if topic in name)
        if len(matches) == 1:
            return self._run(matches[0])
        if len(matches) > 1:
            return f"Multiple matches for '{topic}': {', '.join(matches)}. Be more specific."

        return (
            f"No help found for '{topic}'.\n"
            f"Available topics: {', '.join(sorted(available))}"
        )


def _registered_tools() -> dict[str, Any]:
    """Return local developer tools plus MCP schemas visible right now."""
    registered: dict[str, Any] = {}
    try:
        from infinidev.tools import get_tools_for_role

        registered.update(
            (tool.name, tool) for tool in get_tools_for_role("developer")
        )
    except Exception:
        pass

    # An agent and its local role list can be built before MCP warmup lands.
    # Ask the non-blocking discovery cache directly as well, so help never
    # denies a tool that appeared later in the same session.
    try:
        for tool_class in discover_mcp_tool_classes():
            tool = tool_class()
            registered.setdefault(tool.name, tool)
    except Exception:
        pass
    return registered


def _pseudo_tool_schemas() -> dict[str, dict[str, Any]]:
    """Return engine-handled tools that do not have local tool classes."""
    from infinidev.engine.schema_sanitizer import (
        ADD_NOTE_SCHEMA,
        ADD_SESSION_NOTE_SCHEMA,
        STEP_COMPLETE_SCHEMA,
        THINK_SCHEMA,
    )

    schemas = [
        STEP_COMPLETE_SCHEMA,
        ADD_NOTE_SCHEMA,
        ADD_SESSION_NOTE_SCHEMA,
        THINK_SCHEMA,
    ]
    return {schema["function"]["name"]: schema["function"] for schema in schemas}


def _render_overview(
    registered: dict[str, Any], pseudo: dict[str, dict[str, Any]]
) -> str:
    """List only categories that contain a tool available in this run."""
    available = set(registered) | set(pseudo)
    lines = ["Available help categories:"]
    for category, names in _CATEGORY_INDEX.items():
        if available.intersection(names):
            lines.append(f"  {category:<13} — {_CATEGORY_DESCRIPTIONS[category]}")
    if any(getattr(tool, "mcp_server", None) for tool in registered.values()):
        lines.append("  mcp           — Tools published by configured MCP servers")
    lines.extend([
        "",
        'Call help(context="<category>") for a tool list, or '
        'help(context="<tool_name>") for its exact parameters.',
    ])
    return "\n".join(lines)


def _render_category(
    category: str,
    registered: dict[str, Any],
    pseudo: dict[str, dict[str, Any]],
) -> str:
    """Render a compact category listing from live descriptions and schemas."""
    if category == "mcp":
        names = sorted(
            name
            for name, tool in registered.items()
            if getattr(tool, "mcp_server", None)
        )
    else:
        names = [
            name
            for name in _CATEGORY_INDEX[category]
            if name in registered or name in pseudo
        ]
    if not names:
        return f"No {category} tools are available in this run."

    lines = [f"{category.upper()} TOOLS", ""]
    for name in names:
        if name in registered:
            tool = registered[name]
            signature = _pydantic_signature(name, tool.args_schema)
            description = (tool.description or "").strip()
        else:
            function = pseudo[name]
            signature = _json_signature(name, function.get("parameters", {}))
            description = (function.get("description") or "").strip()
        lines.append(f"  {signature}")
        if description:
            lines.append(f"    {_first_sentence(description)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_registered_tool(tool: Any) -> str:
    """Build detailed help for one registered tool from its Pydantic schema."""
    lines = [f"# {tool.name}", "", (tool.description or "").strip()]
    server = getattr(tool, "mcp_server", None)
    if server:
        lines.append(f"\nProvided by the {server!r} MCP server.")

    fields = getattr(tool.args_schema, "model_fields", None) or {}
    if fields:
        lines.append("\n## Parameters")
        for field_name, field in fields.items():
            required = "required" if field.is_required() else "optional"
            described = (field.description or "").strip()
            lines.append(
                f"- **{field_name}** ({required})"
                + (f" — {described}" if described else "")
            )
    else:
        lines.append("\nTakes no arguments.")

    try:
        from infinidev.prompts.tool_hints import TOOL_DESCRIPTIONS

        example = TOOL_DESCRIPTIONS.get(tool.name, ("", ""))[1]
    except Exception:
        example = ""
    if example:
        lines.extend(["\n## Example", example])
    return "\n".join(lines)


def _render_schema_tool(function: dict[str, Any]) -> str:
    """Render an engine pseudo-tool from its OpenAI function schema."""
    name = function["name"]
    lines = [f"# {name}", "", (function.get("description") or "").strip()]
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    if not properties:
        lines.append("\nTakes no arguments.")
        return "\n".join(lines)

    lines.append("\n## Parameters")
    for field_name, field in properties.items():
        status = "required" if field_name in required else "optional"
        described = (field.get("description") or "").strip()
        lines.append(
            f"- **{field_name}** ({status})"
            + (f" — {described}" if described else "")
        )
    return "\n".join(lines)


def _pydantic_signature(name: str, schema: type[BaseModel]) -> str:
    fields = getattr(schema, "model_fields", None) or {}
    args = [
        field_name if field.is_required() else f"{field_name}?"
        for field_name, field in fields.items()
    ]
    return f"{name}({', '.join(args)})"


def _json_signature(name: str, parameters: dict[str, Any]) -> str:
    required = set(parameters.get("required", []))
    args = [
        field_name if field_name in required else f"{field_name}?"
        for field_name in parameters.get("properties", {})
    ]
    return f"{name}({', '.join(args)})"


def _first_sentence(text: str) -> str:
    compact = " ".join(text.split())
    head, separator, _ = compact.partition(". ")
    return head + "." if separator else compact


_CATEGORY_DESCRIPTIONS = {
    "file": "Read, search, and inspect workspace files",
    "edit": "Create or change files and refactor symbols",
    "code_intel": "Navigate and analyze indexed code",
    "git": "Inspect branches and commit changes",
    "shell": "Run foreground, background, and Python commands",
    "knowledge": "Store, search, validate, and report findings",
    "web": "Search and fetch web content",
    "docs": "Manage cached library documentation",
    "planning": "Manage execution steps and test output",
    "communication": "Send progress or questions to the user",
    "meta": "Inspect tools and recover earlier context",
    "protocol": "Preserve state and finish loop steps",
}
