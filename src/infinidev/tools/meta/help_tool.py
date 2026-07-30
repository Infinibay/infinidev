"""Meta-tool that provides detailed help and examples for all tools."""

from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.meta.help_input import HelpInput
from infinidev.tools.meta.help_content import HELP_CONTENT, _CATEGORY_INDEX


class HelpTool(InfinibayBaseTool):
    name: str = "help"
    description: str = "Get detailed help and examples for any tool."
    args_schema: Type[BaseModel] = HelpInput

    def _run(self, context: str | None = None) -> str:
        if context is not None:
            context = context.strip().lower()

        # Direct match
        if context in HELP_CONTENT:
            return HELP_CONTENT[context]

        # Try matching as category index key
        if context in _CATEGORY_INDEX:
            return HELP_CONTENT.get(context, f"No help available for category: {context}")

        # Fuzzy: search for context as substring in keys
        matches = [k for k in HELP_CONTENT if k and context and context in k]
        if len(matches) == 1:
            return HELP_CONTENT[matches[0]]
        if len(matches) > 1:
            return f"Multiple matches for '{context}': {', '.join(matches)}. Be more specific."

        # Nothing hand-written. Tools discovered at runtime from an MCP
        # server are never in HELP_CONTENT — a static file cannot document
        # what a server has not advertised yet — so render their own schema
        # instead of answering "no help found" about a tool the model can
        # plainly see in its toolbox.
        live = _render_live_tool(context)
        if live:
            return live

        available = sorted(k for k in HELP_CONTENT if k is not None)
        return (
            f"No help found for '{context}'.\n"
            f"Available topics: {', '.join(available)}"
        )


def _render_live_tool(name: str | None) -> str | None:
    """Build help for a registered tool from its description and schema."""
    if not name:
        return None
    tool = _find_registered_tool(name)
    if tool is None:
        return None

    lines = [f"# {tool.name}", "", (tool.description or "").strip()]
    server = getattr(tool, "mcp_server", None)
    if server:
        lines.append(f"\nProvided by the {server!r} MCP server.")

    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None) or {}
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
    return "\n".join(lines)


def _find_registered_tool(name: str):
    """Locate a live tool instance by name, without building the full toolset."""
    try:
        from infinidev.tools.mcp_bridge import discover_mcp_tool_classes

        for cls in discover_mcp_tool_classes():
            if cls.model_fields["name"].default == name:
                return cls()
    except Exception:
        pass
    return None
