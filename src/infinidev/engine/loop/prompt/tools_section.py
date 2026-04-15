"""Render the ``## Available Tools`` section for non-function-calling models.

When the active LLM doesn't support native function calling, we embed
tool descriptions directly in the system prompt and instruct the model
to reply with a JSON object. Extracted from ``loop/context.py`` so this
purely-formatting logic can be unit-tested without spinning up an
engine or prompt builder.
"""

from __future__ import annotations

from typing import Any


def build_tools_prompt_section(
    tool_schemas: list[dict[str, Any]],
    *,
    small_model: bool = False,
) -> str:
    """Render tool schemas as a text section for non-FC models.

    When the model doesn't support native function calling, tool
    descriptions are embedded directly in the system prompt. The model
    is instructed to respond with a JSON object containing a
    ``tool_calls`` array.

    For small models, uses a compact grouped format with fewer details.
    """
    if small_model:
        return _build_tools_prompt_small(tool_schemas)

    lines = [
        "## Available Tools",
        "",
        "CRITICAL: You MUST respond ONLY with a raw JSON object. No markdown, no code fences, no explanation.",
        "Do NOT use <|tool_call>, <tool_call>, or any XML/special token syntax.",
        "Your ENTIRE response must be valid JSON in this exact format:",
        "",
        '{"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}',
        "",
        "Example — read a file then mark step complete:",
        "",
        '{"tool_calls": [{"name": "read_file", "arguments": {"path": "src/main.py"}}, {"name": "step_complete", "arguments": {"summary": "Read the file", "status": "continue"}}]}',
        "",
        "When done with the current step, call \"step_complete\".",
        "",
        "---",
        "",
    ]

    for schema in tool_schemas:
        func = schema.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"### {name}")
        if desc:
            lines.append(desc)

        props = params.get("properties", {})
        required = set(params.get("required", []))
        if props:
            lines.append("Parameters:")
            for pname, pschema in props.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req_marker = " (required)" if pname in required else ""
                lines.append(f"  - `{pname}` ({ptype}{req_marker}): {pdesc}")

        lines.append("")

    return "\n".join(lines)


def _build_tools_prompt_small(tool_schemas: list[dict[str, Any]]) -> str:
    """Compact tool prompt for small models — grouped by category, minimal details."""
    _GROUPS = {
        "READING": {"read_file", "partial_read", "list_directory", "glob", "code_search",
                    "project_structure", "list_symbols", "search_symbols", "get_symbol_code",
                    "find_definition", "find_references"},
        "EDITING": {"replace_lines", "create_file", "edit_symbol", "add_symbol",
                    "remove_symbol", "add_content_after_line", "add_content_before_line"},
        "SHELL": {"execute_command"},
        "GIT": {"git_branch", "git_commit", "git_diff", "git_status"},
        "WEB": {"web_search", "web_fetch"},
        "KNOWLEDGE": {"record_finding", "search_findings", "read_findings"},
        "STEP MANAGEMENT": {"step_complete", "add_note", "add_session_note", "add_step",
                            "modify_step", "remove_step"},
    }

    available = {}
    for schema in tool_schemas:
        func = schema.get("function", {})
        name = func.get("name", "unknown")
        available[name] = func

    lines = [
        "## Tools",
        "",
        "Respond with a JSON tool call. Format:",
        '{"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}',
        "",
        "Example:",
        '{"tool_calls": [{"name": "read_file", "arguments": {"file_path": "src/main.py"}}]}',
        "",
    ]

    for group_name, group_tools in _GROUPS.items():
        present = [n for n in group_tools if n in available]
        if not present:
            continue
        tool_list = ", ".join(present)
        lines.append(f"**{group_name}**: {tool_list}")

    lines.append("")

    _KEY_TOOLS = ["read_file", "replace_lines", "create_file", "execute_command",
                  "step_complete", "add_note", "add_step", "glob", "code_search"]
    for name in _KEY_TOOLS:
        if name not in available:
            continue
        func = available[name]
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))
        req_params = [f"{p}" for p in required if p in props]
        if req_params:
            lines.append(f"- **{name}**({', '.join(req_params)})")

    return "\n".join(lines)
