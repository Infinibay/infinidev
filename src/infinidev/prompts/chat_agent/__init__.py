"""Chat-agent prompt package — exports the builder that renders the
system prompt with live tool lists injected from the tool registry."""

from __future__ import annotations

from infinidev.prompts.chat_agent.system import CHAT_AGENT_SYSTEM_PROMPT_TEMPLATE

__all__ = [
    "CHAT_AGENT_SYSTEM_PROMPT_TEMPLATE",
    "build_chat_agent_system_prompt",
]


_CATEGORY_LABELS: dict[str, str] = {
    "file": "Write & modify files",
    "code_intel": "Edit code symbols (functions, methods, classes)",
    "git": "Git operations",
    "shell": "Run shell commands & Python code",
    "knowledge": "Knowledge base (record/update findings, reports)",
    "docs": "Documentation management",
    "web": "Web access (search, fetch)",
    "meta": "Planning & meta tools",
    "chat": "Direct user messaging",
}

# Tools that exist in the developer toolset but are terminators or
# plan-management — listing them confuses the chat agent about what
# "the developer can do" for user-facing tasks.
_HIDDEN_DEV_TOOLS: frozenset[str] = frozenset({
    "respond", "escalate", "step_complete", "emit_plan",
    "add_step", "modify_step", "remove_step",
})


def _tool_category(tool: object) -> str:
    """Infer a tool's category from its module path.

    Tools live under ``infinidev.tools.<category>.<module>``. Using the
    module path instead of a hardcoded map keeps categorization in
    sync with the code layout — add a new ``infinidev.tools.foo``
    package and tools there auto-render under a "Foo" category.
    """
    mod = getattr(tool.__class__, "__module__", "") or ""
    parts = mod.split(".")
    if len(parts) >= 3 and parts[0] == "infinidev" and parts[1] == "tools":
        return parts[2]
    return "other"


def _render_chat_toolbox(chat_tools: list) -> str:
    """Render the chat agent's read-only toolbox as a comma-separated
    backtick-wrapped list, excluding the two terminator tools."""
    names = sorted(
        t.name for t in chat_tools
        if t.name not in ("respond", "escalate")
    )
    return ", ".join(f"``{n}``" for n in names)


def _render_developer_toolset(
    dev_tools: list, chat_tools: list,
) -> str:
    """Render the developer-only tools (those the chat agent does NOT
    have) grouped by category as a markdown bullet list."""
    chat_names = {t.name for t in chat_tools}
    extras = [
        t for t in dev_tools
        if t.name not in chat_names and t.name not in _HIDDEN_DEV_TOOLS
    ]

    groups: dict[str, list[str]] = {}
    for t in extras:
        groups.setdefault(_tool_category(t), []).append(t.name)

    # Order categories by the label map (stable, readable order);
    # unknown categories fall to the end alphabetically.
    ordered = [
        c for c in _CATEGORY_LABELS
        if c in groups
    ] + sorted(c for c in groups if c not in _CATEGORY_LABELS)

    lines = []
    for cat in ordered:
        label = _CATEGORY_LABELS.get(cat, cat.replace("_", " ").capitalize())
        names = ", ".join(f"``{n}``" for n in sorted(groups[cat]))
        lines.append(f"  * **{label}**: {names}")
    return "\n".join(lines)


def build_chat_agent_system_prompt() -> str:
    """Render the chat-agent system prompt with live tool lists.

    Called once per chat turn from ``orchestration.chat_agent`` to
    assemble the system message. Uses the current tool registry so
    additions/removals to either tier show up without touching the
    prompt text.
    """
    # Lazy import keeps this package importable before ``tools`` is
    # initialized (relevant during interpreter startup / tests).
    from infinidev.tools import get_tools_for_role

    chat_tools = get_tools_for_role("chat_agent")
    dev_tools = get_tools_for_role("developer")

    return CHAT_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        chat_agent_toolbox=_render_chat_toolbox(chat_tools),
        developer_toolset=_render_developer_toolset(dev_tools, chat_tools),
    )
