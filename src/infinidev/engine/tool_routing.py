"""Deterministic task-level capability routing for the developer toolbox."""

from __future__ import annotations

import re
from typing import Any


_CORE = frozenset({
    # A compact developer baseline.  It supports ordinary inspect-edit-test
    # work, while less common/destructive operations arrive through an
    # explicit capability request instead of occupying every model turn.
    "read_file", "create_file", "edit_file", "apply_file_patch",
    "list_directory", "code_search", "glob",
    "git_diff", "git_status", "execute_command",
    "send_message", "describe_tool", "recall_context",
    "add_step", "modify_step", "remove_step",
    "declare_test_command", "tail_test_output", "request_capability",
})

_CAPABILITY_TOOLS: dict[str, frozenset[str]] = {
    "web": frozenset({"web_search", "web_fetch", "code_search_web"}),
    "knowledge": frozenset({
        "record_finding", "validate_finding", "reject_finding",
        "update_finding", "delete_finding", "write_report", "read_report",
        "delete_report", "search_knowledge", "summarize_findings",
    }),
    "docs": frozenset({
        "delete_documentation", "find_documentation", "update_documentation",
    }),
    "git_mutation": frozenset({"git_branch", "git_commit"}),
    "background": frozenset({
        "run_in_background", "background_status", "stop_background_task",
        "wait_for_background_task",
    }),
    "advanced_refactor": frozenset({
        "rename_symbol", "move_symbol", "find_similar_methods",
        "search_by_docstring", "iter_symbols", "project_stats",
    }),
    "file_management": frozenset({
        "delete_file", "move_file", "preview_changes", "rollback_task_changes",
    }),
    "vision": frozenset({"view_image"}),
    "code_interpreter": frozenset({"code_interpreter"}),
    "image_generation": frozenset({"generate_image"}),
}

_CAPABILITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "web": re.compile(
        r"\b(web|internet|online|browse|url|website|latest release|current version)\b|https?://",
        re.IGNORECASE,
    ),
    "knowledge": re.compile(
        r"\b(research|investigat|finding|evidence|report|audit|literature|sources?)\b",
        re.IGNORECASE,
    ),
    "docs": re.compile(
        r"\b(documentation|docs?|library reference|api reference)\b",
        re.IGNORECASE,
    ),
    "git_mutation": re.compile(
        r"\b(git commit|commit (?:the|these|changes)|create (?:a )?branch|checkout branch|git branch)\b",
        re.IGNORECASE,
    ),
    "background": re.compile(
        r"\b(background|daemon|server|watch mode|long[- ]running|tail logs?)\b",
        re.IGNORECASE,
    ),
    "advanced_refactor": re.compile(
        r"\b(refactor|rename (?:the )?(?:symbol|class|function|method)|move (?:the )?(?:symbol|class|function|method)|call graph|similar methods?)\b",
        re.IGNORECASE,
    ),
    "file_management": re.compile(
        r"\b(?:delete|remove|move|rename)\b.{0,24}"
        r"\b(?:files?|directories|directory|folders?|paths?)\b|"
        r"\b(?:rollback|revert)\b.{0,24}\b(?:files?|changes?|workspace|task)\b|"
        r"\b(?:git\s+mv|mv|rm)\s+[^\s]",
        re.IGNORECASE,
    ),
    "vision": re.compile(
        r"\b(image|screenshot|png|jpe?g|visual|photo)\b",
        re.IGNORECASE,
    ),
    "code_interpreter": re.compile(
        r"\b(dataframe|notebook|python analysis|plot|chart|visuali[sz]e)\b",
        re.IGNORECASE,
    ),
    "image_generation": re.compile(
        r"\b(generate|create|edit|render)\b.{0,30}\b(image|illustration|poster|graphic)\b",
        re.IGNORECASE,
    ),
}

_USER_LITERAL_TASK_RE = re.compile(
    r'<task\s+authority="USER_LITERAL">\s*(.*?)\s*</task>',
    re.DOTALL,
)
_RESULT_KIND_SUFFIX_RE = re.compile(
    r"\nRequested result kind:\s*[^\n]+\s*$",
    re.IGNORECASE,
)


def _routing_description(description: str) -> str:
    """Keep engine protocol prose out of capability inference.

    TaskAdapter wraps the literal request together with rolling-plan policy in
    one description. Words such as ``remove`` in that policy previously
    enabled destructive file-management tools for every Task. When the
    authority block is present, route from its user-authored body only.
    """
    match = _USER_LITERAL_TASK_RE.search(description)
    if match is None:
        return description
    return _RESULT_KIND_SUFFIX_RE.sub("", match.group(1)).strip()


def task_capabilities(
    description: str,
    initial_plan: Any | None = None,
    *,
    task_profile: Any | None = None,
) -> set[str]:
    """Infer optional capabilities from user text plus planner decomposition."""

    parts = [_routing_description(description)]
    if initial_plan is not None:
        parts.append(str(getattr(initial_plan, "overview", "")))
        for step in getattr(initial_plan, "steps", ()):
            parts.extend(
                str(getattr(step, field, ""))
                for field in ("title", "detail", "expected_output")
            )
    corpus = "\n".join(parts)
    capabilities = {
        capability
        for capability, pattern in _CAPABILITY_PATTERNS.items()
        if pattern.search(corpus)
    }
    operations = set(getattr(task_profile, "operations", ()) or ())
    authority = set(getattr(task_profile, "authority", ()) or ())
    if "refactor" in operations:
        capabilities.add("advanced_refactor")
    if "research" in operations:
        capabilities.add("knowledge")
    if "docs" in operations:
        capabilities.add("docs")
    if authority & {"commit", "publish"}:
        capabilities.add("git_mutation")
    return capabilities


def select_developer_tools(
    tools: list[Any],
    description: str,
    initial_plan: Any | None = None,
    *,
    task_profile: Any | None = None,
) -> list[Any]:
    """Return the smallest toolbox that preserves the task's inferred capabilities."""

    enabled_names = set(_CORE)
    for capability in task_capabilities(
        description, initial_plan, task_profile=task_profile,
    ):
        enabled_names.update(_CAPABILITY_TOOLS[capability])

    selected: list[Any] = []
    for tool in tools:
        # Configured MCP tools stay visible: their server is an explicit user
        # extension and ToolEffects/permissions constrain sensitive calls.
        if getattr(tool, "mcp_server", None) or tool.name in enabled_names:
            selected.append(tool)
    return selected


def expand_capability_tools(
    current_tools: list[Any],
    available_tools: list[Any],
    capability: str,
) -> list[Any]:
    """Add one known optional group without treating availability as permission."""

    if capability not in _CAPABILITY_TOOLS:
        raise ValueError(f"Unknown capability: {capability}")
    names = _CAPABILITY_TOOLS[capability]
    existing = {tool.name for tool in current_tools}
    additions = [
        tool for tool in available_tools
        if tool.name in names and tool.name not in existing
    ]
    return [*current_tools, *additions]
