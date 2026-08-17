"""Structured side effects shared by tool routing, permissions, and scheduling."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolUseConstraints(BaseModel):
    """Compact model-facing guidance that survives description compression."""

    model_config = ConfigDict(frozen=True)

    use_when: tuple[str, ...] = ()
    do_not_use_when: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    common_failures: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not any(self.model_dump().values())

    def summary(self) -> str:
        parts: list[str] = []
        labels = (
            ("Use when", self.use_when),
            ("Do not use when", self.do_not_use_when),
            ("Preconditions", self.preconditions),
            ("Common failure", self.common_failures),
        )
        for label, values in labels:
            if values:
                parts.append(f"{label}: {'; '.join(values)}.")
        return " ".join(parts)


class ToolEffects(BaseModel):
    """Machine-readable effects of one tool invocation."""

    model_config = ConfigDict(frozen=True)

    reads_workspace: bool = False
    reads_internal_state: bool = False
    reads_external_state: bool = False
    writes_workspace: bool = False
    mutates_git: bool = False
    accesses_network: bool = False
    mutates_internal_state: bool = False
    mutates_external_state: bool = False
    runs_process: bool = False
    communicates_user: bool = False
    destructive: bool = False
    may_cost_money: bool = False
    handles_secrets: bool = False

    @property
    def is_empty(self) -> bool:
        """Whether no effect metadata has been declared."""

        return not any(self.model_dump().values())

    def summary(self) -> str:
        """Compact model-facing effect statement, empty for pure terminators."""

        labels = {
            "reads_workspace": "reads workspace",
            "reads_internal_state": "reads Infinidev state",
            "reads_external_state": "reads external state",
            "writes_workspace": "writes workspace",
            "mutates_git": "mutates Git",
            "accesses_network": "accesses network",
            "mutates_internal_state": "updates Infinidev state",
            "mutates_external_state": "mutates external state",
            "runs_process": "runs a process",
            "communicates_user": "messages the user",
            "destructive": "destructive",
            "may_cost_money": "may incur cost",
            "handles_secrets": "may handle secrets",
        }
        active = [labels[field] for field, enabled in self.model_dump().items() if enabled]
        return ", ".join(active)

    @property
    def needs_explicit_broker(self) -> bool:
        """Whether the generic broker must authorize the operation."""

        return any(
            (
                self.mutates_git,
                self.mutates_external_state,
                self.destructive,
                self.may_cost_money,
                self.handles_secrets,
            )
        )


def check_effect_permission(
    tool_name: str,
    effects: ToolEffects,
    arguments: dict[str, Any],
    *,
    tool: Any | None = None,
) -> str | None:
    """Authorize sensitive effects or return a model-facing denial."""

    if not effects.needs_explicit_broker:
        return None

    from infinidev.config.settings import settings

    # MCP tools have their own permission gate (``MCP_PERMISSION``) that runs
    # inside the bridge's ``_run``. When the user has explicitly opted into an
    # MCP server they trust, ``MCP_PERMISSION=auto_approve`` is the
    # authoritative decision — running the generic effect-broker too would
    # re-prompt the user for every call (e.g. ``ken_remember``). Honour the
    # MCP gate first, then fall through to ``TOOL_EFFECTS_PERMISSION``.
    is_mcp = bool(getattr(tool, "is_mcp_tool", False))
    if is_mcp:
        mcp_mode = str(getattr(settings, "MCP_PERMISSION", "auto_approve") or "auto_approve")
        if mcp_mode == "auto_approve":
            return None
        if mcp_mode == "deny":
            return f"Tool operation denied: MCP_PERMISSION=deny ({tool_name})"
        # ``ask``/``auto`` fall through to the generic broker below.

    mode = str(getattr(settings, "TOOL_EFFECTS_PERMISSION", "auto") or "auto")
    if mode == "auto_approve":
        return None
    if mode not in {"auto", "ask"}:
        return f"Tool operation denied: invalid TOOL_EFFECTS_PERMISSION={mode!r}"

    from infinidev.tools.permission import (
        is_permission_handler_registered,
        request_permission,
    )

    if not is_permission_handler_registered():
        return (
            f"Tool '{tool_name}' has sensitive effects and requires confirmation, "
            "but no approval UI is available. Set "
            "TOOL_EFFECTS_PERMISSION=auto_approve to allow it non-interactively."
        )

    labels = [field for field, enabled in effects.model_dump().items() if enabled]
    details = json.dumps(
        {"effects": labels, "arguments": arguments},
        ensure_ascii=False,
        default=str,
    )
    approved = request_permission(
        tool_name=tool_name,
        description=f"Allow tool effects: {', '.join(labels)}",
        details=details,
    )
    if not approved:
        return f"Tool operation denied by user: {tool_name}"
    return None


_READS_WORKSPACE = frozenset({
    "read_file", "list_directory", "code_search", "glob", "view_image",
    "git_diff", "git_status", "find_references", "list_symbols",
    "search_symbols", "get_symbol_code", "project_structure", "analyze_code",
    "find_similar_methods", "search_by_docstring", "iter_symbols", "project_stats",
    "preview_changes",
})
_WRITES_WORKSPACE = frozenset({
    "create_file", "edit_file", "rename_symbol", "move_symbol",
    "delete_file", "move_file", "apply_file_patch", "rollback_task_changes",
})
_READS_INTERNAL = frozenset({
    "background_status", "wait_for_background_task", "search_findings",
    "read_report", "search_knowledge", "summarize_findings", "find_documentation",
    "describe_tool", "recall_context", "tail_test_output",
    "history_search", "history_read", "history_trace",
})
_NETWORK = frozenset({
    "web_search", "web_fetch", "code_search_web", "generate_image",
    "update_documentation",
})
_INTERNAL_MUTATIONS = frozenset({
    "record_finding", "validate_finding", "reject_finding", "update_finding",
    "delete_finding", "write_report", "delete_report", "delete_documentation",
    "update_documentation", "add_step", "modify_step", "remove_step",
    "declare_test_command", "tail_test_output", "request_capability",
})
_PROCESS = frozenset({
    "execute_command", "code_interpreter", "run_in_background",
    "stop_background_task", "wait_for_background_task", "background_status",
})
_DESTRUCTIVE = frozenset({
    "delete_file", "delete_finding", "delete_report", "delete_documentation",
    "rollback_task_changes",
})


def local_effects_for_name(name: str) -> ToolEffects:
    """Structured metadata for the finite built-in tool catalog."""

    return ToolEffects(
        reads_workspace=name in _READS_WORKSPACE,
        reads_internal_state=name in _READS_INTERNAL,
        writes_workspace=name in _WRITES_WORKSPACE,
        accesses_network=name in _NETWORK,
        mutates_internal_state=name in _INTERNAL_MUTATIONS,
        runs_process=name in _PROCESS,
        communicates_user=name == "send_message",
        destructive=name in _DESTRUCTIVE,
        may_cost_money=name == "generate_image",
    )


def constraints_for_tool(
    name: str,
    effects: ToolEffects,
    *,
    remote: bool = False,
) -> ToolUseConstraints:
    """Derive conservative use boundaries from effects plus known ambiguities."""

    use_when: list[str] = []
    avoid: list[str] = []
    preconditions: list[str] = []
    failures: list[str] = []

    if effects.reads_workspace:
        use_when.append("workspace evidence is needed")
    if effects.reads_internal_state:
        use_when.append("stored Infinidev context can answer a named active-task question")
    if effects.reads_external_state:
        use_when.append("current external state is required")
    if effects.writes_workspace:
        use_when.append("the active task requires this workspace change")
        avoid.append("the task is read-only or only asks for analysis")
    if effects.runs_process:
        use_when.append("execution provides verification or necessary inspection")
    if effects.accesses_network:
        use_when.append("local evidence is insufficient or freshness matters")
        avoid.append("arguments contain secrets not intended for the remote service")
    if effects.mutates_internal_state:
        use_when.append("the task needs durable Infinidev state updated")
    if effects.communicates_user:
        use_when.append("the user needs an update or a genuinely blocking question")
    if effects.mutates_git:
        preconditions.append("the user explicitly requested the Git mutation")
        preconditions.append("review status and diff before staging or switching")
        avoid.append("Git mutation was merely inferred from finishing the task")
    if effects.mutates_external_state:
        preconditions.append("the user authorized this exact external effect")
    if effects.destructive:
        preconditions.append("the exact target is resolved and recovery is understood")
        avoid.append("a narrower reversible operation satisfies the task")
    if effects.may_cost_money:
        preconditions.append("cost-bearing execution is authorized")
    if effects.handles_secrets:
        preconditions.append("secret exposure is necessary and destination-approved")
    if remote:
        failures.append("server annotations may be incomplete; host permission still applies")

    specific: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
        "code_search": (
            ("searching literal text, strings, or non-symbol patterns",),
            ("a known symbol definition or reference query has a code-intel tool",),
            (),
        ),
        "execute_command": (
            ("running builds, tests, or bounded shell inspection",),
            ("a first-class file tool can perform the requested edit",),
            ("command classification or timeout can reject the call",),
        ),
        "code_interpreter": (
            ("in-process Python analysis is simpler than shell execution",),
            ("the task needs arbitrary system-shell mutation",),
            (),
        ),
        "recall_context": (
            ("raw output from an earlier task step was evicted",),
            ("searching durable findings/reports or current workspace state",),
            ("retrieved content may be stale and is advisory",),
        ),
        "search_knowledge": (
            ("searching durable findings or reports across steps/sessions",),
            ("retrieving raw output evicted from the current task",),
            ("semantic mode needs a query; reports use text mode",),
        ),
        "git_commit": (
            ("the user explicitly requested a commit",),
            ("completion alone is being treated as commit authorization",),
            ("files must be explicit unless include_all=true",),
        ),
        "describe_tool": (
            ("a tool contract or category is unclear",),
            ("the schema already answers the question",),
            (),
        ),
    }
    extra_use, extra_avoid, extra_failures = specific.get(name, ((), (), ()))
    use_when.extend(extra_use)
    avoid.extend(extra_avoid)
    failures.extend(extra_failures)
    if not use_when:
        use_when.append("its declared purpose directly matches the active task")

    return ToolUseConstraints(
        use_when=tuple(dict.fromkeys(use_when)),
        do_not_use_when=tuple(dict.fromkeys(avoid)),
        preconditions=tuple(dict.fromkeys(preconditions)),
        common_failures=tuple(dict.fromkeys(failures)),
    )


def apply_local_effect_defaults(tool: Any) -> Any:
    """Fill built-in metadata without overwriting explicit class declarations."""

    if getattr(tool, "mcp_server", None):
        return tool
    current = getattr(tool, "effects", None)
    inferred = local_effects_for_name(str(getattr(tool, "name", "")))
    if not isinstance(current, ToolEffects) or current.is_empty:
        tool.effects = inferred
    constraints = getattr(tool, "use_constraints", None)
    if not isinstance(constraints, ToolUseConstraints) or constraints.is_empty:
        tool.use_constraints = constraints_for_tool(
            str(getattr(tool, "name", "")),
            getattr(tool, "effects", inferred),
        )
    return tool
