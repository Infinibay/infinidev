"""Live help categories and code-interpreter bridge documentation."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from infinidev.code_intel import interpreter_api


_CATEGORY_INDEX = {
    "file": [
        "read_file", "list_directory", "glob", "code_search", "view_image",
    ],
    "edit": [
        "create_file", "edit_file", "delete_file", "move_file",
        "apply_file_patch", "preview_changes", "rollback_task_changes",
        "rename_symbol", "move_symbol",
    ],
    "code_intel": [
        "get_symbol_code", "list_symbols", "search_symbols", "find_references",
        "project_structure", "analyze_code", "find_similar_methods",
        "search_by_docstring", "iter_symbols", "project_stats",
    ],
    "git": ["git_branch", "git_commit", "git_diff", "git_status"],
    "shell": [
        "execute_command", "code_interpreter", "run_in_background",
        "background_status", "stop_background_task", "wait_for_background_task",
    ],
    "knowledge": [
        "write_report", "read_report", "read_command_output", "delete_report",
    ],
    "web": ["web_search", "web_fetch", "code_search_web"],
    "docs": [
        "find_documentation", "update_documentation", "delete_documentation",
    ],
    "planning": [
        "add_step", "modify_step", "remove_step", "declare_test_command",
        "tail_test_output",
    ],
    "communication": ["send_message"],
    "meta": ["describe_tool", "recall_context", "request_capability"],
    "protocol": ["step_complete", "add_note", "add_session_note"],
}


def _one_sentence(function: Callable[..., Any]) -> str:
    """Return the first sentence of the function's live docstring."""
    compact = " ".join((inspect.getdoc(function) or "").split())
    if not compact:
        return "Read-only code-intelligence query."
    head, separator, _ = compact.partition(". ")
    return head + "." if separator else compact


def _call_signature(name: str, function: Callable[..., Any]) -> str:
    """Render parameters without the Python return-arrow notation."""
    signature = inspect.signature(function).replace(
        return_annotation=inspect.Signature.empty,
    )
    return f"{name}{signature}"


def _return_annotation(function: Callable[..., Any]) -> str:
    """Render the live return annotation as plain text."""
    annotation = function.__annotations__.get("return")
    if annotation is None:
        return "unspecified"
    return annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)


def _bridge_overview(functions: dict[str, Callable[..., Any]]) -> str:
    lines = [
        "CODE INTERPRETER BRIDGE",
        "",
        "The code_interpreter process pre-imports these read-only project queries:",
        "",
    ]
    lines.extend(f"  {_call_signature(name, function)}" for name, function in functions.items())
    lines.extend([
        "",
        "Call describe_tool(context=\"code_interpreter.<function_name>\") for the live "
        "summary and return type.",
    ])
    return "\n".join(lines)


def _bridge_topic(name: str, function: Callable[..., Any]) -> str:
    return (
        f"{_call_signature(name, function)}\n\n"
        f"{_one_sentence(function)}\n"
        f"Returns: {_return_annotation(function)}."
    )


def _build_help_content() -> dict[str, str]:
    """Build bridge help from the exact API exported to the subprocess."""
    functions = {
        name: getattr(interpreter_api, name)
        for name in interpreter_api.__all__
    }
    content = {"code_interpreter": _bridge_overview(functions)}
    content.update({
        f"code_interpreter.{name}": _bridge_topic(name, function)
        for name, function in functions.items()
    })
    return content


HELP_CONTENT: dict[str, str] = _build_help_content()
