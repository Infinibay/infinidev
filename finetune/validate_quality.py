#!/usr/bin/env python3
"""Validate structured fine-tuning scenarios against the live tool surface."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from finetune.tool_catalog import get_training_schema_map


SCENARIOS_DIR = Path(__file__).parent / "scenarios_v3"
REQUIRED_KEYS = {"repo", "lang", "type", "prompt", "turns", "format_version"}
TOOL_SCHEMAS = get_training_schema_map()
EDIT_TOOLS = {"create_file", "edit_file", "rename_symbol", "move_symbol"}
EXPLORATION_TOOLS = {
    "read_file",
    "glob",
    "list_directory",
    "code_search",
    "project_structure",
    "list_symbols",
    "search_symbols",
    "find_references",
    "get_symbol_code",
}
PLACEHOLDER_PATTERNS = ("lorem ipsum", "foo bar baz placeholder", "TODO: implement")
_PYTHON_TYPES: dict[str, type[Any] | tuple[type[Any], ...]] = {
    "array": list,
    "boolean": bool,
    "integer": int,
    "null": type(None),
    "number": (int, float),
    "object": dict,
    "string": str,
}


def _matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Return whether a value matches the schema shapes used by tool inputs."""
    alternatives = schema.get("anyOf") or schema.get("oneOf")
    if alternatives:
        return any(_matches_schema(value, option) for option in alternatives)

    expected = schema.get("type")
    if isinstance(expected, list):
        return any(_matches_schema(value, {"type": item}) for item in expected)
    if expected not in _PYTHON_TYPES:
        return True
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, _PYTHON_TYPES[expected])


def _argument_errors(turn_index: int, name: str, arguments: Any) -> list[str]:
    """Validate one call's arguments against its live public schema."""
    if not isinstance(arguments, dict):
        return [f"Turn {turn_index}: {name}.arguments is not an object"]

    parameters = TOOL_SCHEMAS[name].get("parameters", {})
    properties = parameters.get("properties", {})
    errors: list[str] = []
    for field_name in parameters.get("required", []):
        if field_name not in arguments:
            errors.append(f"Turn {turn_index}: {name} missing required arg '{field_name}'")

    for field_name, value in arguments.items():
        field_schema = properties.get(field_name)
        if field_schema is None:
            continue
        if not _matches_schema(value, field_schema):
            expected = field_schema.get("type", "declared schema")
            errors.append(
                f"Turn {turn_index}: {name}.{field_name} has type "
                f"{type(value).__name__}; expected {expected}"
            )
            continue
        if isinstance(value, str) and not value.strip() and field_name in parameters.get(
            "required", []
        ):
            errors.append(f"Turn {turn_index}: {name}.{field_name} is empty")
        if "enum" in field_schema and value not in field_schema["enum"]:
            errors.append(
                f"Turn {turn_index}: {name}.{field_name}={value!r} is not one of "
                f"{field_schema['enum']}"
            )
        if isinstance(value, str) and len(value) < field_schema.get("minLength", 0):
            errors.append(
                f"Turn {turn_index}: {name}.{field_name} is shorter than "
                f"{field_schema['minLength']} characters"
            )
    return errors


def _collect_tool_calls(
    turns: list[dict[str, Any]], errors: list[str]
) -> list[tuple[int, str, dict[str, Any]]]:
    """Collect structured tool calls while reporting malformed call objects."""
    calls: list[tuple[int, str, dict[str, Any]]] = []
    for turn_index, turn in enumerate(turns):
        if turn.get("role") != "assistant":
            continue
        tool_calls = turn.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            errors.append(f"Turn {turn_index}: assistant turn has no structured tool_calls")
            continue
        for call_index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                errors.append(
                    f"Turn {turn_index}, call {call_index}: tool call is not an object"
                )
                continue
            name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(name, str) or not name:
                errors.append(f"Turn {turn_index}, call {call_index}: missing tool name")
                continue
            if name not in TOOL_SCHEMAS:
                errors.append(
                    f"Turn {turn_index}: tool '{name}' is not in the live developer schema"
                )
                continue
            errors.extend(_argument_errors(turn_index, name, arguments))
            calls.append((turn_index, name, arguments if isinstance(arguments, dict) else {}))
    return calls


def validate_scenario(data: Any, *, source: str = "<memory>") -> dict[str, list[str]]:
    """Validate one decoded scenario and return errors and quality warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        return {"errors": [f"{source}: scenario is not an object"], "warnings": []}

    missing_keys = REQUIRED_KEYS - data.keys()
    if missing_keys:
        errors.append(f"Missing required keys: {sorted(missing_keys)}")
    if data.get("format_version") != "v3_structured":
        errors.append(f"Wrong format_version: {data.get('format_version')!r}")

    turns = data.get("turns")
    if not isinstance(turns, list) or not turns:
        errors.append("turns must be a non-empty list")
        return {"errors": errors, "warnings": warnings}
    if not isinstance(turns[0], dict) or turns[0].get("role") != "user":
        errors.append("First turn must be a user turn")

    previous_role: str | None = None
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            errors.append(f"Turn {turn_index}: turn is not an object")
            continue
        role = turn.get("role")
        if role not in {"user", "assistant", "tool"}:
            errors.append(f"Turn {turn_index}: unknown role {role!r}")
        elif role == "assistant" and previous_role not in {"user", "tool"}:
            errors.append(f"Turn {turn_index}: assistant does not follow user or tool")
        elif role == "tool" and previous_role != "assistant":
            errors.append(f"Turn {turn_index}: tool does not follow assistant")
        elif role == "tool" and not isinstance(turn.get("content"), str):
            errors.append(f"Turn {turn_index}: tool content is not a string")
        previous_role = role if isinstance(role, str) else None

    calls = _collect_tool_calls(turns, errors)
    if not calls:
        errors.append("Scenario contains no valid tool calls")
        return {"errors": errors, "warnings": warnings}

    completions = [(index, args) for index, name, args in calls if name == "step_complete"]
    if not completions:
        errors.append("Scenario contains no step_complete call")
    elif completions[-1][1].get("status") != "done":
        errors.append("Last step_complete call does not have status='done'")
    elif not completions[-1][1].get("final_answer"):
        errors.append("Last step_complete(done) call has no final_answer")

    first_edit = next((index for index, name, _ in calls if name in EDIT_TOOLS), None)
    if first_edit is not None:
        explored = any(
            index < first_edit and name in EXPLORATION_TOOLS for index, name, _ in calls
        )
        if not explored:
            warnings.append("No repository observation precedes the first edit")
        verified = any(
            index > first_edit and name == "execute_command" for index, name, _ in calls
        )
        if not verified:
            warnings.append("No execute_command verification follows the first edit")

    previous_call: tuple[str, str] | None = None
    for turn_index, name, arguments in calls:
        fingerprint = (name, json.dumps(arguments, sort_keys=True))
        if fingerprint == previous_call:
            errors.append(f"Turn {turn_index}: consecutive identical call to {name}")
        previous_call = fingerprint

        content = arguments.get("content") if name == "create_file" else None
        if name == "edit_file":
            content = arguments.get("new_string")
        if isinstance(content, str) and len(content) > 3_000:
            warnings.append(f"Turn {turn_index}: {name} writes {len(content)} characters")
        if isinstance(content, str):
            lowered = content.lower()
            for pattern in PLACEHOLDER_PATTERNS:
                if pattern.lower() in lowered:
                    warnings.append(
                        f"Turn {turn_index}: {name} contains placeholder text {pattern!r}"
                    )

    return {"errors": errors, "warnings": warnings}


def validate_file(path: Path) -> dict[str, list[str]]:
    """Load and validate the first scenario in a JSONL file."""
    try:
        with path.open(encoding="utf-8") as handle:
            line = handle.readline()
    except OSError as exc:
        return {"errors": [f"Cannot read {path}: {exc}"], "warnings": []}
    if not line.strip():
        return {"errors": ["File is empty"], "warnings": []}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"errors": [f"Invalid JSON: {exc}"], "warnings": []}
    return validate_scenario(data, source=str(path))


def main(argv: list[str] | None = None) -> int:
    """Validate selected scenarios and return a process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    pattern = f"{args[0]}*.jsonl" if args else "*.jsonl"
    files = sorted(SCENARIOS_DIR.glob(pattern))
    if not files:
        print(f"No files found in {SCENARIOS_DIR} for {pattern!r}")
        return 1

    error_categories: Counter[str] = Counter()
    warning_categories: Counter[str] = Counter()
    files_with_errors = 0
    files_with_warnings = 0
    for path in files:
        result = validate_file(path)
        if result["errors"]:
            files_with_errors += 1
        if result["warnings"]:
            files_with_warnings += 1
        if result["errors"] or result["warnings"]:
            print(f"\n{path.name}")
        for message in result["errors"]:
            error_categories[message.split(":", 1)[-1].strip()] += 1
            print(f"  ERROR: {message}")
        for message in result["warnings"]:
            warning_categories[message.split(":", 1)[-1].strip()] += 1
            print(f"  WARN: {message}")

    print(
        f"\nValidated {len(files)} files: {files_with_errors} with errors, "
        f"{files_with_warnings} with warnings."
    )
    if error_categories:
        print("Top errors:")
        for message, count in error_categories.most_common(10):
            print(f"  {count:4d}  {message}")
    if warning_categories:
        print("Top warnings:")
        for message, count in warning_categories.most_common(10):
            print(f"  {count:4d}  {message}")
    return 1 if files_with_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
