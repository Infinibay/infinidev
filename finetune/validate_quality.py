#!/usr/bin/env python3
"""Quality audit for training examples in scenarios_v3/."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent / "scenarios_v3"

REQUIRED_KEYS = {"repo", "lang", "type", "prompt", "turns", "format_version"}

# Tool schemas: tool_name -> {required: [(key, type)], optional: [(key, type)]}
TOOL_SCHEMAS = {
    "read_file": {"required": [("path", str)]},
    "partial_read": {"required": [("path", str), ("start_line", int), ("end_line", int)]},
    "create_file": {"required": [("path", str), ("content", str)]},
    "replace_lines": {"required": [("file_path", str), ("content", str), ("start_line", int), ("end_line", int)]},
    "add_content_after_line": {"required": [("file_path", str), ("line_number", int), ("content", str)]},
    "add_content_before_line": {"required": [("file_path", str), ("line_number", int), ("content", str)]},
    "edit_symbol": {"required": [("symbol", str), ("new_code", str)]},
    "add_symbol": {"required": [("file_path", str), ("code", str)]},
    "remove_symbol": {"required": [("symbol", str)]},
    "list_directory": {"required": []},
    "glob": {"required": [("pattern", str)]},
    "code_search": {"required": [("pattern", str)]},
    "execute_command": {"required": [("command", str)]},
    "help": {"required": []},
    "add_note": {"required": [("note", str)]},
    "think": {"required": [("reasoning", str)]},
    "step_complete": {"required": [("summary", str), ("status", str)]},
    "send_message": {"required": [("message", str)]},
    "web_search": {"required": [("query", str)]},
    "web_fetch": {"required": [("url", str)]},
    "code_search_web": {"required": [("query", str)]},
    "record_finding": {"required": [("title", str), ("content", str)]},
    "search_findings": {"required": [("query", str)]},
    "git_status": {"required": []},
    "git_diff": {"required": []},
    "git_branch": {"required": [("branch_name", str)]},
    "git_commit": {"required": [("message", str)]},
    "analyze_code": {"required": [("file_path", str)]},
    "search_symbols": {"required": [("query", str)]},
    "find_references": {"required": [("name", str)]},
    "get_symbol_code": {"required": [("name", str)]},
    "list_symbols": {"required": [("file_path", str)]},
    "project_structure": {"required": []},
    "rename_symbol": {"required": [("symbol", str), ("new_name", str)]},
    "move_symbol": {"required": [("symbol", str), ("target_file", str)]},
}

VALID_STATUSES = {"continue", "done", "blocked", "explore"}
EDIT_TOOLS = {"create_file", "replace_lines", "edit_symbol", "add_symbol", "remove_symbol",
              "add_content_after_line", "add_content_before_line"}
EXPLORE_TOOLS = {"read_file", "partial_read", "glob", "list_directory", "code_search",
                 "project_structure", "list_symbols", "search_symbols", "find_references",
                 "get_symbol_code"}

PLACEHOLDER_PATTERNS = ["lorem ipsum", "foo bar baz placeholder", "TODO: implement"]


def validate_file(filepath: Path) -> dict:
    """Validate a single training file. Returns dict with errors and warnings."""
    errors = []
    warnings = []
    fname = filepath.name

    # 1. STRUCTURAL VALIDITY
    try:
        with open(filepath) as f:
            raw = f.readline()
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return {"errors": errors, "warnings": warnings}

    # Required keys
    missing_keys = REQUIRED_KEYS - set(data.keys())
    if missing_keys:
        # format_version missing is common, separate it
        if missing_keys == {"format_version"}:
            warnings.append("Missing format_version key")
        else:
            non_fv = missing_keys - {"format_version"}
            if non_fv:
                errors.append(f"Missing required keys: {non_fv}")
            if "format_version" in missing_keys:
                warnings.append("Missing format_version key")

    # format_version check
    fv = data.get("format_version")
    if fv is not None and fv != "v3_structured":
        errors.append(f"Wrong format_version: {fv}")

    turns = data.get("turns", [])
    if not isinstance(turns, list):
        errors.append("turns is not a list")
        return {"errors": errors, "warnings": warnings}

    if len(turns) < 6:
        errors.append(f"Too few turns: {len(turns)} (need >= 6)")
        return {"errors": errors, "warnings": warnings}

    # First turn must be user with <task>
    if turns[0].get("role") != "user":
        errors.append(f"First turn role is '{turns[0].get('role')}', expected 'user'")
    first_content = turns[0].get("content", "")
    if "<task>" not in str(first_content):
        errors.append("First turn does not contain <task>")

    # Turn alternation: user, then assistant/tool alternating
    expected_roles_after_first = []  # build expected pattern
    prev_role = None
    for i, turn in enumerate(turns):
        role = turn.get("role")
        if i == 0:
            if role != "user":
                pass  # already flagged
        else:
            if role == "assistant":
                if prev_role == "assistant":
                    errors.append(f"Turn {i}: consecutive assistant turns (prev was also assistant)")
                # Check tool_calls
                if "tool_calls" not in turn:
                    errors.append(f"Turn {i}: assistant turn missing tool_calls")
                else:
                    tc = turn["tool_calls"]
                    if not isinstance(tc, list):
                        errors.append(f"Turn {i}: tool_calls is not a list")
                    else:
                        for j, call in enumerate(tc):
                            if not isinstance(call, dict):
                                errors.append(f"Turn {i}, call {j}: tool call is not a dict")
                            elif "name" not in call or "arguments" not in call:
                                errors.append(f"Turn {i}, call {j}: missing name or arguments")
            elif role == "tool":
                if prev_role != "assistant":
                    errors.append(f"Turn {i}: tool turn not preceded by assistant (prev={prev_role})")
                if "content" not in turn:
                    errors.append(f"Turn {i}: tool turn missing content")
                elif not isinstance(turn["content"], str):
                    errors.append(f"Turn {i}: tool content is not a string")
            elif role == "user":
                pass  # multi-turn might have user turns
            else:
                errors.append(f"Turn {i}: unknown role '{role}'")
        prev_role = role

    # 2. TOOL CALL CORRECTNESS
    all_tool_calls = []  # (turn_idx, name, args)
    for i, turn in enumerate(turns):
        if turn.get("role") == "assistant" and "tool_calls" in turn:
            for j, tc in enumerate(turn.get("tool_calls", [])):
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                all_tool_calls.append((i, name, args))

                if name not in TOOL_SCHEMAS:
                    warnings.append(f"Turn {i}: unknown tool '{name}'")
                    continue

                schema = TOOL_SCHEMAS[name]
                for param_name, param_type in schema["required"]:
                    if param_name not in args:
                        errors.append(f"Turn {i}: {name} missing required arg '{param_name}'")
                    elif args[param_name] is None:
                        errors.append(f"Turn {i}: {name} has null required arg '{param_name}'")
                    elif not isinstance(args[param_name], param_type):
                        # Allow int where str expected if it's a number-as-string scenario
                        if param_type == int and isinstance(args[param_name], (float, str)):
                            try:
                                int(args[param_name])
                            except (ValueError, TypeError):
                                errors.append(f"Turn {i}: {name}.{param_name} wrong type: expected {param_type.__name__}, got {type(args[param_name]).__name__}")
                        elif param_type == str and isinstance(args[param_name], (int, float)):
                            warnings.append(f"Turn {i}: {name}.{param_name} is {type(args[param_name]).__name__}, expected str")
                        else:
                            errors.append(f"Turn {i}: {name}.{param_name} wrong type: expected {param_type.__name__}, got {type(args[param_name]).__name__}")

                # step_complete status validation
                if name == "step_complete" and "status" in args:
                    if args["status"] not in VALID_STATUSES:
                        errors.append(f"Turn {i}: step_complete status '{args['status']}' not in {VALID_STATUSES}")

                # Empty/null required args
                for param_name, param_type in schema["required"]:
                    if param_name in args and isinstance(args[param_name], str) and args[param_name].strip() == "":
                        warnings.append(f"Turn {i}: {name}.{param_name} is empty string")

    # 3. BEHAVIORAL QUALITY

    # Does example end with step_complete status=done?
    last_step_complete = None
    for i, name, args in reversed(all_tool_calls):
        if name == "step_complete":
            last_step_complete = (i, args)
            break

    if last_step_complete is None:
        errors.append("No step_complete found in entire example")
    elif last_step_complete[1].get("status") != "done":
        errors.append(f"Last step_complete has status='{last_step_complete[1].get('status')}', expected 'done'")

    # Does every step_complete with status=done have final_answer?
    for i, name, args in all_tool_calls:
        if name == "step_complete" and args.get("status") == "done":
            if "final_answer" not in args:
                warnings.append(f"Turn {i}: step_complete(done) missing final_answer")

    # Exploration before editing
    first_edit_idx = None
    has_explore_before = False
    for i, name, args in all_tool_calls:
        if name in EXPLORE_TOOLS:
            if first_edit_idx is None:
                has_explore_before = True
        if name in EDIT_TOOLS and first_edit_idx is None:
            first_edit_idx = i
    if first_edit_idx is not None and not has_explore_before:
        # Check if it's purely a create-from-scratch scenario (no existing files to explore)
        warnings.append("No exploration (read_file/glob/list_directory) before first edit")

    # Huge code blocks
    for i, name, args in all_tool_calls:
        if name in ("create_file", "replace_lines"):
            content = args.get("content", "")
            if isinstance(content, str) and len(content) > 3000:
                warnings.append(f"Turn {i}: {name} has large content ({len(content)} chars)")

    # Consecutive identical tool calls
    prev_call = None
    for i, name, args in all_tool_calls:
        cur = (name, json.dumps(args, sort_keys=True))
        if prev_call == cur:
            errors.append(f"Turn {i}: consecutive identical tool call: {name}")
        prev_call = cur

    # Verification step (execute_command after edit)
    has_edit = any(name in EDIT_TOOLS for _, name, _ in all_tool_calls)
    has_verify = False
    if has_edit:
        edit_seen = False
        for _, name, _ in all_tool_calls:
            if name in EDIT_TOOLS:
                edit_seen = True
            if edit_seen and name == "execute_command":
                has_verify = True
                break
        if not has_verify:
            warnings.append("No verification (execute_command) after edits")

    # 4. CONTENT QUALITY SPOT-CHECKS

    # Placeholder content in create_file
    for i, name, args in all_tool_calls:
        if name == "create_file":
            content = str(args.get("content", "")).lower()
            for pat in PLACEHOLDER_PATTERNS:
                if pat in content:
                    warnings.append(f"Turn {i}: create_file contains placeholder text: '{pat}'")

    # Unrealistic tool results
    for i, turn in enumerate(turns):
        if turn.get("role") == "tool":
            content = turn.get("content", "")
            # Find what tool this is a response to
            if i > 0 and turns[i-1].get("role") == "assistant":
                prev_tools = [tc.get("name") for tc in turns[i-1].get("tool_calls", [])]
                if "read_file" in prev_tools and (content.strip() == "" or content.strip() == "OK"):
                    warnings.append(f"Turn {i}: read_file result is empty or just 'OK'")

    # Short think reasoning
    for i, name, args in all_tool_calls:
        if name == "think":
            reasoning = args.get("reasoning", "")
            if isinstance(reasoning, str) and len(reasoning) < 20:
                warnings.append(f"Turn {i}: think has very short reasoning ({len(reasoning)} chars): '{reasoning}'")

    # Short step_complete summaries
    for i, name, args in all_tool_calls:
        if name == "step_complete":
            summary = args.get("summary", "")
            if isinstance(summary, str) and len(summary) < 30:
                warnings.append(f"Turn {i}: step_complete has short summary ({len(summary)} chars): '{summary}'")

    return {"errors": errors, "warnings": warnings}


def main():
    files = sorted(SCENARIOS_DIR.glob("*.jsonl"))
    print(f"Validating {len(files)} files in {SCENARIOS_DIR}\n")

    results = {}
    total_errors = 0
    total_warnings = 0
    error_counts = defaultdict(int)
    warning_counts = defaultdict(int)
    files_to_delete = []
    files_to_fix = []

    for filepath in files:
        result = validate_file(filepath)
        fname = filepath.name
        n_err = len(result["errors"])
        n_warn = len(result["warnings"])
        total_errors += n_err
        total_warnings += n_warn

        for e in result["errors"]:
            # Extract meaningful category
            if "consecutive assistant turns" in e:
                cat = "consecutive assistant turns"
            elif "consecutive identical tool call" in e:
                cat = "consecutive identical tool call"
            elif "missing required arg" in e:
                cat = "missing required arg"
            elif "has null required arg" in e:
                cat = "null required arg"
            elif "wrong type" in e:
                cat = "wrong argument type"
            elif "tool turn missing content" in e:
                cat = "tool turn missing content"
            elif "missing tool_calls" in e:
                cat = "assistant turn missing tool_calls"
            elif "not preceded by assistant" in e:
                cat = "tool not preceded by assistant"
            elif "No step_complete" in e:
                cat = "no step_complete at all"
            elif "Last step_complete" in e:
                cat = "last step_complete not done"
            elif "Missing required keys" in e:
                cat = "missing required keys"
            elif "step_complete status" in e:
                cat = "invalid step_complete status"
            elif "First turn" in e:
                cat = "first turn issue"
            elif "Too few turns" in e:
                cat = "too few turns"
            elif "Invalid JSON" in e:
                cat = "invalid JSON"
            else:
                cat = e[:60]
            error_counts[cat] += 1
        for w in result["warnings"]:
            if "format_version" in w:
                cat = "missing format_version"
            elif "empty string" in w:
                cat = "empty string arg"
            elif "large content" in w:
                cat = "large content block"
            elif "No exploration" in w:
                cat = "no exploration before edit"
            elif "No verification" in w:
                cat = "no verification after edit"
            elif "final_answer" in w:
                cat = "step_complete(done) missing final_answer"
            elif "short reasoning" in w:
                cat = "short think reasoning"
            elif "short summary" in w:
                cat = "short step_complete summary"
            elif "unknown tool" in w:
                cat = "unknown tool name"
            elif "placeholder" in w:
                cat = "placeholder content"
            elif "unrealistic" in w.lower() or "just 'OK'" in w:
                cat = "unrealistic tool result"
            else:
                cat = w[:60]
            warning_counts[cat] += 1

        if n_err > 0 or n_warn > 0:
            results[fname] = result

        # Classify: delete if critical structural errors
        has_critical = False
        for e in result["errors"]:
            if any(kw in e for kw in ["Invalid JSON", "Missing required keys",
                                       "Too few turns", "First turn",
                                       "consecutive identical tool call",
                                       "consecutive assistant turns",
                                       "missing required arg",
                                       "has null required arg",
                                       "wrong type",
                                       "tool turn missing content",
                                       "missing tool_calls",
                                       "not preceded by assistant"]):
                has_critical = True
                break
        # Also critical if no step_complete done
        for e in result["errors"]:
            if "No step_complete found" in e:
                has_critical = True
            if "Last step_complete has status=" in e:
                has_critical = True

        if has_critical:
            files_to_delete.append(fname)
        elif n_err > 0:
            # Has non-critical errors -- still needs attention
            files_to_fix.append(fname)
        elif n_warn > 0:
            files_to_fix.append(fname)

    # Output per-file issues
    print("=" * 80)
    print("PER-FILE ISSUES (files with problems only)")
    print("=" * 80)
    for fname, result in sorted(results.items()):
        print(f"\n--- {fname} ---")
        for e in result["errors"]:
            print(f"  ERROR: {e}")
        for w in result["warnings"]:
            print(f"  WARN:  {w}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total files:    {len(files)}")
    print(f"Files OK:       {len(files) - len(results)}")
    print(f"Files w/issues: {len(results)}")
    print(f"Total errors:   {total_errors}")
    print(f"Total warnings: {total_warnings}")

    print(f"\nTop error categories:")
    for cat, count in sorted(error_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count:4d}  {cat}")

    print(f"\nTop warning categories:")
    for cat, count in sorted(warning_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count:4d}  {cat}")

    print(f"\n{'=' * 80}")
    print(f"FILES TO DELETE ({len(files_to_delete)} critical errors):")
    print("=" * 80)
    for f in sorted(files_to_delete):
        print(f"  {f}")

    print(f"\n{'=' * 80}")
    print(f"FILES TO FIX ({len(files_to_fix)} warnings only):")
    print("=" * 80)
    for f in sorted(files_to_fix):
        print(f"  {f}")

    # Return counts for scripting
    return len(files_to_delete), len(files_to_fix)


if __name__ == "__main__":
    n_del, n_fix = main()
    sys.exit(1 if n_del > 0 else 0)
