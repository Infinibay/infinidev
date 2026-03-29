#!/usr/bin/env python3
"""Validate v3 training examples for quality and correctness.

Usage:
    python finetune/validate_examples.py                    # validate all
    python finetune/validate_examples.py flask_001_bugfix   # validate one
"""

import json
import os
import re
import sys
from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent / "scenarios_v3"
REPOS_DIR = Path(__file__).parent / "repos"

VALID_TOOLS = {
    "think", "step_complete", "help", "read_file", "partial_read",
    "create_file", "replace_lines", "add_content_after_line", "add_content_before_line",
    "edit_symbol", "add_symbol", "remove_symbol", "rename_symbol", "move_symbol",
    "get_symbol_code", "list_symbols", "search_symbols", "find_references",
    "analyze_code", "list_directory", "glob", "code_search", "project_structure",
    "execute_command", "add_note", "send_message",
    "git_status", "git_diff", "git_branch", "git_commit",
}

# Tools that MUST return non-empty content
CONTENT_TOOLS = {
    "read_file", "partial_read", "code_search", "find_references",
    "list_symbols", "search_symbols", "get_symbol_code", "list_directory",
    "glob", "project_structure", "execute_command", "help", "analyze_code",
}

# Tools that return empty (engine processes internally)
EMPTY_TOOLS = {"think", "step_complete"}


class ValidationError:
    def __init__(self, file: str, msg: str, severity: str = "error"):
        self.file = file
        self.msg = msg
        self.severity = severity

    def __str__(self):
        icon = "ERROR" if self.severity == "error" else "WARN"
        return f"  [{icon}] {self.file}: {self.msg}"


def validate_file(filepath: Path) -> list[ValidationError]:
    """Validate a single scenario file."""
    errors = []
    name = filepath.stem

    try:
        with open(filepath) as f:
            line = f.readline().strip()
            if not line:
                errors.append(ValidationError(name, "Empty file"))
                return errors
            data = json.loads(line)
    except json.JSONDecodeError as e:
        errors.append(ValidationError(name, f"Invalid JSON: {e}"))
        return errors

    turns = data.get("turns", [])
    if not turns:
        errors.append(ValidationError(name, "No turns"))
        return errors

    # ── Basic structure ───────────────────────────────────────────────
    if turns[0].get("role") != "user":
        errors.append(ValidationError(name, "First turn must be 'user'"))

    # ── Turn alternation ──────────────────────────────────────────────
    for i in range(1, len(turns)):
        prev = turns[i-1]["role"]
        curr = turns[i]["role"]
        if prev == "assistant" and curr not in ("tool",):
            errors.append(ValidationError(name, f"Turn {i+1}: expected 'tool' after 'assistant', got '{curr}'"))
        if prev == "tool" and curr not in ("assistant", "user"):
            errors.append(ValidationError(name, f"Turn {i+1}: expected 'assistant' or 'user' after 'tool', got '{curr}'"))
        if prev == "user" and curr not in ("assistant",):
            errors.append(ValidationError(name, f"Turn {i+1}: expected 'assistant' after 'user', got '{curr}'"))

    # ── Extract all tool calls (supports both XML and structured formats) ──
    all_tool_calls = []
    assistant_turns = [t for t in turns if t["role"] == "assistant"]
    tool_turns = [t for t in turns if t["role"] == "tool"]

    for at in assistant_turns:
        # Structured format: tool_calls as JSON array
        if "tool_calls" in at:
            for tc in at["tool_calls"]:
                if isinstance(tc, dict):
                    all_tool_calls.append(tc)
        # XML format: <tool_call>...</tool_call> in content
        elif "content" in at:
            content = at["content"]
            for m in re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
                try:
                    tc = json.loads(m.group(1))
                    all_tool_calls.append(tc)
                except json.JSONDecodeError:
                    errors.append(ValidationError(name, f"Invalid tool_call JSON: {m.group(1)[:100]}", "warn"))

    tool_names = [tc.get("name", "") for tc in all_tool_calls]
    tool_name_set = set(tool_names)

    # ── Valid tool names ──────────────────────────────────────────────
    for tn in tool_name_set:
        if tn and tn not in VALID_TOOLS and tn != "_raw":
            errors.append(ValidationError(name, f"Unknown tool: '{tn}'", "warn"))

    # ── help("edit") early ────────────────────────────────────────────
    help_calls = [tc for tc in all_tool_calls if tc.get("name") == "help"]
    if not help_calls:
        errors.append(ValidationError(name, "Missing help() call"))
    elif help_calls[0].get("arguments", {}).get("context") != "edit":
        errors.append(ValidationError(name, "First help() should be help('edit')", "warn"))

    # ── think() count ─────────────────────────────────────────────────
    think_count = tool_names.count("think")
    if think_count < 3:
        errors.append(ValidationError(name, f"Only {think_count} think() calls (min 3)"))

    # ── add_note() ────────────────────────────────────────────────────
    if "add_note" not in tool_name_set:
        errors.append(ValidationError(name, "Missing add_note() call", "warn"))

    # ── step_complete checks ──────────────────────────────────────────
    step_completes = [tc for tc in all_tool_calls if tc.get("name") == "step_complete"]
    if len(step_completes) < 2:
        errors.append(ValidationError(name, f"Only {len(step_completes)} step_complete calls (min 2: continue + done)"))

    # Check first step_complete has next_steps
    continue_steps = [sc for sc in step_completes if sc.get("arguments", {}).get("status") == "continue"]
    if continue_steps:
        first_continue = continue_steps[0]
        next_steps = first_continue.get("arguments", {}).get("next_steps", [])
        if len(next_steps) < 3:
            errors.append(ValidationError(name, f"First step_complete(continue) has {len(next_steps)} next_steps (min 3)", "warn"))

    # Check last step_complete is done with final_answer
    if step_completes:
        last_sc = step_completes[-1]
        if last_sc.get("arguments", {}).get("status") != "done":
            errors.append(ValidationError(name, "Last step_complete should be status='done'"))
        if not last_sc.get("arguments", {}).get("final_answer"):
            errors.append(ValidationError(name, "Last step_complete(done) missing final_answer"))

    # ── Tool results quality ──────────────────────────────────────────
    for i, tt in enumerate(tool_turns):
        content = tt.get("content", "") or ""

        # Check for placeholder truncation
        if "..." in content and len(content) > 20:
            # Allow "..." in error messages and short strings, flag in file content
            lines_with_dots = [l for l in content.split("\n") if l.strip() == "..."]
            if lines_with_dots:
                errors.append(ValidationError(name, f"Tool turn {i+1}: contains '...' placeholder line (truncated content)", "warn"))

    # ── Turn count by complexity ──────────────────────────────────────
    complexity = "simple"
    if "complex" in name:
        complexity = "complex"
    elif "medium" in name:
        complexity = "medium"

    min_turns = {"simple": 15, "medium": 25, "complex": 35}
    if len(turns) < min_turns.get(complexity, 15):
        errors.append(ValidationError(name, f"Only {len(turns)} turns for {complexity} example (min {min_turns[complexity]})"))

    # ── Step count ────────────────────────────────────────────────────
    min_steps = {"simple": 3, "medium": 4, "complex": 6}
    if len(step_completes) < min_steps.get(complexity, 3):
        errors.append(ValidationError(name, f"Only {len(step_completes)} steps for {complexity} (min {min_steps[complexity]})", "warn"))

    return errors


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target:
        files = list(SCENARIOS_DIR.glob(f"{target}*.jsonl"))
    else:
        files = sorted(SCENARIOS_DIR.glob("*.jsonl"))

    if not files:
        print(f"No files found in {SCENARIOS_DIR}")
        return

    total_errors = 0
    total_warnings = 0

    # ── Per-file validation ───────────────────────────────────────────
    for f in files:
        errs = validate_file(f)
        file_errors = [e for e in errs if e.severity == "error"]
        file_warnings = [e for e in errs if e.severity == "warn"]

        if errs:
            status = "FAIL" if file_errors else "WARN"
            print(f"[{status}] {f.stem}")
            for e in errs:
                print(f"  {e}")
        else:
            print(f"[ OK ] {f.stem}")

        total_errors += len(file_errors)
        total_warnings += len(file_warnings)

    # ── Global tool coverage ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Files: {len(files)}")
    print(f"Errors: {total_errors}, Warnings: {total_warnings}")

    if files:
        # Count tool usage across all files
        all_tools_used: dict[str, int] = {}
        for f in files:
            with open(f) as fh:
                data = json.loads(fh.readline())
            for t in data.get("turns", []):
                if t["role"] == "assistant":
                    for m in re.finditer(r'"name": "(\w+)"', t["content"]):
                        tool = m.group(1)
                        all_tools_used[tool] = all_tools_used.get(tool, 0) + 1

        print(f"\nTool coverage ({len(all_tools_used)} tools used):")
        for tool in sorted(VALID_TOOLS):
            count = all_tools_used.get(tool, 0)
            status = "OK" if count > 0 else "MISSING"
            print(f"  {tool:30s} {count:3d}  [{status}]")

    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
