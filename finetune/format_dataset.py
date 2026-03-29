"""Format training examples into Qwen ChatML format for fine-tuning.

Output format is JSONL where each line is a training example with
messages in Qwen's ChatML format: <|im_start|>role\ncontent<|im_end|>
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finetune.config import OUTPUT_DIR, DATASET_DIR, CHATML_SYSTEM, CHATML_USER, CHATML_ASSISTANT, CHATML_END


# ── System prompt template ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert software engineer and technical researcher. You work methodically: understand the problem, plan your approach, execute precisely, and verify your work.

## Tool Usage — IMPORTANT

The editing tools work DIFFERENTLY from what you may expect. Call help("edit") before your first edit to learn the correct workflow.

### Reading
- read_file(path): Read file with line numbers.
- partial_read(path, start_line, end_line): Read specific lines.
- get_symbol_code(name): Get source code of a symbol by name.
- list_symbols(file_path): List symbols in a file.
- search_symbols(query): Search symbols across project.
- find_references(name): Find all usages of a symbol.

### Writing — always read_file FIRST
- create_file(path, content): Create NEW files only. Fails if exists.
- replace_lines(file_path, content, start_line, end_line): Replace line range.
- add_content_after_line(file_path, line_number, content): Insert after line.
- add_content_before_line(file_path, line_number, content): Insert before line.
- edit_symbol(symbol, new_code): Replace method/function by name.
- add_symbol(code, file_path, class_name?): Add method to class/file.
- remove_symbol(symbol): Remove method/function.
- rename_symbol(symbol, new_name): Rename everywhere.
- move_symbol(symbol, target_file): Move to another file/class.

### Other
- analyze_code(file_path?): Detect broken imports, undefined symbols, unused code.
- help(context?): Get detailed help and examples for any tool.
- execute_command(command): Run shell commands.
- add_note(note): Save key info for later steps. Persists across steps.
- think(reasoning): Reason before acting. The user sees this — use it to communicate what you're doing.

## Rules

1. ALWAYS read files before editing them.
2. DO NOT re-read files in <opened-files> — they are already current.
3. Call help("edit") before your first edit.
4. Use think() to inform the user what you're doing and why.
5. Call step_complete after every step.
6. Use add_note for key findings that you'll need later.
7. Run tests after code changes.
8. Each step should be 1-8 tool calls."""


# ── Tool schemas (simplified for training) ────────────────────────────────────

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "partial_read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path", "start_line", "end_line"]}}},
    {"type": "function", "function": {"name": "create_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "replace_lines", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["file_path", "content", "start_line", "end_line"]}}},
    {"type": "function", "function": {"name": "add_content_after_line", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "line_number": {"type": "integer"}, "content": {"type": "string"}}, "required": ["file_path", "line_number", "content"]}}},
    {"type": "function", "function": {"name": "add_content_before_line", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "line_number": {"type": "integer"}, "content": {"type": "string"}}, "required": ["file_path", "line_number", "content"]}}},
    {"type": "function", "function": {"name": "edit_symbol", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "new_code": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["symbol", "new_code"]}}},
    {"type": "function", "function": {"name": "add_symbol", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "code": {"type": "string"}, "class_name": {"type": "string"}}, "required": ["file_path", "code"]}}},
    {"type": "function", "function": {"name": "remove_symbol", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "rename_symbol", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "new_name": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["symbol", "new_name"]}}},
    {"type": "function", "function": {"name": "move_symbol", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "target_file": {"type": "string"}, "target_class": {"type": "string"}, "after_line": {"type": "integer"}}, "required": ["symbol", "target_file"]}}},
    {"type": "function", "function": {"name": "get_symbol_code", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "list_symbols", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "kind": {"type": "string"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "search_symbols", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "kind": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "find_references", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "analyze_code", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "checks": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "help", "parameters": {"type": "object", "properties": {"context": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "execute_command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "add_note", "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}}},
    {"type": "function", "function": {"name": "think", "parameters": {"type": "object", "properties": {"reasoning": {"type": "string"}}, "required": ["reasoning"]}}},
    {"type": "function", "function": {"name": "step_complete", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "status": {"type": "string", "enum": ["continue", "done", "blocked"]}, "next_steps": {"type": "array"}, "final_answer": {"type": "string"}}, "required": ["summary", "status"]}}},
]


def _format_tool_calls(tool_calls: list[dict]) -> str:
    """Format tool calls as the assistant response.

    Uses Qwen's tool call format:
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    parts = []
    for tc in tool_calls:
        # Add think before if present
        think_before = tc.get("think_before", "")
        if think_before:
            parts.append(f'<tool_call>{{"name": "think", "arguments": {{"reasoning": {json.dumps(think_before)}}}}}</tool_call>')

        # Add the actual tool call
        call = {"name": tc["name"], "arguments": tc["arguments"]}
        parts.append(f"<tool_call>{json.dumps(call, ensure_ascii=False)}</tool_call>")

    return "\n".join(parts)


def format_example(example: dict) -> dict:
    """Convert a training example to Qwen ChatML messages format.

    Returns a dict with 'messages' key for standard training format.
    """
    messages = []

    # System message with tools
    system_content = SYSTEM_PROMPT + "\n\n## Available Tools\n```json\n" + json.dumps(TOOL_SCHEMAS, indent=2) + "\n```"
    messages.append({"role": "system", "content": system_content})

    # For each step, create a user turn (prompt/context) and assistant turn (tool calls)
    for step_idx, step_calls in enumerate(example.get("steps", [])):
        if step_idx == 0:
            # First step: user provides the task
            user_content = f"<task>\n{example['prompt']}\n</task>"
        else:
            # Subsequent steps: user provides updated context
            user_content = "<context>Continue with the next step from your plan.</context>"

        messages.append({"role": "user", "content": user_content})

        # Assistant response: tool calls
        assistant_content = _format_tool_calls(step_calls)
        messages.append({"role": "assistant", "content": assistant_content})

    return {
        "messages": messages,
        "metadata": {
            "scenario_type": example.get("scenario_type", ""),
            "repo": example.get("repo", ""),
        },
    }


def format_all():
    """Convert all training examples to JSONL dataset."""
    examples_file = OUTPUT_DIR / "training_examples.json"
    if not examples_file.exists():
        print(f"Examples not found: {examples_file}. Run generate_examples.py first.")
        return

    with open(examples_file) as f:
        examples = json.load(f)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Format each example
    formatted = []
    for example in examples:
        try:
            formatted.append(format_example(example))
        except Exception as e:
            print(f"  [error] {example.get('scenario_type')}/{example.get('repo')}: {e}")

    # Write JSONL
    output_file = DATASET_DIR / "infinidev_train.jsonl"
    with open(output_file, "w") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Also write a split for validation (10%)
    split_idx = max(1, len(formatted) // 10)
    val_items = formatted[:split_idx]
    train_items = formatted[split_idx:]

    train_file = DATASET_DIR / "train.jsonl"
    val_file = DATASET_DIR / "val.jsonl"

    with open(train_file, "w") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset written:")
    print(f"  {output_file} ({len(formatted)} examples)")
    print(f"  {train_file} ({len(train_items)} train)")
    print(f"  {val_file} ({len(val_items)} val)")


if __name__ == "__main__":
    format_all()
