#!/usr/bin/env python3
"""Build training dataset from structured v3 scenarios.

v2 changes:
- qwen_native format: 1 tool call per assistant turn, Qwen-native tags
- Label masking info: marks which tokens are assistant (trainable) vs context (masked)
- Tools schema in system prompt matching Ollama's injection format
- Explicit stop instruction after tool calls

Usage:
    python -m finetune.build_dataset --format qwen_native   # recommended
    python -m finetune.build_dataset --format raw            # structured JSON
"""

import argparse
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCENARIOS_DIR = BASE_DIR / "scenarios_v3"
DATASET_DIR = BASE_DIR / "output" / "dataset"


# ── Tools schema (matches Ollama injection format) ────────────────────────────

TOOLS_SCHEMA = [
    {"name": "read_file", "description": "Read file with line numbers", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "partial_read", "description": "Read line range", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path", "start_line", "end_line"]}},
    {"name": "create_file", "description": "Create new file (fails if exists)", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "replace_lines", "description": "Replace line range in file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["file_path", "content", "start_line", "end_line"]}},
    {"name": "add_content_after_line", "description": "Insert after line", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "line_number": {"type": "integer"}, "content": {"type": "string"}}, "required": ["file_path", "line_number", "content"]}},
    {"name": "add_content_before_line", "description": "Insert before line", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "line_number": {"type": "integer"}, "content": {"type": "string"}}, "required": ["file_path", "line_number", "content"]}},
    {"name": "edit_symbol", "description": "Replace method/function by name", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "new_code": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["symbol", "new_code"]}},
    {"name": "add_symbol", "description": "Add method to class/file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "code": {"type": "string"}, "class_name": {"type": "string"}}, "required": ["file_path", "code"]}},
    {"name": "remove_symbol", "description": "Remove method/function", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "rename_symbol", "description": "Rename everywhere", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "new_name": {"type": "string"}}, "required": ["symbol", "new_name"]}},
    {"name": "move_symbol", "description": "Move to another file", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "target_file": {"type": "string"}}, "required": ["symbol", "target_file"]}},
    {"name": "get_symbol_code", "description": "Get source of symbol", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "list_symbols", "description": "List symbols in file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
    {"name": "search_symbols", "description": "Search symbols across project", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "find_references", "description": "Find all usages", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "analyze_code", "description": "Detect code issues", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": []}},
    {"name": "help", "description": "Get help for tools", "parameters": {"type": "object", "properties": {"context": {"type": "string"}}, "required": []}},
    {"name": "execute_command", "description": "Run shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "add_note", "description": "Save info for later", "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}},
    {"name": "think", "description": "Reason before acting (user sees this)", "parameters": {"type": "object", "properties": {"reasoning": {"type": "string"}}, "required": ["reasoning"]}},
    {"name": "step_complete", "description": "Signal step done", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "status": {"type": "string"}, "next_steps": {"type": "array", "items": {"type": "object", "properties": {"op": {"type": "string"}, "index": {"type": "integer"}, "title": {"type": "string"}, "description": {"type": "string"}}}}, "final_answer": {"type": "string"}}, "required": ["summary", "status"]}},
    {"name": "send_message", "description": "Message to user", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
    {"name": "list_directory", "description": "List files at path", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}},
    {"name": "glob", "description": "Find files by pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "code_search", "description": "Search text/regex in files", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "project_structure", "description": "Show directory tree", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}},
    {"name": "git_status", "description": "Git status", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "git_diff", "description": "Git diff", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "git_branch", "description": "Git branch ops", "parameters": {"type": "object", "properties": {"branch_name": {"type": "string"}}, "required": ["branch_name"]}},
    {"name": "git_commit", "description": "Git commit", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
    {"name": "web_search", "description": "Search the web using DuckDuckGo", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}}, "required": ["query"]}},
    {"name": "web_fetch", "description": "Fetch readable content from URL", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "format": {"type": "string"}}, "required": ["url"]}},
    {"name": "code_search_web", "description": "Search web for code examples and API docs", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "language": {"type": "string"}, "num_results": {"type": "integer"}}, "required": ["query"]}},
]


def build_system_prompt_with_tools() -> str:
    """Build system prompt with tools schema in Qwen/Ollama format."""
    tools_json = "\n".join(
        json.dumps({"type": "function", "function": t}, ensure_ascii=False)
        for t in TOOLS_SCHEMA
    )
    return f"""You are an expert software engineer. You work methodically: understand → plan → execute → verify.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

IMPORTANT: After calling a tool, STOP immediately and wait for the result. Do NOT generate tool outputs yourself. The system will execute the tool and return the real result."""


def expand_tool_calls(scenario: dict) -> list[dict]:
    """Expand multi-tool-call turns into individual assistant/tool pairs.

    Input: [
        {"role": "assistant", "tool_calls": [tc1, tc2, tc3]},
        {"role": "tool", "content": "result1\\n---\\nresult2\\n---\\nresult3"}
    ]
    Output: [
        {"role": "assistant", "tool_calls": [tc1]},
        {"role": "tool", "content": "result1"},
        {"role": "assistant", "tool_calls": [tc2]},
        {"role": "tool", "content": "result2"},
        {"role": "assistant", "tool_calls": [tc3]},
        {"role": "tool", "content": "result3"},
    ]
    """
    turns = scenario.get("turns", [])
    expanded = []

    i = 0
    while i < len(turns):
        turn = turns[i]

        if turn.get("role") == "assistant" and len(turn.get("tool_calls", [])) > 1:
            tool_calls = turn["tool_calls"]
            # Look ahead for matching tool result
            tool_result = ""
            if i + 1 < len(turns) and turns[i + 1].get("role") == "tool":
                tool_result = turns[i + 1].get("content", "")
                i += 1  # skip the tool turn

            # Split results by --- separator
            results = [r.strip() for r in tool_result.split("\n---\n")] if "---" in tool_result else [tool_result]

            # Pad results if fewer than tool calls
            while len(results) < len(tool_calls):
                results.append("")

            # Create individual pairs
            for tc, result in zip(tool_calls, results):
                expanded.append({"role": "assistant", "tool_calls": [tc]})
                expanded.append({"role": "tool", "content": result})
        else:
            expanded.append(turn)

        i += 1

    return expanded


def _gemma4_escape(s: str) -> str:
    """Escape a string value for Gemma 4 tool call syntax.

    Gemma 4 uses <|"|> instead of regular quotes inside tool call arguments.
    """
    return s.replace('"', '<|"|>')


def _gemma4_encode_value(value) -> str:
    """Encode a value in Gemma 4's tool call argument format.

    Strings become <|"|>value<|"|>, numbers/bools stay as-is,
    lists and dicts are recursively encoded.
    """
    if isinstance(value, str):
        return f'<|"|>{_gemma4_escape(value)}<|"|>'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        items = ",".join(_gemma4_encode_value(v) for v in value)
        return f"[{items}]"
    elif isinstance(value, dict):
        pairs = ",".join(
            f"{k}:{_gemma4_encode_value(v)}" for k, v in value.items()
        )
        return "{" + pairs + "}"
    else:
        return f'<|"|>{_gemma4_escape(str(value))}<|"|>'


def _gemma4_tool_declarations() -> str:
    """Build Gemma 4 tool declarations from TOOLS_SCHEMA.

    Format: <|tool>declaration:name{description:<|"|>...<|"|>,parameters:{...}}<|tool|>
    """
    decls = []
    for tool in TOOLS_SCHEMA:
        name = tool["name"]
        desc = _gemma4_escape(tool.get("description", ""))
        params = tool.get("parameters", {})

        # Build parameters in Gemma 4 format
        props = params.get("properties", {})
        required = params.get("required", [])

        prop_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string").upper()
            pdesc = _gemma4_escape(pinfo.get("description", ""))
            parts = [f"type:<|\"|\u003e{ptype}<|\"|>"]
            if pdesc:
                parts.append(f"description:<|\"|\u003e{pdesc}<|\"|>")
            if "enum" in pinfo:
                enum_items = ",".join(f'<|"|>{_gemma4_escape(e)}<|"|>' for e in pinfo["enum"])
                parts.append(f"enum:[{enum_items}]")
            prop_parts.append(f"{pname}:{{{','.join(parts)}}}")

        req_items = ",".join(f'<|"|>{r}<|"|>' for r in required)

        decl = (
            f'<|tool>declaration:{name}{{'
            f'description:<|"|>{desc}<|"|>,'
            f'parameters:{{properties:{{{",".join(prop_parts)}}},'
            f'required:[{req_items}],type:<|"|>OBJECT<|"|>}}'
            f'}}<|tool|>'
        )
        decls.append(decl)
    return "".join(decls)


def format_gemma4(scenario: dict) -> str:
    """Format scenario in Gemma 4 native format — NO system prompt instructions.

    Only tool declarations in the system turn, no behavioral instructions.
    Uses Gemma 4's native tool_call/tool_response markers.
    """
    expanded = expand_tool_calls(scenario)
    parts = []

    # System turn: ONLY tool declarations, no instructions
    tool_decls = _gemma4_tool_declarations()
    parts.append(f"<|turn>system\n{tool_decls}<|turn|>")

    for turn in expanded:
        role = turn.get("role", "")

        if role == "user":
            parts.append(f"<|turn>user\n{turn.get('content', '')}<|turn|>")

        elif role == "assistant":
            tool_calls = turn.get("tool_calls", [])
            if tool_calls:
                tc = tool_calls[0]
                name = tc["name"]
                args = tc.get("arguments", {})
                # Encode arguments in Gemma 4 format: key:<|"|>value<|"|>
                arg_parts = ",".join(
                    f"{k}:{_gemma4_encode_value(v)}"
                    for k, v in args.items()
                )
                call_str = f"<|tool_call>call:{name}{{{arg_parts}}}<|tool_call|>"
                parts.append(f"<|turn>model\n{call_str}<|turn|>")
            else:
                content = turn.get("content", "")
                parts.append(f"<|turn>model\n{content}<|turn|>")

        elif role == "tool":
            # Tool response: feed back as tool_response in the model turn
            content = turn.get("content", "")
            # Gemma 4 puts tool_response inline in the model's turn context
            parts.append(f"<|tool_response>{content}<|tool_response|>")

    return "\n".join(parts)


def format_gemma4_bare(scenario: dict) -> str:
    """Format scenario for Gemma 4 with NO system turn at all.

    Completely bare — no instructions, no tool declarations.
    The model learns tool patterns purely from examples.
    Tool calls use Gemma 4 native format.
    """
    expanded = expand_tool_calls(scenario)
    parts = []

    for turn in expanded:
        role = turn.get("role", "")

        if role == "user":
            parts.append(f"<|turn>user\n{turn.get('content', '')}<|turn|>")

        elif role == "assistant":
            tool_calls = turn.get("tool_calls", [])
            if tool_calls:
                tc = tool_calls[0]
                name = tc["name"]
                args = tc.get("arguments", {})
                arg_parts = ",".join(
                    f"{k}:{_gemma4_encode_value(v)}"
                    for k, v in args.items()
                )
                call_str = f"<|tool_call>call:{name}{{{arg_parts}}}<|tool_call|>"
                parts.append(f"<|turn>model\n{call_str}<|turn|>")
            else:
                content = turn.get("content", "")
                parts.append(f"<|turn>model\n{content}<|turn|>")

        elif role == "tool":
            content = turn.get("content", "")
            parts.append(f"<|tool_response>{content}<|tool_response|>")

    return "\n".join(parts)


def format_qwen_native(scenario: dict) -> str:
    """Format scenario as Qwen-native ChatML with proper tool calling format.

    Key differences from old format:
    - 1 tool call per assistant turn
    - <tool_call>\\n{json}\\n</tool_call> format with newlines
    - <tool_response>\\n{content}\\n</tool_response> for results
    - System prompt includes tools schema
    - <|im_end|> immediately after </tool_call>
    """
    expanded = expand_tool_calls(scenario)
    parts = []

    # System prompt with tools
    system = build_system_prompt_with_tools()
    parts.append(f"<|im_start|>system\n{system}<|im_end|>")

    for turn in expanded:
        role = turn.get("role", "")

        if role == "user":
            parts.append(f"<|im_start|>user\n{turn.get('content', '')}<|im_end|>")

        elif role == "assistant":
            tool_calls = turn.get("tool_calls", [])
            if tool_calls:
                tc = tool_calls[0]  # 1 per turn after expansion
                tc_json = json.dumps(tc, ensure_ascii=False)
                parts.append(f"<|im_start|>assistant\n<tool_call>\n{tc_json}\n</tool_call><|im_end|>")
            else:
                content = turn.get("content", "")
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        elif role == "tool":
            content = turn.get("content", "")
            parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>")

    return "\n".join(parts)


def compute_assistant_mask(text: str, tokenizer) -> list[int]:
    """Compute a mask indicating which tokens are from assistant turns.

    Returns list of 1 (assistant/trainable) or 0 (context/masked).
    Used to create labels where non-assistant tokens are -100.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    mask = [0] * len(tokens)

    # Find assistant turn boundaries in the text
    # <|im_start|>assistant\n...<|im_end|>
    pos = 0
    while True:
        start = text.find("<|im_start|>assistant\n", pos)
        if start == -1:
            break
        end = text.find("<|im_end|>", start)
        if end == -1:
            break
        end += len("<|im_end|>")

        # Find token indices for this range
        prefix = text[:start]
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        span = text[:end]
        span_tokens = len(tokenizer.encode(span, add_special_tokens=False))

        for j in range(prefix_tokens, min(span_tokens, len(mask))):
            mask[j] = 1

        pos = end

    return mask


def build_dataset(fmt: str = "qwen_native"):
    """Read all scenario files and build the dataset."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    all_examples = []
    scenario_files = sorted(SCENARIOS_DIR.glob("*.jsonl"))

    if not scenario_files:
        print(f"No scenario files found in {SCENARIOS_DIR}")
        return

    for sf in scenario_files:
        with open(sf) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    scenario = json.loads(line)

                    if fmt == "qwen_native":
                        text = format_qwen_native(scenario)
                    elif fmt == "gemma4":
                        text = format_gemma4(scenario)
                    elif fmt == "gemma4_bare":
                        text = format_gemma4_bare(scenario)
                    elif fmt == "raw":
                        text = json.dumps(scenario, ensure_ascii=False)
                    else:
                        text = format_qwen_native(scenario)  # default

                    all_examples.append({
                        "text": text,
                        "metadata": {
                            "repo": scenario.get("repo", sf.stem),
                            "lang": scenario.get("lang", ""),
                            "type": scenario.get("type", ""),
                        },
                    })
                except json.JSONDecodeError as e:
                    print(f"  [error] {sf.name}:{line_num}: {e}")
                except Exception as e:
                    print(f"  [error] {sf.name}:{line_num}: {e}")

    print(f"Loaded {len(all_examples)} examples from {len(scenario_files)} files (format: {fmt})")

    random.seed(42)
    random.shuffle(all_examples)

    split_idx = max(1, len(all_examples) // 10)
    val_examples = all_examples[:split_idx]
    train_examples = all_examples[split_idx:]

    train_path = DATASET_DIR / f"train_{fmt}.jsonl"
    val_path = DATASET_DIR / f"val_{fmt}.jsonl"

    for path, data in [(train_path, train_examples), (val_path, val_examples)]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    type_counts = {}
    lang_counts = {}
    for ex in all_examples:
        t = ex["metadata"]["type"]
        l = ex["metadata"]["lang"]
        type_counts[t] = type_counts.get(t, 0) + 1
        lang_counts[l] = lang_counts.get(l, 0) + 1

    print(f"\nDataset ({fmt}):")
    print(f"  {train_path.name}: {len(train_examples)} train")
    print(f"  {val_path.name}: {len(val_examples)} val")
    print(f"\nBy type: {json.dumps(type_counts, indent=2)}")
    print(f"\nBy language: {json.dumps(lang_counts, indent=2)}")

    # Sample check
    if train_examples:
        sample = train_examples[0]["text"]
        if fmt.startswith("gemma4"):
            tc_count = sample.count("<|tool_call>call:")
            tc_closed = sample.count("<|tool_call|>")
            print(f"\nSample check (first example):")
            print(f"  Tool calls: {tc_count}")
            print(f"  Properly closed (<|tool_call|>): {tc_closed}")
            print(f"  Match: {'YES' if tc_count == tc_closed else 'NO — mismatch!'}")
            has_system = "<|turn>system" in sample
            print(f"  System turn: {'YES' if has_system else 'NO (bare mode)'}")
        else:
            tc_count = sample.count("<tool_call>")
            im_end_after_tc = sample.count("</tool_call><|im_end|>")
            print(f"\nSample check (first example):")
            print(f"  Tool calls: {tc_count}")
            print(f"  Properly closed (</tool_call><|im_end|>): {im_end_after_tc}")
            print(f"  Match: {'YES' if tc_count == im_end_after_tc else 'NO — some tool calls not properly closed!'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", "-f", default="qwen_native",
                        choices=["qwen_native", "gemma4", "gemma4_bare", "raw"])
    args = parser.parse_args()
    build_dataset(args.format)


if __name__ == "__main__":
    main()
