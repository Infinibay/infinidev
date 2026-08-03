#!/usr/bin/env python3
"""Build model-specific training datasets from validated structured scenarios.

Usage:
    python -m finetune.build_dataset --format qwen_native   # recommended
    python -m finetune.build_dataset --format raw            # structured JSON
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from finetune.tool_catalog import get_training_tool_schemas
from finetune.validate_quality import validate_scenario

BASE_DIR = Path(__file__).parent
SCENARIOS_DIR = BASE_DIR / "scenarios_v3"
DATASET_DIR = BASE_DIR / "output" / "dataset"


class DatasetValidationError(ValueError):
    """Raised when source scenarios cannot produce a trustworthy dataset."""


def _function_schemas() -> list[dict[str, Any]]:
    """Return the inner function objects used by model-specific formatters."""
    return [schema["function"] for schema in get_training_tool_schemas()]


def build_system_prompt_with_tools() -> str:
    """Build system prompt with tools schema in Qwen/Ollama format."""
    tools_json = "\n".join(
        json.dumps({"type": "function", "function": schema}, ensure_ascii=False)
        for schema in _function_schemas()
    )
    return f"""You are Infinidev's implementation agent.

Ground every action in the user's task, repository evidence from this turn, or a tool result.

# Tools

The runtime exposes these function schemas:
<tools>
{tools_json}
</tools>

Inspect the files that establish the requested behavior before changing state. After each change,
run the smallest command that exercises that behavior. Report completion only when the tool output
contains concrete evidence that the requested result exists.

Return exactly one function call in each assistant turn, using this format:
<tool_call>
{{"name": "read_file", "arguments": {{"file_path": "src/main.py"}}}}
</tool_call>

The tool call ends the assistant turn. Wait for the runtime result before deciding the next action.
Never invent tool output. Finish with step_complete only after verification succeeds."""


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
            results = (
                [result.strip() for result in tool_result.split("\n---\n")]
                if "---" in tool_result
                else [tool_result]
            )

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
    """Build Gemma 4 tool declarations from live function schemas.

    Format: <|tool>declaration:name{description:<|"|>...<|"|>,parameters:{...}}<|tool|>
    """
    decls = []
    for tool in _function_schemas():
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
                parts.append(
                    f"<|im_start|>assistant\n<tool_call>\n{tc_json}\n</tool_call><|im_end|>"
                )
            else:
                content = turn.get("content", "")
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        elif role == "tool":
            content = turn.get("content", "")
            parts.append(
                f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>"
            )

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


def _load_validated_scenarios() -> list[tuple[Path, dict[str, Any]]]:
    """Load every scenario and reject the batch if any input is invalid."""
    scenario_files = sorted(SCENARIOS_DIR.glob("*.jsonl"))
    if not scenario_files:
        raise DatasetValidationError(f"No scenario files found in {SCENARIOS_DIR}")

    scenarios: list[tuple[Path, dict[str, Any]]] = []
    problems: list[str] = []
    for sf in scenario_files:
        with sf.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    scenario = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(f"{sf.name}:{line_num}: invalid JSON: {e}")
                    continue
                result = validate_scenario(scenario, source=f"{sf.name}:{line_num}")
                if result["errors"]:
                    problems.extend(
                        f"{sf.name}:{line_num}: {message}" for message in result["errors"]
                    )
                    continue
                scenarios.append((sf, scenario))

    if problems:
        shown = "\n".join(f"  - {problem}" for problem in problems[:20])
        omitted = len(problems) - min(len(problems), 20)
        suffix = f"\n  - ... {omitted} additional errors" if omitted else ""
        raise DatasetValidationError(
            f"Refusing to build from {len(problems)} validation errors:\n{shown}{suffix}\n"
            "Run `uv run python -m finetune.validate_quality` for the full audit."
        )
    return scenarios


def build_dataset(fmt: str = "qwen_native") -> tuple[Path, Path]:
    """Build train and validation files from a fully valid scenario batch."""
    scenarios = _load_validated_scenarios()
    all_examples: list[dict[str, Any]] = []
    for source_file, scenario in scenarios:
        if fmt == "qwen_native":
            text = format_qwen_native(scenario)
        elif fmt == "gemma4":
            text = format_gemma4(scenario)
        elif fmt == "gemma4_bare":
            text = format_gemma4_bare(scenario)
        elif fmt == "raw":
            text = json.dumps(scenario, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported dataset format: {fmt}")

        all_examples.append({
            "text": text,
            "metadata": {
                "repo": scenario.get("repo", source_file.stem),
                "lang": scenario.get("lang", ""),
                "type": scenario.get("type", ""),
            },
        })

    print(f"Loaded {len(all_examples)} validated examples (format: {fmt})")

    random.seed(42)
    random.shuffle(all_examples)

    split_idx = max(1, len(all_examples) // 10)
    val_examples = all_examples[:split_idx]
    train_examples = all_examples[split_idx:]

    train_path = DATASET_DIR / f"train_{fmt}.jsonl"
    val_path = DATASET_DIR / f"val_{fmt}.jsonl"

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for path, data in [(train_path, train_examples), (val_path, val_examples)]:
        with path.open("w", encoding="utf-8") as f:
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
            print(f"  Closing markers (<|tool_call|>): {tc_closed}")
            print(f"  Match: {'YES' if tc_count == tc_closed else 'NO — mismatch!'}")
            has_system = "<|turn>system" in sample
            print(f"  System turn: {'YES' if has_system else 'NO (bare mode)'}")
        else:
            tc_count = sample.count("<tool_call>")
            im_end_after_tc = sample.count("</tool_call><|im_end|>")
            print(f"\nSample check (first example):")
            print(f"  Tool calls: {tc_count}")
            print(f"  Closing markers (</tool_call><|im_end|>): {im_end_after_tc}")
            match = "YES" if tc_count == im_end_after_tc else "NO - marker count differs!"
            print(f"  Match: {match}")
    return train_path, val_path


def main() -> int:
    """Parse CLI arguments and return a process exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", "-f", default="qwen_native",
                        choices=["qwen_native", "gemma4", "gemma4_bare", "raw"])
    args = parser.parse_args()
    try:
        build_dataset(args.format)
    except DatasetValidationError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
