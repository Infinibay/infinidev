#!/usr/bin/env python3
"""Convert v3 scenarios from embedded <tool_call> XML to structured JSON format.

Before: {"role": "assistant", "content": "<tool_call>{\"name\":...}</tool_call>"}
After:  {"role": "assistant", "tool_calls": [{"name":..., "arguments":...}]}

Usage:
    python finetune/convert_to_structured.py              # convert all
    python finetune/convert_to_structured.py flask_001     # convert one
"""

import json
import os
import re
import sys
from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent / "scenarios_v3"


def extract_tool_calls(content: str) -> list[dict]:
    """Extract tool calls from <tool_call>...</tool_call> tags."""
    calls = []
    for m in re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
        try:
            tc = json.loads(m.group(1))
            calls.append(tc)
        except json.JSONDecodeError:
            # Try fixing newlines
            fixed = m.group(1).replace('\n', '\\n')
            try:
                tc = json.loads(fixed)
                calls.append(tc)
            except json.JSONDecodeError:
                # Keep as raw text fallback
                calls.append({"name": "_raw", "arguments": {"text": m.group(1)[:200]}})
    return calls


def convert_file(filepath: Path) -> dict:
    """Convert a single scenario file to structured format."""
    with open(filepath) as f:
        data = json.loads(f.readline())

    new_turns = []
    stats = {"converted": 0, "raw_fallback": 0}

    for turn in data["turns"]:
        if turn["role"] == "assistant":
            tool_calls = extract_tool_calls(turn["content"])
            if tool_calls:
                new_turn = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                stats["converted"] += len(tool_calls)
                # Check for raw fallbacks
                stats["raw_fallback"] += sum(1 for tc in tool_calls if tc.get("name") == "_raw")
            else:
                # No tool calls found — keep as text content
                new_turn = {"role": "assistant", "content": turn["content"]}
            new_turns.append(new_turn)

        elif turn["role"] == "tool":
            # Tool results stay as content strings
            new_turns.append({
                "role": "tool",
                "content": turn["content"],
            })

        else:
            # User turns stay as-is
            new_turns.append(turn)

    data["turns"] = new_turns
    data["format_version"] = "v3_structured"
    return data, stats


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target:
        files = list(SCENARIOS_DIR.glob(f"{target}*.jsonl"))
    else:
        files = sorted(SCENARIOS_DIR.glob("*.jsonl"))

    if not files:
        print(f"No files found in {SCENARIOS_DIR}")
        return

    total_converted = 0
    total_fallback = 0

    for f in files:
        data, stats = convert_file(f)

        # Write back
        with open(f, 'w') as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + '\n')

        total_converted += stats["converted"]
        total_fallback += stats["raw_fallback"]

        status = "OK" if stats["raw_fallback"] == 0 else f"WARN ({stats['raw_fallback']} raw)"
        print(f"  [{status}] {f.stem}: {stats['converted']} tool calls")

    print(f"\nTotal: {total_converted} tool calls converted, {total_fallback} raw fallbacks")


if __name__ == "__main__":
    main()
