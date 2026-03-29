#!/usr/bin/env python3
"""Fix common issues in generated training examples.

Fixes:
1. Unclosed <tool_call> tags
2. Unescaped newlines in JSON strings inside tool_calls
3. "..." placeholder lines in tool results
4. Normalize tool call format: "tool"→"name", "args"→"arguments", "topic"→"context"

Usage:
    python finetune/fix_examples.py                    # fix all in scenarios_v3/
    python finetune/fix_examples.py flask_001           # fix one file
"""

import json
import re
import sys
from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent / "scenarios_v3"


def fix_unclosed_tool_calls(content: str) -> tuple[str, int]:
    """Fix <tool_call> tags that are missing </tool_call>."""
    fixed = 0
    if '<tool_call>' not in content:
        return content, 0

    parts = content.split('<tool_call>')
    new_parts = [parts[0]]

    for part in parts[1:]:
        if '</tool_call>' in part:
            new_parts.append('<tool_call>' + part)
        else:
            brace_start = part.find('{')
            if brace_start >= 0:
                depth = 0
                end = -1
                for i, ch in enumerate(part[brace_start:], brace_start):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end > 0:
                    json_str = part[:end]
                    remainder = part[end:]
                    new_parts.append(f'<tool_call>{json_str}</tool_call>{remainder}')
                    fixed += 1
                else:
                    new_parts.append('<tool_call>' + part)
            else:
                new_parts.append('<tool_call>' + part)

    return ''.join(new_parts), fixed


def fix_newlines_in_tool_calls(content: str) -> tuple[str, int]:
    """Fix unescaped newlines inside JSON string values in tool_calls."""
    fixed = 0

    def fix_match(match):
        nonlocal fixed
        raw = match.group(1)
        try:
            json.loads(raw)
            return match.group(0)
        except json.JSONDecodeError:
            # Replace literal newlines inside JSON strings
            result = ''
            in_string = False
            escape_next = False
            for ch in raw:
                if escape_next:
                    result += ch
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    result += ch
                    continue
                if ch == '"':
                    in_string = not in_string
                    result += ch
                    continue
                if in_string and ch == '\n':
                    result += '\\n'
                    fixed += 1
                    continue
                result += ch

            try:
                json.loads(result)
                return f'<tool_call>{result}</tool_call>'
            except json.JSONDecodeError:
                return match.group(0)

    return re.sub(r'<tool_call>(.*?)</tool_call>', fix_match, content, flags=re.DOTALL), fixed


def fix_placeholder_dots(content: str) -> tuple[str, int]:
    """Remove standalone '...' placeholder lines from tool results."""
    lines = content.split('\n')
    cleaned = [l for l in lines if l.strip() != '...']
    removed = len(lines) - len(cleaned)
    return '\n'.join(cleaned), removed


def normalize_tool_call_format(content: str) -> tuple[str, int]:
    """Normalize tool call JSON: 'tool'→'name', 'args'→'arguments', 'topic'→'context'."""
    fixed = 0

    def fix_match(match):
        nonlocal fixed
        raw = match.group(1)
        try:
            tc = json.loads(raw)
        except json.JSONDecodeError:
            return match.group(0)

        changed = False
        # Rename "tool" → "name"
        if "tool" in tc and "name" not in tc:
            tc["name"] = tc.pop("tool")
            changed = True
        # Rename "args" → "arguments"
        if "args" in tc and "arguments" not in tc:
            tc["arguments"] = tc.pop("args")
            changed = True
        # Rename "topic" → "context" in help args
        if tc.get("name") == "help" and isinstance(tc.get("arguments"), dict):
            args = tc["arguments"]
            if "topic" in args and "context" not in args:
                args["context"] = args.pop("topic")
                changed = True
        # Rename "reasoning" key missing in think
        # Rename "path" → "file_path" for replace_lines, add_content_*
        if tc.get("name") in ("replace_lines", "add_content_after_line", "add_content_before_line"):
            args = tc.get("arguments", {})
            if "path" in args and "file_path" not in args:
                args["file_path"] = args.pop("path")
                changed = True

        if changed:
            fixed += 1
            return f'<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>'
        return match.group(0)

    return re.sub(r'<tool_call>(.*?)</tool_call>', fix_match, content, flags=re.DOTALL), fixed


def fix_file(filepath: Path) -> int:
    """Fix a single scenario file. Returns number of fixes."""
    with open(filepath) as f:
        data = json.loads(f.readline())

    total_fixes = 0

    for t in data['turns']:
        if t['role'] == 'assistant':
            if 'content' in t and '<tool_call>' in (t.get('content') or ''):
                content, n = fix_unclosed_tool_calls(t['content'])
                total_fixes += n
                t['content'] = content

                content, n = fix_newlines_in_tool_calls(t['content'])
                total_fixes += n
                t['content'] = content

                content, n = normalize_tool_call_format(t['content'])
                total_fixes += n
                t['content'] = content
            elif 'tool_calls' in t:
                for tc in t['tool_calls']:
                    if isinstance(tc, dict):
                        if 'tool' in tc and 'name' not in tc:
                            tc['name'] = tc.pop('tool')
                            total_fixes += 1
                        if 'args' in tc and 'arguments' not in tc:
                            tc['arguments'] = tc.pop('args')
                            total_fixes += 1

        if t['role'] == 'tool' and 'content' in t:
            content, n = fix_placeholder_dots(t.get('content') or '')
            total_fixes += n
            t['content'] = content

    # Fix turn alternation: merge consecutive same-role turns
    fixed_turns = [data['turns'][0]]
    for t in data['turns'][1:]:
        prev = fixed_turns[-1]
        if t['role'] == prev['role']:
            # Merge: prefer extending tool_calls list or concatenating content
            if 'tool_calls' in prev and 'tool_calls' in t:
                prev['tool_calls'].extend(t['tool_calls'])
            elif 'content' in prev and 'content' in t:
                prev['content'] = (prev.get('content') or '') + '\n' + (t.get('content') or '')
            total_fixes += 1
        else:
            fixed_turns.append(t)
    data['turns'] = fixed_turns

    with open(filepath, 'w') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

    return total_fixes


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target:
        files = list(SCENARIOS_DIR.glob(f"{target}*.jsonl"))
    else:
        files = sorted(SCENARIOS_DIR.glob("*.jsonl"))

    if not files:
        print(f"No files found in {SCENARIOS_DIR}")
        return

    total = 0
    for f in files:
        fixes = fix_file(f)
        if fixes:
            print(f"  Fixed {fixes} issues in {f.stem}")
        total += fixes

    print(f"\nTotal fixes: {total}")


if __name__ == "__main__":
    main()
