#!/usr/bin/env python3
"""
Patch training examples in finetune/scenarios_v3/ to:
1. Ensure every example starts with file discovery (list_directory or glob) before any read_file
2. Add file-not-found -> glob -> found recovery to ~30% of examples
3. Fix any template-style step_complete summaries

Excludes batch4_* and batch5_* files.
"""

import json
import os
import random
import re
import sys
from pathlib import Path

random.seed(42)  # Reproducible

SCENARIOS_DIR = Path(__file__).parent / "scenarios_v3"

# Tools that count as "file discovery"
DISCOVERY_TOOLS = {"list_directory", "glob", "project_structure"}

# Tools that should come AFTER discovery
NEEDS_DISCOVERY_BEFORE = {"read_file", "partial_read", "help"}

# Directory listings by language
DIR_LISTINGS = {
    "python": {
        ".": ".\n  pyproject.toml\n  README.md\n  src/\n  tests/\n  requirements.txt",
        "src/": "src/\n  __init__.py\n  app.py\n  models.py\n  utils.py\n  config.py",
    },
    "typescript": {
        ".": ".\n  package.json\n  tsconfig.json\n  src/\n  tests/\n  node_modules/\n  README.md",
        "src/": "src/\n  index.ts\n  app.ts\n  types.ts\n  utils.ts",
    },
    "rust": {
        ".": ".\n  Cargo.toml\n  Cargo.lock\n  src/\n  tests/\n  README.md",
        "src/": "src/\n  main.rs\n  lib.rs\n  models.rs\n  utils.rs",
    },
    "go": {
        ".": ".\n  go.mod\n  go.sum\n  main.go\n  handlers.go\n  models.go\n  store.go\n  README.md",
    },
    "c": {
        ".": ".\n  Makefile\n  README.md\n  src/\n  include/\n  tests/",
        "src/": "src/\n  main.c\n  utils.c\n  parser.c",
    },
}

# Template summary patterns to detect
TEMPLATE_PATTERNS = [
    re.compile(r"Read:\s*files\+findings", re.IGNORECASE),
    re.compile(r"Changed:\s*files\+edits", re.IGNORECASE),
    re.compile(r"Remaining:\s*next work", re.IGNORECASE),
    re.compile(r"Decisions:\s*key choices", re.IGNORECASE),
    re.compile(r"Skip empty categories", re.IGNORECASE),
    re.compile(r"\{files\}", re.IGNORECASE),
    re.compile(r"\{edits\}", re.IGNORECASE),
    re.compile(r"\{findings\}", re.IGNORECASE),
]


def get_lang(data):
    """Extract the language from example metadata."""
    lang = data.get("lang", "python")
    # Normalize
    if lang in ("ts", "typescript"):
        return "typescript"
    if lang in ("rs", "rust"):
        return "rust"
    return lang


def get_dir_listing(lang, path="."):
    """Get a realistic directory listing for the given language."""
    listings = DIR_LISTINGS.get(lang, DIR_LISTINGS["python"])
    # Try exact match, then "." fallback
    if path in listings:
        return listings[path]
    if path.rstrip("/") + "/" in listings:
        return listings[path.rstrip("/") + "/"]
    # Default
    return listings.get(".", ".\n  README.md\n  src/\n  tests/")


def find_first_tool_index(turns):
    """Find index of first assistant turn with tool_calls after the user turn."""
    for i, t in enumerate(turns):
        if t["role"] == "assistant" and "tool_calls" in t:
            return i
    return -1


def get_first_tool_name(turns):
    """Get the name of the first tool called."""
    idx = find_first_tool_index(turns)
    if idx < 0:
        return None
    return turns[idx]["tool_calls"][0]["name"]


def needs_discovery_prefix(turns):
    """Check if this example needs a file discovery call prepended."""
    first_tool = get_first_tool_name(turns)
    if first_tool is None:
        return False
    if first_tool in DISCOVERY_TOOLS:
        return False
    if first_tool in NEEDS_DISCOVERY_BEFORE:
        return True
    # For execute_command, code_search, etc. -- also add discovery
    # unless it's web_search, send_message, think, search_findings
    if first_tool in ("web_search", "send_message", "think", "search_findings",
                       "record_finding", "web_fetch"):
        return False
    # For execute_command, code_search -- add discovery
    if first_tool in ("execute_command", "code_search"):
        return True
    return False


def add_discovery_prefix(turns, lang, data):
    """Insert a list_directory or glob call before the first tool call."""
    idx = find_first_tool_index(turns)
    if idx < 0:
        return turns, False

    # Decide whether to use list_directory or glob (60/40 split)
    use_glob = random.random() < 0.4

    if use_glob:
        # Build a glob pattern based on the language
        ext_map = {
            "python": "**/*.py",
            "typescript": "**/*.ts",
            "rust": "**/*.rs",
            "go": "**/*.go",
            "c": "**/*.c",
        }
        pattern = ext_map.get(lang, "**/*.py")

        # Build a realistic glob result from repo context
        repo = data.get("repo", "project")
        glob_results = _build_glob_results(lang, turns)

        assistant_turn = {
            "role": "assistant",
            "tool_calls": [{"name": "glob", "arguments": {"pattern": pattern}}],
        }
        tool_turn = {"role": "tool", "content": glob_results}
    else:
        # Determine a good path for list_directory
        # Look at what the first read_file targets to pick a relevant directory
        first_read_path = _find_first_read_path(turns)
        if first_read_path:
            dir_path = os.path.dirname(first_read_path)
            if not dir_path or dir_path == ".":
                dir_path = "."
        else:
            dir_path = "."

        listing = _build_dir_listing(lang, dir_path, turns)

        assistant_turn = {
            "role": "assistant",
            "tool_calls": [
                {"name": "list_directory", "arguments": {"path": dir_path}}
            ],
        }
        tool_turn = {"role": "tool", "content": listing}

    # Insert before the first tool call
    new_turns = turns[:idx] + [assistant_turn, tool_turn] + turns[idx:]
    return new_turns, True


def _find_first_read_path(turns):
    """Find the path from the first read_file call."""
    for t in turns:
        if t["role"] == "assistant" and "tool_calls" in t:
            for tc in t["tool_calls"]:
                if tc["name"] in ("read_file", "partial_read"):
                    return tc["arguments"].get("path", "")
    return ""


def _build_dir_listing(lang, dir_path, turns):
    """Build a realistic directory listing incorporating actual file paths from the example."""
    # Collect all file paths mentioned in the example
    files_mentioned = set()
    for t in turns:
        if t["role"] == "assistant" and "tool_calls" in t:
            for tc in t["tool_calls"]:
                args = tc.get("arguments", {})
                for key in ("path", "file_path"):
                    if key in args:
                        files_mentioned.add(args[key])

    if dir_path == ".":
        # Build root listing
        dirs_seen = set()
        root_files = set()
        for fp in files_mentioned:
            parts = fp.strip("./").split("/")
            if len(parts) > 1:
                dirs_seen.add(parts[0])
            else:
                root_files.add(parts[0])

        # Add language-specific defaults
        defaults = {
            "python": ({"src", "tests"}, {"pyproject.toml", "README.md", "requirements.txt"}),
            "typescript": ({"src", "tests"}, {"package.json", "tsconfig.json", "README.md"}),
            "rust": ({"src", "tests"}, {"Cargo.toml", "README.md"}),
            "go": (set(), {"go.mod", "main.go", "README.md"}),
            "c": ({"src", "include", "tests"}, {"Makefile", "README.md"}),
        }
        default_dirs, default_files = defaults.get(lang, ({"src"}, {"README.md"}))
        dirs_seen = dirs_seen | default_dirs
        root_files = root_files | default_files

        lines = ["."]
        for d in sorted(dirs_seen):
            lines.append(f"  {d}/")
        for f in sorted(root_files):
            lines.append(f"  {f}")
        return "\n".join(lines)
    else:
        # Build subdirectory listing
        prefix = dir_path.rstrip("/") + "/"
        entries = set()
        for fp in files_mentioned:
            if fp.startswith(prefix):
                remainder = fp[len(prefix):]
                parts = remainder.split("/")
                if len(parts) > 1:
                    entries.add(parts[0] + "/")
                else:
                    entries.add(parts[0])

        if not entries:
            return get_dir_listing(lang, dir_path)

        lines = [dir_path.rstrip("/") + "/"]
        for e in sorted(entries):
            lines.append(f"  {e}")
        return "\n".join(lines)


def _build_glob_results(lang, turns):
    """Build realistic glob results from files mentioned in the example."""
    ext_map = {"python": ".py", "typescript": ".ts", "rust": ".rs", "go": ".go", "c": ".c"}
    ext = ext_map.get(lang, ".py")

    files_mentioned = []
    for t in turns:
        if t["role"] == "assistant" and "tool_calls" in t:
            for tc in t["tool_calls"]:
                args = tc.get("arguments", {})
                for key in ("path", "file_path"):
                    if key in args and args[key].endswith(ext):
                        files_mentioned.append(args[key])

    if files_mentioned:
        # Deduplicate preserving order
        seen = set()
        unique = []
        for f in files_mentioned:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return "\n".join(unique)

    # Fallback
    fallbacks = {
        "python": "src/app.py\nsrc/models.py\nsrc/utils.py\ntests/test_app.py",
        "typescript": "src/index.ts\nsrc/app.ts\nsrc/types.ts\ntests/app.test.ts",
        "rust": "src/main.rs\nsrc/lib.rs\ntests/integration_test.rs",
        "go": "main.go\nhandlers.go\nmodels.go\nstore.go",
        "c": "src/main.c\nsrc/utils.c\ntests/test_main.c",
    }
    return fallbacks.get(lang, fallbacks["python"])


def add_file_not_found_recovery(turns, lang):
    """Add a file-not-found -> glob -> read recovery pattern to one read_file call.

    Returns (modified_turns, was_modified).
    """
    # Find a read_file call that we can modify (not the very first one after discovery)
    read_indices = []
    for i, t in enumerate(turns):
        if t["role"] == "assistant" and "tool_calls" in t:
            for tc in t["tool_calls"]:
                if tc["name"] in ("read_file", "partial_read") and "path" in tc.get("arguments", {}):
                    read_indices.append(i)

    if len(read_indices) < 2:
        # Need at least 2 read_file calls -- modify the first one is risky
        # if there's only 1, try it anyway
        if len(read_indices) == 0:
            return turns, False
        target_idx = read_indices[0]
    else:
        # Pick the second read_file (skip the first which might be right after discovery)
        target_idx = read_indices[1]

    # Get the assistant turn and its tool response
    assistant_turn = turns[target_idx]
    tc = None
    for candidate in assistant_turn["tool_calls"]:
        if candidate["name"] in ("read_file", "partial_read"):
            tc = candidate
            break
    if not tc:
        return turns, False

    correct_path = tc["arguments"]["path"]
    filename = os.path.basename(correct_path)
    dir_part = os.path.dirname(correct_path)

    # Create a plausible wrong path
    wrong_path = _make_wrong_path(correct_path, lang)
    if wrong_path == correct_path:
        return turns, False

    # Find the corresponding tool response
    tool_idx = target_idx + 1
    if tool_idx >= len(turns) or turns[tool_idx]["role"] != "tool":
        return turns, False

    original_tool_content = turns[tool_idx]["content"]

    # Build the recovery sequence
    # 1. Original assistant turn but with wrong path
    wrong_assistant = json.loads(json.dumps(assistant_turn))
    for candidate in wrong_assistant["tool_calls"]:
        if candidate["name"] in ("read_file", "partial_read"):
            candidate["arguments"]["path"] = wrong_path
            break

    # 2. File not found response
    wrong_tool = {
        "role": "tool",
        "content": f"File not found: /workspace/{wrong_path}",
    }

    # 3. Think about the error
    think_assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "think",
                "arguments": {
                    "reasoning": f"File not at {wrong_path}. Let me search for it."
                },
            }
        ],
    }
    think_tool = {"role": "tool", "content": "[Thought recorded]"}

    # 4. Glob to find the file
    glob_assistant = {
        "role": "assistant",
        "tool_calls": [
            {"name": "glob", "arguments": {"pattern": f"**/{filename}"}}
        ],
    }
    glob_tool = {"role": "tool", "content": correct_path}

    # 5. Read the correct file
    correct_assistant = json.loads(json.dumps(assistant_turn))
    # Keep the original tool_calls (with correct path)

    correct_tool = {"role": "tool", "content": original_tool_content}

    # Replace the original assistant+tool pair with the recovery sequence
    new_turns = (
        turns[:target_idx]
        + [
            wrong_assistant,
            wrong_tool,
            think_assistant,
            think_tool,
            glob_assistant,
            glob_tool,
            correct_assistant,
            correct_tool,
        ]
        + turns[tool_idx + 1 :]
    )

    return new_turns, True


def _make_wrong_path(correct_path, lang):
    """Create a plausible wrong path from the correct one."""
    parts = correct_path.split("/")
    filename = parts[-1]

    if len(parts) >= 3:
        # e.g., "src/taskflow/manager.py" -> "taskflow/manager.py" (drop first dir)
        return "/".join(parts[1:])
    elif len(parts) == 2:
        # e.g., "src/app.py" -> "app.py" (drop dir)
        # or "src/app.py" -> "lib/app.py" (wrong dir)
        dir_part = parts[0]
        wrong_dirs = {
            "src": "lib",
            "lib": "src",
            "app": "src",
            "tests": "test",
            "test": "tests",
            "pkg": "src",
            "cmd": "src",
            "internal": "pkg",
        }
        new_dir = wrong_dirs.get(dir_part, "src" if dir_part != "src" else "lib")
        return f"{new_dir}/{filename}"
    else:
        # Single file like "main.go" -> "src/main.go"
        return f"src/{filename}"


def has_template_summary(summary):
    """Check if a step_complete summary looks like a template."""
    for pat in TEMPLATE_PATTERNS:
        if pat.search(summary):
            return True
    return False


def fix_template_summary(summary, turns, data):
    """Generate a specific summary based on context.

    Since we found no actual template summaries in the data,
    this is a safety net that returns the original if no fix is needed.
    """
    if not has_template_summary(summary):
        return summary

    # Collect context
    repo = data.get("repo", "project")
    lang = get_lang(data)
    files_read = []
    files_modified = []
    for t in turns:
        if t["role"] == "assistant" and "tool_calls" in t:
            for tc in t["tool_calls"]:
                args = tc.get("arguments", {})
                name = tc["name"]
                path = args.get("path", args.get("file_path", ""))
                if name in ("read_file", "partial_read") and path:
                    files_read.append(path)
                elif name in ("replace_lines", "create_file", "add_symbol",
                              "edit_symbol", "add_content_after_line",
                              "add_content_before_line") and path:
                    files_modified.append(path)

    # Build a specific summary
    parts = []
    if files_read:
        parts.append(f"Read {', '.join(files_read[:3])}")
    if files_modified:
        parts.append(f"Modified {', '.join(files_modified[:3])}")
    if not parts:
        parts.append(f"Explored {repo} project structure")

    return ". ".join(parts) + "."


def validate_turn_alternation(turns):
    """Validate that turns follow valid patterns.

    Valid sequences:
    - user -> assistant -> tool -> assistant -> tool -> ...
    - Multi-turn: ... tool -> user -> assistant -> tool -> ... (user follow-up)
    - send_message -> tool -> user (user responds to question)
    """
    if not turns:
        return True, ""
    if turns[0]["role"] != "user":
        return False, "First turn must be 'user'"

    for i in range(1, len(turns)):
        curr = turns[i]["role"]
        prev = turns[i - 1]["role"]

        if curr == "user" and prev != "tool":
            return False, f"'user' turn at index {i} not preceded by 'tool'"
        if curr == "tool" and prev != "assistant":
            return False, f"'tool' turn at index {i} not preceded by 'assistant'"
        if curr == "assistant" and prev not in ("user", "tool"):
            return False, f"'assistant' turn at index {i} preceded by '{prev}'"

    return True, ""


def process_file(filepath):
    """Process a single .jsonl file. Returns stats dict."""
    stats = {
        "modified": False,
        "discovery_added": False,
        "fnf_added": False,
        "summaries_fixed": 0,
        "errors": [],
    }

    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception as e:
        stats["errors"].append(f"Read error: {e}")
        return stats

    new_lines = []
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            new_lines.append("\n")
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            stats["errors"].append(f"JSON parse error line {line_num}: {e}")
            new_lines.append(line + "\n")
            continue

        if "turns" not in data:
            new_lines.append(line + "\n")
            continue

        turns = data["turns"]
        lang = get_lang(data)
        changed = False

        # Step 1: Add file discovery if needed
        if needs_discovery_prefix(turns):
            turns, added = add_discovery_prefix(turns, lang, data)
            if added:
                stats["discovery_added"] = True
                changed = True

        # Step 2: Fix template summaries
        for t in turns:
            if t["role"] == "assistant" and "tool_calls" in t:
                for tc in t["tool_calls"]:
                    if tc["name"] == "step_complete" and "summary" in tc.get("arguments", {}):
                        old_summary = tc["arguments"]["summary"]
                        if has_template_summary(old_summary):
                            new_summary = fix_template_summary(old_summary, turns, data)
                            if new_summary != old_summary:
                                tc["arguments"]["summary"] = new_summary
                                stats["summaries_fixed"] += 1
                                changed = True

        if changed:
            data["turns"] = turns

        # Validate turn alternation
        valid, err = validate_turn_alternation(data["turns"])
        if not valid:
            stats["errors"].append(f"Turn alternation error: {err}")
            # Don't write changes if invalid
            new_lines.append(line + "\n")
            continue

        if changed:
            stats["modified"] = True

        new_lines.append(json.dumps(data, ensure_ascii=False) + "\n")

    # Write back
    if stats["modified"]:
        with open(filepath, "w") as f:
            f.writelines(new_lines)

    return stats


def main():
    files = sorted(SCENARIOS_DIR.glob("*.jsonl"))
    # Exclude batch4_ and batch5_
    files = [f for f in files if not f.name.startswith("batch4_") and not f.name.startswith("batch5_")]

    print(f"Found {len(files)} files to process (excluding batch4_*/batch5_*)")

    # Phase 1: Analyze
    print("\n=== Phase 1: Analysis ===")
    needs_discovery = []
    has_recovery = []
    has_template = []

    for filepath in files:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "turns" not in data:
                    continue
                turns = data["turns"]
                fname = filepath.name

                # Check discovery
                if needs_discovery_prefix(turns):
                    needs_discovery.append(fname)

                # Check existing recovery
                text = json.dumps(data)
                if "File not found" in text or "No such file" in text:
                    has_recovery.append(fname)

                # Check template summaries
                for t in turns:
                    if t["role"] == "assistant" and "tool_calls" in t:
                        for tc in t["tool_calls"]:
                            if tc["name"] == "step_complete" and "summary" in tc.get("arguments", {}):
                                if has_template_summary(tc["arguments"]["summary"]):
                                    has_template.append(fname)
                                    break

    print(f"  Examples needing discovery prefix: {len(needs_discovery)}")
    print(f"  Examples already having file-not-found recovery: {len(has_recovery)}")
    print(f"  Examples with template summaries: {len(has_template)}")

    # Phase 2: Fix discovery and summaries
    print("\n=== Phase 2: Fixing discovery prefixes and summaries ===")
    total_modified = 0
    total_discovery = 0
    total_summaries = 0
    total_errors = 0

    for filepath in files:
        stats = process_file(filepath)
        if stats["modified"]:
            total_modified += 1
        if stats["discovery_added"]:
            total_discovery += 1
        total_summaries += stats["summaries_fixed"]
        if stats["errors"]:
            total_errors += len(stats["errors"])
            for err in stats["errors"]:
                print(f"  ERROR in {filepath.name}: {err}")

    print(f"  Files modified for discovery: {total_discovery}")
    print(f"  Summaries fixed: {total_summaries}")
    print(f"  Errors: {total_errors}")

    # Phase 3: Add file-not-found recovery to ~30% of examples
    print("\n=== Phase 3: Adding file-not-found recovery ===")

    # Reload to get updated versions
    candidates = []
    for filepath in files:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "turns" not in data:
                    continue
                text = json.dumps(data)
                # Skip if already has recovery
                if "File not found" in text or "No such file" in text:
                    continue
                # Must have at least one read_file
                has_read = False
                for t in data["turns"]:
                    if t["role"] == "assistant" and "tool_calls" in t:
                        for tc in t["tool_calls"]:
                            if tc["name"] in ("read_file", "partial_read"):
                                has_read = True
                                break
                if has_read:
                    candidates.append(filepath)

    # Target ~30% of all examples, not just candidates
    target_count = max(1, int(len(files) * 0.30))
    # But cap at number of candidates
    target_count = min(target_count, len(candidates))

    # Randomly select
    selected = random.sample(candidates, target_count) if candidates else []

    fnf_added = 0
    fnf_errors = 0

    for filepath in selected:
        with open(filepath) as f:
            line = f.readline().strip()

        data = json.loads(line)
        lang = get_lang(data)
        turns = data["turns"]

        new_turns, added = add_file_not_found_recovery(turns, lang)
        if added:
            data["turns"] = new_turns
            # Validate
            valid, err = validate_turn_alternation(data["turns"])
            if valid:
                with open(filepath, "w") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                fnf_added += 1
            else:
                print(f"  SKIP {filepath.name}: turn alternation error after FNF: {err}")
                fnf_errors += 1
        else:
            fnf_errors += 1

    print(f"  Candidates for FNF recovery: {len(candidates)}")
    print(f"  Target count (~30%): {target_count}")
    print(f"  Successfully added FNF recovery: {fnf_added}")
    print(f"  Failed/skipped: {fnf_errors}")

    # Phase 4: Final validation
    print("\n=== Phase 4: Final Validation ===")
    valid_count = 0
    invalid_count = 0
    for filepath in files:
        with open(filepath) as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "turns" in data:
                        valid, err = validate_turn_alternation(data["turns"])
                        if valid:
                            valid_count += 1
                        else:
                            invalid_count += 1
                            print(f"  INVALID {filepath.name}:{line_num}: {err}")
                    else:
                        valid_count += 1
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    print(f"  JSON ERROR {filepath.name}:{line_num}: {e}")

    print(f"  Valid examples: {valid_count}")
    print(f"  Invalid examples: {invalid_count}")

    # Final report
    print("\n=== SUMMARY ===")
    print(f"Total files processed: {len(files)}")
    print(f"Files modified (discovery prefix): {total_discovery}")
    print(f"Files modified (FNF recovery): {fnf_added}")
    print(f"Total files modified: {total_discovery + fnf_added}")
    print(f"Template summaries fixed: {total_summaries}")
    print(f"All examples valid JSON: {'YES' if invalid_count == 0 else 'NO'}")
    print(f"All turn alternation correct: {'YES' if invalid_count == 0 else 'NO'}")


if __name__ == "__main__":
    main()
