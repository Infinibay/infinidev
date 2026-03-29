"""Generate training examples from scenarios.

Each example is a complete conversation turn with:
- System prompt (identity + tool schemas)
- User prompt (task + context)
- Assistant response (perfect tool call sequence)
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finetune.config import OUTPUT_DIR, SCENARIOS_DIR, REPOS_DIR, MAX_FILE_CONTENT


@dataclass
class ToolCall:
    """A single tool call in the sequence."""
    name: str
    arguments: dict
    think_before: str = ""  # think() content before this call


@dataclass
class TrainingExample:
    """A complete training example."""
    scenario_type: str
    repo: str
    prompt: str
    steps: list[list[ToolCall]]  # List of steps, each step is a list of tool calls
    system_prompt: str = ""


def _read_file_numbered(path: str, max_lines: int = 500) -> str:
    """Read a file and format with line numbers like read_file does."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        return "\n".join(f"{i+1:>6}\t{line.rstrip()}" for i, line in enumerate(lines))
    except Exception:
        return ""


def _get_file_content(path: str) -> str:
    """Read raw file content."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > MAX_FILE_CONTENT:
            content = content[:MAX_FILE_CONTENT]
        return content
    except Exception:
        return ""


# ── Example generators by scenario type ──────────────────────────────────────


def gen_bug_fix_example(scenario: dict) -> TrainingExample:
    """Generate a bug fix example with perfect tool calls."""
    target_file = scenario["target_files"][0]
    symbol = scenario["target_symbols"][0] if scenario["target_symbols"] else ""
    rel_path = _rel(target_file, scenario["repo"])
    lines = scenario.get("target_lines", {}).get(target_file, [1, 10])

    step1 = [
        ToolCall(
            name="help",
            arguments={"context": "edit"},
            think_before="I need to fix a bug. Let me first learn how the editing tools work.",
        ),
        ToolCall(
            name="read_file",
            arguments={"path": target_file},
            think_before=f"Now let me read {rel_path} to see the current code and find the bug.",
        ),
        ToolCall(
            name="add_note",
            arguments={"note": f"Bug is in {symbol}() at {rel_path}:{lines[0]}-{lines[1]}. Need to add None check."},
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Read {rel_path}. Found {symbol}() at lines {lines[0]}-{lines[1]}. The method doesn't handle None — crashes with AttributeError. Will add a None guard.",
                "status": "continue",
                "next_steps": [
                    {"op": "add", "index": 1, "description": f"Fix None handling in {symbol}() and run tests"},
                ],
            },
        ),
    ]

    step2 = [
        ToolCall(
            name="edit_symbol",
            arguments={
                "symbol": symbol,
                "new_code": f"    def {symbol.split('.')[-1]}(self, value):\n        if value is None:\n            return None\n        # ... rest of implementation\n        return value\n",
                "file_path": target_file,
            },
            think_before=f"The file is already in <opened-files>, so I don't need to re-read it. I'll fix {symbol}() by adding a None check at the start.",
        ),
        ToolCall(
            name="execute_command",
            arguments={"command": f"cd {os.path.dirname(target_file)} && python -m pytest -x -q 2>&1 | tail -20"},
            think_before="Let me run the tests to verify the fix.",
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Fixed None handling in {symbol}(). Added guard clause at start. Tests pass.",
                "status": "done",
                "final_answer": f"Fixed the bug in `{symbol}()` in `{rel_path}`. Added a None check at the start of the method that returns None instead of crashing with AttributeError. Tests pass.",
            },
        ),
    ]

    return TrainingExample(
        scenario_type="bug_fix",
        repo=scenario["repo"],
        prompt=scenario["prompt"],
        steps=[step1, step2],
    )


def gen_add_method_example(scenario: dict) -> TrainingExample:
    """Generate an 'add method to class' example."""
    target_file = scenario["target_files"][0]
    class_name = scenario.get("extra", {}).get("class_name", "MyClass")
    rel_path = _rel(target_file, scenario["repo"])

    step1 = [
        ToolCall(
            name="help",
            arguments={"context": "edit"},
            think_before="I need to add a method to a class. Let me learn the correct tool workflow first.",
        ),
        ToolCall(
            name="read_file",
            arguments={"path": target_file},
            think_before=f"Let me read {rel_path} to see the {class_name} class structure.",
        ),
        ToolCall(
            name="add_note",
            arguments={"note": f"{class_name} is in {rel_path}. Need to add to_dict() method."},
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Read {rel_path}. {class_name} class found. Will add to_dict() method.",
                "status": "continue",
                "next_steps": [
                    {"op": "add", "index": 1, "description": f"Add to_dict() to {class_name} and test"},
                ],
            },
        ),
    ]

    step2 = [
        ToolCall(
            name="add_symbol",
            arguments={
                "file_path": target_file,
                "code": "def to_dict(self):\n    return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}\n",
                "class_name": class_name,
            },
            think_before=f"The file is in <opened-files>. I'll add to_dict() to {class_name} using add_symbol — it auto-handles indentation.",
        ),
        ToolCall(
            name="execute_command",
            arguments={"command": f"cd {os.path.dirname(target_file)} && python -c \"from {Path(target_file).stem} import {class_name}; print('{class_name}.to_dict exists:', hasattr({class_name}, 'to_dict'))\""},
            think_before="Quick verification that the method was added correctly.",
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Added to_dict() to {class_name}. Returns dict of public attributes. Verified import works.",
                "status": "done",
                "final_answer": f"Added `to_dict()` method to `{class_name}` in `{rel_path}`. The method returns a dictionary of all public (non-underscore) instance attributes.",
            },
        ),
    ]

    return TrainingExample(
        scenario_type="add_method",
        repo=scenario["repo"],
        prompt=scenario["prompt"],
        steps=[step1, step2],
    )


def gen_add_function_example(scenario: dict) -> TrainingExample:
    """Generate an 'add function' example."""
    target_file = scenario["target_files"][0]
    rel_path = _rel(target_file, scenario["repo"])

    step1 = [
        ToolCall(
            name="help",
            arguments={"context": "edit"},
            think_before="I need to add a new function. Let me check the tools first.",
        ),
        ToolCall(
            name="read_file",
            arguments={"path": target_file},
            think_before=f"Let me read {rel_path} to see the existing structure and find the right place to add the function.",
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Read {rel_path}. Will add validate_input() function at the end of the file.",
                "status": "continue",
                "next_steps": [
                    {"op": "add", "index": 1, "description": "Add validate_input() function and test"},
                ],
            },
        ),
    ]

    step2 = [
        ToolCall(
            name="add_content_after_line",
            arguments={
                "file_path": target_file,
                "line_number": 0,  # Will be adjusted based on actual file length
                "content": "\n\ndef validate_input(data: dict) -> bool:\n    \"\"\"Validate input data.\"\"\"\n    if not isinstance(data, dict):\n        return False\n    name = data.get('name')\n    if not isinstance(name, str) or not name.strip():\n        return False\n    value = data.get('value')\n    if not isinstance(value, (int, float)) or value <= 0:\n        return False\n    return True\n",
            },
            think_before=f"The file is in <opened-files>. I'll add the function at the end using add_content_after_line.",
        ),
        ToolCall(
            name="execute_command",
            arguments={"command": f"cd {os.path.dirname(target_file)} && python -c \"from {Path(target_file).stem} import validate_input; print(validate_input({{'name': 'test', 'value': 1}}))\""},
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Added validate_input() to {rel_path}. Validates name (non-empty string) and value (positive number). Import check passes.",
                "status": "done",
                "final_answer": f"Added `validate_input(data: dict) -> bool` to `{rel_path}`. It checks that 'name' is a non-empty string and 'value' is a positive number. Returns True if valid, False otherwise.",
            },
        ),
    ]

    return TrainingExample(
        scenario_type="add_function",
        repo=scenario["repo"],
        prompt=scenario["prompt"],
        steps=[step1, step2],
    )


def gen_modify_method_example(scenario: dict) -> TrainingExample:
    """Generate a 'modify method' example."""
    target_file = scenario["target_files"][0]
    symbol = scenario["target_symbols"][0] if scenario["target_symbols"] else ""
    rel_path = _rel(target_file, scenario["repo"])
    lines = scenario.get("target_lines", {}).get(target_file, [1, 10])

    step1 = [
        ToolCall(
            name="help",
            arguments={"context": "edit"},
            think_before="I need to modify an existing method to add logging. Let me learn the tools.",
        ),
        ToolCall(
            name="read_file",
            arguments={"path": target_file},
            think_before=f"Reading {rel_path} to see {symbol}() and understand its current implementation.",
        ),
        ToolCall(
            name="add_note",
            arguments={"note": f"{symbol}() at {rel_path}:{lines[0]}-{lines[1]}. Will add logger.debug at start."},
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Read {rel_path}. Found {symbol}() at lines {lines[0]}-{lines[1]}. Will add logging.",
                "status": "continue",
                "next_steps": [
                    {"op": "add", "index": 1, "description": f"Add logging to {symbol}()"},
                ],
            },
        ),
    ]

    step2 = [
        ToolCall(
            name="add_content_after_line",
            arguments={
                "file_path": target_file,
                "line_number": lines[0],  # After the def line
                "content": "        logger.debug(f\"{symbol.split('.')[-1]} called with args={{locals()}}\")\n",
            },
            think_before=f"The file is in <opened-files>. I'll insert a logger.debug line right after the def line at line {lines[0]}.",
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Added logger.debug() at the start of {symbol}(). Logs method name and arguments.",
                "status": "done",
                "final_answer": f"Added `logger.debug()` call at the start of `{symbol}()` in `{rel_path}`. It logs the method name and all its arguments.",
            },
        ),
    ]

    return TrainingExample(
        scenario_type="modify_method",
        repo=scenario["repo"],
        prompt=scenario["prompt"],
        steps=[step1, step2],
    )


def gen_refactor_rename_example(scenario: dict) -> TrainingExample:
    """Generate a rename symbol example."""
    target_file = scenario["target_files"][0]
    old_name = scenario.get("extra", {}).get("old_name", "old_func")
    new_name = scenario.get("extra", {}).get("new_name", "new_func")
    rel_path = _rel(target_file, scenario["repo"])

    step1 = [
        ToolCall(
            name="help",
            arguments={"context": "edit"},
            think_before="I need to rename a symbol across the project. Let me check the tools.",
        ),
        ToolCall(
            name="read_file",
            arguments={"path": target_file},
            think_before=f"Reading {rel_path} to see {old_name}() and understand where it's defined.",
        ),
        ToolCall(
            name="find_references",
            arguments={"name": old_name},
            think_before=f"Let me find all references to {old_name} across the project to understand the impact.",
        ),
        ToolCall(
            name="add_note",
            arguments={"note": f"Renaming {old_name} -> {new_name}. Found in {rel_path} with N references."},
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Read {rel_path}. Found {old_name}() definition and all references. Will rename project-wide.",
                "status": "continue",
                "next_steps": [
                    {"op": "add", "index": 1, "description": f"Rename {old_name} to {new_name} and run tests"},
                ],
            },
        ),
    ]

    step2 = [
        ToolCall(
            name="rename_symbol",
            arguments={
                "symbol": old_name,
                "new_name": new_name,
                "file_path": target_file,
            },
            think_before=f"Using rename_symbol to rename {old_name} -> {new_name} across the entire project. This updates the definition, all references, and all imports automatically.",
        ),
        ToolCall(
            name="execute_command",
            arguments={"command": f"cd {os.path.dirname(target_file)} && python -m pytest -x -q 2>&1 | tail -20"},
            think_before="Running tests to verify the rename didn't break anything.",
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Renamed {old_name} -> {new_name} across project. All references and imports updated. Tests pass.",
                "status": "done",
                "final_answer": f"Renamed `{old_name}` to `{new_name}` in `{rel_path}` and updated all references and imports across the project. Tests pass.",
            },
        ),
    ]

    return TrainingExample(
        scenario_type="refactor_rename",
        repo=scenario["repo"],
        prompt=scenario["prompt"],
        steps=[step1, step2],
    )


def gen_refactor_move_example(scenario: dict) -> TrainingExample:
    """Generate a move symbol example."""
    target_file = scenario["target_files"][0]
    func_name = scenario["target_symbols"][0] if scenario["target_symbols"] else ""
    target_dest = scenario.get("extra", {}).get("target_file", "helpers.py")
    rel_path = _rel(target_file, scenario["repo"])

    step1 = [
        ToolCall(
            name="help",
            arguments={"context": "edit"},
            think_before="I need to move a function to another file. Let me check the tools.",
        ),
        ToolCall(
            name="read_file",
            arguments={"path": target_file},
            think_before=f"Reading {rel_path} to see {func_name}() before moving it.",
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Read {rel_path}. Found {func_name}(). Will move to {target_dest}.",
                "status": "continue",
                "next_steps": [
                    {"op": "add", "index": 1, "description": f"Move {func_name} to {target_dest} and run tests"},
                ],
            },
        ),
    ]

    repo_path = str(REPOS_DIR / scenario["repo"])
    abs_target = os.path.join(repo_path, target_dest)

    step2 = [
        ToolCall(
            name="move_symbol",
            arguments={
                "symbol": func_name,
                "target_file": abs_target,
            },
            think_before=f"Using move_symbol to move {func_name} from {rel_path} to {target_dest}. This extracts the code, inserts it in the target, and updates all imports automatically.",
        ),
        ToolCall(
            name="execute_command",
            arguments={"command": f"cd {repo_path} && python -m pytest -x -q 2>&1 | tail -20"},
            think_before="Running tests to verify the move didn't break anything.",
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Moved {func_name}() from {rel_path} to {target_dest}. Imports updated. Tests pass.",
                "status": "done",
                "final_answer": f"Moved `{func_name}()` from `{rel_path}` to `{target_dest}`. All imports across the project have been updated. Tests pass.",
            },
        ),
    ]

    return TrainingExample(
        scenario_type="refactor_move",
        repo=scenario["repo"],
        prompt=scenario["prompt"],
        steps=[step1, step2],
    )


def gen_add_test_example(scenario: dict) -> TrainingExample:
    """Generate a 'write test' example."""
    target_file = scenario["target_files"][0]
    func_name = scenario["target_symbols"][0] if scenario["target_symbols"] else ""
    test_file = scenario.get("extra", {}).get("test_file", "tests/test_new.py")
    rel_path = _rel(target_file, scenario["repo"])
    repo_path = str(REPOS_DIR / scenario["repo"])

    step1 = [
        ToolCall(
            name="help",
            arguments={"context": "edit"},
            think_before="I need to write tests for a function. Let me check the tool workflow first.",
        ),
        ToolCall(
            name="get_symbol_code",
            arguments={"name": func_name, "file_path": target_file},
            think_before=f"Let me get the source code of {func_name}() to understand what to test.",
        ),
        ToolCall(
            name="add_note",
            arguments={"note": f"Testing {func_name}() from {rel_path}. Will create {test_file} with 3 test cases."},
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Read {func_name}() source. Will create {test_file} with tests for normal, edge, and error cases.",
                "status": "continue",
                "next_steps": [
                    {"op": "add", "index": 1, "description": f"Create {test_file} with 3 test cases"},
                ],
            },
        ),
    ]

    step2 = [
        ToolCall(
            name="create_file",
            arguments={
                "path": os.path.join(repo_path, test_file),
                "content": f"\"\"\"Tests for {func_name}.\"\"\"\nimport pytest\nfrom {Path(target_file).stem} import {func_name}\n\n\ndef test_{func_name}_normal():\n    \"\"\"Test with normal input.\"\"\"\n    result = {func_name}()\n    assert result is not None\n\n\ndef test_{func_name}_edge_case():\n    \"\"\"Test edge case.\"\"\"\n    pass  # TODO: implement\n\n\ndef test_{func_name}_error():\n    \"\"\"Test error handling.\"\"\"\n    with pytest.raises(Exception):\n        {func_name}(None)\n",
            },
            think_before=f"I'll create {test_file} with 3 test cases covering normal input, edge cases, and error handling.",
        ),
        ToolCall(
            name="execute_command",
            arguments={"command": f"cd {repo_path} && python -m pytest {test_file} -v 2>&1 | tail -20"},
            think_before="Running the new tests to see which pass and which need adjustment.",
        ),
        ToolCall(
            name="step_complete",
            arguments={
                "summary": f"Created {test_file} with 3 test cases. Ran tests to verify.",
                "status": "done",
                "final_answer": f"Created `{test_file}` with 3 test cases for `{func_name}()`:\n- `test_{func_name}_normal`: Tests normal input\n- `test_{func_name}_edge_case`: Tests edge cases\n- `test_{func_name}_error`: Tests error handling with invalid input",
            },
        ),
    ]

    return TrainingExample(
        scenario_type="add_test",
        repo=scenario["repo"],
        prompt=scenario["prompt"],
        steps=[step1, step2],
    )


# ── Registry ──────────────────────────────────────────────────────────────────

EXAMPLE_GENERATORS = {
    "bug_fix": gen_bug_fix_example,
    "add_method": gen_add_method_example,
    "add_function": gen_add_function_example,
    "modify_method": gen_modify_method_example,
    "refactor_rename": gen_refactor_rename_example,
    "refactor_move": gen_refactor_move_example,
    "add_test": gen_add_test_example,
}


def _rel(path: str, repo_name: str) -> str:
    """Get relative path from repo root."""
    repo_path = str(REPOS_DIR / repo_name)
    try:
        return os.path.relpath(path, repo_path)
    except ValueError:
        return path


def generate_all():
    """Generate training examples from scenarios."""
    scenarios_file = SCENARIOS_DIR / "all_scenarios.json"
    if not scenarios_file.exists():
        print(f"Scenarios not found: {scenarios_file}. Run generate_scenarios.py first.")
        return

    with open(scenarios_file) as f:
        scenarios = json.load(f)

    examples = []
    for scenario in scenarios:
        stype = scenario["scenario_type"]
        gen = EXAMPLE_GENERATORS.get(stype)
        if not gen:
            print(f"  [skip] No generator for {stype}")
            continue

        try:
            example = gen(scenario)
            examples.append(asdict(example))
        except Exception as e:
            print(f"  [error] {stype}/{scenario['repo']}: {e}")

    output_file = OUTPUT_DIR / "training_examples.json"
    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2, default=str)

    print(f"Generated {len(examples)} training examples -> {output_file}")


if __name__ == "__main__":
    generate_all()
