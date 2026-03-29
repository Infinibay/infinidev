"""Generate task scenarios from indexed repos.

Each scenario is a concrete task description with metadata about which files,
symbols, and line numbers are involved. These are used by generate_examples.py
to craft perfect tool call sequences.
"""

import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finetune.config import REPOS, REPOS_DIR, OUTPUT_DIR, SCENARIOS_DIR


@dataclass
class Scenario:
    """A single task scenario with ground truth."""
    repo: str
    project_id: int
    scenario_type: str
    prompt: str                          # The user's task description
    target_files: list[str]              # Files that need to be read/modified
    target_symbols: list[str] = field(default_factory=list)  # Symbols involved
    target_lines: dict = field(default_factory=dict)         # {file: [line_start, line_end]}
    expected_tools: list[str] = field(default_factory=list)  # Tool sequence hints
    extra: dict = field(default_factory=dict)                # Scenario-specific data


def _get_symbols(conn, project_id: int, kind: str = "", limit: int = 50) -> list[dict]:
    """Query symbols from the index."""
    sql = "SELECT name, qualified_name, kind, file_path, line_start, line_end, signature, parent_symbol FROM ci_symbols WHERE project_id = ?"
    params = [project_id]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY RANDOM() LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "name": r[0], "qualified_name": r[1], "kind": r[2],
            "file_path": r[3], "line_start": r[4], "line_end": r[5],
            "signature": r[6], "parent_symbol": r[7],
        }
        for r in rows
    ]


def _get_classes(conn, project_id: int, limit: int = 20) -> list[dict]:
    return _get_symbols(conn, project_id, kind="class", limit=limit)


def _get_methods(conn, project_id: int, limit: int = 30) -> list[dict]:
    return _get_symbols(conn, project_id, kind="method", limit=limit)


def _get_functions(conn, project_id: int, limit: int = 30) -> list[dict]:
    return _get_symbols(conn, project_id, kind="function", limit=limit)


def _get_imports(conn, project_id: int, limit: int = 30) -> list[dict]:
    sql = "SELECT source, name, file_path, line FROM ci_imports WHERE project_id = ? ORDER BY RANDOM() LIMIT ?"
    rows = conn.execute(sql, [project_id, limit]).fetchall()
    return [{"source": r[0], "name": r[1], "file_path": r[2], "line": r[3]} for r in rows]


def _read_file_lines(path: str, start: int, end: int) -> str:
    """Read specific lines from a file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[start - 1:end])
    except Exception:
        return ""


# ── Scenario generators ──────────────────────────────────────────────────────


def gen_bug_fix(conn, project_id: int, repo_name: str, repo_path: str) -> list[Scenario]:
    """Generate bug fix scenarios: introduce a plausible bug, task is to fix it."""
    scenarios = []
    methods = _get_methods(conn, project_id, limit=10)

    for m in methods[:3]:
        if not m["line_start"] or not m["line_end"]:
            continue
        code = _read_file_lines(m["file_path"], m["line_start"], m["line_end"])
        if len(code) < 20 or len(code) > 2000:
            continue

        rel_path = os.path.relpath(m["file_path"], repo_path)
        symbol = m["qualified_name"] or m["name"]

        scenarios.append(Scenario(
            repo=repo_name,
            project_id=project_id,
            scenario_type="bug_fix",
            prompt=f"There's a bug in {symbol}() in {rel_path}. The method doesn't handle None values correctly — it crashes with AttributeError when a None argument is passed. Fix it.",
            target_files=[m["file_path"]],
            target_symbols=[symbol],
            target_lines={m["file_path"]: [m["line_start"], m["line_end"]]},
            expected_tools=["help", "read_file", "think", "edit_symbol", "execute_command", "step_complete"],
            extra={"original_code": code},
        ))

    return scenarios


def gen_add_method(conn, project_id: int, repo_name: str, repo_path: str) -> list[Scenario]:
    """Generate 'add method to class' scenarios."""
    scenarios = []
    classes = _get_classes(conn, project_id, limit=10)

    for cls in classes[:3]:
        if not cls["file_path"]:
            continue

        rel_path = os.path.relpath(cls["file_path"], repo_path)
        class_name = cls["name"]

        scenarios.append(Scenario(
            repo=repo_name,
            project_id=project_id,
            scenario_type="add_method",
            prompt=f"Add a `to_dict()` method to the `{class_name}` class in {rel_path}. It should return a dictionary representation of the instance's public attributes.",
            target_files=[cls["file_path"]],
            target_symbols=[class_name],
            target_lines={cls["file_path"]: [cls["line_start"], cls["line_end"] or cls["line_start"] + 20]},
            expected_tools=["help", "read_file", "think", "add_symbol", "execute_command", "step_complete"],
            extra={"class_name": class_name},
        ))

    return scenarios


def gen_add_function(conn, project_id: int, repo_name: str, repo_path: str) -> list[Scenario]:
    """Generate 'add standalone function' scenarios."""
    scenarios = []
    # Find files with functions (likely utility modules)
    functions = _get_functions(conn, project_id, limit=10)
    seen_files = set()

    for func in functions[:5]:
        if func["file_path"] in seen_files:
            continue
        seen_files.add(func["file_path"])

        rel_path = os.path.relpath(func["file_path"], repo_path)

        scenarios.append(Scenario(
            repo=repo_name,
            project_id=project_id,
            scenario_type="add_function",
            prompt=f"Add a `validate_input(data: dict) -> bool` function to {rel_path}. It should check that 'name' key exists and is a non-empty string, and 'value' key exists and is a positive number. Return True if valid, False otherwise.",
            target_files=[func["file_path"]],
            expected_tools=["help", "read_file", "think", "add_content_after_line", "execute_command", "step_complete"],
        ))

        if len(scenarios) >= 3:
            break

    return scenarios


def gen_modify_method(conn, project_id: int, repo_name: str, repo_path: str) -> list[Scenario]:
    """Generate 'modify existing method' scenarios."""
    scenarios = []
    methods = _get_methods(conn, project_id, limit=10)

    for m in methods[:3]:
        if not m["line_start"] or not m["line_end"]:
            continue
        if (m["line_end"] - m["line_start"]) < 3:
            continue

        rel_path = os.path.relpath(m["file_path"], repo_path)
        symbol = m["qualified_name"] or m["name"]

        scenarios.append(Scenario(
            repo=repo_name,
            project_id=project_id,
            scenario_type="modify_method",
            prompt=f"Modify {symbol}() in {rel_path} to add logging. Add a `logger.debug()` call at the start of the method that logs the method name and its arguments.",
            target_files=[m["file_path"]],
            target_symbols=[symbol],
            target_lines={m["file_path"]: [m["line_start"], m["line_end"]]},
            expected_tools=["help", "read_file", "think", "edit_symbol", "step_complete"],
        ))

    return scenarios


def gen_refactor_rename(conn, project_id: int, repo_name: str, repo_path: str) -> list[Scenario]:
    """Generate rename scenarios."""
    scenarios = []
    functions = _get_functions(conn, project_id, limit=10)

    for func in functions[:3]:
        if func["name"].startswith("_") or len(func["name"]) < 4:
            continue

        rel_path = os.path.relpath(func["file_path"], repo_path)
        old_name = func["name"]
        new_name = old_name + "_v2"  # Simple rename

        scenarios.append(Scenario(
            repo=repo_name,
            project_id=project_id,
            scenario_type="refactor_rename",
            prompt=f"Rename the function `{old_name}` to `{new_name}` in {rel_path} and update all references across the project.",
            target_files=[func["file_path"]],
            target_symbols=[old_name],
            expected_tools=["help", "read_file", "find_references", "think", "rename_symbol", "execute_command", "step_complete"],
            extra={"old_name": old_name, "new_name": new_name},
        ))

    return scenarios


def gen_refactor_move(conn, project_id: int, repo_name: str, repo_path: str) -> list[Scenario]:
    """Generate move symbol scenarios."""
    scenarios = []
    functions = _get_functions(conn, project_id, limit=10)

    for func in functions[:2]:
        rel_path = os.path.relpath(func["file_path"], repo_path)
        target_rel = os.path.dirname(rel_path) + "/helpers.py"

        scenarios.append(Scenario(
            repo=repo_name,
            project_id=project_id,
            scenario_type="refactor_move",
            prompt=f"Move the function `{func['name']}` from {rel_path} to {target_rel}. Update all imports.",
            target_files=[func["file_path"]],
            target_symbols=[func["name"]],
            expected_tools=["help", "read_file", "think", "move_symbol", "execute_command", "step_complete"],
            extra={"target_file": target_rel},
        ))

    return scenarios


def gen_add_test(conn, project_id: int, repo_name: str, repo_path: str) -> list[Scenario]:
    """Generate 'write test' scenarios."""
    scenarios = []
    functions = _get_functions(conn, project_id, limit=10)

    for func in functions[:3]:
        if func["name"].startswith("_"):
            continue

        rel_path = os.path.relpath(func["file_path"], repo_path)
        test_path = "tests/test_" + os.path.basename(rel_path)

        scenarios.append(Scenario(
            repo=repo_name,
            project_id=project_id,
            scenario_type="add_test",
            prompt=f"Write unit tests for the `{func['name']}` function in {rel_path}. Create {test_path} with at least 3 test cases covering normal input, edge cases, and error handling.",
            target_files=[func["file_path"]],
            target_symbols=[func["name"]],
            expected_tools=["help", "read_file", "get_symbol_code", "think", "create_file", "execute_command", "step_complete"],
            extra={"test_file": test_path},
        ))

    return scenarios


# ── Registry ──────────────────────────────────────────────────────────────────

GENERATORS = {
    "bug_fix": gen_bug_fix,
    "add_method": gen_add_method,
    "add_function": gen_add_function,
    "modify_method": gen_modify_method,
    "refactor_rename": gen_refactor_rename,
    "refactor_move": gen_refactor_move,
    "add_test": gen_add_test,
}


def generate_all():
    """Generate scenarios for all repos."""
    import sqlite3
    from infinidev.config.settings import settings

    ft_db = str(OUTPUT_DIR / "finetune.db")
    if not os.path.exists(ft_db):
        print(f"DB not found: {ft_db}. Run index_repos.py first.")
        return

    settings.DB_PATH = ft_db
    conn = sqlite3.connect(ft_db)

    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    all_scenarios = []

    for i, repo in enumerate(REPOS):
        name = repo["name"]
        repo_path = str(REPOS_DIR / name)
        if not os.path.exists(repo_path):
            continue

        project_id = i + 1
        print(f"\n[{name}] Generating scenarios...")

        for scenario_type, generator in GENERATORS.items():
            try:
                scenarios = generator(conn, project_id, name, repo_path)
                all_scenarios.extend(scenarios)
                print(f"  {scenario_type}: {len(scenarios)} scenarios")
            except Exception as e:
                print(f"  {scenario_type}: error — {e}")

    conn.close()

    # Save scenarios
    output_file = SCENARIOS_DIR / "all_scenarios.json"
    with open(output_file, "w") as f:
        json.dump([asdict(s) for s in all_scenarios], f, indent=2, default=str)

    print(f"\nTotal: {len(all_scenarios)} scenarios saved to {output_file}")


if __name__ == "__main__":
    generate_all()
