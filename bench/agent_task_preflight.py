#!/usr/bin/env python3
"""Prove each agent-task verifier rejects pristine state and accepts a reference solution."""

from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from bench.agent_task_eval import load_tasks
from bench.agent_task_review import fixture_sha256
from bench.agent_task_run import changed_paths


def _verify(command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        shlex.split(command.replace("{python}", sys.executable)),
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def _overlay(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"reference solution is missing: {source}")
    for path in source.rglob("*"):
        if path.is_file():
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def build_preflight(
    tasks_path: Path, fixture_root: Path, reference_root: Path
) -> dict[str, object]:
    """Run both negative and positive verifier controls for every task."""
    records = []
    for task in load_tasks(tasks_path):
        fixture = fixture_root / task.repository_fixture
        with tempfile.TemporaryDirectory(prefix="infinidev-agent-preflight-") as temp:
            workspace = Path(temp) / "repo"
            shutil.copytree(fixture, workspace)
            pristine = _verify(task.verify_command, workspace)
            _overlay(reference_root / task.id, workspace)
            solved = _verify(task.verify_command, workspace)
            changed = changed_paths(fixture, workspace)
        forbidden = [
            path
            for path in changed
            if any(fnmatch.fnmatch(path, pattern) for pattern in task.forbidden_changed_paths)
        ]
        missing = [path for path in task.expected_changed_paths if path not in changed]
        passed = pristine.returncode != 0 and solved.returncode == 0 and not forbidden and not missing
        records.append(
            {
                "task_id": task.id,
                "fixture_sha256": fixture_sha256(fixture),
                "pristine_verify_exit_code": pristine.returncode,
                "reference_verify_exit_code": solved.returncode,
                "reference_changed_paths": changed,
                "forbidden_reference_changes": forbidden,
                "missing_expected_reference_changes": missing,
                "passed": passed,
                "pristine_stdout": pristine.stdout,
                "pristine_stderr": pristine.stderr,
                "reference_stdout": solved.stdout,
                "reference_stderr": solved.stderr,
            }
        )
    return {
        "schema_version": 1,
        "task_count": len(records),
        "all_passed": all(bool(record["passed"]) for record in records),
        "records": records,
        "interpretation_boundary": (
            "Reference solutions prove verifier reachability and reject the pristine fixture. They "
            "are never copied into model workspaces and do not prove the task rubric is semantically fair."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("fixture_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = build_preflight(args.tasks, args.fixture_root, args.reference_root)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
