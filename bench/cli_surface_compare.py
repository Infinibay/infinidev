#!/usr/bin/env python3
"""Compare Codex and Infinidev one-shot CLIs on isolated repository tasks."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bench.agent_task_eval import AgentTask, load_tasks


MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
SURFACES = ("codex", "infinidev")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_IGNORED_PARTS = frozenset(
    {
        ".agents",
        ".codex",
        ".git",
        ".infinidev",
        ".ken",
        ".pytest_cache",
        ".venv",
        "__pycache__",
    }
)
_RUNTIME_DIR_NAMES = frozenset(
    {".infinidev", ".pytest_cache", ".venv", "__pycache__"}
)


@dataclass(frozen=True)
class CliObservation:
    """One CLI execution plus independent repository verification."""

    task_id: str
    model: str
    surface: str
    success: bool
    cli_exit_code: int
    verify_exit_code: int
    latency_seconds: float
    ken_prepared: bool
    ken_prepare_seconds: float
    changed_paths: tuple[str, ...]
    unexpected_changed_paths: tuple[str, ...]
    missing_expected_paths: tuple[str, ...]
    tool_calls: int | None
    failed_tool_calls: int | None
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_output_tokens: int | None
    token_metric: str
    selected_policies: tuple[str, ...]
    artifact_dir: str
    error: str


def _files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _IGNORED_PARTS & set(relative.parts):
            continue
        result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _changed(before: dict[str, str], after: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    )


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _git_init(workspace: Path) -> None:
    commands = (
        ("git", "init", "-q"),
        ("git", "config", "user.name", "Infinidev benchmark"),
        ("git", "config", "user.email", "benchmark@localhost"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "fixture baseline"),
    )
    for command in commands:
        subprocess.run(command, cwd=workspace, check=True, capture_output=True, text=True)


def _remove_runtime_artifacts(workspace: Path) -> None:
    """Remove generated state from a disposable fixture copy before commit."""
    directories = [
        path
        for path in workspace.rglob("*")
        if path.is_dir()
        and (
            path.name in _RUNTIME_DIR_NAMES
            or path.name.startswith("pytest-cache-files-")
        )
    ]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if path.exists():
            shutil.rmtree(path)


def _prepare_ken(workspace: Path, artifact_dir: Path, ken_cli: str) -> float:
    """Build the fixture's semantic index before measuring Infinidev."""
    started = time.perf_counter()
    completed = subprocess.run(
        [ken_cli, "install", ".", "--embed"],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    elapsed = time.perf_counter() - started
    (artifact_dir / "ken-install.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ken install . --embed failed with exit code {completed.returncode}"
        )
    return elapsed


def parse_codex_jsonl(text: str) -> dict[str, Any]:
    """Extract exact usage and completed tool-like items from Codex JSONL."""
    tool_ids: set[str] = set()
    failed_ids: set[str] = set()
    usage: dict[str, int] = {}
    for raw_line in text.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {str(key): int(value or 0) for key, value in event["usage"].items()}
        if event.get("type") != "item.completed" or not isinstance(event.get("item"), dict):
            continue
        item = event["item"]
        if item.get("type") not in {"command_execution", "file_change"}:
            continue
        item_id = str(item.get("id", ""))
        tool_ids.add(item_id)
        if item.get("status") == "failed" or (
            isinstance(item.get("exit_code"), int) and item["exit_code"] != 0
        ):
            failed_ids.add(item_id)
    return {
        "tool_calls": len(tool_ids),
        "failed_tool_calls": len(failed_ids),
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
        "token_metric": "provider-reported-exact" if usage else "unavailable",
        "selected_policies": (),
    }


def parse_infinidev_output(text: str) -> dict[str, Any]:
    """Extract observable CLI counters without presenting them as exact usage."""
    clean = _ANSI.sub("", text)
    tool_lines = [line for line in clean.splitlines() if line.lstrip().startswith("▸ ")]
    token_values: list[int] = []
    for line in tool_lines:
        match = re.search(r"\b(\d+(?:\.\d+)?)ktk\b", line)
        if match:
            token_values.append(round(float(match.group(1)) * 1000))
    selected: tuple[str, ...] = ()
    match = re.search(r"^Task policies:\s*(.+)$", clean, re.MULTILINE)
    if match:
        selected = tuple(item.strip() for item in match.group(1).split(",") if item.strip())
    return {
        "tool_calls": len(tool_lines),
        "failed_tool_calls": sum("✗" in line for line in tool_lines),
        "input_tokens": max(token_values) if token_values else None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_output_tokens": None,
        "token_metric": "observable-developer-loop-lower-bound",
        "selected_policies": selected,
    }


def _command(
    surface: str,
    model: str,
    prompt: str,
    workspace: Path,
    *,
    codex_cli: str,
    infinidev_cli: str,
) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    if surface == "codex":
        return (
            [
                codex_cli,
                "exec",
                "--json",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--approve-for-me",
                "-c",
                'model_reasoning_effort="medium"',
                "-m",
                model,
                "-C",
                str(workspace),
                prompt,
            ],
            environment,
        )

    environment.update(
        {
            "INFINIDEV_ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED": "false",
            "INFINIDEV_CONTEXT_RANK_ENABLED": "true",
            "INFINIDEV_EXECUTE_COMMANDS_PERMISSION": "auto_approve",
            "INFINIDEV_FILE_OPERATIONS_PERMISSION": "auto_approve",
            "INFINIDEV_KEN_SESSION_ENABLED": "true",
            "INFINIDEV_LLM_NUM_RETRIES": "0",
            "INFINIDEV_LOOP_MAX_ITERATIONS": "12",
            "INFINIDEV_LOOP_MAX_TOTAL_TOOL_CALLS": "80",
            "INFINIDEV_TASK_POLICIES_EMBEDDINGS_ENABLED": "true",
            "INFINIDEV_TASK_POLICIES_ENABLED": "true",
            "INFINIDEV_TASK_POLICIES_EVIDENCE_GATED": "false",
            "INFINIDEV_TASK_POLICIES_SHADOW_MODE": "false",
            "INFINIDEV_TASK_POLICIES_SHOW_SELECTION": "true",
            "INFINIDEV_THINKING_BUDGET": "medium",
            "INFINIDEV_TOOL_EFFECTS_PERMISSION": "auto_approve",
        }
    )
    return (
        [
            infinidev_cli,
            "--no-tui",
            "--provider",
            "openai_subscription",
            "--model",
            model,
            "--prompt",
            prompt,
        ],
        environment,
    )


def run_one(
    task: AgentTask,
    model: str,
    surface: str,
    artifact_root: Path,
    fixture_root: Path,
    *,
    codex_cli: str,
    infinidev_cli: str,
    ken_cli: str,
    timeout_seconds: float,
) -> CliObservation:
    """Run one surface in a preserved, baseline-committed workspace."""
    artifact_dir = artifact_root / model / surface
    workspace = artifact_dir / "workspace"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(fixture_root / task.repository_fixture, workspace)
    _remove_runtime_artifacts(workspace)
    _git_init(workspace)
    ken_prepare_seconds = 0.0
    if surface == "infinidev":
        ken_prepare_seconds = _prepare_ken(workspace, artifact_dir, ken_cli)
    before = _files(workspace)
    prompt = f"{task.request}\n\nUse `{sys.executable} -m pytest -q` for verification."
    command, environment = _command(
        surface,
        model,
        prompt,
        workspace,
        codex_cli=codex_cli,
        infinidev_cli=infinidev_cli,
    )
    if surface == "infinidev":
        environment["INFINIDEV_LOG_FILE"] = str(artifact_dir / "infinidev.log")
        environment["INFINIDEV_LOG_LEVEL"] = "DEBUG"
    started = time.perf_counter()
    error = ""
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        cli_exit = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        cli_exit = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        error = f"CLI timeout after {timeout_seconds:.0f}s"
    latency = time.perf_counter() - started
    (artifact_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (artifact_dir / "stderr.log").write_text(stderr, encoding="utf-8")

    verify_env = os.environ.copy()
    verify_env["PYTHONPATH"] = str(workspace)
    verified = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        env=verify_env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    (artifact_dir / "verify.log").write_text(
        verified.stdout + verified.stderr, encoding="utf-8"
    )
    changed = _changed(before, _files(workspace))
    expected = task.expected_changed_paths
    unexpected = tuple(path for path in changed if not _matches(path, expected))
    missing = tuple(
        pattern
        for pattern in expected
        if not any(fnmatch.fnmatch(path, pattern) for path in changed)
    )
    parsed = (
        parse_codex_jsonl(stdout)
        if surface == "codex"
        else parse_infinidev_output(stdout + "\n" + stderr)
    )
    success = cli_exit == 0 and verified.returncode == 0 and not unexpected and not missing
    return CliObservation(
        task_id=task.id,
        model=model,
        surface=surface,
        success=success,
        cli_exit_code=cli_exit,
        verify_exit_code=verified.returncode,
        latency_seconds=latency,
        ken_prepared=surface == "infinidev",
        ken_prepare_seconds=ken_prepare_seconds,
        changed_paths=changed,
        unexpected_changed_paths=unexpected,
        missing_expected_paths=missing,
        artifact_dir=str(artifact_dir),
        error=error,
        **parsed,
    )


def _write_report(path: Path, observations: list[CliObservation]) -> None:
    lines = [
        "# Codex CLI vs Infinidev CLI",
        "",
        "One execution per model and surface. Token fields are not cross-surface equivalent: ",
        "Codex reports provider totals, while Infinidev currently exposes only an observable ",
        "developer-loop lower bound.",
        "",
        "Infinidev workspaces are prepared with `ken install . --embed` before the timed run.",
        "The preparation duration is reported separately.",
        "",
        "| Model | Surface | Pass | Changed | Tools | Failed | Tokens | Seconds | Ken prep | Policies |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in observations:
        token = item.input_tokens if item.input_tokens is not None else "n/a"
        lines.append(
            f"| {item.model} | {item.surface} | {'yes' if item.success else 'no'} | "
            f"{', '.join(item.changed_paths) or 'none'} | {item.tool_calls} | "
            f"{item.failed_tool_calls} | {token} | {item.latency_seconds:.2f} | "
            f"{item.ken_prepare_seconds:.2f} | "
            f"{', '.join(item.selected_policies) or 'none'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--fixture-root", type=Path, default=Path("bench/agent_task_fixtures"))
    parser.add_argument("--codex-cli", default=shutil.which("codex") or "codex")
    parser.add_argument("--infinidev-cli", default=shutil.which("infinidev") or "infinidev")
    parser.add_argument("--ken-cli", default=shutil.which("ken") or "ken")
    parser.add_argument("--model", action="append", choices=MODELS, default=[])
    parser.add_argument("--surface", action="append", choices=SURFACES, default=[])
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    matches = [task for task in load_tasks(args.tasks) if task.id == args.task_id]
    if len(matches) != 1:
        parser.error(f"expected exactly one task with id {args.task_id!r}")
    if args.artifact_root.exists():
        parser.error("artifact root must not exist")
    args.artifact_root.mkdir(parents=True)

    observations: list[CliObservation] = []
    selected_models = tuple(args.model) or MODELS
    selected_surfaces = tuple(args.surface) or SURFACES
    for index, model in enumerate(selected_models):
        normal_order = tuple(item for item in SURFACES if item in selected_surfaces)
        order = normal_order if index % 2 == 0 else tuple(reversed(normal_order))
        for surface in order:
            print(f"running {model}/{surface}", flush=True)
            observation = run_one(
                matches[0],
                model,
                surface,
                args.artifact_root,
                args.fixture_root,
                codex_cli=args.codex_cli,
                infinidev_cli=args.infinidev_cli,
                ken_cli=args.ken_cli,
                timeout_seconds=args.timeout_seconds,
            )
            observations.append(observation)
            with (args.artifact_root / "observations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(observation), sort_keys=True) + "\n")
            print(
                f"finished {model}/{surface}: success={observation.success} "
                f"tools={observation.tool_calls} seconds={observation.latency_seconds:.2f}",
                flush=True,
            )
    _write_report(args.artifact_root / "report.md", observations)


if __name__ == "__main__":
    main()
