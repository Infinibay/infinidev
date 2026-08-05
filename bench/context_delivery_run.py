#!/usr/bin/env python3
"""Run paired context-delivery tasks through Infinidev's real developer loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping

try:
    from bench.context_delivery_eval import (
        CONDITIONS,
        ContextObservation,
        ContextTask,
        file_sha256,
        load_condition_manifest,
        load_tasks,
    )
except ModuleNotFoundError:  # Direct ``python bench/context_delivery_run.py`` execution.
    from context_delivery_eval import (  # type: ignore[no-redef]
        CONDITIONS,
        ContextObservation,
        ContextTask,
        file_sha256,
        load_condition_manifest,
        load_tasks,
    )


MIN_TASK_INTERVAL_SECONDS = 2.0
DEFAULT_VERIFY_TIMEOUT_SECONDS = 120.0
_IGNORED_PARTS = frozenset({".git", ".infinidev", "__pycache__", ".pytest_cache"})


@dataclass(frozen=True)
class RunConfig:
    """Immutable provider and loop settings for one paired campaign."""

    provider: str
    model: str
    model_identity: str
    repetitions: int = 1
    min_task_interval_seconds: float = MIN_TASK_INTERVAL_SECONDS
    max_iterations: int = 12
    max_total_tool_calls: int = 80
    max_tool_calls_per_action: int = 20
    verify_timeout_seconds: float = DEFAULT_VERIFY_TIMEOUT_SECONDS

    @classmethod
    def from_path(cls, path: Path) -> RunConfig:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("context delivery run config must be an object")
        config = cls(
            provider=str(value.get("provider", "")).strip(),
            model=str(value.get("model", "")).strip(),
            model_identity=str(value.get("model_identity", "")).strip(),
            repetitions=int(value.get("repetitions", 1)),
            min_task_interval_seconds=float(
                value.get("min_task_interval_seconds", MIN_TASK_INTERVAL_SECONDS)
            ),
            max_iterations=int(value.get("max_iterations", 12)),
            max_total_tool_calls=int(value.get("max_total_tool_calls", 80)),
            max_tool_calls_per_action=int(value.get("max_tool_calls_per_action", 20)),
            verify_timeout_seconds=float(
                value.get("verify_timeout_seconds", DEFAULT_VERIFY_TIMEOUT_SECONDS)
            ),
        )
        if not all((config.provider, config.model, config.model_identity)):
            raise ValueError("run config needs provider, model, and immutable model_identity")
        if config.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if config.min_task_interval_seconds < MIN_TASK_INTERVAL_SECONDS:
            raise ValueError("min_task_interval_seconds must be at least 2.0")
        if min(
            config.max_iterations,
            config.max_total_tool_calls,
            config.max_tool_calls_per_action,
        ) < 1:
            raise ValueError("loop budgets must be positive")
        if config.verify_timeout_seconds <= 0:
            raise ValueError("verify_timeout_seconds must be positive")
        return config


@contextmanager
def single_flight_lock(path: Path | None = None) -> Iterator[None]:
    """Reject a second campaign instead of risking concurrent subscription use."""
    from infinidev.engine.subscription_safety import (
        SUBSCRIPTION_LOCK_PATH,
        subscription_single_flight,
    )

    with subscription_single_flight(path or SUBSCRIPTION_LOCK_PATH):
        yield


def fixture_files(fixture: Path) -> list[Path]:
    """Return the exact stable corpus order, rejecting binary fixture content."""
    if not fixture.is_dir():
        raise ValueError(f"repository fixture does not exist: {fixture}")
    paths = [
        path
        for path in fixture.rglob("*")
        if path.is_file() and not (_IGNORED_PARTS & set(path.relative_to(fixture).parts))
    ]
    return sorted(paths, key=lambda path: path.relative_to(fixture).as_posix())


def build_full_corpus(
    fixture: Path,
    *,
    relevant_paths: tuple[str, ...] = (),
    relevant_position: str = "none",
) -> tuple[str, tuple[str, ...]]:
    """Serialize every declared fixture file with visible path boundaries."""
    files = fixture_files(fixture)
    if relevant_position not in {"none", "front", "middle", "end"}:
        raise ValueError(f"invalid relevant evidence position: {relevant_position}")
    relevant_set = set(relevant_paths)
    relevant = [path for path in files if path.relative_to(fixture).as_posix() in relevant_set]
    distractors = [path for path in files if path not in relevant]
    if relevant_position != "none" and relevant_set and len(relevant) != len(relevant_set):
        missing = sorted(relevant_set - {path.relative_to(fixture).as_posix() for path in relevant})
        raise ValueError(f"required corpus files are absent from fixture: {missing}")
    if relevant_position == "front":
        files = [*relevant, *distractors]
    elif relevant_position == "middle":
        midpoint = len(distractors) // 2
        files = [*distractors[:midpoint], *relevant, *distractors[midpoint:]]
    elif relevant_position == "end":
        files = [*distractors, *relevant]
    blocks: list[str] = []
    paths: list[str] = []
    for path in files:
        relative = path.relative_to(fixture).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"fixture corpus contains non-UTF-8 file: {relative}") from exc
        paths.append(relative)
        blocks.append(f"--- FILE: {relative} ---\n{content.rstrip()}")
    if not blocks:
        raise ValueError(f"repository fixture is empty: {fixture}")
    return "\n\n".join(blocks), tuple(paths)


def _delivered_evidence(
    task: ContextTask, condition: str, engine: object, corpus_paths: tuple[str, ...]
) -> tuple[str, ...]:
    if condition == "baseline":
        return ()
    targets: set[str] = set(corpus_paths if condition == "full" else ())
    if condition == "ranked":
        targets.update(getattr(engine, "_cr_delivered_targets", ()))
        if not targets:
            result = getattr(engine, "_cr_cached_result", None)
            for collection in ("files", "symbols", "findings"):
                for item in getattr(result, collection, ()) if result is not None else ():
                    targets.add(str(getattr(item, "target", "")))
    delivered = []
    for evidence in task.required_evidence:
        evidence_path = evidence.split(":", 1)[0]
        if evidence in targets or any(
            target == evidence_path or target.endswith(f"/{evidence_path}")
            for target in targets
        ):
            delivered.append(evidence)
    return tuple(delivered)


@contextmanager
def configured_runtime(config: RunConfig, condition: str) -> Iterator[None]:
    """Apply and restore only the global settings varied by this experiment."""
    from infinidev.config.model_capabilities import _reset_capabilities
    from infinidev.config.settings import settings

    names = ("LLM_PROVIDER", "LLM_MODEL", "CONTEXT_RANK_ENABLED")
    previous = {name: getattr(settings, name) for name in names}
    settings.LLM_PROVIDER = config.provider
    settings.LLM_MODEL = config.model
    settings.CONTEXT_RANK_ENABLED = condition == "ranked"
    _reset_capabilities()
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)
        _reset_capabilities()


@contextmanager
def workspace(path: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    old_workspace = os.environ.get("INFINIDEV_WORKSPACE")
    os.environ["INFINIDEV_WORKSPACE"] = str(path)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if old_workspace is None:
            os.environ.pop("INFINIDEV_WORKSPACE", None)
        else:
            os.environ["INFINIDEV_WORKSPACE"] = old_workspace


def _verify(command: str, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    command = command.replace("{python}", sys.executable)
    return subprocess.run(
        shlex.split(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_one(
    task: ContextTask,
    condition: str,
    repetition: int,
    *,
    config: RunConfig,
    fixture_root: Path,
    artifact_root: Path,
    dataset_sha256: str,
    manifest_sha256: str,
    condition_sha256: str,
) -> ContextObservation:
    """Run one fresh fixture through LoopEngine, then verify it deterministically."""
    from infinidev.agents.base import InfinidevAgent
    from infinidev.engine.loop.engine import LoopEngine

    resolved_fixture_root = fixture_root.resolve()
    source = (resolved_fixture_root / task.repository_fixture).resolve()
    try:
        source.relative_to(resolved_fixture_root)
    except ValueError as exc:
        raise ValueError(
            f"repository fixture escapes fixture root: {task.repository_fixture}"
        ) from exc
    run_name = f"{task.id}.r{repetition}.{condition}"
    artifact_dir = artifact_root / run_name
    artifact_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    error = ""
    verify_exit: int | None = None
    final_answer = ""
    verify_stdout = ""
    verify_stderr = ""
    relevant_paths = tuple(item.split(":", 1)[0] for item in task.required_evidence)
    corpus, corpus_paths = build_full_corpus(
        source,
        relevant_paths=relevant_paths,
        relevant_position=task.relevant_evidence_position,
    )
    with tempfile.TemporaryDirectory(prefix="infinidev-context-task-") as temp:
        run_workspace = Path(temp) / "repo"
        shutil.copytree(source, run_workspace)
        instance_seed = f"{run_name}:{run_workspace.resolve()}"
        project_id = int.from_bytes(
            hashlib.sha256(instance_seed.encode()).digest()[:4], "big"
        )
        instance_id = f"{run_name}-{project_id:08x}"
        engine = LoopEngine()
        agent = InfinidevAgent(
            agent_id=f"context-eval-{instance_id}", role="developer", project_id=project_id
        )
        try:
            with configured_runtime(config, condition), workspace(run_workspace):
                agent.activate_context(session_id=f"context-eval-{instance_id}")
                if condition == "ranked":
                    from infinidev.code_intel.indexer import index_directory

                    index_directory(
                        project_id, str(run_workspace), wait_for_embeddings=True
                    )
                    engine._cr_hooks.start(instance_id, task.id, task.request)
                final_answer = engine.execute(
                    agent,
                    (task.request, "Implement the request and leave the verifier passing."),
                    verbose=False,
                    max_iterations=config.max_iterations,
                    max_total_tool_calls=config.max_total_tool_calls,
                    max_tool_calls_per_action=config.max_tool_calls_per_action,
                    context_corpus=corpus if condition == "full" else None,
                    allow_llm_retries=False,
                )
                verified = _verify(
                    task.verify_command, run_workspace, config.verify_timeout_seconds
                )
                verify_exit = verified.returncode
                verify_stdout = verified.stdout
                verify_stderr = verified.stderr
                if condition == "ranked":
                    engine._cr_hooks.finish()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            agent.deactivate()
        changed_dir = artifact_dir / "workspace"
        shutil.copytree(run_workspace, changed_dir)

    state = getattr(engine, "_last_state", None)
    context_items = _delivered_evidence(task, condition, engine, corpus_paths)
    artifact = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "condition_manifest_sha256": manifest_sha256,
        "condition_sha256": condition_sha256,
        "task": asdict(task),
        "condition": condition,
        "repetition": repetition,
        "model_identity": config.model_identity,
        "run_config": asdict(config),
        "final_answer": final_answer,
        "engine_status": getattr(engine, "_last_status", ""),
        "plan_steps": engine.get_plan_steps(),
        "action_records": [
            record.model_dump(mode="json")
            for record in getattr(state, "history", ())
        ],
        "prompt_composition_history": list(
            getattr(state, "prompt_composition_history", ())
        ),
        "request_payload_history": list(
            getattr(state, "request_payload_history", ())
        ),
        "changed_files_summary": engine.get_changed_files_summary(),
        "file_change_reasons": engine.get_file_change_reasons(),
        "verify_exit_code": verify_exit,
        "verify_stdout": verify_stdout,
        "verify_stderr": verify_stderr,
        "error": error,
        "context_items": context_items,
        "context_corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "isolated_project_id": project_id,
        "isolated_instance_id": instance_id,
    }
    artifact_path = artifact_dir / "run.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ContextObservation(
        task_id=task.id,
        condition=condition,
        repetition=repetition,
        model_identity=config.model_identity,
        dataset_sha256=dataset_sha256,
        condition_manifest_sha256=manifest_sha256,
        condition_sha256=condition_sha256,
        success=not error and verify_exit == 0,
        verify_exit_code=verify_exit,
        prompt_tokens=int(getattr(state, "total_prompt_tokens", 0)),
        completion_tokens=int(getattr(state, "total_completion_tokens", 0)),
        latency_seconds=time.perf_counter() - started,
        tool_calls=int(getattr(state, "total_tool_calls", 0)),
        context_items=context_items,
        error=error,
        run_artifact=str(artifact_path),
    )


def run_campaign(
    tasks_path: Path,
    conditions_path: Path,
    config_path: Path,
    observations_path: Path,
    artifact_root: Path,
    *,
    fixture_root: Path,
    split: str,
    include_drafts: bool,
) -> None:
    config = RunConfig.from_path(config_path)
    dataset_sha = file_sha256(tasks_path)
    manifest_sha, condition_hashes = load_condition_manifest(
        conditions_path, dataset_sha256=dataset_sha
    )
    tasks = [
        task for task in load_tasks(tasks_path)
        if task.split == split and (include_drafts or task.review_status == "approved")
    ]
    if not tasks:
        raise ValueError(f"no selected context tasks for split: {split}")
    if observations_path.exists() and observations_path.stat().st_size:
        raise ValueError("observations output must be new and empty; campaigns never resume implicitly")
    last_started: float | None = None
    with single_flight_lock():
        for repetition in range(config.repetitions):
            for task in tasks:
                for condition in CONDITIONS:
                    now = time.monotonic()
                    if last_started is not None:
                        remaining = config.min_task_interval_seconds - (now - last_started)
                        if remaining > 0:
                            time.sleep(remaining)
                    last_started = time.monotonic()
                    row = run_one(
                        task, condition, repetition,
                        config=config, fixture_root=fixture_root,
                        artifact_root=artifact_root,
                        dataset_sha256=dataset_sha,
                        manifest_sha256=manifest_sha,
                        condition_sha256=condition_hashes[condition],
                    )
                    _append_jsonl(observations_path, asdict(row))
                    if row.error:
                        raise RuntimeError(
                            f"stopped after provider/runtime error in {task.id}/{condition}: "
                            f"{row.error}"
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("conditions", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--fixture-root", type=Path, default=Path("bench"))
    parser.add_argument("--split", choices=("calibration", "validation"), default="validation")
    parser.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args()
    run_campaign(
        args.tasks, args.conditions, args.config, args.observations,
        args.artifact_root, fixture_root=args.fixture_root,
        split=args.split, include_drafts=args.include_drafts,
    )


if __name__ == "__main__":
    main()
