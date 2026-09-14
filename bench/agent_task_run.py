#!/usr/bin/env python3
"""Run provider-neutral baseline/candidate agent tasks in isolated repositories."""

from __future__ import annotations

import argparse
import fnmatch
from functools import lru_cache
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from bench.agent_task_eval import (
    AgentTask,
    AgentTaskObservation,
    file_sha256,
    load_condition_manifest,
    load_tasks,
)


MIN_REQUEST_INTERVAL_SECONDS = 2.0
#: Compiled-cache patterns kept out of the baseline commit and out of any
#: fixture copy. They are not source, and a run that recompiles them reports
#: them as changes.
_CACHE_PATTERNS = (
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
)


_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".infinidev",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "build",
        "node_modules",
        "target",
        # A workspace-local interpreter the agent creates for itself. The
        # fixtures ask for pytest, ``python`` on PATH is the system one, and a
        # PEP 668 refusal leads models to `python -m venv .venv` inside the
        # task repository. Left in, its thousands of library files become
        # "changed paths" and drown the artifact copy, hiding the change the
        # task was about. Observed on MiniMax-M3 with pricing-rounding.
        ".venv",
        "venv",
        # Ken's index and daemon state. It writes them next to the workspace it
        # indexes, so they land in every changed-path diff: the recorded
        # orchestrator runs show .ken/ken.db, .ken/vectors/* and the daemon
        # files counted as the agent's own changes, inflating changed_lines and
        # the artifact copy.
        ".ken",
    }
)
_IGNORED_PART_SUFFIXES = (".egg-info",)
_IGNORED_GENERATED_PATHS = frozenset(
    {
        "test/test_default",
        "test/test_links",
        "test/test_strict",
        "test/test_strict_links",
    }
)
_CONDITIONS = ("baseline", "candidate")
_TREATMENTS = ("adaptive_behavior", "prompt_calibration", "task_policy")
TASK_POLICY_TREATMENT_MARKER = "runtime:conditional-task-policies-v1"
ADAPTIVE_BEHAVIOR_TREATMENT_MARKER = "runtime:adaptive-observable-behavior-v1"


@dataclass(frozen=True)
class AgentTaskRunConfig:
    """One provider route and bounded execution policy shared by paired tasks."""

    provider: str
    model: str
    model_identity: str
    base_url: str = ""
    api_key_env: str = ""
    repetitions: int = 1
    min_request_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS
    max_iterations: int = 12
    max_total_tool_calls: int = 80
    max_tool_calls_per_action: int = 20
    verify_timeout_seconds: float = 120.0
    treatment: str = "prompt_calibration"
    task_policy_prediction_report: str = ""
    task_policy_prediction_report_sha256: str = ""
    # Prompt-style variant under test. Empty keeps the product default. The
    # setting is applied in-process because ``./.infinidev/settings.json``
    # outranks environment variables in pydantic-settings, so an env override
    # would silently measure the wrong style.
    prompt_style: str = ""
    #: Run the whole orchestration pipeline instead of the loop directly. This
    #: is what a user gets: chat agent, optional elaboration, engine selection,
    #: the developer loop and the review phase. Needed to measure the engine
    #: mode the product actually defaults to, which the direct loop cannot.
    pipeline_mode: bool = False
    # Arbitrary ``Settings`` fields applied in-process for this run and
    # restored afterwards. The evaluation manifests predate most of the knobs
    # under test, and one entry per setting would mean editing the runner every
    # time a candidate adds one.
    settings_overrides: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> AgentTaskRunConfig:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("agent task run config must be an object")
        config = cls(
            provider=str(value.get("provider", "")).strip(),
            model=str(value.get("model", "")).strip(),
            model_identity=str(value.get("model_identity", "")).strip(),
            base_url=str(value.get("base_url", "")).strip(),
            api_key_env=str(value.get("api_key_env", "")).strip(),
            repetitions=int(value.get("repetitions", 1)),
            min_request_interval_seconds=float(
                value.get("min_request_interval_seconds", MIN_REQUEST_INTERVAL_SECONDS)
            ),
            max_iterations=int(value.get("max_iterations", 12)),
            max_total_tool_calls=int(value.get("max_total_tool_calls", 80)),
            max_tool_calls_per_action=int(value.get("max_tool_calls_per_action", 20)),
            verify_timeout_seconds=float(value.get("verify_timeout_seconds", 120.0)),
            treatment=str(value.get("treatment", "prompt_calibration")).strip(),
            task_policy_prediction_report=str(
                value.get("task_policy_prediction_report", "")
            ).strip(),
            task_policy_prediction_report_sha256=str(
                value.get("task_policy_prediction_report_sha256", "")
            ).strip(),
            prompt_style=str(value.get("prompt_style", "")).strip(),
            pipeline_mode=bool(value.get("pipeline_mode", False)),
            settings_overrides=dict(value.get("settings_overrides", {}) or {}),
        )
        if not all((config.provider, config.model, config.model_identity)):
            raise ValueError("agent task config needs provider, model, and immutable identity")
        from infinidev.config.providers import PROVIDERS

        if config.provider not in PROVIDERS:
            raise ValueError(f"agent task provider is not registered: {config.provider}")
        if config.repetitions < 1:
            raise ValueError("agent task repetitions must be positive")
        if config.min_request_interval_seconds < MIN_REQUEST_INTERVAL_SECONDS:
            raise ValueError("agent task minimum request interval must be at least 2 seconds")
        if min(
            config.max_iterations,
            config.max_total_tool_calls,
            config.max_tool_calls_per_action,
        ) < 1:
            raise ValueError("agent task loop budgets must be positive")
        if config.verify_timeout_seconds <= 0:
            raise ValueError("agent task verifier timeout must be positive")
        if config.treatment not in _TREATMENTS:
            raise ValueError(f"unsupported agent task treatment: {config.treatment}")
        has_prediction_report = bool(config.task_policy_prediction_report)
        has_prediction_sha = bool(config.task_policy_prediction_report_sha256)
        if has_prediction_report != has_prediction_sha:
            raise ValueError(
                "task-policy prediction report path and SHA-256 must be provided together"
            )
        if has_prediction_report and config.treatment != "task_policy":
            raise ValueError("a task-policy prediction report requires task_policy treatment")
        if config.settings_overrides:
            from infinidev.config.settings import Settings

            unknown = sorted(set(config.settings_overrides) - set(Settings.model_fields))
            if unknown:
                raise ValueError(
                    "agent task settings overrides are not settings: "
                    + ", ".join(unknown)
                )
        return config


@contextmanager
def _workspace(path: Path) -> Iterator[None]:
    previous_cwd = Path.cwd()
    previous_workspace = os.environ.get("INFINIDEV_WORKSPACE")
    os.environ["INFINIDEV_WORKSPACE"] = str(path)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if previous_workspace is None:
            os.environ.pop("INFINIDEV_WORKSPACE", None)
        else:
            os.environ["INFINIDEV_WORKSPACE"] = previous_workspace


def copy_workspace(source: Path, destination: Path) -> None:
    """Copy an evaluation checkout without expanding repository symlinks."""
    shutil.copytree(source, destination, symlinks=True)


class _AuxUsage:
    """Model calls the LoopEngine's counters never see.

    Two independent sources, because neither alone is complete:

    * ``by_lane`` comes from ``hooks.notify_token_usage``, which the chat
      agent, the planner and the stage planner call voluntarily. It names the
      lane but is opt-in — the council, the spec elaborator, the review engine,
      the adversarial verifier and the task-policy classifier all call
      ``litellm.completion`` and never report.
    * ``provider`` counts every successful call in the process at the provider
      boundary, through a litellm callback. It has no lane but has no gaps.

    An observation row that carried only the loop's own ``prompt_tokens``
    under-reported exactly the phases a pipeline mode adds, which biased an
    engine comparison toward whichever arm ran fewer of them.
    """

    def __init__(self) -> None:
        self.by_lane: dict[str, int] = {}
        self.provider_prompt = 0
        self.provider_completion = 0
        self.provider_calls = 0

    def reset(self) -> None:
        self.by_lane = {}
        self.provider_prompt = 0
        self.provider_completion = 0
        self.provider_calls = 0

    def add(self, tokens: int, lane: str) -> None:
        if tokens > 0:
            self.by_lane[lane] = self.by_lane.get(lane, 0) + int(tokens)

    def add_provider_call(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        try:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        self.provider_calls += 1
        self.provider_prompt += prompt
        self.provider_completion += completion

    @property
    def total(self) -> int:
        return sum(self.by_lane.values())


#: One accumulator per process. The runner is single-threaded and runs one
#: unit at a time, so a module-level instance is enough and keeps the hooks
#: object a plain subclass with no state to thread through ``run_task``.
_AUX_USAGE = _AuxUsage()
_CALLBACK_REGISTERED = False


def _provider_usage_callback() -> Any:
    """Register a litellm callback that counts every successful call once."""

    global _CALLBACK_REGISTERED
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class _ProviderUsage(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            _AUX_USAGE.add_provider_call(response_obj)

    watcher = _ProviderUsage()
    if not _CALLBACK_REGISTERED:
        litellm.callbacks.append(watcher)
        _CALLBACK_REGISTERED = True
    return watcher


def _metered_hooks() -> Any:
    from infinidev.engine.orchestration import NonInteractiveHooks

    class _MeteredHooks(NonInteractiveHooks):
        def notify_token_usage(self, prompt_tokens: int, lane: str = "chat") -> None:
            _AUX_USAGE.add(prompt_tokens, lane)

    return _MeteredHooks()


def _run_through_pipeline(
    *,
    agent: Any,
    engine: Any,
    request: str,
    instruction: str,
    session_id: str,
    max_iterations: int,
) -> str:
    """Run one user turn the way the product does, and return its reply.

    The direct-loop path bypasses the chat agent, engine selection and the
    review phase, so it can only measure the loop. This runs ``run_task``, the
    same entry point the TUI, the classic CLI and the web server use, with the
    non-interactive hooks the one-shot CLI already relies on.
    """
    from infinidev.engine.analysis.review_engine import ReviewEngine
    from infinidev.engine.orchestration import run_task
    from infinidev.tools.permission import (
        make_noninteractive_permission_handler,
        set_permission_handler,
    )

    from infinidev.config.settings import settings

    hooks = _metered_hooks()
    _provider_usage_callback()
    set_permission_handler(make_noninteractive_permission_handler(request))
    previous_iterations = settings.TASK_MAX_ITERATIONS
    if max_iterations > 0:
        settings.TASK_MAX_ITERATIONS = max_iterations
    try:
        return run_task(
            agent=agent,
            user_input=f"{request}\n\n{instruction}",
            session_id=session_id,
            engine=engine,
            reviewer=ReviewEngine(),
            hooks=hooks,
        ) or ""
    finally:
        settings.TASK_MAX_ITERATIONS = previous_iterations
        set_permission_handler(None)


#: A provider failure that reached the reply instead of raising. The pipeline
#: catches engine exceptions and returns them as the turn's text, so a quota or
#: network failure produced a row with no error, no status, zero tokens and a
#: failed verification — indistinguishable from a task the model could not do.
_PROVIDER_FAILURE = re.compile(
    r"engine failed|APIConnectionError|RateLimitError|AuthenticationError"
    r"|litellm\.|usage limit reached",
    re.IGNORECASE,
)


def provider_failure_in_reply(final_answer: str, prompt_tokens: int) -> str:
    """The provider failure a run's reply is reporting, if that is what it is.

    Keyed on zero prompt tokens as well as the wording: a run that reached the
    model cannot have spent nothing, and a real reply can mention a timeout
    without being one. Six runs against an exhausted plan were recorded as
    ordinary task failures before this existed.
    """
    if prompt_tokens > 0:
        return ""
    text = (final_answer or "").strip()
    if not text or not _PROVIDER_FAILURE.search(text):
        return ""
    return text[:300]


def ensure_project(project_id: int, name: str) -> None:
    """Register the project row the task's tools write under.

    ``findings``, ``artifacts``, ``tasks`` and the knowledge tables all carry
    ``project_id REFERENCES projects(id)``. The database seeds one "Default
    Project", and the evaluation invents its own id per run, so nothing created
    the row: every write under it failed the foreign key. The taxonomy recorded
    "Failed to record finding: FOREIGN KEY constraint failed" as a recurring
    tool failure, and the whole DB-backed surface was untested behind it.
    """
    from infinidev.db.service import execute_with_retry

    def _insert(conn) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO projects (id, name, description) "
            "VALUES (?, ?, ?)",
            (project_id, name, "agent-task evaluation workspace"),
        )
        conn.commit()

    execute_with_retry(_insert)


def _exclude_generated_caches(workspace: Path) -> None:
    """Keep compiled caches out of the workspace baseline.

    Written to ``.git/info/exclude`` rather than a ``.gitignore`` in the tree:
    the fixture must not gain a file the task did not have, and ``info/exclude``
    is per-checkout and never committed.
    """

    exclude = workspace / ".git" / "info" / "exclude"
    try:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        missing = [
            pattern for pattern in _CACHE_PATTERNS
            if pattern not in existing.splitlines()
        ]
        if missing:
            with exclude.open("a", encoding="utf-8") as handle:
                if existing and not existing.endswith("\n"):
                    handle.write("\n")
                handle.write("\n".join(missing) + "\n")
    except OSError as exc:  # pragma: no cover - a read-only workspace is not fatal
        logger.debug("could not write .git/info/exclude: %s", exc)


def init_git_workspace(workspace: Path) -> bool:
    """Make the task workspace a repository with the fixture committed.

    The agent's own instructions send it to ``git_diff`` and ``git_status`` to
    review what it changed. The fixtures are bare directories, so every one of
    those calls failed against a workspace that is not a repository: 62 of the
    189 failed tool calls across the recorded campaigns, 33% of all failures,
    spent on an environment the request never mentioned. A real checkout is a
    repository, so the harness supplies one.

    Best effort: a host without ``git`` runs the task as before rather than
    failing the campaign.

    Generated caches are excluded from the baseline commit. The fixtures carry
    no ``.gitignore``, so a stray ``__pycache__`` left by someone running pytest
    inside the fixture directory became a *tracked* file, entered the workspace
    baseline, and every later run reported the recompiled ``.pyc`` as a change:
    in one ``wide-sum`` run the reviewer's diff was 7 980 characters of which
    7 370 — 92 % — were compiled-cache noise.
    """
    if not shutil.which("git"):
        return False
    _exclude_generated_caches(workspace)
    commands = (
        ["git", "init", "--quiet"],
        ["git", "add", "-A"],
        [
            "git", "-c", "user.email=evaluation@localhost",
            "-c", "user.name=Evaluation", "commit", "--quiet",
            "-m", "fixture baseline",
        ],
    )
    for command in commands:
        completed = subprocess.run(
            command, cwd=workspace, capture_output=True, text=True, timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            return False
    return True


def _withhold_paths(
    workspace: Path, paths: tuple[str, ...]
) -> dict[str, bytes]:
    """Remove verifier-only files from the agent's workspace.

    Returns the pristine bytes so the verifier can be run against the same
    files the fixture shipped, not the ones the agent found.
    """
    saved: dict[str, bytes] = {}
    for relative in paths:
        target = (workspace / relative).resolve()
        try:
            target.relative_to(workspace.resolve())
        except ValueError as exc:
            raise ValueError(
                f"withheld path escapes the workspace: {relative}"
            ) from exc
        if not target.is_file():
            raise ValueError(f"withheld path does not exist: {relative}")
        saved[relative] = target.read_bytes()
        target.unlink()
    return saved


def _restore_withheld(
    workspace: Path, saved: Mapping[str, bytes]
) -> tuple[str, ...]:
    """Put the verifier-only files back and report any the agent wrote.

    A path that exists again before the restore is a file the agent created
    where it had none, which is the one way a run can reach the withheld
    contract without earning it.
    """
    tampered: list[str] = []
    for relative, content in saved.items():
        target = workspace / relative
        if target.exists():
            tampered.append(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return tuple(tampered)


def _structured_evaluation_task(task: AgentTask, task_profile: object | None) -> object:
    """Build the runtime Task with the benchmark's declared verifier visible."""
    from infinidev.engine.orchestration.task_schema import task_from_free_text

    return task_from_free_text(
        task.request,
        kind=task.category,
        derived_verification_criteria=[
            f"Run `{task.verify_command}` and require exit code 0 before completion."
        ],
        task_profile=task_profile,
    )


@lru_cache(maxsize=4)
def _frozen_task_policy_predictions(
    report_path: str,
    expected_sha256: str,
) -> tuple[dict[str, tuple[tuple[str, float], ...]], str, str]:
    """Load thresholded multi-label predictions from one immutable training report."""
    from bench.task_policy_multilabel_head import METHOD_LABELS

    path = Path(report_path)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "task-policy prediction report SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 1 or not isinstance(folds[0], dict):
        raise ValueError("task-policy prediction report must contain exactly one fold")
    prediction_only = folds[0].get("prediction_only")
    if not isinstance(prediction_only, dict):
        raise ValueError("task-policy prediction report has no frozen predictions")
    ids = prediction_only.get("ids")
    predictions = prediction_only.get("predictions")
    method_scores = prediction_only.get("method_scores")
    if not all(isinstance(value, list) for value in (ids, predictions, method_scores)):
        raise ValueError("task-policy frozen prediction arrays are invalid")
    if not (len(ids) == len(predictions) == len(method_scores)):
        raise ValueError("task-policy frozen prediction arrays have different lengths")
    result: dict[str, tuple[tuple[str, float], ...]] = {}
    positions = {label: index for index, label in enumerate(METHOD_LABELS)}
    for candidate_id, labels, scores in zip(ids, predictions, method_scores, strict=True):
        identifier = str(candidate_id)
        if identifier in result or not isinstance(labels, list) or not isinstance(scores, list):
            raise ValueError("task-policy frozen prediction row is invalid")
        if len(scores) != len(METHOD_LABELS):
            raise ValueError("task-policy frozen score width changed")
        try:
            result[identifier] = tuple(
                (str(label), float(scores[positions[str(label)]])) for label in labels
            )
        except KeyError as exc:
            raise ValueError(f"task-policy frozen prediction has unknown label: {exc.args[0]}") from exc
    model = str(payload.get("model", "")).strip()
    version = str(payload.get("version", "")).strip()
    if not model or not version:
        raise ValueError("task-policy prediction report lacks model identity")
    return result, model, version


def _frozen_task_policy_profile(
    text: str,
    task_id: str,
    *,
    report_path: str,
    expected_sha256: str,
) -> object:
    """Apply frozen mini-model methods while retaining literal authority boundaries."""
    from infinidev.engine.task_policies import resolve_task_profile
    from infinidev.engine.task_policies.models import PolicySelection
    from infinidev.engine.task_policies.registry import POLICIES

    base = resolve_task_profile(text, enable_embeddings=False, enable_llm_fallback=False)
    predictions, model, version = _frozen_task_policy_predictions(
        report_path,
        expected_sha256,
    )
    if task_id not in predictions:
        raise ValueError(f"task-policy prediction report lacks task id: {task_id}")
    policy_by_id = {policy.id: policy for policy in POLICIES}
    selected = []
    rejected = []
    operations = {
        operation
        for operation in base.operations
        if operation not in {
            "bugfix", "feature", "refactor", "research", "review", "performance",
        }
    }
    for policy_id, score in predictions[task_id]:
        policy = policy_by_id.get(policy_id)
        if policy is None:
            raise ValueError(f"task-policy prediction references unknown policy: {policy_id}")
        if policy.requires_modify and "modify" not in base.authority:
            from infinidev.engine.task_policies.models import RejectedPolicyCandidate

            rejected.append(RejectedPolicyCandidate(
                id=policy.id,
                reason="frozen mini-model method lacks literal modify authority",
                score=score,
            ))
            continue
        operations.update(policy.operations)
        selected.append(PolicySelection(
            id=policy.id,
            version=policy.version,
            source="embedding",
            evidence=("nemotron-frozen-prediction",),
            score=score,
            policy_hash=policy.content_hash,
        ))
    sequence = set()
    if operations & {"research"}:
        sequence.add("investigate")
    if "modify" in base.authority and operations & {
        "bugfix", "feature", "refactor", "performance", "docs", "migration",
    }:
        sequence.update(("implement", "verify"))
    if "review" in operations:
        sequence.add("review")
    result = set()
    if "modify" in base.authority and "implement" in sequence:
        result.add("code")
    elif operations & {"research", "review", "performance"}:
        result.add("report")
    ordered_operations = tuple(
        operation
        for operation in (
            "bugfix", "feature", "refactor", "research", "review", "performance",
            "docs", "migration", "security",
        )
        if operation in operations
    )
    ordered_sequence = tuple(
        step
        for step in ("investigate", "implement", "verify", "review", "commit", "publish")
        if step in sequence
    )
    return base.model_copy(update={
        "operations": ordered_operations,
        "result": tuple(value for value in ("code", "report") if value in result),
        "sequence": ordered_sequence,
        "selected_policies": tuple(selected),
        "rejected_candidates": tuple(rejected),
        "semantic_space_id": f"{model}:{expected_sha256[:16]}",
        "semantic_classifier_version": version,
        "semantic_abstained": not bool(selected),
        "semantic_abstention_reason": "" if selected else "frozen prediction abstained",
    })


def _canonical_profile_sha(profile: Mapping[str, object]) -> str:
    weights = {
        str(key): float(value)
        for key, value in dict(profile.get("weights", {})).items()
        if float(value)
    }
    payload = json.dumps(
        {
            "name": str(profile.get("name", "")).strip(),
            "weights": weights,
            "description": str(profile.get("description", "")).strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@contextmanager
def interpreter_on_path() -> Iterator[None]:
    """Make the runner's own interpreter reachable as ``python``.

    Tasks instruct the agent to run the project's tests, and the verifier runs
    them with ``sys.executable``. Without this, an agent that types the obvious
    ``python -m pytest`` gets "command not found" or a PEP 668 refusal from the
    system interpreter, and spends the run building a virtualenv inside the
    repository instead of doing the task. Observed on MiniMax-M3: seven
    ``execute_command`` calls and a workspace-local ``.venv`` for a one-line
    numeric fix that the hidden contract then graded as correct.

    The point is not to help the agent. It is that a harness must provide the
    environment its own tasks describe, or it measures the environment.
    """
    interpreter_bin = str(Path(sys.executable).parent)
    previous = os.environ.get("PATH", "")
    if interpreter_bin in previous.split(os.pathsep):
        yield
        return
    os.environ["PATH"] = (
        interpreter_bin + os.pathsep + previous if previous else interpreter_bin
    )
    try:
        yield
    finally:
        if previous:
            os.environ["PATH"] = previous
        else:
            os.environ.pop("PATH", None)


@contextmanager
def configured_evaluation_runtime(
    config: AgentTaskRunConfig,
    condition: str,
    manifest: Mapping[str, object],
    temp_root: Path,
) -> Iterator[None]:
    """Activate an explicit user profile and optional evaluation-only guidance."""
    from infinidev.config.model_capabilities import _reset_capabilities
    from infinidev.config.providers import get_provider
    from infinidev.config.settings import settings
    from infinidev.engine.subscription_safety import paced_llm_requests

    profile = manifest.get("utility_profile")
    conditions = manifest.get("conditions")
    if not isinstance(profile, dict) or not isinstance(conditions, dict):
        raise ValueError("agent task condition manifest is incomplete")
    profile_sha = _canonical_profile_sha(profile)
    if manifest.get("utility_profile_sha256") != profile_sha:
        raise ValueError("agent task utility profile hash mismatch")
    profile_path = temp_root / "user-profile.json"
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    setting_names = (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_NUM_RETRIES",
        "EXECUTE_COMMANDS_PERMISSION",
        "FILE_OPERATIONS_PERMISSION",
        "TOOL_EFFECTS_PERMISSION",
        "CONTEXT_RANK_ENABLED",
        "PROMPT_STYLE",
        "USER_PREFERENCE_PROFILE",
        "USER_PREFERENCE_PROFILE_SHA256",
        "PROMPT_CALIBRATION_PROFILE",
        "PROMPT_CALIBRATION_MODEL_IDENTITY",
        "PROMPT_CALIBRATION_UTILITY_PROFILE",
        "PROMPT_CALIBRATION_UTILITY_PROFILE_SHA256",
        "TASK_POLICIES_ENABLED",
        "TASK_POLICIES_SHADOW_MODE",
        "TASK_POLICIES_EMBEDDINGS_ENABLED",
        "TASK_POLICIES_LLM_FALLBACK_ENABLED",
        "TASK_POLICIES_EVIDENCE_GATED",
        "ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED",
        "ADAPTIVE_RUNTIME_BEHAVIOR_SHADOW_MODE",
        "ADAPTIVE_RUNTIME_SEMANTIC_SHADOW_ENABLED",
        "ADAPTIVE_RUNTIME_REASONING_ENABLED",
        "ADAPTIVE_RUNTIME_REASONING_SHADOW_MODE",
    )
    previous = {name: getattr(settings, name) for name in setting_names}
    previous.update(
        {name: getattr(settings, name) for name in config.settings_overrides}
    )
    provider = get_provider(config.provider)
    model = config.model
    if config.provider != "openai_subscription" and not model.startswith(provider.prefix):
        model = f"{provider.prefix}{model}"
    api_key = ""
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"agent task API key environment variable is empty: {config.api_key_env}")
    try:
        settings.LLM_PROVIDER = config.provider
        settings.LLM_MODEL = model
        settings.LLM_BASE_URL = config.base_url or provider.default_base_url
        if api_key:
            settings.LLM_API_KEY = api_key
        settings.LLM_NUM_RETRIES = 0
        # The runner owns a fresh disposable workspace and has no approval UI.
        # Leaving interactive permission modes active turns valid verification
        # commands into artificial tool failures and measures workarounds rather
        # than agent behavior.
        settings.EXECUTE_COMMANDS_PERMISSION = "auto_approve"
        settings.FILE_OPERATIONS_PERMISSION = "auto_approve"
        settings.TOOL_EFFECTS_PERMISSION = "auto_approve"
        settings.CONTEXT_RANK_ENABLED = False
        if config.prompt_style:
            settings.PROMPT_STYLE = config.prompt_style
        for override_name, override_value in config.settings_overrides.items():
            setattr(settings, override_name, override_value)
        settings.USER_PREFERENCE_PROFILE = str(profile_path)
        settings.USER_PREFERENCE_PROFILE_SHA256 = profile_sha
        settings.PROMPT_CALIBRATION_PROFILE = ""
        settings.PROMPT_CALIBRATION_MODEL_IDENTITY = config.model_identity
        settings.PROMPT_CALIBRATION_UTILITY_PROFILE = profile["name"]
        settings.PROMPT_CALIBRATION_UTILITY_PROFILE_SHA256 = profile_sha
        if config.treatment == "task_policy":
            candidate = conditions.get("candidate")
            marker = (
                str(candidate.get("system_prompt", ""))
                if isinstance(candidate, dict)
                else ""
            )
            if marker != TASK_POLICY_TREATMENT_MARKER:
                raise ValueError(
                    "task-policy evaluation manifest has the wrong treatment marker"
                )
            settings.TASK_POLICIES_ENABLED = condition == "candidate"
            settings.TASK_POLICIES_SHADOW_MODE = False
            # Candidate exercises the deployed local mini-head. Authority,
            # permissions, and external effects still come only from literals.
            # Evidence gating belongs to production rollout; an evaluation
            # candidate must be able to measure an unapproved fragment without
            # silently turning into the baseline.
            settings.TASK_POLICIES_EMBEDDINGS_ENABLED = condition == "candidate"
            settings.TASK_POLICIES_LLM_FALLBACK_ENABLED = False
            settings.TASK_POLICIES_EVIDENCE_GATED = False
            settings.ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED = False
        elif config.treatment == "adaptive_behavior":
            candidate = conditions.get("candidate")
            marker = (
                str(candidate.get("system_prompt", ""))
                if isinstance(candidate, dict)
                else ""
            )
            if marker != ADAPTIVE_BEHAVIOR_TREATMENT_MARKER:
                raise ValueError(
                    "adaptive-behavior evaluation manifest has the wrong treatment marker"
                )
            settings.TASK_POLICIES_ENABLED = False
            settings.ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED = True
            settings.ADAPTIVE_RUNTIME_BEHAVIOR_SHADOW_MODE = condition == "baseline"
            settings.ADAPTIVE_RUNTIME_SEMANTIC_SHADOW_ENABLED = True
            settings.ADAPTIVE_RUNTIME_REASONING_ENABLED = True
            settings.ADAPTIVE_RUNTIME_REASONING_SHADOW_MODE = condition == "baseline"
        elif condition == "candidate":
            raw_candidate = conditions[condition]
            if not isinstance(raw_candidate, dict):
                raise ValueError("candidate evaluation condition is invalid")
            guidance = str(raw_candidate["system_prompt"])
            guidance_bytes = len(guidance.encode("utf-8"))
            evaluation_profile = {
                "schema_version": 1,
                "deployment_approved": True,
                "evaluation_only": True,
                "provider": config.provider,
                "model": config.model,
                "model_identity": config.model_identity,
                "selected_condition": "candidate",
                "roles": {
                    "developer": {
                        "guidance": guidance,
                        "sha256": hashlib.sha256(guidance.encode()).hexdigest(),
                        "utf8_bytes": guidance_bytes,
                    }
                },
                "validation": {
                    "selection_report_sha256": "evaluation-only",
                    "dataset_sha256": manifest.get("dataset_sha256", ""),
                    "observations_sha256": "",
                    "paired_comparison": {},
                    "objective": "utility",
                    "calibration_role": "developer",
                    "utility_profile": {"name": profile["name"], "sha256": profile_sha},
                },
            }
            evaluation_profile_path = temp_root / "candidate-profile.json"
            evaluation_profile_path.write_text(
                json.dumps(evaluation_profile, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            settings.PROMPT_CALIBRATION_PROFILE = str(evaluation_profile_path)
        _reset_capabilities()
        with paced_llm_requests(config.min_request_interval_seconds):
            yield
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)
        _reset_capabilities()


def _files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() in _IGNORED_GENERATED_PATHS:
            continue
        if _IGNORED_PARTS & set(relative.parts) or any(
            part.endswith(_IGNORED_PART_SUFFIXES) for part in relative.parts
        ):
            continue
        result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def changed_paths(source: Path, result: Path) -> tuple[str, ...]:
    """Return added, removed, or content-changed paths in deterministic order."""
    before = _files(source)
    after = _files(result)
    return tuple(sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path)))


def _verify(command: str, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_root = (cwd / "src").resolve()
    if source_root.is_dir():
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(source_root), existing) if part
        )
    return subprocess.run(
        shlex.split(command.replace("{python}", sys.executable)),
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _pattern_checks(patterns: tuple[str, ...], text: str) -> dict[str, bool]:
    return {pattern: re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None for pattern in patterns}


@contextmanager
def capture_tool_trace(
    records: list[dict[str, object]],
    *,
    project_id: int | None = None,
    agent_id: str = "",
) -> Iterator[None]:
    """Capture exact evaluation tool arguments/results without prompt guidance."""
    from infinidev.engine.hooks.hooks import HookContext, HookEvent, hook_manager

    def _capture(ctx: HookContext) -> None:
        if project_id is not None and ctx.project_id != project_id:
            return
        if agent_id and ctx.agent_id != agent_id:
            return
        result = ctx.result or ""
        failed = '"error"' in result.lower() or "failed:" in result.lower()
        try:
            result_payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            result_payload = None
        if isinstance(result_payload, dict):
            exit_code = result_payload.get("exit_code")
            failed = failed or (
                isinstance(exit_code, int) and exit_code != 0
            ) or result_payload.get("success") is False
        records.append(
            {
                "tool_run_id": str(ctx.metadata.get("tool_run_id", "")),
                "tool_name": ctx.tool_name,
                "arguments": dict(ctx.arguments),
                "result": result[:20_000],
                "result_truncated": len(result) > 20_000,
                "failed": failed,
            }
        )

    hook_manager.register(
        HookEvent.POST_TOOL,
        _capture,
        priority=900,
        name="agent-task-evaluation-trace",
    )
    try:
        yield
    finally:
        hook_manager.unregister(HookEvent.POST_TOOL, _capture)


def run_one(
    task: AgentTask,
    condition: str,
    repetition: int,
    *,
    config: AgentTaskRunConfig,
    manifest: Mapping[str, object],
    fixture_root: Path,
    artifact_root: Path,
    dataset_sha256: str,
    manifest_sha256: str,
    condition_sha256: str,
) -> AgentTaskObservation:
    """Execute one task with a fresh LoopEngine and preserve all review artifacts."""
    from infinidev.agents.base import InfinidevAgent
    from infinidev.engine.loop.engine import LoopEngine

    resolved_fixture_root = fixture_root.resolve()
    source = (resolved_fixture_root / task.repository_fixture).resolve()
    try:
        source.relative_to(resolved_fixture_root)
    except ValueError as exc:
        raise ValueError(f"agent task fixture escapes root: {task.repository_fixture}") from exc
    if not source.is_dir():
        raise ValueError(f"agent task fixture does not exist: {source}")
    run_name = f"{task.id}.r{repetition}.{condition}"
    artifact_dir = artifact_root / run_name
    artifact_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    error = ""
    final_answer = ""
    verify_exit: int | None = None
    verify_stdout = ""
    verify_stderr = ""
    engine_status = ""
    tool_trace: list[dict[str, object]] = []
    engine: LoopEngine | None = None
    agent: InfinidevAgent | None = None
    resolved_profile = None
    with tempfile.TemporaryDirectory(prefix="infinidev-agent-task-") as temp:
        temp_root = Path(temp)
        run_workspace = temp_root / "repo"
        copy_workspace(source, run_workspace)
        withheld = _withhold_paths(run_workspace, task.withheld_paths)
        # After the withhold, so the committed baseline is exactly what the
        # agent starts from and a diff shows only the agent's own work.
        init_git_workspace(run_workspace)
        withheld_tampering: tuple[str, ...] = ()
        seed = f"{run_name}:{run_workspace.resolve()}"
        project_id = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "big")
        instance_id = f"agent-task-{project_id:08x}-{repetition}"
        try:
            with interpreter_on_path(), configured_evaluation_runtime(
                config, condition, manifest, temp_root
            ):
                # Tool resolution consults the active model capabilities. Build
                # both objects only after the evaluation route is installed;
                # otherwise a previous/default provider contaminates the run.
                ensure_project(
                    project_id, f"agent-task::{task.id}::r{repetition}",
                )
                engine = LoopEngine()
                agent = InfinidevAgent(
                    agent_id=f"agent-task-{instance_id}",
                    role="developer",
                    project_id=project_id,
                )
                with (
                    _workspace(run_workspace),
                    capture_tool_trace(
                        tool_trace,
                        project_id=project_id,
                        agent_id=agent.agent_id,
                    ),
                ):
                    structured_task = None
                    if config.treatment in {"adaptive_behavior", "task_policy"}:
                        from infinidev.engine.task_policies import resolve_task_profile
                        from infinidev.config.settings import settings

                        if config.treatment == "task_policy" and condition == "candidate":
                            if config.task_policy_prediction_report:
                                resolved_profile = _frozen_task_policy_profile(
                                    task.request,
                                    task.id,
                                    report_path=config.task_policy_prediction_report,
                                    expected_sha256=(
                                        config.task_policy_prediction_report_sha256
                                    ),
                                )
                            else:
                                resolved_profile = resolve_task_profile(
                                    task.request,
                                    enable_embeddings=settings.TASK_POLICIES_EMBEDDINGS_ENABLED,
                                    enable_llm_fallback=settings.TASK_POLICIES_LLM_FALLBACK_ENABLED,
                                    embedding_threshold=settings.TASK_POLICIES_EMBEDDING_MIN_SCORE,
                                    embedding_margin=settings.TASK_POLICIES_EMBEDDING_MIN_MARGIN,
                                    max_policies=settings.TASK_POLICIES_MAX_SELECTED,
                                )
                        structured_task = _structured_evaluation_task(
                            task, resolved_profile
                        )
                    agent.activate_context(session_id=instance_id)
                    _AUX_USAGE.reset()
                    instruction = (
                        "Complete this isolated evaluation task. Do not create branches, commit, push, "
                        "or modify anything outside the workspace. Leave deterministic verification passing."
                    )
                    if config.pipeline_mode:
                        final_answer = _run_through_pipeline(
                            agent=agent,
                            engine=engine,
                            request=task.request,
                            instruction=instruction,
                            session_id=instance_id,
                            max_iterations=config.max_iterations,
                        )
                    else:
                        final_answer = engine.execute(
                            agent,
                            (task.request, instruction),
                            verbose=False,
                            max_iterations=config.max_iterations,
                            max_total_tool_calls=config.max_total_tool_calls,
                            max_tool_calls_per_action=config.max_tool_calls_per_action,
                            allow_llm_retries=False,
                            # TreeEngine owns separate budgets and would let one
                            # benchmark action escape this runner's declared cap.
                            allow_explore=False,
                            task=structured_task,
                        )
                    withheld_tampering = _restore_withheld(
                        run_workspace, withheld,
                    )
                    verified = _verify(
                        task.verify_command,
                        run_workspace,
                        config.verify_timeout_seconds,
                    )
                    verify_exit = verified.returncode
                    verify_stdout = verified.stdout
                    verify_stderr = verified.stderr
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if agent is not None:
                agent.deactivate()
        engine_status = str(getattr(engine, "_last_status", "")) if engine else ""
        state = getattr(engine, "_last_state", None) if engine else None
        if not error:
            reported = provider_failure_in_reply(
                final_answer, int(getattr(state, "total_prompt_tokens", 0) or 0),
            )
            if reported:
                error = reported
        history = list(getattr(state, "history", ()))
        changed = changed_paths(source, run_workspace)
        forbidden = tuple(
            path
            for path in changed
            if any(fnmatch.fnmatch(path, pattern) for pattern in task.forbidden_changed_paths)
        )
        missing_expected = tuple(path for path in task.expected_changed_paths if path not in changed)
        final_checks = _pattern_checks(task.required_final_patterns, final_answer)
        action_checks = _pattern_checks(
            task.required_action_patterns,
            json.dumps(tool_trace, ensure_ascii=False, sort_keys=True),
        )
        changed_dir = artifact_dir / "workspace"
        copy_workspace(run_workspace, changed_dir)

    # ``final_checks`` is deliberately NOT part of success. It is a regular
    # expression over free text, and it produced false failures in three
    # consecutive rounds: on user-owned-tradeoff ("choice"/"you" where the
    # pattern wanted "choose"/"user") and on wide-sum (three runs marked failed
    # while all six had found the right stage and passed the behavioural
    # verifier). Answer wording is a communication signal, so it is recorded and
    # reported separately rather than allowed to decide whether the work was
    # done.
    success = (
        not error
        and verify_exit == 0
        and not forbidden
        and not missing_expected
        and not withheld_tampering
        and all(action_checks.values())
    )
    artifact = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "condition_manifest_sha256": manifest_sha256,
        "condition_sha256": condition_sha256,
        "task": asdict(task),
        "condition": condition,
        "treatment": config.treatment,
        "task_profile": (
            resolved_profile.event_payload() if resolved_profile is not None else None
        ),
        "repetition": repetition,
        "provider": config.provider,
        "model": config.model,
        "model_identity": config.model_identity,
        "run_config": asdict(config),
        "final_answer": final_answer,
        "engine_status": engine_status,
        "plan_steps": engine.get_plan_steps() if engine else [],
        "action_records": [
            record.model_dump(mode="json") if hasattr(record, "model_dump") else str(record)
            for record in history
        ],
        "tool_trace": tool_trace,
        "malformed_tool_calls": int(
            getattr(state, "malformed_tool_calls", 0)
        ),
        "aux_prompt_tokens": _AUX_USAGE.total,
        "aux_prompt_tokens_by_lane": dict(_AUX_USAGE.by_lane),
        "provider_prompt_tokens": _AUX_USAGE.provider_prompt,
        "provider_completion_tokens": _AUX_USAGE.provider_completion,
        "provider_calls": _AUX_USAGE.provider_calls,
        "malformed_call_reasons": list(
            getattr(state, "malformed_call_reasons", ()) or ()
        ),
        "changed_paths": changed,
        "forbidden_changes": forbidden,
        "missing_expected_changes": missing_expected,
        "withheld_tampering": withheld_tampering,
        "final_pattern_checks": final_checks,
        "final_answer_patterns_ok": all(final_checks.values()),
        "action_pattern_checks": action_checks,
        "prompt_composition_history": list(getattr(state, "prompt_composition_history", ())),
        "request_payload_history": list(getattr(state, "request_payload_history", ())),
        "runtime_behavior_events": list(getattr(state, "runtime_behavior_events", ())),
        "runtime_interventions_given": list(
            getattr(state, "runtime_interventions_given", ())
        ),
        "changed_files_summary": engine.get_changed_files_summary() if engine else "",
        "file_change_reasons": engine.get_file_change_reasons() if engine else {},
        "verify_exit_code": verify_exit,
        "verify_stdout": verify_stdout,
        "verify_stderr": verify_stderr,
        "error": error,
        "rubric": [asdict(item) for item in task.rubric],
    }
    artifact_path = artifact_dir / "run.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return AgentTaskObservation(
        task_id=task.id,
        condition=condition,
        repetition=repetition,
        provider=config.provider,
        model=config.model,
        model_identity=config.model_identity,
        dataset_sha256=dataset_sha256,
        condition_manifest_sha256=manifest_sha256,
        condition_sha256=condition_sha256,
        success=success,
        verify_exit_code=verify_exit,
        engine_status=engine_status,
        changed_paths=changed,
        forbidden_changes=forbidden,
        missing_expected_changes=missing_expected,
        final_pattern_checks=final_checks,
        final_answer_patterns_ok=all(final_checks.values()),
        action_pattern_checks=action_checks,
        prompt_tokens=int(getattr(state, "total_prompt_tokens", 0)),
        completion_tokens=int(getattr(state, "total_completion_tokens", 0)),
        aux_prompt_tokens=_AUX_USAGE.total,
        provider_prompt_tokens=_AUX_USAGE.provider_prompt,
        provider_completion_tokens=_AUX_USAGE.provider_completion,
        provider_calls=_AUX_USAGE.provider_calls,
        latency_seconds=time.perf_counter() - started,
        tool_calls=int(getattr(state, "total_tool_calls", 0)),
        malformed_tool_calls=int(getattr(state, "malformed_tool_calls", 0)),
        error=error,
        run_artifact=str(artifact_path),
    )


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def select_agent_tasks(
    tasks: list[AgentTask],
    *,
    split: str,
    include_drafts: bool,
    task_ids: tuple[str, ...] = (),
) -> list[AgentTask]:
    """Select an explicit ordered task subset, rejecting misspelled ids."""
    requested = set(task_ids)
    known = {task.id for task in tasks}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown agent task ids: {', '.join(sorted(unknown))}")
    return [
        task
        for task in tasks
        if task.split == split
        and (include_drafts or task.review_status == "approved")
        and (not requested or task.id in requested)
    ]


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
    task_ids: tuple[str, ...] = (),
    conditions: tuple[str, ...] = _CONDITIONS,
    acquire_global_lock: bool = True,
    skip_units: frozenset[tuple[str, int, str]] = frozenset(),
) -> None:
    """Run every task/condition serially and stop on the first runtime error.

    ``skip_units`` carries ``(task_id, repetition, condition)`` triples already
    present in the observations file. The runner still stops on the first
    provider error, because a campaign that silently continues past a broken
    route reports a shorter sample as if it were the planned one; what
    ``skip_units`` buys is that the work already paid for is not thrown away
    when the caller resumes.
    """
    from infinidev.engine.subscription_safety import subscription_single_flight

    config = AgentTaskRunConfig.from_path(config_path)
    tasks = load_tasks(tasks_path)
    dataset_sha = file_sha256(tasks_path)
    manifest_sha, condition_hashes, manifest = load_condition_manifest(
        conditions_path, dataset_sha256=dataset_sha
    )
    if manifest.get("model_identity") != config.model_identity:
        raise ValueError("agent task manifest model identity does not match run config")
    if not conditions or any(condition not in _CONDITIONS for condition in conditions):
        raise ValueError("agent task conditions must contain baseline and/or candidate")
    if len(set(conditions)) != len(conditions):
        raise ValueError("agent task conditions must be unique")
    selected = select_agent_tasks(
        tasks,
        split=split,
        include_drafts=include_drafts,
        task_ids=task_ids,
    )
    if not selected:
        raise ValueError(f"no selected agent tasks for split: {split}")
    if (
        observations_path.exists()
        and observations_path.stat().st_size
        and not skip_units
    ):
        raise ValueError("agent task observations output must be new and empty")
    lock = subscription_single_flight() if acquire_global_lock else nullcontext()
    with lock:
        for repetition in range(config.repetitions):
            for task in selected:
                for condition in conditions:
                    if (task.id, repetition, condition) in skip_units:
                        continue
                    row = run_one(
                        task,
                        condition,
                        repetition,
                        config=config,
                        manifest=manifest,
                        fixture_root=fixture_root,
                        artifact_root=artifact_root,
                        dataset_sha256=dataset_sha,
                        manifest_sha256=manifest_sha,
                        condition_sha256=condition_hashes[condition],
                    )
                    _append_jsonl(observations_path, asdict(row))
                    if row.error:
                        raise RuntimeError(
                            f"stopped after runtime/provider error in {task.id}/{condition}: {row.error}"
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("conditions", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--fixture-root", type=Path, default=Path("bench/agent_task_fixtures"))
    parser.add_argument("--split", choices=("calibration", "validation"), default="validation")
    parser.add_argument("--include-drafts", action="store_true")
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="run only this task id; repeat to select more than one",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=_CONDITIONS,
        default=[],
        help="run only this condition; repeat to select both",
    )
    args = parser.parse_args()
    run_campaign(
        args.tasks,
        args.conditions,
        args.config,
        args.observations,
        args.artifact_root,
        fixture_root=args.fixture_root,
        split=args.split,
        include_drafts=args.include_drafts,
        task_ids=tuple(args.task_id),
        conditions=tuple(args.condition) or _CONDITIONS,
    )


if __name__ == "__main__":
    main()
