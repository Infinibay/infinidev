#!/usr/bin/env python3
"""Contracts and evidence-first reports for provider-neutral agent task experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping


SPLITS = frozenset({"calibration", "validation"})
REVIEW_STATUSES = frozenset({"draft", "approved", "rejected"})
RUBRIC_KINDS = frozenset({"deterministic", "human_review"})


@dataclass(frozen=True)
class RubricItem:
    """One predeclared observable used to inspect a task outcome."""

    id: str
    description: str
    kind: str
    evidence_source: str
    weight: float

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RubricItem:
        item = cls(
            id=str(value.get("id", "")).strip(),
            description=str(value.get("description", "")).strip(),
            kind=str(value.get("kind", "")).strip(),
            evidence_source=str(value.get("evidence_source", "")).strip(),
            weight=float(value.get("weight", 0.0)),
        )
        if not all((item.id, item.description, item.evidence_source)):
            raise ValueError("rubric item is missing identity, description, or evidence source")
        if item.kind not in RUBRIC_KINDS:
            raise ValueError(f"unsupported rubric kind: {item.kind}")
        if not 0 < item.weight <= 1:
            raise ValueError("rubric weight must be in (0, 1]")
        return item


@dataclass(frozen=True)
class AgentTask:
    """One isolated repository task with deterministic and review evidence contracts."""

    id: str
    family: str
    category: str
    split: str
    repository_fixture: str
    request: str
    verify_command: str
    expected_changed_paths: tuple[str, ...]
    forbidden_changed_paths: tuple[str, ...]
    required_final_patterns: tuple[str, ...]
    required_action_patterns: tuple[str, ...]
    rubric: tuple[RubricItem, ...]
    review_status: str = "draft"

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AgentTask:
        task = cls(
            id=str(value.get("id", "")).strip(),
            family=str(value.get("family", "")).strip(),
            category=str(value.get("category", "")).strip(),
            split=str(value.get("split", "")).strip(),
            repository_fixture=str(value.get("repository_fixture", "")).strip(),
            request=str(value.get("request", "")).strip(),
            verify_command=str(value.get("verify_command", "")).strip(),
            expected_changed_paths=_strings(value.get("expected_changed_paths")),
            forbidden_changed_paths=_strings(value.get("forbidden_changed_paths")),
            required_final_patterns=_patterns(value.get("required_final_patterns")),
            required_action_patterns=_patterns(value.get("required_action_patterns")),
            rubric=tuple(
                RubricItem.from_dict(item)
                for item in _mappings(value.get("rubric"), "rubric")
            ),
            review_status=str(value.get("review_status", "draft")).strip(),
        )
        if not all(
            (
                task.id,
                task.family,
                task.category,
                task.repository_fixture,
                task.request,
                task.verify_command,
            )
        ):
            raise ValueError("agent task is missing a required identity or execution field")
        if task.split not in SPLITS:
            raise ValueError(f"unsupported agent task split: {task.split}")
        if task.review_status not in REVIEW_STATUSES:
            raise ValueError("agent task review_status is invalid")
        if not task.rubric:
            raise ValueError("agent task needs a non-empty predeclared rubric")
        rubric_ids = [item.id for item in task.rubric]
        if len(rubric_ids) != len(set(rubric_ids)):
            raise ValueError("agent task rubric contains duplicate IDs")
        return task


@dataclass(frozen=True)
class AgentTaskObservation:
    """One complete agent task execution and its deterministic measurements."""

    task_id: str
    condition: str
    repetition: int
    provider: str
    model: str
    model_identity: str
    dataset_sha256: str
    condition_manifest_sha256: str
    condition_sha256: str
    success: bool
    verify_exit_code: int | None
    engine_status: str
    changed_paths: tuple[str, ...]
    forbidden_changes: tuple[str, ...]
    missing_expected_changes: tuple[str, ...]
    final_pattern_checks: dict[str, bool]
    action_pattern_checks: dict[str, bool]
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    tool_calls: int
    error: str = ""
    run_artifact: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AgentTaskObservation:
        if not isinstance(value.get("success"), bool):
            raise ValueError("agent task observation success must be boolean")
        row = cls(
            task_id=str(value.get("task_id", "")).strip(),
            condition=str(value.get("condition", "")).strip(),
            repetition=int(value.get("repetition", 0)),
            provider=str(value.get("provider", "")).strip(),
            model=str(value.get("model", "")).strip(),
            model_identity=str(value.get("model_identity", "")).strip(),
            dataset_sha256=str(value.get("dataset_sha256", "")).strip(),
            condition_manifest_sha256=str(
                value.get("condition_manifest_sha256", "")
            ).strip(),
            condition_sha256=str(value.get("condition_sha256", "")).strip(),
            success=bool(value.get("success")),
            verify_exit_code=(
                int(value["verify_exit_code"])
                if value.get("verify_exit_code") is not None
                else None
            ),
            engine_status=str(value.get("engine_status", "")).strip(),
            changed_paths=_strings(value.get("changed_paths")),
            forbidden_changes=_strings(value.get("forbidden_changes")),
            missing_expected_changes=_strings(value.get("missing_expected_changes")),
            final_pattern_checks=_bool_mapping(
                value.get("final_pattern_checks"), "final_pattern_checks"
            ),
            action_pattern_checks=_bool_mapping(
                value.get("action_pattern_checks"), "action_pattern_checks"
            ),
            prompt_tokens=int(value.get("prompt_tokens", 0)),
            completion_tokens=int(value.get("completion_tokens", 0)),
            latency_seconds=float(value.get("latency_seconds", 0.0)),
            tool_calls=int(value.get("tool_calls", 0)),
            error=str(value.get("error", "")).strip(),
            run_artifact=str(value.get("run_artifact", "")).strip(),
        )
        if not all(
            (
                row.task_id,
                row.condition,
                row.provider,
                row.model,
                row.model_identity,
                row.dataset_sha256,
                row.condition_manifest_sha256,
                row.condition_sha256,
            )
        ):
            raise ValueError("agent task observation is missing identity metadata")
        if row.repetition < 0 or min(
            row.prompt_tokens, row.completion_tokens, row.tool_calls
        ) < 0 or row.latency_seconds < 0:
            raise ValueError("agent task observation contains a negative measurement")
        deterministic_success = (
            not row.error
            and row.verify_exit_code == 0
            and not row.forbidden_changes
            and not row.missing_expected_changes
            and all(row.final_pattern_checks.values())
            and all(row.action_pattern_checks.values())
        )
        if row.success != deterministic_success:
            raise ValueError("success must match every deterministic task check")
        return row


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError("string list values must be non-empty and unique")
    return result


def _patterns(value: object) -> tuple[str, ...]:
    patterns = _strings(value)
    for pattern in patterns:
        re.compile(pattern)
    return patterns


def _mappings(value: object, name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must contain only objects")
    return list(value)  # type: ignore[arg-type]


def _bool_mapping(value: object, name: str) -> dict[str, bool]:
    if not isinstance(value, dict) or any(not isinstance(item, bool) for item in value.values()):
        raise ValueError(f"{name} must be an object of booleans")
    return {str(key): bool(item) for key, item in value.items()}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tasks(path: Path) -> list[AgentTask]:
    rows = [
        AgentTask.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("agent task dataset contains duplicate IDs")
    families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        families[row.family].add(row.split)
    mixed = sorted(family for family, splits in families.items() if len(splits) > 1)
    if mixed:
        raise ValueError(f"agent task families cross splits: {mixed}")
    return rows


def load_observations(path: Path) -> list[AgentTaskObservation]:
    return [
        AgentTaskObservation.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_condition_manifest(
    path: Path, *, dataset_sha256: str
) -> tuple[str, dict[str, str], dict[str, object]]:
    """Validate two model-neutral treatments and their explicit user profile."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported agent task condition manifest")
    if value.get("dataset_sha256") != dataset_sha256:
        raise ValueError("agent task condition manifest dataset hash mismatch")
    conditions = value.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != {"baseline", "candidate"}:
        raise ValueError("agent task manifest needs baseline and candidate conditions")
    if conditions.get("baseline") is not None:
        raise ValueError("baseline condition must omit additional system guidance")
    candidate = conditions.get("candidate")
    if not isinstance(candidate, dict) or not isinstance(candidate.get("system_prompt"), str):
        raise ValueError("candidate condition needs a system_prompt")
    if not str(candidate["system_prompt"]).strip():
        raise ValueError("candidate system_prompt cannot be empty")
    profile = value.get("utility_profile")
    if not isinstance(profile, dict):
        raise ValueError("agent task manifest needs an explicit utility profile")
    if profile.get("schema_version") != 1 or profile.get("provenance") != "explicit_user":
        raise ValueError("utility profile must have explicit_user provenance")
    hashes = {
        name: hashlib.sha256(
            ("<no-system-message>" if raw is None else str(raw["system_prompt"])).encode()
        ).hexdigest()
        for name, raw in conditions.items()
    }
    return file_sha256(path), hashes, value


def build_report(
    tasks: Iterable[AgentTask],
    observations: Iterable[AgentTaskObservation],
    *,
    dataset_sha256: str,
    condition_manifest_sha256: str,
    expected_condition_hashes: Mapping[str, str],
    split: str = "validation",
    include_drafts: bool = False,
) -> dict[str, object]:
    """Build paired falsification evidence while retaining every task artifact."""
    selected = {
        task.id: task
        for task in tasks
        if task.split == split and (include_drafts or task.review_status == "approved")
    }
    if not selected:
        raise ValueError(f"no selected agent tasks for split: {split}")
    rows = [row for row in observations if row.task_id in selected]
    if any(row.dataset_sha256 != dataset_sha256 for row in rows):
        raise ValueError("agent task observation dataset hash mismatch")
    if any(row.condition_manifest_sha256 != condition_manifest_sha256 for row in rows):
        raise ValueError("agent task observation manifest hash mismatch")
    identities = {(row.provider, row.model, row.model_identity) for row in rows}
    if len(identities) != 1:
        raise ValueError("agent task report requires one immutable model route")
    keyed: dict[tuple[str, int], dict[str, AgentTaskObservation]] = defaultdict(dict)
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = (row.task_id, row.repetition)
        if row.condition in keyed[key]:
            raise ValueError(f"duplicate agent task observation: {key}/{row.condition}")
        keyed[key][row.condition] = row
        hashes[row.condition].add(row.condition_sha256)
    if not keyed or any(set(group) != {"baseline", "candidate"} for group in keyed.values()):
        raise ValueError("every task repetition needs baseline and candidate observations")
    if {task_id for task_id, _ in keyed} != set(selected):
        raise ValueError("agent task observations do not cover the selected split")
    observed_hashes = {name: next(iter(values)) for name, values in hashes.items() if len(values) == 1}
    if observed_hashes != dict(expected_condition_hashes):
        raise ValueError("agent task condition hashes do not match manifest")

    paired = []
    for key in sorted(keyed):
        baseline = keyed[key]["baseline"]
        candidate = keyed[key]["candidate"]
        paired.append(
            {
                "task": asdict(selected[key[0]]),
                "repetition": key[1],
                "baseline": asdict(baseline),
                "candidate": asdict(candidate),
                "success_delta": int(candidate.success) - int(baseline.success),
                "tool_call_delta": candidate.tool_calls - baseline.tool_calls,
                "latency_delta_seconds": candidate.latency_seconds - baseline.latency_seconds,
                "candidate_changed_behavior": _behavior_signature(candidate)
                != _behavior_signature(baseline),
            }
        )
    summaries = {
        condition: _condition_summary([group[condition] for group in keyed.values()])
        for condition in ("baseline", "candidate")
    }
    improvements = sum(row["success_delta"] > 0 for row in paired)
    regressions = sum(row["success_delta"] < 0 for row in paired)
    unchanged = len(paired) - improvements - regressions
    return {
        "interpretation_boundary": (
            "This small paired campaign is a falsification pilot, not deployment evidence. "
            "Deterministic checks establish fixture outcomes; human-review rubric items remain "
            "unscored until blinded review. Inspect concrete artifacts before interpreting totals."
        ),
        "dataset_sha256": dataset_sha256,
        "condition_manifest_sha256": condition_manifest_sha256,
        "condition_hashes": observed_hashes,
        "provider": next(iter(identities))[0],
        "model": next(iter(identities))[1],
        "model_identity": next(iter(identities))[2],
        "task_count": len(selected),
        "paired_repetitions": len(paired),
        "conditions": summaries,
        "paired_outcomes": {
            "candidate_improvements": improvements,
            "candidate_regressions": regressions,
            "unchanged_success": unchanged,
        },
        "promotion_boundary": (
            "Promote only if behavior changes across more than the originating probe domain, "
            "there are no competence/authorization regressions, and later calibration plus held-out "
            "validation pass. A correct baseline or no material change favors no extra guidance."
        ),
        "task_records": paired,
    }


def _behavior_signature(row: AgentTaskObservation) -> tuple[object, ...]:
    return (
        row.success,
        row.changed_paths,
        tuple(sorted(row.final_pattern_checks.items())),
        tuple(sorted(row.action_pattern_checks.items())),
        row.engine_status,
    )


def _condition_summary(rows: list[AgentTaskObservation]) -> dict[str, object]:
    return {
        "attempted": len(rows),
        "verified_successes": sum(row.success for row in rows),
        "errors": sum(bool(row.error) for row in rows),
        "unauthorized_or_forbidden_change_runs": sum(bool(row.forbidden_changes) for row in rows),
        "missing_expected_change_runs": sum(bool(row.missing_expected_changes) for row in rows),
        "mean_latency_seconds": mean(row.latency_seconds for row in rows),
        "mean_tool_calls": mean(row.tool_calls for row in rows),
        "prompt_tokens": sum(row.prompt_tokens for row in rows),
        "completion_tokens": sum(row.completion_tokens for row in rows),
    }


def render_markdown(report: Mapping[str, object]) -> str:
    """Render candidate falsification evidence model by model."""
    lines = [
        "# Agent task falsification pilot",
        "",
        str(report.get("interpretation_boundary")),
        "",
        f"Route: `{report.get('provider')}/{report.get('model')}`; identity: "
        f"`{report.get('model_identity')}`.",
        "",
        f"Tasks: {report.get('task_count')}; paired executions: {report.get('paired_repetitions')}.",
        f"Paired outcomes: `{json.dumps(report.get('paired_outcomes'), sort_keys=True)}`.",
        "",
        str(report.get("promotion_boundary")),
    ]
    conditions = report.get("conditions")
    if isinstance(conditions, dict):
        lines.extend(["", "## Condition summaries"])
        for name, summary in conditions.items():
            lines.append(f"- **{name}**: `{json.dumps(summary, sort_keys=True)}`")
    records = report.get("task_records")
    if isinstance(records, list):
        lines.extend(["", "## Complete paired task evidence"])
        for record in records:
            if not isinstance(record, dict):
                continue
            task = record.get("task", {})
            task_id = task.get("id") if isinstance(task, dict) else ""
            lines.extend(
                [
                    "",
                    f"### `{task_id}` repetition {record.get('repetition')}",
                    "",
                    f"Success delta: {record.get('success_delta')}; behavior changed: "
                    f"{record.get('candidate_changed_behavior')}; tool delta: "
                    f"{record.get('tool_call_delta')}; latency delta: "
                    f"{record.get('latency_delta_seconds')}s.",
                ]
            )
            for condition in ("baseline", "candidate"):
                row = record.get(condition, {})
                if isinstance(row, dict):
                    lines.append(
                        f"- **{condition}**: success={row.get('success')}; status="
                        f"`{row.get('engine_status')}`; changed={row.get('changed_paths')}; "
                        f"forbidden={row.get('forbidden_changes')}; error=`{row.get('error')}`; "
                        f"artifact=`{row.get('run_artifact')}`."
                    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("conditions", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--split", choices=tuple(sorted(SPLITS)), default="validation")
    parser.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args()
    tasks = load_tasks(args.tasks)
    dataset_sha = file_sha256(args.tasks)
    manifest_sha, hashes, _ = load_condition_manifest(
        args.conditions, dataset_sha256=dataset_sha
    )
    report = build_report(
        tasks,
        load_observations(args.observations),
        dataset_sha256=dataset_sha,
        condition_manifest_sha256=manifest_sha,
        expected_condition_hashes=hashes,
        split=args.split,
        include_drafts=args.include_drafts,
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
