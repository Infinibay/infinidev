#!/usr/bin/env python3
"""Evaluate baseline, ranked, and full context on held-out repository tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping


CONDITIONS = ("baseline", "ranked", "full")
SPLITS = frozenset({"calibration", "validation"})
POSITIONS = frozenset({"none", "front", "middle", "end"})


@dataclass(frozen=True)
class ContextTask:
    """One deterministic repository task and its required context evidence."""

    id: str
    family: str
    split: str
    repository_fixture: str
    request: str
    verify_command: str
    required_evidence: tuple[str, ...]
    relevant_evidence_position: str = "none"
    review_status: str = "draft"

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ContextTask:
        raw_evidence = value.get("required_evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError("context task required_evidence must be a list")
        task = cls(
            id=str(value.get("id", "")).strip(),
            family=str(value.get("family", "")).strip(),
            split=str(value.get("split", "")).strip(),
            repository_fixture=str(value.get("repository_fixture", "")).strip(),
            request=str(value.get("request", "")).strip(),
            verify_command=str(value.get("verify_command", "")).strip(),
            required_evidence=tuple(str(item).strip() for item in raw_evidence),
            relevant_evidence_position=str(
                value.get("relevant_evidence_position", "none")
            ).strip(),
            review_status=str(value.get("review_status", "draft")).strip(),
        )
        if not all(
            (task.id, task.family, task.repository_fixture, task.request, task.verify_command)
        ):
            raise ValueError("context task is missing a required identity or execution field")
        if task.split not in SPLITS:
            raise ValueError(f"unsupported context task split: {task.split}")
        if task.relevant_evidence_position not in POSITIONS:
            raise ValueError("relevant_evidence_position must be none, front, middle, or end")
        if task.review_status not in {"draft", "approved", "rejected"}:
            raise ValueError("context task review_status is invalid")
        if not task.required_evidence or any(not item for item in task.required_evidence):
            raise ValueError("context task needs non-empty required_evidence")
        if len(task.required_evidence) != len(set(task.required_evidence)):
            raise ValueError("context task required_evidence contains duplicates")
        return task


@dataclass(frozen=True)
class ContextObservation:
    """One end-to-end task outcome under one context-delivery condition."""

    task_id: str
    condition: str
    repetition: int
    model_identity: str
    dataset_sha256: str
    condition_manifest_sha256: str
    condition_sha256: str
    success: bool
    verify_exit_code: int | None
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    tool_calls: int
    context_items: tuple[str, ...]
    error: str = ""
    run_artifact: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ContextObservation:
        if not isinstance(value.get("success"), bool):
            raise ValueError("context observation success must be boolean")
        raw_items = value.get("context_items")
        if not isinstance(raw_items, list):
            raise ValueError("context observation context_items must be a list")
        row = cls(
            task_id=str(value.get("task_id", "")).strip(),
            condition=str(value.get("condition", "")).strip(),
            repetition=int(value.get("repetition", 0)),
            model_identity=str(value.get("model_identity", "")).strip(),
            dataset_sha256=str(value.get("dataset_sha256", "")).strip(),
            condition_manifest_sha256=str(
                value.get("condition_manifest_sha256", "")
            ).strip(),
            condition_sha256=str(value.get("condition_sha256", "")).strip(),
            success=bool(value.get("success", False)),
            verify_exit_code=(
                int(value["verify_exit_code"])
                if value.get("verify_exit_code") is not None
                else None
            ),
            prompt_tokens=int(value.get("prompt_tokens", 0)),
            completion_tokens=int(value.get("completion_tokens", 0)),
            latency_seconds=float(value.get("latency_seconds", 0.0)),
            tool_calls=int(value.get("tool_calls", 0)),
            context_items=tuple(str(item).strip() for item in raw_items),
            error=str(value.get("error", "")).strip(),
            run_artifact=str(value.get("run_artifact", "")).strip(),
        )
        if not all(
            (
                row.task_id,
                row.model_identity,
                row.dataset_sha256,
                row.condition_manifest_sha256,
                row.condition_sha256,
            )
        ):
            raise ValueError("context observation is missing identity metadata")
        if row.condition not in CONDITIONS:
            raise ValueError(f"unsupported context condition: {row.condition}")
        if row.repetition < 0 or min(
            row.prompt_tokens,
            row.completion_tokens,
            row.tool_calls,
        ) < 0 or row.latency_seconds < 0:
            raise ValueError("context observation contains a negative measurement")
        expected_success = not row.error and row.verify_exit_code == 0
        if row.success != expected_success:
            raise ValueError("success must match error-free deterministic verification")
        return row


def load_tasks(path: Path) -> list[ContextTask]:
    return [ContextTask.from_dict(json.loads(line)) for line in _jsonl_lines(path)]


def load_observations(path: Path) -> list[ContextObservation]:
    return [ContextObservation.from_dict(json.loads(line)) for line in _jsonl_lines(path)]


def _jsonl_lines(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield line


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_condition_manifest(
    path: Path, *, dataset_sha256: str
) -> tuple[str, dict[str, str]]:
    """Validate and hash the exact context-delivery treatments."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported context condition manifest")
    if value.get("dataset_sha256") != dataset_sha256:
        raise ValueError("context condition manifest dataset hash mismatch")
    conditions = value.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise ValueError("condition manifest needs baseline, ranked, and full")
    hashes: dict[str, str] = {}
    for name in CONDITIONS:
        condition = conditions[name]
        if not isinstance(condition, dict) or condition.get("context_source") != name:
            raise ValueError(f"invalid context condition definition: {name}")
        encoded = json.dumps(
            condition, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    return file_sha256(path), hashes


def validate_family_atomic_splits(tasks: Iterable[ContextTask]) -> None:
    families: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        families[task.family].add(task.split)
    mixed = sorted(family for family, splits in families.items() if len(splits) > 1)
    if mixed:
        raise ValueError(f"context task families cross calibration/validation: {mixed}")


def build_report(
    tasks: Iterable[ContextTask],
    observations: Iterable[ContextObservation],
    *,
    dataset_sha256: str,
    condition_manifest_sha256: str,
    expected_condition_hashes: Mapping[str, str],
    split: str = "validation",
    include_drafts: bool = False,
) -> dict[str, object]:
    """Build paired outcome and evidence-delivery summaries without hiding tasks."""
    all_tasks = list(tasks)
    validate_family_atomic_splits(all_tasks)
    task_ids = [task.id for task in all_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("context task dataset contains duplicate IDs")
    if split not in SPLITS:
        raise ValueError(f"unsupported context task split: {split}")
    selected = {
        task.id: task
        for task in all_tasks
        if task.split == split and (include_drafts or task.review_status == "approved")
    }
    if not selected:
        raise ValueError(f"no context tasks for split: {split}")
    all_rows = list(observations)
    known_ids = set(task_ids)
    unknown = sorted({row.task_id for row in all_rows if row.task_id not in known_ids})
    if unknown:
        raise ValueError(f"context observations contain unknown task IDs: {unknown}")
    rows = [row for row in all_rows if row.task_id in selected]
    if any(row.dataset_sha256 != dataset_sha256 for row in rows):
        raise ValueError("context observation dataset hash mismatch")
    if any(
        row.condition_manifest_sha256 != condition_manifest_sha256 for row in rows
    ):
        raise ValueError("context observation condition manifest hash mismatch")
    identities = {row.model_identity for row in rows}
    if len(identities) != 1:
        raise ValueError("context evaluation requires one immutable model identity")
    condition_hashes: dict[str, set[str]] = defaultdict(set)
    keyed: dict[tuple[str, int], dict[str, ContextObservation]] = defaultdict(dict)
    for row in rows:
        condition_hashes[row.condition].add(row.condition_sha256)
        key = (row.task_id, row.repetition)
        if row.condition in keyed[key]:
            raise ValueError(f"duplicate context observation: {key}/{row.condition}")
        keyed[key][row.condition] = row
    if not keyed or any(set(group) != set(CONDITIONS) for group in keyed.values()):
        raise ValueError("every task repetition needs baseline, ranked, and full observations")
    if any(len(condition_hashes[name]) != 1 for name in CONDITIONS):
        raise ValueError("each context condition needs one immutable condition hash")
    if set(expected_condition_hashes) != set(CONDITIONS):
        raise ValueError("expected condition hashes are incomplete")
    observed_hashes = {
        name: next(iter(condition_hashes[name])) for name in CONDITIONS
    }
    if observed_hashes != dict(expected_condition_hashes):
        raise ValueError("context observation condition hash mismatch")
    observed_task_ids = {task_id for task_id, _ in keyed}
    if observed_task_ids != set(selected):
        raise ValueError("context observations do not cover the complete selected task split")

    summaries = {
        condition: _condition_summary(
            selected,
            [group[condition] for group in keyed.values()],
        )
        for condition in CONDITIONS
    }
    comparisons = {
        condition: _paired_summary(
            [group["baseline"] for group in keyed.values()],
            [group[condition] for group in keyed.values()],
        )
        for condition in ("ranked", "full")
    }
    task_records = []
    for key in sorted(keyed):
        task = selected[key[0]]
        task_records.append(
            {
                "task": asdict(task),
                "repetition": key[1],
                "conditions": {
                    name: _row_record(task, keyed[key][name]) for name in CONDITIONS
                },
            }
        )
    return {
        "interpretation_boundary": (
            "Deterministic task outcomes measure the tested fixture and verifier, not universal "
            "agent quality. Evidence recall diagnoses delivery, but a recalled item does not prove "
            "the model used it. Inspect every task record before interpreting aggregates."
        ),
        "dataset_sha256": dataset_sha256,
        "split": split,
        "model_identity": next(iter(identities)),
        "condition_manifest_sha256": condition_manifest_sha256,
        "condition_hashes": observed_hashes,
        "task_count": len(selected),
        "paired_repetitions": len(keyed),
        "conditions": summaries,
        "paired_vs_baseline": comparisons,
        "task_records": task_records,
    }


def _row_record(task: ContextTask, row: ContextObservation) -> dict[str, object]:
    supplied = set(row.context_items)
    missing = [item for item in task.required_evidence if item not in supplied]
    return {
        **asdict(row),
        "required_evidence_delivered": [
            item for item in task.required_evidence if item in supplied
        ],
        "required_evidence_omitted": missing,
        "evidence_recall": (len(task.required_evidence) - len(missing))
        / len(task.required_evidence),
        "qualitative_artifact": _qualitative_artifact(row),
    }


def _qualitative_artifact(row: ContextObservation) -> dict[str, object]:
    """Load action-level evidence while rejecting a mismatched run artifact."""
    if not row.run_artifact:
        return {"available": False, "reason": "observation has no run artifact"}
    path = Path(row.run_artifact)
    if not path.is_file():
        return {"available": False, "reason": "run artifact is missing"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"unreadable run artifact: {exc}"}
    if not isinstance(value, dict):
        return {"available": False, "reason": "run artifact is not an object"}
    expected = {
        "dataset_sha256": row.dataset_sha256,
        "condition_manifest_sha256": row.condition_manifest_sha256,
        "condition_sha256": row.condition_sha256,
        "condition": row.condition,
        "repetition": row.repetition,
        "model_identity": row.model_identity,
    }
    mismatched = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    task = value.get("task")
    if not isinstance(task, dict) or task.get("id") != row.task_id:
        mismatched.append("task.id")
    if mismatched:
        return {
            "available": False,
            "reason": f"run artifact identity mismatch: {sorted(set(mismatched))}",
        }
    return {
        "available": True,
        "engine_status": str(value.get("engine_status", "")),
        "final_answer": str(value.get("final_answer", "")),
        "plan_steps": value.get("plan_steps", []),
        "action_records": value.get("action_records", []),
        "changed_files_summary": str(value.get("changed_files_summary", "")),
        "file_change_reasons": value.get("file_change_reasons", {}),
        "verify_stdout": str(value.get("verify_stdout", "")),
        "verify_stderr": str(value.get("verify_stderr", "")),
        "prompt_composition_history": value.get("prompt_composition_history", []),
        "request_payload_history": value.get("request_payload_history", []),
    }


def _condition_summary(
    tasks: Mapping[str, ContextTask], rows: list[ContextObservation]
) -> dict[str, object]:
    records = [_row_record(tasks[row.task_id], row) for row in rows]
    return {
        "attempted": len(rows),
        "verified_successes": sum(row.success for row in rows),
        "provider_errors": sum(bool(row.error) for row in rows),
        "mean_prompt_tokens": mean(row.prompt_tokens for row in rows),
        "mean_completion_tokens": mean(row.completion_tokens for row in rows),
        "mean_latency_seconds": mean(row.latency_seconds for row in rows),
        "mean_tool_calls": mean(row.tool_calls for row in rows),
        "mean_evidence_recall": mean(float(record["evidence_recall"]) for record in records),
    }


def _paired_summary(
    baseline: list[ContextObservation], candidate: list[ContextObservation]
) -> dict[str, object]:
    base_by_key = {(row.task_id, row.repetition): row for row in baseline}
    candidate_by_key = {(row.task_id, row.repetition): row for row in candidate}
    if set(base_by_key) != set(candidate_by_key):
        raise ValueError("paired context conditions do not share task repetitions")
    wins = losses = ties = 0
    for key, base in base_by_key.items():
        other = candidate_by_key[key]
        if other.success and not base.success:
            wins += 1
        elif base.success and not other.success:
            losses += 1
        else:
            ties += 1
    return {
        "paired_n": len(base_by_key),
        "success_wins": wins,
        "success_losses": losses,
        "success_ties": ties,
        "sign_exact_p": _two_sided_sign_p(wins, losses),
    }


def _two_sided_sign_p(wins: int, losses: int) -> float | None:
    discordant = wins + losses
    if not discordant:
        return None
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def render_markdown(report: Mapping[str, object]) -> str:
    """Render concrete repository outcomes before compact condition summaries."""
    lines = [
        "# Held-out context delivery evaluation",
        "",
        str(report["interpretation_boundary"]),
        "",
        f"Model identity: `{report['model_identity']}`. Split: `{report['split']}`.",
        "",
        "## Task-level evidence",
    ]
    for record in report.get("task_records", []):
        if not isinstance(record, dict):
            continue
        task = record["task"]
        lines.extend(
            [
                "",
                f"### `{task['id']}` repetition {record['repetition']}",
                "",
                f"Request: {task['request']}",
                f"Verifier: `{task['verify_command']}`.",
                f"Required evidence: `{list(task['required_evidence'])}`.",
            ]
        )
        for condition in CONDITIONS:
            row = record["conditions"][condition]
            lines.append(
                f"- **{condition}**: success={row['success']}; prompt_tokens={row['prompt_tokens']}; "
                f"tool_calls={row['tool_calls']}; latency={row['latency_seconds']:.3f}s; "
                f"evidence_recall={row['evidence_recall']:.2f}; "
                f"omitted={row['required_evidence_omitted']}; error={row['error']!r}; "
                f"artifact={row['run_artifact']!r}."
            )
            qualitative = row["qualitative_artifact"]
            if qualitative.get("available"):
                final_answer = str(qualitative.get("final_answer", "")).strip()
                changed = str(qualitative.get("changed_files_summary", "")).strip()
                lines.append(
                    f"  - Engine status: `{qualitative.get('engine_status', '')}`; "
                    f"plan: `{qualitative.get('plan_steps', [])}`."
                )
                if final_answer:
                    lines.append(f"  - Final response: {final_answer[:1200]}")
                actions = qualitative.get("action_records", [])
                if isinstance(actions, list):
                    action_summaries = [
                        str(action.get("summary", "")).strip()
                        for action in actions
                        if isinstance(action, dict) and action.get("summary")
                    ]
                    if action_summaries:
                        lines.append(
                            "  - Recorded actions: "
                            + " | ".join(action_summaries[:12])[:2400]
                        )
                if changed:
                    lines.extend(["", changed[:3000], ""])
                payloads = qualitative.get("request_payload_history", [])
                if isinstance(payloads, list) and payloads:
                    largest = max(
                        (item for item in payloads if isinstance(item, dict)),
                        key=lambda item: int(item.get("request_payload_chars", 0)),
                        default=None,
                    )
                    if largest is not None:
                        lines.append(
                            "  - Largest dispatched request: "
                            f"{largest.get('request_payload_chars', 0)} chars across "
                            f"{largest.get('message_count', 0)} messages; role content "
                            f"{largest.get('message_content_chars_by_role', {})}."
                        )
                compositions = qualitative.get("prompt_composition_history", [])
                if isinstance(compositions, list) and compositions:
                    latest = compositions[-1]
                    if isinstance(latest, dict):
                        sections = latest.get("user_sections", {})
                        if isinstance(sections, dict):
                            top_sections = sorted(
                                sections.items(), key=lambda item: int(item[1]), reverse=True
                            )[:5]
                            lines.append(
                                f"  - Final iteration's largest user-prompt sections: {top_sections}."
                            )
            else:
                lines.append(
                    f"  - Qualitative artifact unavailable: {qualitative.get('reason', '')}."
                )
    lines.extend(["", "## Aggregate navigation"])
    for condition in CONDITIONS:
        summary = report["conditions"][condition]
        lines.append(
            f"- **{condition}**: verified {summary['verified_successes']}/{summary['attempted']}; "
            f"mean prompt tokens {summary['mean_prompt_tokens']:.1f}; mean tool calls "
            f"{summary['mean_tool_calls']:.1f}; mean evidence recall "
            f"{summary['mean_evidence_recall']:.2f}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("conditions", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--split", choices=sorted(SPLITS), default="validation")
    parser.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args()
    dataset_sha256 = file_sha256(args.tasks)
    manifest_sha256, condition_hashes = load_condition_manifest(
        args.conditions, dataset_sha256=dataset_sha256
    )
    report = build_report(
        load_tasks(args.tasks),
        load_observations(args.observations),
        dataset_sha256=dataset_sha256,
        condition_manifest_sha256=manifest_sha256,
        expected_condition_hashes=condition_hashes,
        split=args.split,
        include_drafts=args.include_drafts,
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
