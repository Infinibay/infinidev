"""Evaluate paired adaptive-runtime observations against hard cost gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


_GENERATED_SUFFIXES = (".egg-info",)
_METRICS = ("prompt_tokens", "completion_tokens", "tool_calls", "latency_seconds")


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _normalized_paths(paths: list[str]) -> set[str]:
    return {
        path
        for path in paths
        if not any(part.endswith(_GENERATED_SUFFIXES) for part in Path(path).parts)
    }


def evaluate_runtime_campaign(
    observation_paths: list[Path],
    conditions_path: Path,
    intervention_review_path: Path | None = None,
) -> dict[str, Any]:
    """Pair tasks, compute ratios, and apply every declared deployment gate."""
    rows = _load_rows(observation_paths)
    manifest = json.loads(conditions_path.read_text(encoding="utf-8"))
    review_items = []
    if intervention_review_path is not None:
        review_payload = json.loads(intervention_review_path.read_text(encoding="utf-8"))
        review_items = list(review_payload.get("reviews", ()))
    intervention_reviews = {
        (item["task_id"], item["condition"], item["label"]): item
        for item in review_items
    }
    expected_tasks = set(manifest["task_ids"])
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = str(row["task_id"])
        condition = str(row["condition"])
        if condition in by_task[key]:
            raise ValueError(f"duplicate observation: {key}/{condition}")
        by_task[key][condition] = row
    if set(by_task) != expected_tasks:
        raise ValueError("observations do not match the declared task set")
    if any(set(pair) != {"baseline", "candidate"} for pair in by_task.values()):
        raise ValueError("every task needs exactly one baseline and candidate")

    aggregate = {
        condition: {
            metric: sum(float(pair[condition][metric]) for pair in by_task.values())
            for metric in _METRICS
        }
        for condition in ("baseline", "candidate")
    }
    ratios = {
        metric: aggregate["candidate"][metric] / aggregate["baseline"][metric]
        for metric in _METRICS
    }
    per_task: dict[str, dict[str, Any]] = {}
    success_regressions = 0
    scope_regressions = 0
    max_tool_increase = -10**9
    queued_interventions = 0
    unmatched_interventions = 0
    reviewed_interventions = 0
    false_interventions = 0
    unreviewed_interventions = 0
    for task_id, pair in sorted(by_task.items()):
        baseline = pair["baseline"]
        candidate = pair["candidate"]
        success_regressions += int(bool(baseline["success"]) and not candidate["success"])
        artifact = json.loads(Path(candidate["run_artifact"]).read_text(encoding="utf-8"))
        expected = set(artifact["task"]["expected_changed_paths"])
        candidate_paths = _normalized_paths(list(candidate["changed_paths"]))
        unexpected = sorted(candidate_paths - expected)
        scope_regressions += bool(unexpected)
        tool_increase = int(candidate["tool_calls"]) - int(baseline["tool_calls"])
        max_tool_increase = max(max_tool_increase, tool_increase)
        given = list(artifact.get("runtime_interventions_given", ()))
        queued = {
            event.get("label")
            for event in artifact.get("runtime_behavior_events", ())
            if event.get("intervention_queued")
        }
        queued_interventions += len(given)
        unmatched_interventions += sum(label not in queued for label in given)
        for label in given:
            review = intervention_reviews.get((task_id, "candidate", label))
            if review is None:
                unreviewed_interventions += 1
                continue
            reviewed_interventions += 1
            false_interventions += review.get("verdict") == "false_positive"
        per_task[task_id] = {
            "baseline_success": bool(baseline["success"]),
            "candidate_success": bool(candidate["success"]),
            "candidate_unexpected_changed_paths": unexpected,
            "interventions": given,
            "prompt_token_ratio": candidate["prompt_tokens"] / baseline["prompt_tokens"],
            "completion_token_ratio": (
                candidate["completion_tokens"] / baseline["completion_tokens"]
            ),
            "tool_call_delta": tool_increase,
            "latency_ratio": candidate["latency_seconds"] / baseline["latency_seconds"],
        }

    false_intervention_rate = (
        false_interventions / reviewed_interventions if reviewed_interventions else 0.0
    )
    declared = manifest["deployment_gates"]
    gates = {
        "success_regressions": success_regressions <= declared["success_regressions"],
        "scope_regressions": scope_regressions <= declared["scope_regressions"],
        "aggregate_prompt_tokens": (
            ratios["prompt_tokens"] <= declared["aggregate_prompt_token_ratio_max"]
        ),
        "aggregate_completion_tokens": (
            ratios["completion_tokens"]
            <= declared["aggregate_completion_token_ratio_max"]
        ),
        "aggregate_tool_calls": (
            ratios["tool_calls"] <= declared["aggregate_tool_call_ratio_max"]
        ),
        "per_task_tool_calls": (
            max_tool_increase <= declared["per_task_tool_call_increase_max"]
        ),
        "aggregate_latency": (
            ratios["latency_seconds"] <= declared["aggregate_latency_ratio_max"]
        ),
        "detector_false_interventions": (
            unmatched_interventions == 0
            and unreviewed_interventions == 0
            and false_intervention_rate
            <= declared["detector_false_intervention_rate_max"]
        ),
    }
    return {
        "schema_version": 1,
        "task_count": len(by_task),
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "aggregate": aggregate,
        "ratios": ratios,
        "reductions": {metric: 1.0 - ratio for metric, ratio in ratios.items()},
        "success_regressions": success_regressions,
        "scope_regressions": scope_regressions,
        "queued_interventions": queued_interventions,
        "unmatched_interventions": unmatched_interventions,
        "reviewed_interventions": reviewed_interventions,
        "unreviewed_interventions": unreviewed_interventions,
        "false_interventions": false_interventions,
        "false_intervention_rate": false_intervention_rate,
        "per_task": per_task,
        "interpretation_boundary": manifest["interpretation_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conditions", type=Path)
    parser.add_argument("observations", nargs="+", type=Path)
    parser.add_argument("--intervention-review", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        evaluate_runtime_campaign(
            args.observations,
            args.conditions,
            args.intervention_review,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
