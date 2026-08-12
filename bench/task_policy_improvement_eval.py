"""Evaluate prompt-only task-policy pairs against explicit improvement gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


_COST_METRICS = ("prompt_tokens", "completion_tokens", "tool_calls", "latency_seconds")


def _load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _first_composition(artifact: dict[str, Any]) -> dict[str, Any]:
    history = artifact.get("prompt_composition_history", [])
    return dict(history[0]) if history else {}


def evaluate_task_policy_improvement(
    observation_paths: list[Path],
    gates_path: Path,
) -> dict[str, Any]:
    """Pair observations and require quality, isolation, and cost improvement."""
    rows = _load_jsonl(observation_paths)
    specification = json.loads(gates_path.read_text(encoding="utf-8"))
    expected_tasks = set(specification["task_ids"])
    expected_fragments = specification["expected_candidate_fragments"]
    gates_spec = specification["gates"]

    routes = {(row["provider"], row["model_identity"]) for row in rows}
    if len(routes) != 1:
        raise ValueError("observations must use one immutable model route")

    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        task_id = str(row["task_id"])
        condition = str(row["condition"])
        if condition in paired[task_id]:
            raise ValueError(f"duplicate observation: {task_id}/{condition}")
        paired[task_id][condition] = row
    if set(paired) != expected_tasks:
        raise ValueError("observations do not match the declared task set")
    if any(set(pair) != {"baseline", "candidate"} for pair in paired.values()):
        raise ValueError("every task requires one baseline and one candidate")
    for task_id, pair in paired.items():
        campaign_ids = {
            (row["dataset_sha256"], row["condition_manifest_sha256"])
            for row in pair.values()
        }
        if len(campaign_ids) != 1:
            raise ValueError(f"paired task crosses campaign manifests: {task_id}")

    aggregate = {
        condition: {
            metric: sum(float(pair[condition][metric]) for pair in paired.values())
            for metric in _COST_METRICS
        }
        for condition in ("baseline", "candidate")
    }
    ratios = {
        metric: aggregate["candidate"][metric] / max(aggregate["baseline"][metric], 1.0)
        for metric in _COST_METRICS
    }

    success_regressions = 0
    scope_regressions = 0
    isolation_failures: list[str] = []
    fragment_failures: list[str] = []
    per_task: dict[str, Any] = {}
    for task_id, pair in sorted(paired.items()):
        baseline = pair["baseline"]
        candidate = pair["candidate"]
        success_regressions += int(bool(baseline["success"]) and not candidate["success"])
        scope_regressions += int(
            bool(candidate.get("forbidden_changes"))
            or bool(candidate.get("missing_expected_changes"))
        )

        baseline_artifact = json.loads(Path(baseline["run_artifact"]).read_text())
        candidate_artifact = json.loads(Path(candidate["run_artifact"]).read_text())
        baseline_prompt = _first_composition(baseline_artifact)
        candidate_prompt = _first_composition(candidate_artifact)
        observed_fragments = list(candidate_prompt.get("conditional_fragment_ids", ()))

        if baseline_prompt.get("conditional_fragment_ids"):
            isolation_failures.append(f"{task_id}:baseline-has-fragment")
        if baseline_prompt.get("user_chars") != candidate_prompt.get("user_chars"):
            isolation_failures.append(f"{task_id}:user-prompt-differs")
        if baseline_prompt.get("tool_schema_chars") != candidate_prompt.get("tool_schema_chars"):
            isolation_failures.append(f"{task_id}:tool-schema-differs")
        stable_delta = abs(
            int(candidate_prompt.get("stable_system_chars", 0))
            - int(baseline_prompt.get("stable_system_chars", 0))
        )
        if stable_delta > int(gates_spec["stable_system_char_delta_max"]):
            isolation_failures.append(f"{task_id}:stable-system-differs")
        if observed_fragments != list(expected_fragments[task_id]):
            fragment_failures.append(task_id)

        per_task[task_id] = {
            "baseline_success": bool(baseline["success"]),
            "candidate_success": bool(candidate["success"]),
            "candidate_fragments": observed_fragments,
            "stable_system_char_delta": stable_delta,
            "prompt_token_ratio": candidate["prompt_tokens"] / max(
                baseline["prompt_tokens"], 1
            ),
            "completion_token_ratio": candidate["completion_tokens"] / max(
                baseline["completion_tokens"], 1
            ),
            "tool_call_ratio": candidate["tool_calls"] / max(baseline["tool_calls"], 1),
            "latency_ratio": candidate["latency_seconds"] / max(
                baseline["latency_seconds"], 1e-9
            ),
        }

    meaningful = sum(
        ratio <= float(gates_spec["meaningful_reduction_ratio_max"])
        for ratio in ratios.values()
    )
    gates = {
        "success_regressions": success_regressions <= gates_spec["success_regressions"],
        "scope_regressions": scope_regressions <= gates_spec["scope_regressions"],
        "prompt_only_isolation": not isolation_failures,
        "expected_fragments": not fragment_failures,
        "aggregate_prompt_tokens": ratios["prompt_tokens"] <= gates_spec[
            "aggregate_prompt_token_ratio_max"
        ],
        "aggregate_completion_tokens": ratios["completion_tokens"] <= gates_spec[
            "aggregate_completion_token_ratio_max"
        ],
        "aggregate_tool_calls": ratios["tool_calls"] <= gates_spec[
            "aggregate_tool_call_ratio_max"
        ],
        "aggregate_latency": ratios["latency_seconds"] <= gates_spec[
            "aggregate_latency_ratio_max"
        ],
        "measurable_improvement": meaningful >= gates_spec[
            "minimum_meaningfully_improved_metrics"
        ],
    }
    return {
        "schema_version": 1,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "aggregate": aggregate,
        "ratios": ratios,
        "reductions": {metric: 1.0 - ratio for metric, ratio in ratios.items()},
        "meaningfully_improved_metrics": meaningful,
        "success_regressions": success_regressions,
        "scope_regressions": scope_regressions,
        "isolation_failures": isolation_failures,
        "fragment_failures": fragment_failures,
        "per_task": per_task,
        "interpretation_boundary": specification["interpretation_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gates", type=Path)
    parser.add_argument("observations", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        evaluate_task_policy_improvement(args.observations, args.gates),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
