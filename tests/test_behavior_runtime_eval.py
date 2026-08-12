"""Hard-gate tests for paired runtime behavior campaigns."""

from __future__ import annotations

import json
from pathlib import Path

from bench.behavior_runtime_eval import evaluate_runtime_campaign


def test_runtime_eval_applies_quality_scope_and_cost_gates(tmp_path: Path) -> None:
    task_ids = ["a", "b"]
    manifest = {
        "task_ids": task_ids,
        "deployment_gates": {
            "success_regressions": 0,
            "scope_regressions": 0,
            "aggregate_prompt_token_ratio_max": 1.0,
            "aggregate_completion_token_ratio_max": 1.0,
            "aggregate_tool_call_ratio_max": 1.0,
            "per_task_tool_call_increase_max": 1,
            "aggregate_latency_ratio_max": 1.1,
            "detector_false_intervention_rate_max": 0.01,
        },
        "interpretation_boundary": "exploratory",
    }
    conditions = tmp_path / "conditions.json"
    conditions.write_text(json.dumps(manifest), encoding="utf-8")
    observations = tmp_path / "observations.jsonl"
    rows = []
    for task_id in task_ids:
        artifact = tmp_path / f"{task_id}.json"
        artifact.write_text(json.dumps({
            "task": {"expected_changed_paths": ["source.py"]},
            "runtime_interventions_given": ["excessive_discovery"],
            "runtime_behavior_events": [
                {"label": "excessive_discovery", "intervention_queued": True}
            ],
        }), encoding="utf-8")
        common = {
            "task_id": task_id,
            "success": True,
            "changed_paths": ["source.py", "src/pkg.egg-info/PKG-INFO"],
            "completion_tokens": 50,
            "latency_seconds": 10,
            "tool_calls": 10,
            "run_artifact": str(artifact),
        }
        rows.extend((
            {**common, "condition": "baseline", "prompt_tokens": 100},
            {**common, "condition": "candidate", "prompt_tokens": 80},
        ))
    observations.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "reviews": [
            {
                "task_id": task_id,
                "condition": "candidate",
                "label": "excessive_discovery",
                "verdict": "true_positive",
            }
            for task_id in task_ids
        ]
    }), encoding="utf-8")

    report = evaluate_runtime_campaign([observations], conditions, review)

    assert report["all_gates_pass"]
    assert report["ratios"]["prompt_tokens"] == 0.8
    assert report["scope_regressions"] == 0


def test_runtime_eval_refuses_to_pass_unreviewed_interventions(tmp_path: Path) -> None:
    conditions = tmp_path / "conditions.json"
    conditions.write_text(json.dumps({
        "task_ids": ["a"],
        "deployment_gates": {
            "success_regressions": 0,
            "scope_regressions": 0,
            "aggregate_prompt_token_ratio_max": 1.0,
            "aggregate_completion_token_ratio_max": 1.0,
            "aggregate_tool_call_ratio_max": 1.0,
            "per_task_tool_call_increase_max": 1,
            "aggregate_latency_ratio_max": 1.1,
            "detector_false_intervention_rate_max": 0.01,
        },
        "interpretation_boundary": "exploratory",
    }), encoding="utf-8")
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "task": {"expected_changed_paths": ["a.py"]},
        "runtime_interventions_given": ["excessive_discovery"],
        "runtime_behavior_events": [
            {"label": "excessive_discovery", "intervention_queued": True}
        ],
    }), encoding="utf-8")
    observations = tmp_path / "observations.jsonl"
    common = {
        "task_id": "a", "success": True, "changed_paths": ["a.py"],
        "prompt_tokens": 1, "completion_tokens": 1, "tool_calls": 1,
        "latency_seconds": 1, "run_artifact": str(artifact),
    }
    observations.write_text(
        json.dumps({**common, "condition": "baseline"}) + "\n"
        + json.dumps({**common, "condition": "candidate"}) + "\n",
        encoding="utf-8",
    )

    report = evaluate_runtime_campaign([observations], conditions)

    assert not report["all_gates_pass"]
    assert report["unreviewed_interventions"] == 1
