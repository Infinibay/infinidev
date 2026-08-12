"""Hard-gate coverage for prompt-only task-policy improvement campaigns."""

from __future__ import annotations

import json
from pathlib import Path

from bench.task_policy_improvement_eval import evaluate_task_policy_improvement


def _write_pair(tmp_path: Path, *, candidate_prompt_tokens: int = 70) -> tuple[Path, Path]:
    artifacts = {}
    for condition, fragments, dynamic in (
        ("baseline", [], 0),
        ("candidate", ["refactor.developer"], 300),
    ):
        path = tmp_path / f"{condition}.json"
        path.write_text(json.dumps({
            "prompt_composition_history": [{
                "stable_system_chars": 1000 + (2 if condition == "candidate" else 0),
                "dynamic_system_chars": dynamic,
                "conditional_fragment_ids": fragments,
                "user_chars": 500,
                "tool_schema_chars": 700,
            }],
        }), encoding="utf-8")
        artifacts[condition] = path

    observations = tmp_path / "observations.jsonl"
    common = {
        "task_id": "task-a",
        "dataset_sha256": "dataset",
        "condition_manifest_sha256": "manifest",
        "provider": "provider",
        "model_identity": "provider:model:v1",
        "success": True,
        "forbidden_changes": [],
        "missing_expected_changes": [],
        "completion_tokens": 50,
        "tool_calls": 10,
        "latency_seconds": 10,
    }
    rows = [
        {
            **common,
            "condition": "baseline",
            "prompt_tokens": 100,
            "run_artifact": str(artifacts["baseline"]),
        },
        {
            **common,
            "condition": "candidate",
            "prompt_tokens": candidate_prompt_tokens,
            "completion_tokens": 40,
            "tool_calls": 8,
            "latency_seconds": 9,
            "run_artifact": str(artifacts["candidate"]),
        },
    ]
    observations.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({
        "task_ids": ["task-a"],
        "expected_candidate_fragments": {"task-a": ["refactor.developer"]},
        "gates": {
            "success_regressions": 0,
            "scope_regressions": 0,
            "stable_system_char_delta_max": 2,
            "aggregate_prompt_token_ratio_max": 1.0,
            "aggregate_completion_token_ratio_max": 1.0,
            "aggregate_tool_call_ratio_max": 1.0,
            "aggregate_latency_ratio_max": 1.1,
            "meaningful_reduction_ratio_max": 0.95,
            "minimum_meaningfully_improved_metrics": 2,
        },
        "interpretation_boundary": "test",
    }), encoding="utf-8")
    return observations, gates


def test_improvement_eval_passes_isolated_quality_and_cost_win(tmp_path: Path) -> None:
    observations, gates = _write_pair(tmp_path)

    report = evaluate_task_policy_improvement([observations], gates)

    assert report["all_gates_pass"]
    assert report["gates"]["prompt_only_isolation"]
    assert report["ratios"]["prompt_tokens"] == 0.7
    assert report["meaningfully_improved_metrics"] == 4


def test_improvement_eval_rejects_cost_only_replacement(tmp_path: Path) -> None:
    observations, gates = _write_pair(tmp_path, candidate_prompt_tokens=110)

    report = evaluate_task_policy_improvement([observations], gates)

    assert not report["all_gates_pass"]
    assert not report["gates"]["aggregate_prompt_tokens"]
