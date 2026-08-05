from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bench.agent_task_eval import (
    AgentTaskObservation,
    build_report,
    file_sha256,
    load_condition_manifest,
    load_tasks,
)


def _observation(task_id: str, condition: str, condition_hash: str) -> AgentTaskObservation:
    return AgentTaskObservation(
        task_id=task_id,
        condition=condition,
        repetition=0,
        provider="anthropic",
        model="claude-test",
        model_identity="anthropic:claude-test@revision",
        dataset_sha256="dataset",
        condition_manifest_sha256="manifest",
        condition_sha256=condition_hash,
        success=True,
        verify_exit_code=0,
        engine_status="complete",
        changed_paths=("PLAN.md",),
        forbidden_changes=(),
        missing_expected_changes=(),
        final_pattern_checks={"plan": True},
        action_pattern_checks={},
        prompt_tokens=100,
        completion_tokens=20,
        latency_seconds=2.0,
        tool_calls=3,
    )


def test_checked_in_pilot_has_six_distinct_categories_and_reviewed_rubrics() -> None:
    tasks = load_tasks(Path("bench/agent_task_pilot.tasks.jsonl"))
    assert len(tasks) == 6
    assert len({task.category for task in tasks}) == 6
    assert all(task.review_status == "draft" for task in tasks)
    assert all(any(item.kind == "deterministic" for item in task.rubric) for task in tasks)
    assert all(any(item.kind == "human_review" for item in task.rubric) for task in tasks)


def test_condition_manifest_is_provider_neutral_and_hash_bound(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(Path("bench/agent_task_pilot.tasks.jsonl").read_text(), encoding="utf-8")
    profile = {
        "schema_version": 1,
        "provenance": "explicit_user",
        "name": "control",
        "description": "Prefer visible control.",
        "weights": {"user_control": 1.0},
    }
    manifest = {
        "schema_version": 1,
        "dataset_sha256": file_sha256(tasks),
        "utility_profile": profile,
        "conditions": {"baseline": None, "candidate": {"system_prompt": "Prefer checkpoints."}},
    }
    path = tmp_path / "conditions.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _, hashes, loaded = load_condition_manifest(path, dataset_sha256=file_sha256(tasks))
    assert set(hashes) == {"baseline", "candidate"}
    assert loaded["utility_profile"]["name"] == "control"
    assert "provider" not in loaded


def test_report_requires_complete_pairs_and_retains_route() -> None:
    task = load_tasks(Path("bench/agent_task_pilot.tasks.jsonl"))[0]
    task = replace(task, review_status="approved")
    rows = [_observation(task.id, "baseline", "base"), _observation(task.id, "candidate", "cand")]
    report = build_report(
        [task],
        rows,
        dataset_sha256="dataset",
        condition_manifest_sha256="manifest",
        expected_condition_hashes={"baseline": "base", "candidate": "cand"},
    )
    assert report["provider"] == "anthropic"
    assert report["task_count"] == 1
    assert report["paired_outcomes"] == {
        "candidate_improvements": 0,
        "candidate_regressions": 0,
        "unchanged_success": 1,
    }

    with pytest.raises(ValueError, match="baseline and candidate"):
        build_report(
            [task],
            rows[:1],
            dataset_sha256="dataset",
            condition_manifest_sha256="manifest",
            expected_condition_hashes={"baseline": "base", "candidate": "cand"},
        )


def test_observation_rejects_success_when_a_deterministic_check_failed() -> None:
    value = {
        **_observation("p", "baseline", hashlib.sha256(b"x").hexdigest()).__dict__,
        "forbidden_changes": ["tests/test_x.py"],
        "changed_paths": ["tests/test_x.py"],
        "missing_expected_changes": [],
        "final_pattern_checks": {"done": True},
        "action_pattern_checks": {},
    }
    with pytest.raises(ValueError, match="success must match"):
        AgentTaskObservation.from_dict(value)
