from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bench.context_delivery_eval import (
    ContextObservation,
    ContextTask,
    build_report,
    file_sha256,
    load_condition_manifest,
    load_tasks,
    render_markdown,
    validate_family_atomic_splits,
)


def _tasks() -> list[ContextTask]:
    return [
        ContextTask(
            id="task-1",
            family="migration-placement",
            split="validation",
            repository_fixture="fixtures/migration",
            request="Repair rollback handling.",
            verify_command="pytest -q",
            required_evidence=("src/migrate.py:rollback", "tests/test_migrate.py"),
            relevant_evidence_position="middle",
            review_status="approved",
        )
    ]


def _row(condition: str, *, success: bool, repetition: int = 0) -> ContextObservation:
    return ContextObservation(
        task_id="task-1",
        condition=condition,
        repetition=repetition,
        model_identity="provider/model@revision",
        dataset_sha256="dataset",
        condition_manifest_sha256="manifest",
        condition_sha256=f"hash-{condition}",
        success=success,
        verify_exit_code=0 if success else 1,
        prompt_tokens={"baseline": 100, "ranked": 180, "full": 1000}[condition],
        completion_tokens=20,
        latency_seconds=1.0,
        tool_calls=2,
        context_items=(
            ("src/migrate.py:rollback", "tests/test_migrate.py")
            if condition != "baseline"
            else ()
        ),
        run_artifact=f"runs/{condition}.json",
    )


def test_report_preserves_tasks_and_compares_paired_success() -> None:
    report = build_report(
        _tasks(),
        [_row("baseline", success=False), _row("ranked", success=True), _row("full", success=True)],
        dataset_sha256="dataset",
        condition_manifest_sha256="manifest",
        expected_condition_hashes={name: f"hash-{name}" for name in ("baseline", "ranked", "full")},
    )

    assert report["conditions"]["ranked"]["verified_successes"] == 1
    assert report["conditions"]["ranked"]["mean_evidence_recall"] == 1.0
    assert report["paired_vs_baseline"]["ranked"]["success_wins"] == 1
    assert report["task_records"][0]["conditions"]["baseline"][
        "required_evidence_omitted"
    ] == ["src/migrate.py:rollback", "tests/test_migrate.py"]
    markdown = render_markdown(report)
    assert "Repair rollback handling" in markdown
    assert "runs/ranked.json" in markdown


def test_rejects_incomplete_or_duplicate_condition_pairs() -> None:
    with pytest.raises(ValueError, match="baseline, ranked, and full"):
        build_report(
            _tasks(),
            [_row("baseline", success=False), _row("ranked", success=True)],
            dataset_sha256="dataset",
            condition_manifest_sha256="manifest",
            expected_condition_hashes={name: f"hash-{name}" for name in ("baseline", "ranked", "full")},
        )
    with pytest.raises(ValueError, match="duplicate context observation"):
        build_report(
            _tasks(),
            [
                _row("baseline", success=False),
                _row("baseline", success=False),
                _row("ranked", success=True),
                _row("full", success=True),
            ],
            dataset_sha256="dataset",
            condition_manifest_sha256="manifest",
            expected_condition_hashes={name: f"hash-{name}" for name in ("baseline", "ranked", "full")},
        )


def test_rejects_family_leakage_and_dataset_drift() -> None:
    leaked = [
        _tasks()[0],
        ContextTask(
            id="task-2",
            family="migration-placement",
            split="calibration",
            repository_fixture="fixtures/migration",
            request="Same family variant.",
            verify_command="pytest -q",
            required_evidence=("src/migrate.py",),
        ),
    ]
    with pytest.raises(ValueError, match="cross calibration/validation"):
        validate_family_atomic_splits(leaked)
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        build_report(
            _tasks(),
            [_row("baseline", success=False), _row("ranked", success=True), _row("full", success=True)],
            dataset_sha256="different",
            condition_manifest_sha256="manifest",
            expected_condition_hashes={name: f"hash-{name}" for name in ("baseline", "ranked", "full")},
        )


def test_observation_success_must_match_deterministic_verifier() -> None:
    with pytest.raises(ValueError, match="success must match"):
        ContextObservation.from_dict(
            {
                "task_id": "task-1",
                "condition": "ranked",
                "repetition": 0,
                "model_identity": "provider/model@revision",
                "dataset_sha256": "dataset",
                "condition_manifest_sha256": "manifest",
                "condition_sha256": "hash",
                "success": True,
                "verify_exit_code": 1,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "latency_seconds": 1,
                "tool_calls": 1,
                "context_items": [],
            }
        )
    with pytest.raises(ValueError, match="success must be boolean"):
        ContextObservation.from_dict(
            {
                "task_id": "task-1",
                "condition": "ranked",
                "repetition": 0,
                "model_identity": "provider/model@revision",
                "dataset_sha256": "dataset",
                "condition_manifest_sha256": "manifest",
                "condition_sha256": "hash",
                "success": "false",
                "verify_exit_code": 1,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "latency_seconds": 1,
                "tool_calls": 1,
                "context_items": [],
            }
        )


def test_condition_manifest_binds_each_treatment_and_dataset(tmp_path) -> None:
    path = tmp_path / "conditions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_sha256": "dataset",
                "conditions": {
                    name: {"context_source": name, "description": f"{name} treatment"}
                    for name in ("baseline", "ranked", "full")
                },
            }
        )
    )

    manifest_hash, hashes = load_condition_manifest(path, dataset_sha256="dataset")

    assert len(manifest_hash) == 64
    assert set(hashes) == {"baseline", "ranked", "full"}
    assert len(set(hashes.values())) == 3
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        load_condition_manifest(path, dataset_sha256="changed")


def test_checked_in_examples_are_hash_bound_and_family_atomic() -> None:
    tasks_path = Path("bench/context_delivery_tasks.example.jsonl")
    conditions_path = Path("bench/context_delivery_conditions.example.json")
    tasks = load_tasks(tasks_path)

    validate_family_atomic_splits(tasks)
    manifest_hash, condition_hashes = load_condition_manifest(
        conditions_path, dataset_sha256=file_sha256(tasks_path)
    )

    assert {task.relevant_evidence_position for task in tasks} == {
        "front", "middle", "end"
    }
    assert len(manifest_hash) == 64
    assert set(condition_hashes) == {"baseline", "ranked", "full"}


def test_report_includes_identity_checked_action_level_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "ranked.json"
    artifact.write_text(
        json.dumps(
            {
                "dataset_sha256": "dataset",
                "condition_manifest_sha256": "manifest",
                "condition_sha256": "hash-ranked",
                "condition": "ranked",
                "repetition": 0,
                "model_identity": "provider/model@revision",
                "task": {"id": "task-1"},
                "engine_status": "done",
                "final_answer": "Reversed the compensating operations.",
                "plan_steps": [{"title": "Repair rollback", "status": "done"}],
                "action_records": [{"summary": "Inspected and repaired rollback."}],
                "changed_files_summary": "### src/migrate.py\n```diff\n+ reversed\n```",
                "file_change_reasons": {"src/migrate.py": ["reverse rollback"]},
                "verify_stdout": "1 passed",
                "verify_stderr": "",
                "prompt_composition_history": [
                    {"iteration": 0, "user_sections": {"task": 20, "plan": 40}}
                ],
                "request_payload_history": [
                    {
                        "request_payload_chars": 900,
                        "message_count": 4,
                        "message_content_chars_by_role": {"tool": 500, "user": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        _row("baseline", success=False),
        replace(_row("ranked", success=True), run_artifact=str(artifact)),
        _row("full", success=True),
    ]

    report = build_report(
        _tasks(), rows,
        dataset_sha256="dataset",
        condition_manifest_sha256="manifest",
        expected_condition_hashes={name: f"hash-{name}" for name in ("baseline", "ranked", "full")},
    )

    details = report["task_records"][0]["conditions"]["ranked"]["qualitative_artifact"]
    assert details["available"] is True
    assert details["final_answer"] == "Reversed the compensating operations."
    markdown = render_markdown(report)
    assert "Inspected and repaired rollback" in markdown
    assert "Reversed the compensating operations" in markdown
    assert "+ reversed" in markdown
    assert "Largest dispatched request: 900 chars" in markdown
    assert "('plan', 40)" in markdown
