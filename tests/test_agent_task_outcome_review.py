from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.agent_task_outcome_review import (
    build_outcome_report,
    export_blind_packet,
    render_review_template,
)


def _source_report(tmp_path: Path) -> Path:
    task_records = []
    for index, task_id in enumerate(("plan", "review")):
        rows = {}
        for condition in ("baseline", "candidate"):
            artifact_dir = tmp_path / f"{task_id}.{condition}"
            workspace = artifact_dir / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "RESULT.md").write_text(f"{task_id} evidence", encoding="utf-8")
            artifact = {
                "final_answer": f"Completed {task_id}",
                "engine_status": "complete",
                "action_records": [{"tool": "read_file"}],
                "changed_paths": ["RESULT.md"],
                "forbidden_changes": [],
                "missing_expected_changes": [],
                "verify_exit_code": 0,
                "verify_stdout": "ok",
                "verify_stderr": "",
                "error": "",
                "run_config": {"provider": "anthropic"},
                "prompt_composition_history": [f"secret {condition} guidance"],
            }
            artifact_path = artifact_dir / "run.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            rows[condition] = {
                "success": True,
                "run_artifact": str(artifact_path),
                "tool_calls": 2,
                "latency_seconds": 2.0,
            }
        task_records.append(
            {
                "task": {
                    "id": task_id,
                    "category": task_id,
                    "request": f"Do {task_id}",
                    "rubric": [
                        {
                            "id": "quality",
                            "description": "Evidence is useful.",
                            "kind": "human_review",
                            "evidence_source": "RESULT.md",
                            "weight": 1.0,
                        }
                    ],
                },
                "repetition": 0,
                "baseline": rows["baseline"],
                "candidate": rows["candidate"],
                "success_delta": 0,
                "candidate_changed_behavior": True,
            }
        )
    report = {
        "paired_outcomes": {
            "candidate_improvements": 0,
            "candidate_regressions": 0,
            "unchanged_success": 2,
        },
        "conditions": {
            "baseline": {
                "attempted": 2,
                "verified_successes": 2,
                "errors": 0,
                "unauthorized_or_forbidden_change_runs": 0,
                "mean_latency_seconds": 2.0,
                "mean_tool_calls": 2.0,
            },
            "candidate": {
                "attempted": 2,
                "verified_successes": 2,
                "errors": 0,
                "unauthorized_or_forbidden_change_runs": 0,
                "mean_latency_seconds": 2.1,
                "mean_tool_calls": 2.0,
            },
        },
        "task_records": task_records,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _reviews(
    packet: dict[str, object], key: dict[str, object], *, candidate_score: str
) -> list[dict[str, object]]:
    mappings = {
        (item["task_id"], item["repetition"]): item["mapping"] for item in key["records"]
    }
    rows = []
    for record in packet["records"]:
        mapping = mappings[(record["task_id"], record["repetition"])]
        for label in ("A", "B"):
            rows.append(
                {
                    "packet_sha256": key["packet_sha256"],
                    "task_id": record["task_id"],
                    "repetition": record["repetition"],
                    "variant": label,
                    "rubric_id": "quality",
                    "score": candidate_score if mapping[label] == "candidate" else "not_met",
                    "reviewer_identity": "blind-reviewer",
                    "rationale": "The preserved evidence directly supports this score.",
                }
            )
    return rows


def test_export_packet_separates_condition_key_and_excludes_prompt_payload(tmp_path: Path) -> None:
    report_path = _source_report(tmp_path)
    packet, key = export_blind_packet(report_path)
    rendered = json.dumps(packet)
    assert packet["candidate_blind"] is True
    assert "secret baseline guidance" not in rendered
    assert "secret candidate guidance" not in rendered
    assert '"mapping"' not in rendered
    assert all(
        set(item["mapping"].values()) == {"baseline", "candidate"}
        for item in key["records"]
    )
    template = render_review_template(packet, key["packet_sha256"])
    assert len(template.splitlines()) == 4
    assert "REPLACE_WITH_met_not_met_or_unclear" in template


def test_two_domain_preference_gain_advances_only_to_larger_calibration(tmp_path: Path) -> None:
    report_path = _source_report(tmp_path)
    source = json.loads(report_path.read_text())
    packet, key = export_blind_packet(report_path)
    result = build_outcome_report(
        source,
        packet,
        key,
        _reviews(packet, key, candidate_score="met"),
        source_report_sha256=packet["source_report_sha256"],
    )
    assert result["decision"] == "advance_to_larger_calibration"
    assert result["deployment_authorized"] is False
    assert result["human_preference_delta"] == 1.0


def test_incomplete_review_and_competence_regression_fail_closed(tmp_path: Path) -> None:
    report_path = _source_report(tmp_path)
    source = json.loads(report_path.read_text())
    packet, key = export_blind_packet(report_path)
    reviews = _reviews(packet, key, candidate_score="met")
    with pytest.raises(ValueError, match="incomplete"):
        build_outcome_report(
            source,
            packet,
            key,
            reviews[:-1],
            source_report_sha256=packet["source_report_sha256"],
        )

    source["paired_outcomes"]["candidate_regressions"] = 1
    result = build_outcome_report(
        source,
        packet,
        key,
        reviews,
        source_report_sha256=packet["source_report_sha256"],
    )
    assert result["decision"] == "discard_competence_regression"
