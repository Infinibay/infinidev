from __future__ import annotations

from pathlib import Path

from bench.agent_task_completion_audit import audit_completed_campaign


def test_checked_in_pilot_has_all_36_runs_reviews_and_decisions() -> None:
    report = audit_completed_campaign(
        Path("bench/agent_task_pilot.approved.jsonl"),
        Path("bench/runs/20260804-agent-task-pilot"),
    )

    assert report["all_passed"] is True
    assert report["evidence_counts"]["observations"] == 36
    assert report["evidence_counts"]["run_artifacts"] == 36
    assert len(report["routes"]) == 3
    assert all(route["passed"] for route in report["routes"])
