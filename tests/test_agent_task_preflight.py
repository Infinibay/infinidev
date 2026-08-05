from __future__ import annotations

from pathlib import Path

from bench.agent_task_preflight import build_preflight


def test_all_pilot_verifiers_have_negative_and_positive_controls() -> None:
    report = build_preflight(
        Path("bench/agent_task_pilot.tasks.jsonl"),
        Path("bench/agent_task_fixtures"),
        Path("bench/agent_task_reference_solutions"),
    )
    assert report["task_count"] == 6
    assert report["all_passed"] is True
    assert all(record["pristine_verify_exit_code"] != 0 for record in report["records"])
    assert all(record["reference_verify_exit_code"] == 0 for record in report["records"])
