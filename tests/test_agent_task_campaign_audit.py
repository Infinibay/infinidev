from __future__ import annotations

from pathlib import Path

from bench.agent_task_campaign_audit import audit_campaign


def test_checked_in_campaign_is_exactly_36_safe_sequential_executions() -> None:
    routes = [
        (
            Path(f"bench/agent_task_pilot.{name}.conditions.json"),
            Path(f"bench/agent_task_run.gpt-5.6-{name}.json"),
        )
        for name in ("sol", "terra", "luna")
    ]
    report = audit_campaign(Path("bench/agent_task_pilot.tasks.jsonl"), routes)
    assert report["all_passed"] is True
    assert report["task_count"] == 6
    assert report["route_count"] == 3
    assert report["planned_executions"] == 36
    assert report["execution_authorized"] is False
