from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.agent_task_eval import file_sha256
from bench.agent_task_multi_run import CampaignRoute, validate_campaign


def _approved_campaign(tmp_path: Path) -> tuple[Path, list[CampaignRoute]]:
    tasks = tmp_path / "tasks.approved.jsonl"
    tasks.write_text(
        Path("bench/agent_task_pilot.tasks.jsonl")
        .read_text(encoding="utf-8")
        .replace('"review_status":"draft"', '"review_status":"approved"'),
        encoding="utf-8",
    )
    routes = []
    for name in ("sol", "terra", "luna"):
        source = Path(f"bench/agent_task_pilot.{name}.conditions.json")
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["dataset_sha256"] = file_sha256(tasks)
        conditions = tmp_path / f"{name}.conditions.json"
        conditions.write_text(json.dumps(manifest), encoding="utf-8")
        routes.append(
            CampaignRoute(
                name,
                conditions,
                Path(f"bench/agent_task_run.gpt-5.6-{name}.json"),
            )
        )
    return tasks, routes


def test_multi_campaign_requires_reviewed_tasks() -> None:
    routes = [
        CampaignRoute(
            name,
            Path(f"bench/agent_task_pilot.{name}.conditions.json"),
            Path(f"bench/agent_task_run.gpt-5.6-{name}.json"),
        )
        for name in ("sol", "terra", "luna")
    ]
    with pytest.raises(ValueError, match="six approved"):
        validate_campaign(Path("bench/agent_task_pilot.tasks.jsonl"), routes, Path("unused"))


def test_multi_campaign_freezes_exactly_36_executions(tmp_path: Path) -> None:
    tasks, routes = _approved_campaign(tmp_path)
    plan = validate_campaign(tasks, routes, tmp_path / "new-output")
    assert plan["task_count"] == 6
    assert plan["planned_executions"] == 36
    assert plan["parallel_requests"] is False
    assert len(plan["routes"]) == 3


def test_multi_campaign_refuses_nonempty_output(tmp_path: Path) -> None:
    tasks, routes = _approved_campaign(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "partial.jsonl").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        validate_campaign(tasks, routes, output)
