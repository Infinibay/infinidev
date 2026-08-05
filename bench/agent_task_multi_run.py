#!/usr/bin/env python3
"""Run a reviewed multi-model agent-task pilot under one global sequential lock."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from bench.agent_task_eval import (
    build_report,
    file_sha256,
    load_condition_manifest,
    load_observations,
    load_tasks,
    render_markdown,
)
from bench.agent_task_run import AgentTaskRunConfig, run_campaign


@dataclass(frozen=True)
class CampaignRoute:
    """One named model route and its model-specific condition manifest."""

    name: str
    conditions_path: Path
    config_path: Path


def validate_campaign(
    tasks_path: Path,
    routes: list[CampaignRoute],
    output_root: Path,
) -> dict[str, object]:
    """Fail before provider calls unless the complete campaign is immutable and approved."""
    tasks = [task for task in load_tasks(tasks_path) if task.split == "validation"]
    if len(tasks) != 6 or any(task.review_status != "approved" for task in tasks):
        raise ValueError("multi-model pilot requires exactly six approved validation tasks")
    if len(routes) != 3 or len({route.name for route in routes}) != 3:
        raise ValueError("multi-model pilot requires exactly three uniquely named routes")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("multi-model pilot output root must be new or empty")
    dataset_sha = file_sha256(tasks_path)
    records = []
    identities: set[str] = set()
    for route in routes:
        config = AgentTaskRunConfig.from_path(route.config_path)
        _, hashes, manifest = load_condition_manifest(
            route.conditions_path, dataset_sha256=dataset_sha
        )
        if manifest.get("model_identity") != config.model_identity:
            raise ValueError(f"manifest/config identity mismatch: {route.name}")
        if manifest.get("task_ids") != [task.id for task in tasks]:
            raise ValueError(f"manifest task order mismatch: {route.name}")
        if config.repetitions != 1 or manifest.get("repetitions") != 1:
            raise ValueError(f"pilot route must use one repetition: {route.name}")
        contract = manifest.get("execution_contract")
        if not isinstance(contract, dict) or not all(
            (
                contract.get("fresh_workspace_per_execution") is True,
                contract.get("fresh_agent_session_per_execution") is True,
                contract.get("parallel_requests") is False,
                float(contract.get("minimum_llm_request_interval_seconds", 0)) >= 2.0,
                contract.get("automatic_llm_retries") is False,
                contract.get("stop_on_first_runtime_or_provider_error") is True,
            )
        ):
            raise ValueError(f"pilot execution contract is unsafe: {route.name}")
        if config.model_identity in identities:
            raise ValueError(f"duplicate model identity: {config.model_identity}")
        identities.add(config.model_identity)
        records.append(
            {
                "name": route.name,
                "provider": config.provider,
                "model": config.model,
                "model_identity": config.model_identity,
                "conditions_path": str(route.conditions_path),
                "conditions_sha256": file_sha256(route.conditions_path),
                "config_path": str(route.config_path),
                "config_sha256": file_sha256(route.config_path),
                "planned_executions": len(tasks) * len(hashes) * config.repetitions,
            }
        )
    planned = sum(int(record["planned_executions"]) for record in records)
    if planned != 36:
        raise ValueError(f"multi-model pilot must contain exactly 36 executions, got {planned}")
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_sha,
        "task_count": len(tasks),
        "planned_executions": planned,
        "parallel_requests": False,
        "routes": records,
    }


def run_multi_campaign(
    tasks_path: Path,
    routes: list[CampaignRoute],
    output_root: Path,
    *,
    fixture_root: Path,
) -> None:
    """Execute all routes serially, preserving completed evidence on the first failure."""
    from infinidev.engine.subscription_safety import subscription_single_flight

    plan = validate_campaign(tasks_path, routes, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "campaign-plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tasks = load_tasks(tasks_path)
    dataset_sha = file_sha256(tasks_path)
    with subscription_single_flight():
        for route in routes:
            route_root = output_root / route.name
            route_root.mkdir()
            observations = route_root / "observations.jsonl"
            artifacts = route_root / "artifacts"
            run_campaign(
                tasks_path,
                route.conditions_path,
                route.config_path,
                observations,
                artifacts,
                fixture_root=fixture_root,
                split="validation",
                include_drafts=False,
                acquire_global_lock=False,
            )
            manifest_sha, hashes, _ = load_condition_manifest(
                route.conditions_path, dataset_sha256=dataset_sha
            )
            report = build_report(
                tasks,
                load_observations(observations),
                dataset_sha256=dataset_sha,
                condition_manifest_sha256=manifest_sha,
                expected_condition_hashes=hashes,
            )
            (route_root / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (route_root / "report.md").write_text(render_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--fixture-root", type=Path, default=Path("bench/agent_task_fixtures"))
    parser.add_argument(
        "--route",
        nargs=3,
        action="append",
        metavar=("NAME", "CONDITIONS", "CONFIG"),
        required=True,
    )
    args = parser.parse_args()
    routes = [
        CampaignRoute(name, Path(conditions), Path(config))
        for name, conditions, config in args.route
    ]
    run_multi_campaign(
        args.tasks,
        routes,
        args.output_root,
        fixture_root=args.fixture_root,
    )


if __name__ == "__main__":
    main()
