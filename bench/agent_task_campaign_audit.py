#!/usr/bin/env python3
"""Audit a frozen multi-model agent-task campaign without contacting providers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.agent_task_eval import file_sha256, load_condition_manifest, load_tasks
from bench.agent_task_run import AgentTaskRunConfig


def audit_campaign(
    tasks_path: Path,
    routes: list[tuple[Path, Path]],
) -> dict[str, object]:
    """Validate task, condition, route, pacing, and call-count invariants."""
    tasks = load_tasks(tasks_path)
    dataset_sha = file_sha256(tasks_path)
    task_ids = [task.id for task in tasks if task.split == "validation"]
    records: list[dict[str, object]] = []
    for conditions_path, config_path in routes:
        config = AgentTaskRunConfig.from_path(config_path)
        _, condition_hashes, manifest = load_condition_manifest(
            conditions_path, dataset_sha256=dataset_sha
        )
        contract = manifest.get("execution_contract")
        if not isinstance(contract, dict):
            raise ValueError(f"execution contract is missing: {conditions_path}")
        checks = {
            "model_identity_matches": manifest.get("model_identity") == config.model_identity,
            "task_ids_match": manifest.get("task_ids") == task_ids,
            "two_conditions": set(condition_hashes) == {"baseline", "candidate"},
            "one_repetition": config.repetitions == 1 and manifest.get("repetitions") == 1,
            "sequential": contract.get("parallel_requests") is False,
            "isolated_workspace": contract.get("fresh_workspace_per_execution") is True,
            "isolated_session": contract.get("fresh_agent_session_per_execution") is True,
            "paced_at_least_two_seconds": (
                config.min_request_interval_seconds >= 2.0
                and float(contract.get("minimum_llm_request_interval_seconds", 0)) >= 2.0
            ),
            "no_retries": contract.get("automatic_llm_retries") is False,
            "stop_on_error": contract.get("stop_on_first_runtime_or_provider_error") is True,
        }
        planned = len(task_ids) * len(condition_hashes) * config.repetitions
        checks["planned_count_matches"] = manifest.get("planned_executions") == planned
        records.append(
            {
                "provider": config.provider,
                "model": config.model,
                "model_identity": config.model_identity,
                "conditions_path": str(conditions_path),
                "config_path": str(config_path),
                "planned_executions": planned,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_sha,
        "task_count": len(task_ids),
        "route_count": len(records),
        "planned_executions": sum(int(record["planned_executions"]) for record in records),
        "all_passed": bool(records) and all(bool(record["passed"]) for record in records),
        "records": records,
        "execution_authorized": False,
        "remaining_gates": [
            "Independent candidate-blind approval of all six tasks and human rubrics.",
            "Regenerate manifests against the approved dataset bytes.",
            "Explicit authorization immediately before provider-backed execution.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--route", nargs=2, action="append", metavar=("CONDITIONS", "CONFIG"), required=True
    )
    args = parser.parse_args()
    report = audit_campaign(
        args.tasks,
        [(Path(conditions), Path(config)) for conditions, config in args.route],
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
