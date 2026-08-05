#!/usr/bin/env python3
"""Fail-closed post-execution audit for the 36-run falsification pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bench.agent_task_eval import file_sha256, load_observations, load_tasks


EXPECTED_CATEGORIES = {
    "decomposition_and_planning",
    "implementation_strategy",
    "test_strategy",
    "code_review",
    "decision_support_for_users",
    "recovery_from_tool_errors",
}
ALLOWED_DECISIONS = {
    "discard_provider_or_runtime_regression",
    "discard_competence_regression",
    "discard_authorization_regression",
    "discard_preference_regression",
    "discard_no_effect",
    "prefer_baseline_no_guidance",
    "discard_single_domain_effect",
    "discard_efficiency_regression",
    "advance_to_larger_calibration",
    "inconclusive_rewrite_or_repeat",
}


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha_json(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit_completed_campaign(
    tasks_path: Path,
    campaign_root: Path,
    *,
    repository_root: Path = Path("."),
) -> dict[str, object]:
    """Prove completed-call, artifact, blind-review, and decision invariants."""
    tasks = load_tasks(tasks_path)
    plan = _json(campaign_root / "campaign-plan.json")
    checks: dict[str, bool] = {
        "six_approved_held_out_tasks": (
            len(tasks) == 6
            and all(task.split == "validation" and task.review_status == "approved" for task in tasks)
        ),
        "six_required_categories": {task.category for task in tasks} == EXPECTED_CATEGORIES,
        "dataset_hash_matches_plan": plan.get("dataset_sha256") == file_sha256(tasks_path),
        "exactly_36_planned": plan.get("planned_executions") == 36,
        "three_routes_planned": isinstance(plan.get("routes"), list)
        and len(plan["routes"]) == 3,
        "sequential_plan": plan.get("parallel_requests") is False,
    }
    route_results: list[dict[str, object]] = []
    seen_models: set[str] = set()
    observed_keys: set[tuple[str, str, str, int]] = set()
    total_observations = 0
    total_artifacts = 0
    total_review_records = 0
    routes = plan.get("routes")
    if not isinstance(routes, list):
        routes = []
    for route in routes:
        if not isinstance(route, dict):
            continue
        name = str(route.get("name", ""))
        route_root = campaign_root / name
        observations_path = route_root / "observations.jsonl"
        report_path = route_root / "report.json"
        packet_path = route_root / "blind-packet.json"
        key_path = route_root / "condition-key.json"
        outcome_path = route_root / "outcome-decision.json"
        observations = load_observations(observations_path)
        report = _json(report_path)
        packet = _json(packet_path)
        key = _json(key_path)
        outcome = _json(outcome_path)
        expected_pairs = {(task.id, condition, 0) for task in tasks for condition in ("baseline", "candidate")}
        actual_pairs = {(row.task_id, row.condition, row.repetition) for row in observations}
        artifacts_exist = all(
            bool(row.run_artifact) and (repository_root / row.run_artifact).is_file()
            for row in observations
        )
        route_keys = {
            (row.model_identity, row.task_id, row.condition, row.repetition)
            for row in observations
        }
        overlap = bool(observed_keys & route_keys)
        observed_keys.update(route_keys)
        review_records = outcome.get("review_records")
        review_count = len(review_records) if isinstance(review_records, list) else 0
        route_checks = {
            "twelve_observations": len(observations) == 12,
            "six_baseline_candidate_pairs": actual_pairs == expected_pairs,
            "identity_matches_plan": bool(observations)
            and all(row.model_identity == route.get("model_identity") for row in observations),
            "dataset_hash_matches": bool(observations)
            and all(row.dataset_sha256 == plan.get("dataset_sha256") for row in observations),
            "condition_manifest_hash_matches": bool(observations)
            and all(row.condition_manifest_sha256 == route.get("conditions_sha256") for row in observations),
            "no_runtime_or_provider_errors": all(not row.error for row in observations),
            "all_artifacts_exist": artifacts_exist,
            "no_duplicate_execution_keys": not overlap and len(route_keys) == len(observations),
            "report_has_six_pairs": report.get("paired_repetitions") == 6,
            "blind_packet_is_bound": packet.get("candidate_blind") is True
            and packet.get("source_report_sha256") == file_sha256(report_path),
            "condition_key_is_bound": key.get("source_report_sha256") == file_sha256(report_path)
            and key.get("packet_sha256") == _sha_json(packet),
            "complete_human_review": review_count > 0
            and outcome.get("source_report_sha256") == file_sha256(report_path)
            and outcome.get("packet_sha256") == _sha_json(packet),
            "decision_is_preregistered": outcome.get("decision") in ALLOWED_DECISIONS,
            "deployment_remains_unauthorized": outcome.get("deployment_authorized") is False,
        }
        seen_models.add(str(route.get("model_identity", "")))
        total_observations += len(observations)
        total_artifacts += sum(
            bool(row.run_artifact) and (repository_root / row.run_artifact).is_file()
            for row in observations
        )
        total_review_records += review_count
        route_results.append(
            {
                "name": name,
                "model_identity": route.get("model_identity"),
                "observations": len(observations),
                "decision": outcome.get("decision"),
                "checks": route_checks,
                "passed": all(route_checks.values()),
            }
        )
    checks.update(
        {
            "three_distinct_models": len(seen_models) == 3 and "" not in seen_models,
            "exactly_36_observations": total_observations == 36,
            "exactly_36_run_artifacts": total_artifacts == 36,
            "campaign_dossier_exists": (campaign_root / "CAMPAIGN_DOSSIER.json").is_file(),
            "decision_maps_exist": (campaign_root / "MODEL_DECISION_MAPS.md").is_file(),
            "interpreted_analysis_exists": (campaign_root / "PILOT_RESULTS_ANALYSIS.md").is_file(),
            "human_reviews_exist_for_all_routes": total_review_records > 0
            and all(result["checks"]["complete_human_review"] for result in route_results),
        }
    )
    all_passed = all(checks.values()) and len(route_results) == 3 and all(
        result["passed"] for result in route_results
    )
    return {
        "schema_version": 1,
        "campaign_root": str(campaign_root),
        "all_passed": all_passed,
        "checks": checks,
        "routes": route_results,
        "evidence_counts": {
            "tasks": len(tasks),
            "observations": total_observations,
            "run_artifacts": total_artifacts,
            "blind_human_review_records": total_review_records,
        },
        "interpretation_boundary": (
            "Passing proves completion and integrity of this one-run falsification pilot. It does "
            "not authorize deployment, establish repeatability, or validate hidden reasoning."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = audit_completed_campaign(
        args.tasks, args.campaign_root, repository_root=args.repository_root
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
