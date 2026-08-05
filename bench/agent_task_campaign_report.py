#!/usr/bin/env python3
"""Consolidate provider-neutral agent-task reports into an evidence-first dossier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def _ratio(candidate: object, baseline: object) -> float | None:
    try:
        denominator = float(baseline)
        if denominator == 0:
            return None
        return float(candidate) / denominator
    except (TypeError, ValueError):
        return None


def _artifact_evidence(path: object) -> dict[str, object]:
    artifact_path = Path(str(path))
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    final_checks = raw.get("final_pattern_checks", {})
    action_checks = raw.get("action_pattern_checks", {})
    success = (
        raw.get("verify_exit_code") == 0
        and not raw.get("error")
        and not raw.get("forbidden_changes")
        and not raw.get("missing_expected_changes")
        and isinstance(final_checks, dict)
        and all(final_checks.values())
        and isinstance(action_checks, dict)
        and all(action_checks.values())
    )
    return {
        "artifact_path": str(artifact_path),
        "success": success,
        "final_answer": raw.get("final_answer", ""),
        "changed_paths": raw.get("changed_paths", []),
        "forbidden_changes": raw.get("forbidden_changes", []),
        "missing_expected_changes": raw.get("missing_expected_changes", []),
        "verification": {
            "exit_code": raw.get("verify_exit_code"),
            "stdout": raw.get("verify_stdout", ""),
            "stderr": raw.get("verify_stderr", ""),
            "final_pattern_checks": final_checks,
            "action_pattern_checks": action_checks,
        },
        "tool_trace": raw.get("tool_trace", []),
        "action_records": raw.get("action_records", []),
    }


def build_campaign_dossier(
    campaign_plan: Mapping[str, object],
    reports: list[Mapping[str, object]],
    *,
    outcomes: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build a cross-model dossier while retaining concrete behavioral evidence."""
    planned = int(campaign_plan.get("planned_executions", 0))
    routes = campaign_plan.get("routes")
    if planned != 36 or not isinstance(routes, list) or len(routes) != 3:
        raise ValueError("dossier requires the frozen 36-execution campaign plan")
    if len(reports) != 3:
        raise ValueError("dossier requires exactly three completed route reports")
    if any(int(report.get("paired_repetitions", 0)) != 6 for report in reports):
        raise ValueError("every dossier route must contain six paired task results")

    outcome_map = outcomes or {}
    model_records = []
    total_observed = 0
    for report in reports:
        model = str(report.get("model", ""))
        conditions = report.get("conditions")
        task_records = report.get("task_records")
        if not isinstance(conditions, dict) or not isinstance(task_records, list):
            raise ValueError(f"route report is incomplete: {model}")
        baseline = conditions.get("baseline")
        candidate = conditions.get("candidate")
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise ValueError(f"route summaries are incomplete: {model}")
        categories = []
        for pair in task_records:
            if not isinstance(pair, dict) or not isinstance(pair.get("task"), dict):
                raise ValueError(f"route task record is invalid: {model}")
            task = pair["task"]
            baseline_row = pair.get("baseline")
            candidate_row = pair.get("candidate")
            if not isinstance(baseline_row, dict) or not isinstance(candidate_row, dict):
                raise ValueError(f"route pair is incomplete: {model}/{task.get('id')}")
            categories.append(
                {
                    "task_id": task.get("id"),
                    "category": task.get("category"),
                    "request": task.get("request"),
                    "rubric": task.get("rubric"),
                    "deterministic_delta": pair.get("success_delta"),
                    "behavior_signature_changed": pair.get("candidate_changed_behavior"),
                    "tool_call_delta": pair.get("tool_call_delta"),
                    "latency_delta_seconds": pair.get("latency_delta_seconds"),
                    "baseline": _artifact_evidence(baseline_row.get("run_artifact")),
                    "candidate": _artifact_evidence(candidate_row.get("run_artifact")),
                }
            )
        total_observed += len(categories) * 2
        model_records.append(
            {
                "provider": report.get("provider"),
                "model": model,
                "model_identity": report.get("model_identity"),
                "deterministic_outcomes": report.get("paired_outcomes"),
                "efficiency": {
                    "latency_ratio": _ratio(
                        candidate.get("mean_latency_seconds"), baseline.get("mean_latency_seconds")
                    ),
                    "tool_call_ratio": _ratio(
                        candidate.get("mean_tool_calls"), baseline.get("mean_tool_calls")
                    ),
                    "prompt_token_ratio": _ratio(
                        candidate.get("prompt_tokens"), baseline.get("prompt_tokens")
                    ),
                    "completion_token_ratio": _ratio(
                        candidate.get("completion_tokens"), baseline.get("completion_tokens")
                    ),
                },
                "condition_summaries": conditions,
                "reviewed_outcome": outcome_map.get(model),
                "category_maps": categories,
            }
        )
    if total_observed != planned:
        raise ValueError(f"dossier evidence count mismatch: expected {planned}, got {total_observed}")
    return {
        "schema_version": 1,
        "interpretation_boundary": (
            "This dossier maps observable decisions and work traces, not hidden reasoning or a "
            "model's literal mental state. One run per condition can falsify a candidate but cannot "
            "authorize deployment or establish stable model-wide traits."
        ),
        "campaign_plan": dict(campaign_plan),
        "observed_executions": total_observed,
        "models": model_records,
    }


def _pct(value: object) -> str:
    return "n/a" if value is None else f"{(float(value) - 1.0) * 100:+.1f}%"


def render_markdown(dossier: Mapping[str, object]) -> str:
    """Render a readable model-by-model, category-by-category evidence map."""
    lines = [
        "# Agent-task model decision maps",
        "",
        str(dossier.get("interpretation_boundary")),
        "",
        f"Observed executions: {dossier.get('observed_executions')}.",
    ]
    models = dossier.get("models")
    if not isinstance(models, list):
        raise ValueError("dossier has no model records")
    for record in models:
        if not isinstance(record, dict):
            continue
        lines.extend(["", f"## {record.get('model')}", ""])
        lines.append(f"Deterministic pairs: `{json.dumps(record.get('deterministic_outcomes'), sort_keys=True)}`.")
        efficiency = record.get("efficiency")
        if isinstance(efficiency, dict):
            lines.extend(
                [
                    "",
                    "Candidate relative to baseline: "
                    f"latency {_pct(efficiency.get('latency_ratio'))}, "
                    f"tool calls {_pct(efficiency.get('tool_call_ratio'))}, "
                    f"input tokens {_pct(efficiency.get('prompt_token_ratio'))}, "
                    f"output tokens {_pct(efficiency.get('completion_token_ratio'))}.",
                ]
            )
        outcome = record.get("reviewed_outcome")
        if isinstance(outcome, dict):
            lines.extend(
                [
                    "",
                    f"Reviewed decision: `{outcome.get('decision')}`.",
                    f"Reasons: {outcome.get('decision_reasons')}",
                ]
            )
        categories = record.get("category_maps")
        if not isinstance(categories, list):
            continue
        for category in categories:
            if not isinstance(category, dict):
                continue
            lines.extend(
                [
                    "",
                    f"### {category.get('category')} — `{category.get('task_id')}`",
                    "",
                    f"Request: {category.get('request')}",
                    "",
                    "Observed delta: "
                    f"success={category.get('deterministic_delta')}, "
                    f"signature_changed={category.get('behavior_signature_changed')}, "
                    f"tools={category.get('tool_call_delta')}, "
                    f"latency={category.get('latency_delta_seconds')}s.",
                ]
            )
            for condition in ("baseline", "candidate"):
                evidence = category.get(condition)
                if not isinstance(evidence, dict):
                    continue
                lines.extend(
                    [
                        "",
                        f"#### {condition}",
                        "",
                        f"Artifact: `{evidence.get('artifact_path')}`",
                        f"Changed paths: `{evidence.get('changed_paths')}`; "
                        f"forbidden: `{evidence.get('forbidden_changes')}`.",
                        "",
                        str(evidence.get("final_answer", "")),
                    ]
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("--route", action="append", required=True)
    parser.add_argument("--outcome", action="append", nargs=2, metavar=("MODEL", "PATH"))
    args = parser.parse_args()
    campaign_plan = json.loads(
        (args.campaign_root / "campaign-plan.json").read_text(encoding="utf-8")
    )
    reports = [
        json.loads((args.campaign_root / route / "report.json").read_text(encoding="utf-8"))
        for route in args.route
    ]
    outcomes = {
        model: json.loads(Path(path).read_text(encoding="utf-8"))
        for model, path in (args.outcome or [])
    }
    dossier = build_campaign_dossier(campaign_plan, reports, outcomes=outcomes)
    args.output_json.write_text(
        json.dumps(dossier, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(dossier), encoding="utf-8")


if __name__ == "__main__":
    main()
