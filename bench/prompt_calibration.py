#!/usr/bin/env python3
"""Select a validated prompt condition for one immutable model identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping


MAX_DEPLOYMENT_GUIDANCE_BYTES = 4 * 1024


@dataclass(frozen=True)
class CalibrationPolicy:
    """Hard gates applied before ranking prompt candidates."""

    min_paired_n: int = 100
    min_accuracy: float = 0.0
    max_error_rate: float = 0.02
    max_latency_ratio: float = 1.25
    max_tool_call_ratio: float = 1.25
    max_p_value: float = 0.05
    min_accuracy_delta: float = 0.0
    objective: str = "accuracy"
    min_preference_paired_n: int = 100
    min_mean_utility: float = -1.0
    min_utility_delta: float = 0.0
    max_normative_accuracy_regression: float = 0.0


@dataclass(frozen=True)
class Selection:
    """Auditable result of calibrating one model on one evaluation report."""

    model_identity: str
    selected_condition: str
    baseline_condition: str
    eligible_conditions: tuple[str, ...]
    rejected_conditions: Mapping[str, tuple[str, ...]]
    report_sha256: str
    calibrated_at: str
    objective: str = "accuracy"
    utility_profile_name: str = ""
    utility_profile_sha256: str = ""


def select_condition(
    report: dict[str, object],
    *,
    model_identity: str,
    baseline: str,
    policy: CalibrationPolicy,
    report_sha256: str = "",
) -> Selection:
    """Gate candidates and select the strongest validated condition lexicographically."""
    if policy.objective not in {"accuracy", "utility"}:
        raise ValueError("calibration objective must be accuracy or utility")
    conditions = _mapping(report.get("conditions"), "conditions")
    comparisons = _mapping(report.get("paired_vs_baseline"), "paired_vs_baseline")
    if baseline not in conditions:
        raise ValueError(f"baseline condition is missing: {baseline}")
    base = _mapping(conditions[baseline], f"conditions.{baseline}")

    eligible = [baseline]
    rejected: dict[str, tuple[str, ...]] = {}
    for name, raw_metrics in conditions.items():
        if name == baseline:
            continue
        metrics = _mapping(raw_metrics, f"conditions.{name}")
        paired = _mapping(comparisons.get(name), f"paired_vs_baseline.{name}")
        reasons = _rejection_reasons(metrics, base, paired, policy)
        if reasons:
            rejected[name] = tuple(reasons)
        else:
            eligible.append(name)

    utility_profile: Mapping[str, object] = {}
    if policy.objective == "utility":
        utility_profile = _mapping(report.get("utility_profile"), "utility_profile")
    selected = max(
        eligible,
        key=lambda name: _rank(_mapping(conditions[name], name), policy.objective),
    )
    return Selection(
        model_identity=model_identity,
        selected_condition=selected,
        baseline_condition=baseline,
        eligible_conditions=tuple(sorted(eligible)),
        rejected_conditions=rejected,
        report_sha256=report_sha256,
        calibrated_at=datetime.now(UTC).isoformat(),
        objective=policy.objective,
        utility_profile_name=str(utility_profile.get("name", "")),
        utility_profile_sha256=str(utility_profile.get("sha256", "")),
    )


def _rejection_reasons(
    metrics: Mapping[str, object],
    baseline: Mapping[str, object],
    paired: Mapping[str, object],
    policy: CalibrationPolicy,
) -> list[str]:
    reasons: list[str] = []
    if policy.objective == "accuracy":
        if _number(paired, "paired_n") < policy.min_paired_n:
            reasons.append("insufficient paired validation samples")
        if _number(paired, "accuracy_delta") <= policy.min_accuracy_delta:
            reasons.append("paired accuracy improvement below floor")
        if _number(paired, "mcnemar_exact_p") > policy.max_p_value:
            reasons.append("paired improvement is not statistically significant")
    else:
        if _number(paired, "preference_paired_n") < policy.min_preference_paired_n:
            reasons.append("insufficient paired preference samples")
        if _number(paired, "mean_utility_delta") <= policy.min_utility_delta:
            reasons.append("paired utility improvement below floor")
        if _number(paired, "utility_sign_exact_p") > policy.max_p_value:
            reasons.append("paired utility improvement is not statistically significant")
        if _number(metrics, "mean_preference_utility") < policy.min_mean_utility:
            reasons.append("mean preference utility below floor")
        candidate_accuracy = _number(metrics, "accuracy")
        baseline_accuracy = _number(baseline, "accuracy")
        if candidate_accuracy < baseline_accuracy - policy.max_normative_accuracy_regression:
            reasons.append("normative accuracy regression above ceiling")
    if _number(metrics, "accuracy") < policy.min_accuracy:
        reasons.append("accuracy below floor")
    attempted = _number(metrics, "attempted")
    error_rate = _number(metrics, "errors") / attempted if attempted else 1.0
    if error_rate > policy.max_error_rate:
        reasons.append("error rate above ceiling")
    _ratio_gate(metrics, baseline, "mean_latency_seconds", policy.max_latency_ratio, reasons)
    _ratio_gate(metrics, baseline, "mean_tool_calls", policy.max_tool_call_ratio, reasons)
    return reasons


def _ratio_gate(
    metrics: Mapping[str, object],
    baseline: Mapping[str, object],
    key: str,
    maximum: float,
    reasons: list[str],
) -> None:
    candidate = metrics.get(key)
    base = baseline.get(key)
    if isinstance(candidate, (int, float)) and isinstance(base, (int, float)) and base > 0:
        if candidate / base > maximum:
            reasons.append(f"{key} exceeds baseline ratio")


def _rank(metrics: Mapping[str, object], objective: str) -> tuple[float, ...]:
    common = (
        _optional_number(metrics.get("accuracy"), -1.0),
        _optional_number(metrics.get("perturbation_success"), -1.0),
        -_optional_number(metrics.get("brier"), 1.0),
        -_optional_number(metrics.get("ece"), 1.0),
        -_optional_number(metrics.get("mean_latency_seconds"), float("inf")),
        -_optional_number(metrics.get("mean_tool_calls"), float("inf")),
    )
    if objective == "utility":
        return (
            _optional_number(metrics.get("mean_preference_utility"), -1.0),
            -_optional_number(metrics.get("mean_preference_regret"), float("inf")),
            *common,
        )
    return common


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"report field must be an object: {name}")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"report metric must be numeric: {key}")
    return float(value)


def _optional_number(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def build_deployment_profile(
    selection: Selection,
    report: dict[str, object],
    run_config: dict[str, object],
    *,
    provider: str,
    roles: list[str],
    deployment_approved: bool,
) -> dict[str, object]:
    """Compile a selected condition into a hash-bound runtime profile."""
    if run_config.get("model_identity") != selection.model_identity:
        raise ValueError("run config model_identity does not match selection")
    model = str(run_config.get("model", ""))
    if not model or not provider:
        raise ValueError("profile needs provider and model route")
    allowed_roles = {"chat_agent", "planner", "developer"}
    if len(roles) != 1 or roles[0] not in allowed_roles:
        raise ValueError("a deployment profile must authorize exactly one calibrated role")
    calibration_role = str(run_config.get("calibration_role", "")).strip()
    if calibration_role != roles[0]:
        raise ValueError("deployment role does not match the run config calibration_role")
    if run_config.get("prompt_layer") != "behavior":
        raise ValueError("preference calibration may deploy only to the behavior layer")
    if run_config.get("evidence_kind") != "preference_behavior":
        raise ValueError("deployment evidence is not a preference-behavior study")
    conditions = _mapping(run_config.get("conditions"), "run_config.conditions")
    raw_condition = conditions.get(selection.selected_condition)
    if isinstance(raw_condition, str):
        guidance = raw_condition
    elif isinstance(raw_condition, dict) and isinstance(
        raw_condition.get("system_prompt"), str
    ):
        guidance = str(raw_condition["system_prompt"])
    else:
        raise ValueError("selected condition is missing from run config")
    guidance_bytes = len(guidance.encode("utf-8"))
    if not guidance.strip() or guidance_bytes > MAX_DEPLOYMENT_GUIDANCE_BYTES:
        raise ValueError("selected guidance is empty or exceeds the compact runtime limit")
    guidance_hash = hashlib.sha256(guidance.encode()).hexdigest()
    condition_hashes = _mapping(report.get("condition_hashes"), "condition_hashes")
    if condition_hashes.get(selection.selected_condition) != guidance_hash:
        raise ValueError("selected condition hash does not match validation report")
    comparisons = _mapping(report.get("paired_vs_baseline"), "paired_vs_baseline")
    comparison = comparisons.get(selection.selected_condition, {})
    return {
        "schema_version": 2,
        "prompt_layer": "behavior",
        "evidence_kind": "preference_behavior",
        "deployment_approved": deployment_approved,
        "provider": provider,
        "model": model,
        "model_identity": selection.model_identity,
        "selected_condition": selection.selected_condition,
        "roles": {
            role: {
                "prompt_layer": "behavior",
                "guidance": guidance,
                "sha256": guidance_hash,
                "utf8_bytes": guidance_bytes,
            }
            for role in roles
        },
        "validation": {
            "selection_report_sha256": selection.report_sha256,
            "dataset_sha256": report.get("dataset_sha256", ""),
            "observations_sha256": report.get("observations_sha256", ""),
            "paired_comparison": comparison,
            "objective": selection.objective,
            "calibration_role": calibration_role,
            "utility_profile": {
                "name": selection.utility_profile_name,
                "sha256": selection.utility_profile_sha256,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--model", required=True, help="immutable provider/model/version identity")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--min-paired-n", type=int, default=100)
    parser.add_argument("--objective", choices=("accuracy", "utility"), default="accuracy")
    parser.add_argument("--min-preference-paired-n", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--provider")
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--approve-deployment", action="store_true")
    args = parser.parse_args()

    payload = args.report.read_bytes()
    report = json.loads(payload)
    selection = select_condition(
        report,
        model_identity=args.model,
        baseline=args.baseline,
        policy=CalibrationPolicy(
            min_paired_n=args.min_paired_n,
            objective=args.objective,
            min_preference_paired_n=args.min_preference_paired_n,
        ),
        report_sha256=hashlib.sha256(payload).hexdigest(),
    )
    rendered = json.dumps(asdict(selection), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.profile_output:
        if not args.run_config or not args.provider or not args.role:
            parser.error(
                "--profile-output requires --run-config, --provider, and --role"
            )
        run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
        profile = build_deployment_profile(
            selection,
            report,
            run_config,
            provider=args.provider,
            roles=args.role,
            deployment_approved=args.approve_deployment,
        )
        args.profile_output.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
