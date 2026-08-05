from __future__ import annotations

import hashlib

import pytest

from bench.prompt_calibration import (
    CalibrationPolicy,
    build_deployment_profile,
    select_condition,
)


def _report() -> dict[str, object]:
    return {
        "conditions": {
            "base": {
                "attempted": 200,
                "errors": 1,
                "accuracy": 0.70,
                "perturbation_success": 0.60,
                "brier": 0.22,
                "ece": 0.15,
                "mean_latency_seconds": 10.0,
                "mean_tool_calls": 4.0,
            },
            "concise": {
                "attempted": 200,
                "errors": 1,
                "accuracy": 0.76,
                "perturbation_success": 0.69,
                "brier": 0.18,
                "ece": 0.11,
                "mean_latency_seconds": 10.5,
                "mean_tool_calls": 4.1,
            },
        },
        "paired_vs_baseline": {
            "concise": {
                "paired_n": 199,
                "wins": 20,
                "losses": 8,
                "ties": 171,
                "accuracy_delta": 12 / 199,
                "mcnemar_exact_p": 0.0357,
            }
        },
    }


def test_selects_candidate_that_passes_validation_gates() -> None:
    selection = select_condition(
        _report(),
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(min_paired_n=100),
    )
    assert selection.selected_condition == "concise"
    assert selection.eligible_conditions == ("base", "concise")


def test_rejects_candidate_with_too_few_paired_samples() -> None:
    report = _report()
    report["paired_vs_baseline"]["concise"]["paired_n"] = 12  # type: ignore[index]
    selection = select_condition(
        report,
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(min_paired_n=100),
    )
    assert selection.selected_condition == "base"
    assert selection.rejected_conditions["concise"] == (
        "insufficient paired validation samples",
    )


def test_rejects_candidate_that_is_too_slow_even_when_more_accurate() -> None:
    report = _report()
    report["conditions"]["concise"]["mean_latency_seconds"] = 20.0  # type: ignore[index]
    selection = select_condition(
        report,
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(min_paired_n=100, max_latency_ratio=1.25),
    )
    assert selection.selected_condition == "base"
    assert "mean_latency_seconds exceeds baseline ratio" in selection.rejected_conditions[
        "concise"
    ]


def test_compiles_hash_bound_opt_in_runtime_profile() -> None:
    report = _report()
    guidance = "Inspect evidence before acting."
    report["condition_hashes"] = {"concise": hashlib.sha256(guidance.encode()).hexdigest()}
    report["dataset_sha256"] = "dataset-hash"
    report["observations_sha256"] = "observation-hash"
    selection = select_condition(
        report,
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(min_paired_n=100),
        report_sha256="report-hash",
    )
    profile = build_deployment_profile(
        selection,
        report,
        {
            "model": "provider/model",
            "model_identity": "provider/model@revision",
            "calibration_role": "developer",
            "prompt_layer": "behavior",
            "evidence_kind": "preference_behavior",
            "conditions": {"concise": {"system_prompt": guidance}},
        },
        provider="provider",
        roles=["developer"],
        deployment_approved=True,
    )
    assert profile["roles"]["developer"]["guidance"] == guidance
    assert profile["roles"]["developer"]["utf8_bytes"] == len(guidance.encode())
    assert profile["schema_version"] == 2
    assert profile["prompt_layer"] == "behavior"
    assert profile["roles"]["developer"]["prompt_layer"] == "behavior"
    assert profile["deployment_approved"] is True
    assert profile["validation"]["dataset_sha256"] == "dataset-hash"
    assert profile["validation"]["calibration_role"] == "developer"


def test_profile_compilation_cannot_copy_one_validation_across_roles() -> None:
    report = _report()
    guidance = "Inspect evidence before acting."
    report["condition_hashes"] = {"concise": hashlib.sha256(guidance.encode()).hexdigest()}
    selection = select_condition(
        report,
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(min_paired_n=100),
    )

    with pytest.raises(ValueError, match="exactly one calibrated role"):
        build_deployment_profile(
            selection,
            report,
            {
                "model": "provider/model",
                "model_identity": "provider/model@revision",
                "calibration_role": "developer",
                "prompt_layer": "behavior",
                "evidence_kind": "preference_behavior",
                "conditions": {"concise": guidance},
            },
            provider="provider",
            roles=["developer", "planner"],
            deployment_approved=True,
        )


def test_profile_compilation_rejects_non_compact_guidance() -> None:
    report = _report()
    guidance = "x" * (4 * 1024 + 1)
    report["condition_hashes"] = {"concise": hashlib.sha256(guidance.encode()).hexdigest()}
    selection = select_condition(
        report,
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(min_paired_n=100),
    )

    with pytest.raises(ValueError, match="compact runtime limit"):
        build_deployment_profile(
            selection,
            report,
            {
                "model": "provider/model",
                "model_identity": "provider/model@revision",
                "calibration_role": "developer",
                "prompt_layer": "behavior",
                "evidence_kind": "preference_behavior",
                "conditions": {"concise": guidance},
            },
            provider="provider",
            roles=["developer"],
            deployment_approved=True,
        )


def test_profile_compilation_rejects_condition_not_bound_to_report() -> None:
    report = _report()
    report["condition_hashes"] = {"concise": "wrong"}
    selection = select_condition(
        report,
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(min_paired_n=100),
    )
    with pytest.raises(ValueError, match="hash does not match"):
        build_deployment_profile(
            selection,
            report,
            {
                "model": "provider/model",
                "model_identity": "provider/model@revision",
                "calibration_role": "developer",
                "prompt_layer": "behavior",
                "evidence_kind": "preference_behavior",
                "conditions": {"concise": "Guidance"},
            },
            provider="provider",
            roles=["developer"],
            deployment_approved=False,
        )


def test_profile_compilation_rejects_a_role_not_evaluated_by_the_run() -> None:
    report = _report()
    guidance = "Inspect evidence before acting."
    report["condition_hashes"] = {"concise": hashlib.sha256(guidance.encode()).hexdigest()}
    selection = select_condition(
        report,
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(min_paired_n=100),
    )

    with pytest.raises(ValueError, match="calibration_role"):
        build_deployment_profile(
            selection,
            report,
            {
                "model": "provider/model",
                "model_identity": "provider/model@revision",
                "calibration_role": "developer",
                "prompt_layer": "behavior",
                "evidence_kind": "preference_behavior",
                "conditions": {"concise": guidance},
            },
            provider="provider",
            roles=["planner"],
            deployment_approved=True,
        )


def test_utility_objective_selects_user_aligned_condition_without_safety_regression() -> None:
    report = _report()
    report["utility_profile"] = {
        "name": "high-control",
        "sha256": "profile-hash",
        "weights": {"interaction": 1.0, "user_control": 1.0},
    }
    report["conditions"]["base"].update(  # type: ignore[index]
        {"mean_preference_utility": 0.1, "mean_preference_regret": 0.7}
    )
    report["conditions"]["concise"].update(  # type: ignore[index]
        {"mean_preference_utility": 0.8, "mean_preference_regret": 0.1}
    )
    report["conditions"]["concise"]["accuracy"] = 0.70  # type: ignore[index]
    report["paired_vs_baseline"]["concise"].update(  # type: ignore[index]
        {
            "preference_paired_n": 150,
            "mean_utility_delta": 0.7,
            "utility_sign_exact_p": 0.001,
        }
    )
    selection = select_condition(
        report,
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(
            objective="utility",
            min_preference_paired_n=100,
        ),
    )
    assert selection.selected_condition == "concise"
    assert selection.objective == "utility"
    assert selection.utility_profile_name == "high-control"


def test_utility_objective_rejects_preference_gain_that_harms_normative_accuracy() -> None:
    report = _report()
    report["utility_profile"] = {"name": "fast", "sha256": "p", "weights": {"speed": 1}}
    report["conditions"]["base"].update(  # type: ignore[index]
        {"mean_preference_utility": 0.0, "mean_preference_regret": 0.8}
    )
    report["conditions"]["concise"].update(  # type: ignore[index]
        {"mean_preference_utility": 0.9, "mean_preference_regret": 0.0}
    )
    report["conditions"]["concise"]["accuracy"] = 0.60  # type: ignore[index]
    report["paired_vs_baseline"]["concise"].update(  # type: ignore[index]
        {
            "preference_paired_n": 150,
            "mean_utility_delta": 0.9,
            "utility_sign_exact_p": 0.001,
        }
    )
    selection = select_condition(
        report,
        model_identity="provider/model@revision",
        baseline="base",
        policy=CalibrationPolicy(objective="utility", min_preference_paired_n=100),
    )
    assert selection.selected_condition == "base"
    assert "normative accuracy regression above ceiling" in selection.rejected_conditions[
        "concise"
    ]
