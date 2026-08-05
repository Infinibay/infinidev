from __future__ import annotations

import pytest

from bench.prompt_candidate_pool import compile_candidate_pool


def _brief() -> dict[str, object]:
    return {
        "model": "Sol",
        "utility_profile": {"name": "control", "sha256": "profile-sha"},
        "stable_profile_conflicts_to_test": [
            {
                "probe_id": "p1",
                "category": "planning",
                "profile_best_actions": ["show checkpoint"],
            }
        ],
        "unstable_profile_hypotheses": [
            {
                "probe_id": "p2",
                "category": "interaction",
                "profile_best_actions": ["ask user"],
            }
        ],
        "normative_evidence": [
            {
                "probe_id": "p3",
                "category": "safety",
                "status": "stable_match",
                "draft_expected_action": "confirm target",
            },
            {
                "probe_id": "p4",
                "category": "safety",
                "status": "requires_independent_review_or_more_evidence",
                "draft_expected_action": "confirm target",
            },
        ],
    }


def _pool() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_brief_sha256": "brief-sha",
        "model": "Sol",
        "model_identity": "provider:sol@revision",
        "utility_profile_sha256": "profile-sha",
        "calibration_role": "developer",
        "candidates": [
            {
                "name": "visible-checkpoint",
                "kind": "preference_compensation",
                "guidance_style": "advisory",
                "guidance": "When the active profile values control, prefer a visible checkpoint.",
                "evidence_probe_ids": ["p1"],
                "rationale": "The stable raw prior skipped the profile-preferred checkpoint.",
                "expected_effect": "More visible control on consequential plan boundaries.",
                "regression_risks": ["May add unnecessary interaction."],
            }
        ],
    }


def test_compiles_inert_evidence_bound_candidate_condition() -> None:
    compiled = compile_candidate_pool(_pool(), _brief(), brief_sha256="brief-sha")
    assert compiled["deployment_approved"] is False
    assert compiled["candidates"][0]["evidence_actions"] == {"p1": ["show checkpoint"]}
    assert compiled["run_config_fragment"]["conditions"]["current"] is None
    assert compiled["run_config_fragment"]["conditions"]["visible-checkpoint"] == {
        "system_prompt": "When the active profile values control, prefer a visible checkpoint."
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_brief_sha256", "stale", "source brief hash"),
        ("utility_profile_sha256", "other", "utility profile hash"),
        ("calibration_role", "all", "calibration_role"),
    ],
)
def test_rejects_identity_or_scope_drift(field: str, value: str, message: str) -> None:
    pool = _pool()
    pool[field] = value
    with pytest.raises(ValueError, match=message):
        compile_candidate_pool(pool, _brief(), brief_sha256="brief-sha")


def test_rejects_unbound_or_absolute_preference_guidance() -> None:
    pool = _pool()
    pool["candidates"][0]["evidence_probe_ids"] = ["p2"]
    with pytest.raises(ValueError, match="does not support preference_compensation"):
        compile_candidate_pool(pool, _brief(), brief_sha256="brief-sha")

    pool = _pool()
    pool["candidates"][0]["guidance"] = "Always pause for user approval."
    with pytest.raises(ValueError, match="absolute language"):
        compile_candidate_pool(pool, _brief(), brief_sha256="brief-sha")


def test_stable_normative_strength_cannot_be_miscast_as_remediation() -> None:
    pool = _pool()
    pool["candidates"][0].update(
        {
            "kind": "normative_remediation",
            "guidance_style": "contract",
            "evidence_probe_ids": ["p3"],
        }
    )
    with pytest.raises(ValueError, match="does not support normative_remediation"):
        compile_candidate_pool(pool, _brief(), brief_sha256="brief-sha")
