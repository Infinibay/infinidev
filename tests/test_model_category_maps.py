from __future__ import annotations

from bench.model_category_maps import build_category_maps, evidence_label


def test_evidence_label_separates_stability_unique_mode_and_tie() -> None:
    assert evidence_label({"exactly_stable": True, "balanced_modal_keys": ["A"]}) == (
        "stable_prior_4_of_4"
    )
    assert evidence_label({"exactly_stable": False, "balanced_modal_keys": ["A"]}) == (
        "position_sensitive_unique_mode"
    )
    assert evidence_label({"exactly_stable": False, "balanced_modal_keys": ["A", "B"]}) == (
        "unresolved_modal_tie"
    )


def test_pivots_probe_first_analysis_to_model_then_category() -> None:
    analysis = {
        "selection_boundary": "selected sample",
        "models": {"Sol": {"stable_probes": 1, "unstable_probes": 0}},
        "records": [
            {
                "probe_id": "p",
                "family": "f",
                "category": "planning",
                "evaluation_mode": "preference",
                "scenario": "Plan",
                "classification": "stable_shared",
                "models": {
                    "Sol": {
                        "exactly_stable": True,
                        "balanced_modal_keys": ["A"],
                        "balanced_modal_actions": ["act"],
                        "balanced_counts": {"A": 4},
                        "fixed_actions": ["ask"],
                        "fixed_to_balanced_relation": "changed_unique",
                    }
                },
            }
        ],
    }
    report = build_category_maps(analysis)
    category = report["models"]["Sol"]["categories"]["planning"]
    assert category["probe_count"] == 1
    assert category["stable_policies"] == [{"probe_id": "p", "policy": ["act"]}]
