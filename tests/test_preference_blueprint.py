from __future__ import annotations

from bench.model_behavior import Probe
from bench.preference_blueprint import audit_blueprint, remaining_targets


def test_blueprint_requires_enough_variants_and_axis_tension() -> None:
    value = {
        "variants_per_family": 2,
        "families": [
            {
                "category": "planning",
                "family": "planning-depth",
                "tradeoff": "Detailed plan versus iterative planning.",
                "information_sought": "Preferred planning investment.",
                "axes": ["speed", "quality"],
                "variant_axis": "Reorder facts.",
            },
            {
                "category": "planning",
                "family": "planning-control",
                "tradeoff": "Checkpoints versus autonomy.",
                "information_sought": "Preferred collaboration cadence.",
                "axes": ["interaction", "autonomy"],
                "variant_axis": "Rephrase facts.",
            },
        ],
    }
    report = audit_blueprint(value, {"planning": 4})
    assert report["passes"] is True
    assert report["planned_probes"] == 4


def test_blueprint_reports_category_shortfall_and_unknown_axis() -> None:
    value = {
        "variants_per_family": 2,
        "families": [
            {
                "category": "planning",
                "family": "planning-depth",
                "tradeoff": "A versus B.",
                "information_sought": "Preference.",
                "axes": ["speed", "magic"],
                "variant_axis": "Wording.",
            }
        ],
    }
    report = audit_blueprint(value, {"planning": 4})
    assert report["passes"] is False
    assert report["shortfalls"] == {"planning": 2}
    assert report["issues"]["planning-depth"] == ["unknown_axis"]


def test_remaining_targets_credits_existing_authored_preferences() -> None:
    probe = Probe(
        "existing", "interaction", "?", {"A": "x", "B": "y"}, None,
        evaluation_mode="preference",
        choice_effects={"A": {"speed": 1.0}, "B": {"quality": 1.0}},
    )
    assert remaining_targets(
        {probe.id: probe}, {"interaction": 1, "planning": 2}
    ) == {"planning": 2}


def test_blueprint_ignores_already_materialized_families() -> None:
    value = {
        "variants_per_family": 2,
        "families": [
            {
                "category": "done-category",
                "family": "done-family",
                "tradeoff": "Already done.",
                "information_sought": "Already known.",
                "axes": ["speed", "quality"],
                "variant_axis": "Wording.",
            },
            {
                "category": "planning",
                "family": "remaining-family",
                "tradeoff": "Speed versus depth.",
                "information_sought": "Planning preference.",
                "axes": ["speed", "quality"],
                "variant_axis": "Wording.",
            },
        ],
    }
    report = audit_blueprint(value, {"planning": 2}, {"done-family"})
    assert report["passes"] is True
    assert report["families"] == 1
    assert report["planned_probes"] == 2
