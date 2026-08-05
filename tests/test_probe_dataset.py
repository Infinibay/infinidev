from __future__ import annotations

import pytest

from bench.model_behavior import Probe
from bench.probe_dataset import (
    audit_dataset,
    load_preference_axis_targets,
    load_preference_category_targets,
    load_targets,
)


def _probe(identifier: str, category: str, group: str, split: str) -> Probe:
    return Probe(
        identifier,
        category,
        identifier,
        {"A": "x", "B": "y"},
        "A",
        group,
        (),
        None,
        None,
        split,
        "approved",
        "A follows the stated evidence.",
        "test-reviewer",
        "",
        {
            "hypothesis": "The model uses evidence.",
            "decisive_information": "The location is unknown.",
            "variant_axis": "wording",
            "failure_signal": "The model guesses a file.",
            "calibration_use": "Add search-first guidance.",
            "distractor_rationales": {"B": "B guesses."},
        },
    )


def test_audit_reports_category_shortfall() -> None:
    report = audit_dataset({"one": _probe("one", "tools", "g1", "calibration")}, {"tools": 2})
    assert report["missing_to_target"] == {"tools": 1}
    assert report["categories"]["tools"]["authored_total"] == 1
    assert report["categories"]["tools"]["authored_normative"] == 1
    assert report["categories"]["tools"]["authored_preference"] == 0
    assert report["passes"] is False


def test_audit_rejects_family_leakage_between_splits() -> None:
    probes = {
        "one": _probe("one", "tools", "same", "calibration"),
        "two": _probe("two", "tools", "same", "validation"),
    }
    report = audit_dataset(probes, {"tools": 2})
    assert report["group_split_leakage"] == ["same"]
    assert report["passes"] is False


def test_audit_passes_balanced_dataset_without_leakage() -> None:
    probes = {
        "one": _probe("one", "tools", "train-family", "calibration"),
        "two": _probe("two", "tools", "heldout-family", "validation"),
    }
    assert audit_dataset(probes, {"tools": 2})["passes"] is True


def test_drafts_do_not_count_toward_release_target() -> None:
    probe = Probe("one", "tools", "unique", {"A": "x", "B": "y"}, "A")
    report = audit_dataset({"one": probe}, {"tools": 1})
    assert report["approved"] == 0
    assert report["missing_to_target"] == {"tools": 1}


def test_approved_probe_needs_rationale_and_reviewer() -> None:
    probe = Probe(
        "one", "tools", "unique", {"A": "x", "B": "y"}, "A",
        review_status="approved",
    )
    report = audit_dataset({"one": probe}, {"tools": 1})
    assert report["approval_metadata_issues"] == ["one"]


def test_audit_detects_normalized_duplicate_questions() -> None:
    one = _probe("one", "tools", "g1", "calibration")
    two = _probe("two", "tools", "g2", "validation")
    two = Probe(
        two.id, two.category, "ONE!", two.choices, two.answer, two.group,
        two.tags, two.scenario, two.user_request, two.split, two.review_status,
        two.gold_rationale, two.reviewer,
    )
    report = audit_dataset({"one": one, "two": two}, {"tools": 2})
    assert report["duplicate_questions"] == [["one", "two"]]


def test_load_targets_accepts_described_taxonomy(tmp_path) -> None:
    path = tmp_path / "taxonomy.json"
    path.write_text(
        '{"categories":{"tools":{"target":20,"objective":"Choose tools."}}}'
    )
    assert load_targets(path) == {"tools": 20}


def test_loads_and_validates_preference_axis_targets(tmp_path) -> None:
    path = tmp_path / "taxonomy.json"
    path.write_text(
        '{"categories":{"tools":1},'
        '"preference_axis_targets":{"autonomy":5,"quality":7}}'
    )
    assert load_preference_axis_targets(path) == {"autonomy": 5, "quality": 7}

    path.write_text(
        '{"categories":{"tools":1},"preference_axis_targets":{"magic":5}}'
    )
    with pytest.raises(ValueError, match="unknown preference utility axes"):
        load_preference_axis_targets(path)


def test_loads_and_validates_per_category_preference_targets(tmp_path) -> None:
    path = tmp_path / "taxonomy.json"
    path.write_text(
        '{"categories":{"planning":{"target":20,"preference_target":4}}}'
    )
    assert load_preference_category_targets(path) == {"planning": 4}

    path.write_text(
        '{"preference_per_category_target":4,'
        '"categories":{"planning":{"target":20},"review":{"target":20}}}'
    )
    assert load_preference_category_targets(path) == {"planning": 4, "review": 4}

    path.write_text(
        '{"categories":{"planning":{"target":2,"preference_target":3}}}'
    )
    with pytest.raises(ValueError, match="preference_target"):
        load_preference_category_targets(path)


def test_release_scale_audit_checks_split_choices_groups_and_answer_balance() -> None:
    probes = {
        f"p{index}": Probe(
            f"p{index}",
            "tools",
            f"question {index}",
            {"A": "x", "B": "y"},
            "A",
            f"single-{index}",
            split="calibration",
            review_status="approved",
            gold_rationale="Evidence supports A.",
            reviewer="reviewer",
            analysis={
                "hypothesis": "Uses evidence.",
                "decisive_information": "A is supported.",
                "variant_axis": "wording",
                "failure_signal": "Selects another option.",
                "calibration_use": "Test evidence guidance.",
                "distractor_rationales": {"B": "Unsupported."},
            },
        )
        for index in range(10)
    }
    report = audit_dataset(probes, {"tools": 10})
    assert report["split_shortfalls"]["tools"]["validation"] == 2
    assert len(report["choice_count_issues"]) == 10
    assert report["answer_balance_issues"] == ["tools"]
    assert len(report["group_size_issues"]) == 10


def test_audit_reports_missing_per_question_analysis() -> None:
    probe = Probe("one", "tools", "unique", {"A": "x", "B": "y"}, "A")
    issues = audit_dataset({"one": probe}, {"tools": 1})["analysis_metadata_issues"]
    assert "hypothesis" in issues["one"]
    assert "distractor_rationales" in issues["one"]


def test_audit_reports_preference_axis_coverage() -> None:
    probe = Probe(
        "preference-one",
        "interaction",
        "Choose a safe cadence.",
        {"A": "Frequent updates", "B": "One final update"},
        None,
        evaluation_mode="preference",
        choice_effects={
            "A": {"interaction": 1.0, "user_control": 0.8},
            "B": {"interaction": -1.0, "autonomy": 0.8},
        },
        analysis={
            "hypothesis": "Uses the active profile.",
            "decisive_information": "Both choices are safe.",
            "variant_axis": "wording",
            "failure_signal": "Ignores the profile.",
            "calibration_use": "Select profile-specific guidance.",
            "preference_tradeoff": "Interaction versus autonomy.",
            "choice_rationales": {"A": "More contact.", "B": "More autonomy."},
        },
    )
    report = audit_dataset({probe.id: probe}, {"interaction": 1})
    assert report["preference_axis_counts"]["interaction"] == 1
    assert report["preference_axis_counts"]["autonomy"] == 1
    assert report["preference_axis_counts"]["quality"] == 0
    assert report["preference_axes_by_category"]["interaction"] == {
        "autonomy": 1,
        "interaction": 1,
        "user_control": 1,
    }


def test_preference_release_target_counts_only_approved_probes() -> None:
    draft = Probe(
        "draft-preference",
        "interaction",
        "Choose a cadence.",
        {"A": "Frequent", "B": "Sparse"},
        None,
        evaluation_mode="preference",
        choice_effects={
            "A": {"interaction": 1.0},
            "B": {"interaction": -1.0},
        },
        analysis={
            "hypothesis": "Uses profile.",
            "decisive_information": "Both are safe.",
            "variant_axis": "wording",
            "failure_signal": "Ignores profile.",
            "calibration_use": "Calibrate cadence.",
            "preference_tradeoff": "More or less interaction.",
            "choice_rationales": {"A": "More.", "B": "Less."},
        },
    )
    report = audit_dataset(
        {draft.id: draft}, {"interaction": 1}, {"interaction": 1}
    )
    assert report["preference_axis_counts"]["interaction"] == 1
    assert report["approved_preference_axis_counts"]["interaction"] == 0
    assert report["preference_axis_shortfalls"] == {"interaction": 1}
    assert report["passes"] is False


def test_preference_category_target_reports_authored_and_approved_shortfalls() -> None:
    probe = Probe(
        "planning-preference",
        "planning",
        "Choose a cadence.",
        {"A": "Fast", "B": "Thorough"},
        None,
        evaluation_mode="preference",
        choice_effects={
            "A": {"speed": 1.0, "quality": -0.2},
            "B": {"speed": -0.2, "quality": 1.0},
        },
        analysis={
            "hypothesis": "Uses profile.",
            "decisive_information": "Both choices satisfy hard requirements.",
            "variant_axis": "wording",
            "failure_signal": "Universalizes one trade-off.",
            "calibration_use": "Calibrate planning depth.",
            "preference_tradeoff": "Speed versus planning depth.",
            "choice_rationales": {"A": "Faster.", "B": "More thorough."},
        },
    )
    report = audit_dataset(
        {probe.id: probe}, {"planning": 1}, preference_category_targets={"planning": 1}
    )
    assert report["authored_preference_category_shortfalls"] == {}
    assert report["approved_preference_category_shortfalls"] == {"planning": 1}
