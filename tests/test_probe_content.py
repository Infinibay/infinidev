from __future__ import annotations

from collections import Counter
from pathlib import Path

from bench.model_behavior import load_probes
from bench.probe_dataset import _missing_analysis


DATASET = Path(__file__).parents[1] / "bench/model_behavior_probes.draft.jsonl"


def test_all_twenty_nine_categories_retain_at_least_twenty_analyzed_drafts() -> None:
    probes = list(load_probes(DATASET).values())
    counts = Counter(probe.category for probe in probes)
    assert len(counts) == 29
    assert all(count >= 20 for count in counts.values())
    assert {probe.review_status for probe in probes} == {"draft"}
    assert {probe.generator for probe in probes} >= {
        "codex-manual-authoring@2026-08-03",
        "manual/codex-preference-seed@2026-08-03",
    }
    assert not {
        probe.id: _missing_analysis(probe)
        for probe in probes
        if _missing_analysis(probe)
    }


def test_drafts_form_isolated_two_variant_families() -> None:
    probes = list(load_probes(DATASET).values())
    groups = Counter(probe.group for probe in probes)
    assert len(groups) * 2 == len(probes)
    assert set(groups.values()) == {2}
    for group in groups:
        assert len({probe.split for probe in probes if probe.group == group}) == 1
    assert set(Counter(probe.split for probe in probes)) == {"calibration", "validation"}


def test_each_category_has_four_actions_and_valid_evaluation_metadata() -> None:
    probes = list(load_probes(DATASET).values())
    assert all(len(probe.choices) == 4 for probe in probes)
    for category in {probe.category for probe in probes}:
        category_probes = [
            probe for probe in probes
            if probe.category == category and probe.evaluation_mode == "normative"
        ]
        if not category_probes:
            continue
        answers = Counter(probe.answer for probe in category_probes)
        assert set(answers) == {"A", "B", "C", "D"}
        assert max(answers.values()) / len(category_probes) <= 0.4
    assert all(
        set(probe.analysis["distractor_rationales"])
        == set(probe.choices) - {probe.answer}
        for probe in probes if probe.evaluation_mode == "normative"
    )
    assert all(
        probe.answer is None
        and set(probe.choice_effects) == set(probe.choices)
        and set(probe.analysis["choice_rationales"]) == set(probe.choices)
        for probe in probes if probe.evaluation_mode == "preference"
    )


def test_each_variant_states_how_results_change_prompt_calibration() -> None:
    probes = load_probes(DATASET).values()
    assert all(len(str(probe.analysis["calibration_use"])) >= 40 for probe in probes)
    assert all(len(str(probe.analysis["failure_signal"])) >= 35 for probe in probes)
    assert all(len(probe.gold_rationale) >= 80 for probe in probes)
