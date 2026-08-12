"""Integrity checks for the held-out task-policy evaluation corpus."""

from __future__ import annotations

from bench.task_policy_eval import build_holdout_corpus, evaluate_profiles


def test_holdout_corpus_has_hundreds_of_unique_examples() -> None:
    examples = build_holdout_corpus()

    assert len(examples) >= 200
    assert len({example.id for example in examples}) == len(examples)
    assert len({example.text for example in examples}) == len(examples)


def test_offline_report_tracks_authority_and_per_policy_metrics() -> None:
    report = evaluate_profiles(build_holdout_corpus())

    assert report["examples"] >= 200
    assert report["authority_exact_match"] >= 0.85
    assert report["false_write_authority_rate"] == 0.0
    assert "bugfix.root_cause" in report["per_policy"]
    assert "precision" in report["per_policy"]["bugfix.root_cause"]
