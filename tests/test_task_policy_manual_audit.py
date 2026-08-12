"""Tests for the read-only manual-dataset auditor."""

from __future__ import annotations

from bench.task_policy_manual_audit import (
    load_examples,
    semantic_contract_report,
    structural_report,
)


def test_manual_audit_reports_current_corpus_without_generation() -> None:
    report = structural_report(load_examples())

    assert report["examples"] == 854
    assert min(report["policies"].values()) >= 40
    assert report["single_policies"] == {
        "bugfix.root_cause": 72,
        "feature.contract_first": 60,
        "performance.measure_first": 60,
        "refactor.preserve_behavior": 72,
        "research.evidence_first": 78,
        "review.read_only": 78,
    }
    assert report["policy_combinations"] == {
        "bugfix.root_cause + performance.measure_first": 18,
        "bugfix.root_cause + performance.measure_first + refactor.preserve_behavior": 18,
        "bugfix.root_cause + refactor.preserve_behavior": 18,
        "bugfix.root_cause + research.evidence_first": 18,
        "feature.contract_first + performance.measure_first": 18,
        "feature.contract_first + performance.measure_first + research.evidence_first": 18,
        "feature.contract_first + refactor.preserve_behavior": 18,
        "feature.contract_first + research.evidence_first": 20,
        "performance.measure_first + review.read_only": 18,
        "research.evidence_first + review.read_only": 18,
    }
    assert report["cardinality"] == {"0": 252, "1": 420, "2": 146, "3": 36}
    assert report["projects"] == 278
    assert report["users"] == 294
    assert report["styles"] == 372
    assert report["duplicate_ids"] == 0
    assert report["duplicate_texts"] == 0
    assert report["duplicate_scenarios"] == 0
    assert report["max_project"]["share"] < 0.10
    assert report["max_user"]["share"] < 0.15
    assert report["max_style"]["share"] < 0.15


def test_manual_examples_satisfy_machine_verifiable_semantic_contract() -> None:
    report = semantic_contract_report(load_examples())

    assert report["violation_count"] == 0, report["violations"][:10]


def test_semantic_contract_rejects_read_only_modification_policy() -> None:
    row = {
        "id": "invalid-review-fix",
        "policies": ["review.read_only", "bugfix.root_cause"],
        "authority": ["answer", "diagnose", "read_only"],
        "uncategorized_reason": None,
    }

    report = semantic_contract_report([row])

    assert {item["rule"] for item in report["violations"]} == {
        "read_only_conflict",
        "requires_modify",
    }
