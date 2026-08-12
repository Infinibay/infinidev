"""Contract tests for the manually authored task-policy corpus."""

from __future__ import annotations

from collections import Counter

from bench.task_policy_manual_audit import load_examples


POLICIES = {
    "bugfix.root_cause",
    "feature.contract_first",
    "refactor.preserve_behavior",
    "research.evidence_first",
    "review.read_only",
    "performance.measure_first",
}
DIFFICULTIES = {
    "D0_explicit",
    "D1_paraphrase",
    "D2_overlap",
    "D3_composed",
    "D4_pragmatic",
    "D5_contextual",
}
REQUIRED_FIELDS = {
    "id",
    "batch",
    "text",
    "context_before",
    "policies",
    "uncategorized_reason",
    "scenario_family",
    "contrast_family",
    "user_type",
    "project_type",
    "programming_languages",
    "language",
    "locale",
    "linguistic_features",
    "style",
    "difficulty",
    "authority",
    "constraints",
    "split",
    "source",
    "author",
    "review_status",
    "rationale",
    "contrast_note",
}


def _rows() -> list[dict[str, object]]:
    return load_examples()


def test_manual_dataset_rows_are_complete_drafts() -> None:
    rows = _rows()

    assert len(rows) >= 201
    for row in rows:
        assert REQUIRED_FIELDS <= row.keys()
        assert row["source"] == "manually_authored_synthetic"
        assert row["review_status"] == "draft"
        assert row["split"] == "calibration"
        assert row["difficulty"] in DIFFICULTIES
        assert row["rationale"]
        assert row["contrast_note"]
        assert row["author"] == "Codex"
        assert isinstance(row["context_before"], list)
        assert "{" not in str(row["text"])
        assert "}" not in str(row["text"])


def test_manual_dataset_labels_and_uncategorized_are_consistent() -> None:
    rows = _rows()
    policy_counts: Counter[str] = Counter()

    for row in rows:
        policies = row["policies"]
        assert isinstance(policies, list)
        assert set(policies) <= POLICIES
        policy_counts.update(policies)
        if policies:
            assert row["uncategorized_reason"] is None
        else:
            assert row["uncategorized_reason"]

    assert set(policy_counts) == POLICIES
    assert min(policy_counts.values()) >= 30


def test_manual_dataset_first_batch_has_real_cross_axis_diversity() -> None:
    rows = _rows()
    ids = [str(row["id"]) for row in rows]
    texts = [str(row["text"]).casefold() for row in rows]
    scenarios = [str(row["scenario_family"]) for row in rows]
    difficult = {
        "D2_overlap",
        "D3_composed",
        "D4_pragmatic",
        "D5_contextual",
    }
    project_counts = Counter(row["project_type"] for row in rows)
    user_counts = Counter(row["user_type"] for row in rows)
    style_counts = Counter(row["style"] for row in rows)

    assert len(ids) == len(set(ids))
    assert ids == [f"manual-cal-{index:03d}" for index in range(1, len(rows) + 1)]
    assert len(texts) == len(set(texts))
    assert len(scenarios) == len(set(scenarios))
    assert len({row["project_type"] for row in rows}) >= 18
    assert len({row["user_type"] for row in rows}) >= 15
    assert len({row["style"] for row in rows}) >= 16
    assert len({row["language"] for row in rows}) >= 6
    assert sum(row["language"] != "en" for row in rows) / len(rows) >= 0.45
    assert sum(row["difficulty"] in difficult for row in rows) / len(rows) >= 0.35
    assert max(project_counts.values()) / len(rows) <= 0.10
    assert max(user_counts.values()) / len(rows) <= 0.15
    assert max(style_counts.values()) / len(rows) <= 0.15


def test_manual_dataset_does_not_hide_uncategorized_in_one_bucket() -> None:
    rows = _rows()
    reasons = {
        row["uncategorized_reason"]
        for row in rows
        if not row["policies"]
    }

    assert len(reasons) >= 16
    assert {
        "quoted_action",
        "ambiguous_authority",
        "unsupported_method",
        "out_of_domain",
        "insufficient_context",
        "healthy_existing_plan",
    } <= reasons


def test_manual_dataset_covers_zero_one_two_and_three_policy_requests() -> None:
    rows = _rows()
    cardinalities = Counter(len(row["policies"]) for row in rows)

    assert set(cardinalities) >= {0, 1, 2, 3}
    assert cardinalities[2] >= 27
    assert cardinalities[3] >= 3


def test_manual_dataset_batches_record_authorship_iterations() -> None:
    rows = _rows()
    batches = Counter(row["batch"] for row in rows)

    assert batches == {
        "boundary-01": 50,
        "causal-02": 30,
        "pragmatic-03": 30,
        "register-04": 30,
        "mechanisms-05": 61,
        "frontiers-06": 60,
        "abstention-07": 16,
        "composition-08": 20,
        "composition-09": 40,
        "boundaries-10": 30,
        "contrasts-11": 30,
        "composition-12": 40,
        "paired-13": 30,
        "singles-14": 60,
        "singles-15": 42,
        "composition-16": 44,
        "abstention-17": 40,
        "abstention-18": 43,
        "abstention-19": 46,
        "abstention-20": 40,
        "precision-boundaries-21": 48,
        "performance-negatives-22": 24,
    }
