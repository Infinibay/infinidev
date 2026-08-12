"""Tests for the manual-corpus E5 cross-validation diagnostic."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from bench.task_policy_manual_audit import load_examples
from bench.task_policy_manual_cv import (
    _label_signature,
    _precision_first_threshold,
    _predict_independent,
    _rbf_kernel,
    assign_folds,
    render_input,
    load_precomputed_vectors,
    select_independent_thresholds,
    select_joint_thresholds,
    subsample_training_indices,
)
from bench.task_policy_multilabel_head import MultiLabelExample


def test_assign_folds_is_deterministic_and_covers_every_row() -> None:
    rows = load_examples()

    first = assign_folds(rows, 5)
    second = assign_folds(rows, 5)

    assert first == second
    assert len(first) == len(rows)
    assert set(first) == {0, 1, 2, 3, 4}


def test_assign_folds_populates_each_fold_for_well_supported_signatures() -> None:
    rows = load_examples()
    folds = assign_folds(rows, 5)
    signatures: dict[tuple[str, ...], Counter[int]] = {}
    for row, fold in zip(rows, folds, strict=True):
        if row.get("evaluation_group"):
            continue
        signature = tuple(sorted(row["policies"]))
        signatures.setdefault(signature, Counter())[fold] += 1

    for counts in signatures.values():
        if sum(counts.values()) < 5:
            continue
        if sum(counts.values()) < 20:
            continue
        assert all(counts[fold] > 0 for fold in range(5))


def test_assign_folds_does_not_move_existing_rows_when_corpus_grows() -> None:
    rows = load_examples()
    original = assign_folds(rows, 5)
    extended = rows + [{
        "policies": ["bugfix.root_cause"],
        "scenario_family": "future-independent-family",
    }]

    assert assign_folds(extended, 5)[:-1] == original


def test_assign_folds_keeps_evaluation_groups_together() -> None:
    rows = [
        {"policies": ["bugfix.root_cause"], "scenario_family": "a", "evaluation_group": "pair"},
        {"policies": ["refactor.preserve_behavior"], "scenario_family": "b", "evaluation_group": "pair"},
    ] + [
        {"policies": [], "scenario_family": f"other-{index}"}
        for index in range(8)
    ]

    folds = assign_folds(rows, 4)

    assert folds[0] == folds[1]


def test_render_input_preserves_current_message_and_available_context() -> None:
    row = {
        "text": "Seguí.",
        "context_before": ["El plan activo ya es un bugfix."],
    }

    rendered = render_input(row)

    assert "Previous context:" in rendered
    assert "El plan activo ya es un bugfix." in rendered
    assert rendered.endswith("Current user message:\nSeguí.")


def test_assign_folds_rejects_non_diagnostic_fold_counts() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        assign_folds(load_examples(), 2)


def test_training_subsample_is_deterministic_and_signature_stratified() -> None:
    rows = load_examples()
    indices = np.arange(len(rows), dtype=np.int64)

    first = subsample_training_indices(rows, indices, 0.25)
    second = subsample_training_indices(rows, indices, 0.25)

    assert first.tolist() == second.tolist()
    assert len(first) < len(indices)
    selected_signatures = {_label_signature(rows[index]) for index in first}
    assert selected_signatures == {_label_signature(row) for row in rows}


def test_training_subsample_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        subsample_training_indices(load_examples(), np.asarray([0]), 0.0)


def test_independent_prediction_can_emit_zero_one_two_or_three_labels() -> None:
    method_scores = np.asarray([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.8, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.8, 0.7, 0.1, 0.1, 0.1],
    ])
    task_scores = np.asarray([0.1, 0.9, 0.9, 0.9])

    predictions = _predict_independent(
        method_scores,
        task_scores,
        (0.5,) * 6,
        0.5,
    )

    assert [len(prediction) for prediction in predictions] == [0, 1, 2, 3]


def test_independent_threshold_selection_uses_validation_only() -> None:
    rows = [
        MultiLabelExample("zero", "x", (), "validation"),
        MultiLabelExample("bug", "x", ("bugfix.root_cause",), "validation"),
    ]
    method_scores = np.asarray([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
    ])
    task_scores = np.asarray([0.1, 0.9])

    thresholds, task_threshold, report = select_independent_thresholds(
        method_scores,
        task_scores,
        rows,
        minimum_method_precision=0.95,
    )

    assert thresholds[0] > 0.1
    assert 0.0 <= task_threshold <= 1.0
    assert report["exact_match"] == 1.0


def test_precision_first_threshold_keeps_recall_floor_and_prefers_margin() -> None:
    scores = np.asarray([0.1, 0.2, 0.6, 0.8, 0.9])
    expected = np.asarray([False, False, True, True, True])

    threshold = _precision_first_threshold(
        scores,
        expected,
        minimum_precision=1.0,
        minimum_recall=2 / 3,
    )

    assert 0.6 < threshold <= 0.8
    assert int(np.sum(scores >= threshold)) == 2


def test_joint_threshold_selection_rejects_invalid_pass_count() -> None:
    with pytest.raises(ValueError, match="at least one pass"):
        select_joint_thresholds(
            np.zeros((1, 6)),
            np.zeros(1),
            [MultiLabelExample("zero", "x", (), "validation")],
            minimum_method_precision=0.95,
            passes=0,
        )


def test_joint_threshold_selection_preserves_perfect_calibration() -> None:
    rows = [
        MultiLabelExample("zero", "x", (), "validation"),
        MultiLabelExample("bug", "x", ("bugfix.root_cause",), "validation"),
        MultiLabelExample(
            "compound",
            "x",
            ("bugfix.root_cause", "performance.measure_first"),
            "validation",
        ),
    ]
    method_scores = np.asarray([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.9],
    ])
    task_scores = np.asarray([0.1, 0.9, 0.9])

    _, _, report = select_joint_thresholds(
        method_scores,
        task_scores,
        rows,
        minimum_method_precision=0.95,
    )

    assert report["exact_match"] == 1.0


def test_rbf_kernel_is_symmetric_and_peaks_on_identical_rows() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    kernel = _rbf_kernel(vectors, vectors, gamma=1.0)

    assert kernel == pytest.approx(kernel.T)
    assert np.diag(kernel).tolist() == pytest.approx([1.0, 1.0])
    assert kernel[0, 1] < kernel[0, 0]


def test_precomputed_vectors_reject_different_row_ids(tmp_path: Path) -> None:
    archive = tmp_path / "vectors.npz"
    np.savez_compressed(
        archive,
        vectors=np.zeros((1, 2), dtype=np.float32),
        ids=np.asarray(["wrong-id"]),
        model=np.asarray("example/model"),
        prefix=np.asarray(""),
        load_seconds=np.asarray(1.0),
        corpus_seconds=np.asarray(1.0),
        warm_single_p50_ms=np.asarray(1.0),
        warm_single_p95_ms=np.asarray(1.0),
        rss_delta_mib=np.asarray(1.0),
    )

    with pytest.raises(ValueError, match="ids do not match"):
        load_precomputed_vectors(archive, [{"id": "manual-cal-001"}])
