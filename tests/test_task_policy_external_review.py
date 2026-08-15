"""Tests for manually reviewed external task-policy examples."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.task_policy_external_review import (
    clean_external_request,
    load_external_candidates,
    load_external_reviews,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(candidate_id: str = "external:one") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source": {"repo": "owner/repo", "programming_language": "python"},
        "issue_text": "The existing parser crashes when the optional field is absent.",
    }


def test_external_reviews_join_manual_labels_without_using_upstream_hints(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    candidate = _candidate()
    candidate["source"]["upstream_category_hint"] = "feature-request"  # type: ignore[index]
    _write_jsonl(candidates, [candidate])
    _write_jsonl(reviews, [{
        "candidate_id": "external:one",
        "include": True,
        "policies": ["bugfix", "performance"],
        "notes": "The request fixes a crash and explicitly asks to remove repeated work.",
    }])

    loaded = load_external_reviews(candidates, reviews)

    assert len(loaded) == 1
    assert loaded[0].repo == "owner/repo"
    assert loaded[0].policies == (
        "bugfix.root_cause",
        "performance.measure_first",
    )
    assert loaded[0].as_example().split == "external_manual_review"


def test_external_reviews_allow_a_reasoned_uncategorized_decision(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(candidates, [_candidate()])
    _write_jsonl(reviews, [{
        "candidate_id": "external:one",
        "include": True,
        "policies": [],
        "uncategorized_reason": "unsupported_method",
        "notes": "Documentation-only maintenance is outside the current task methods.",
    }])

    loaded = load_external_reviews(candidates, reviews)

    assert loaded[0].policies == ()
    assert loaded[0].uncategorized_reason == "unsupported_method"


def test_external_reviews_allow_out_of_domain_natural_negative(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(candidates, [_candidate()])
    _write_jsonl(reviews, [{
        "candidate_id": "external:one",
        "include": True,
        "policies": [],
        "uncategorized_reason": "out_of_domain",
        "notes": "The word program refers to an organization rather than software.",
    }])

    loaded = load_external_reviews(candidates, reviews)

    assert loaded[0].policies == ()
    assert loaded[0].uncategorized_reason == "out_of_domain"


def test_external_reviews_preserve_model_annotation_weight_inputs(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(candidates, [_candidate()])
    _write_jsonl(reviews, [{
        "candidate_id": "external:one",
        "include": True,
        "policies": ["bugfix"],
        "notes": "A model-labeled contract restoration.",
        "annotation": {"kind": "model", "confidence": 0.82},
    }])

    loaded = load_external_reviews(candidates, reviews)

    assert loaded[0].annotation_kind == "model"
    assert loaded[0].annotation_confidence == 0.82


def test_external_reviews_require_reason_only_for_uncategorized_rows(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(candidates, [_candidate()])
    _write_jsonl(reviews, [{
        "candidate_id": "external:one",
        "include": True,
        "policies": [],
        "notes": "Reviewed manually.",
    }])

    with pytest.raises(ValueError, match="needs uncategorized_reason"):
        load_external_reviews(candidates, reviews)

    _write_jsonl(reviews, [{
        "candidate_id": "external:one",
        "include": True,
        "policies": ["bugfix"],
        "uncategorized_reason": "unsupported_method",
        "notes": "Reviewed manually.",
    }])
    with pytest.raises(ValueError, match="cannot have uncategorized_reason"):
        load_external_reviews(candidates, reviews)


def test_external_reviews_reject_unknown_uncategorized_reason(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(candidates, [_candidate()])
    _write_jsonl(reviews, [{
        "candidate_id": "external:one",
        "include": True,
        "policies": [],
        "uncategorized_reason": "whatever",
        "notes": "Reviewed manually.",
    }])

    with pytest.raises(ValueError, match="unknown uncategorized_reason"):
        load_external_reviews(candidates, reviews)


def test_external_reviews_reject_unknown_labels_and_candidates(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(candidates, [_candidate()])
    _write_jsonl(reviews, [{
        "candidate_id": "external:one",
        "include": True,
        "policies": ["documentation"],
        "notes": "Reviewed manually.",
    }])

    with pytest.raises(ValueError, match="unknown policy label"):
        load_external_reviews(candidates, reviews)

    _write_jsonl(reviews, [{
        "candidate_id": "external:missing",
        "include": True,
        "policies": [],
        "uncategorized_reason": "unsupported_method",
        "notes": "Reviewed manually.",
    }])
    with pytest.raises(ValueError, match="unknown candidate"):
        load_external_reviews(candidates, reviews)


def test_external_reviews_join_multiple_candidate_queues(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(first, [_candidate("external:first")])
    _write_jsonl(second, [_candidate("external:second")])
    _write_jsonl(reviews, [{
        "candidate_id": "external:second",
        "include": True,
        "policies": ["bugfix"],
        "notes": "The second queue contains the reviewed failure.",
    }])

    loaded = load_external_reviews((first, second), reviews)

    assert [item.candidate_id for item in loaded] == ["external:second"]


def test_external_candidates_load_without_review_labels(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [_candidate("external:unlabeled")])

    loaded = load_external_candidates(candidates)

    assert [item.candidate_id for item in loaded] == ["external:unlabeled"]
    assert loaded[0].text.startswith("The existing parser crashes")


def test_external_request_removes_generated_interface_augmentation() -> None:
    text = (
        "The parser rejects valid input.\n\n"
        "New interfaces introduced: Function: generated summary"
    )

    assert clean_external_request(text) == "The parser rejects valid input."
