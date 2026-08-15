from __future__ import annotations

import hashlib
import json

import pytest

from bench.task_policy_training_augmentation import (
    _positive_support,
    augment_training_split,
)


def _payload(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def _candidate(candidate_id: str, repo: str, text: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "issue_text": text,
        "source": {"repo": repo, "dataset": "test"},
    }


def _review(candidate_id: str, *, model: bool = False) -> dict:
    row = {
        "candidate_id": candidate_id,
        "include": True,
        "policies": ["feature"],
        "notes": "Adds a new capability.",
    }
    if model:
        row["annotation"] = {"kind": "model", "model": "MiniMax-M3"}
    return row


def _base_split(path) -> None:
    artifacts = {}
    rows = {
        "training": [_candidate("train", "repo/train", "Add export support.")],
        "calibration": [_candidate("cal", "repo/cal", "Add import support.")],
        "evaluation": [_candidate("eval", "repo/eval", "Add archive support.")],
    }
    for partition, candidates in rows.items():
        reviews = [_review(str(candidates[0]["candidate_id"]))]
        candidate_payload = _payload(candidates)
        review_payload = _payload(reviews)
        (path / f"{partition}_candidates.jsonl").write_bytes(candidate_payload)
        (path / f"{partition}_reviews.jsonl").write_bytes(review_payload)
        artifacts[partition] = {
            "candidates": f"{partition}_candidates.jsonl",
            "candidates_sha256": hashlib.sha256(candidate_payload).hexdigest(),
            "reviews": f"{partition}_reviews.jsonl",
            "reviews_sha256": hashlib.sha256(review_payload).hexdigest(),
        }
    (path / "manifest.json").write_text(json.dumps({"artifacts": artifacts}))


def test_positive_support_normalizes_short_labels() -> None:
    support = _positive_support([
        {"policies": ["performance", "review"]},
        {"policies": ["performance.measure_first", "research"]},
    ])

    assert support == {
        "performance.measure_first": 2,
        "research.evidence_first": 1,
        "review.read_only": 1,
    }


def test_augmentation_adds_model_rows_only_to_training(tmp_path) -> None:
    base = tmp_path / "base"
    output = tmp_path / "output"
    base.mkdir()
    _base_split(base)
    candidates = tmp_path / "new.jsonl"
    reviews = tmp_path / "new_reviews.jsonl"
    candidates.write_bytes(_payload([
        _candidate("new", "repo/new", "Create a batch endpoint."),
    ]))
    reviews.write_bytes(_payload([_review("new", model=True)]))

    manifest_path = augment_training_split(
        base, output, candidate_paths=[candidates], model_review_path=reviews, minimum_positive_support=0
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["artifacts"]["training"]["rows"] == 2
    assert manifest["artifacts"]["calibration"]["rows"] == 1
    assert manifest["training_provenance"] == {"human": 1, "model": 1}
    assert (output / "calibration_candidates.jsonl").read_bytes() == (
        base / "calibration_candidates.jsonl"
    ).read_bytes()


def test_augmentation_rejects_model_family_leaking_into_holdout(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    _base_split(base)
    candidates = tmp_path / "new.jsonl"
    reviews = tmp_path / "new_reviews.jsonl"
    candidates.write_bytes(_payload([
        _candidate("new", "repo/cal", "Create another import endpoint."),
    ]))
    reviews.write_bytes(_payload([_review("new", model=True)]))

    with pytest.raises(ValueError, match="leaks"):
        augment_training_split(
            base,
            tmp_path / "output",
            candidate_paths=[candidates],
            model_review_path=reviews, minimum_positive_support=0,
        )
