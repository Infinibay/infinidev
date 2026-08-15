from __future__ import annotations

import hashlib
import json

import pytest

from bench.task_policy_balanced_augmentation import augment_balanced_split


def _payload(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def _candidate(candidate_id: str, repo: str, text: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "issue_text": text,
        "source": {"repo": repo, "dataset": "test"},
    }


def _review(candidate_id: str, label: str, *, model: bool = False) -> dict:
    row = {
        "candidate_id": candidate_id,
        "include": True,
        "policies": [label],
        "notes": f"Classified as {label}.",
    }
    if model:
        row["annotation"] = {"kind": "model", "model": "MiniMax-M3"}
    return row


def _base_split(path) -> None:
    artifacts = {}
    rows = {
        "training": [_candidate("train", "repo/train", "Add export support.")],
        "calibration": [_candidate("cal", "repo/cal", "Investigate import failures.")],
        "evaluation": [_candidate("eval", "repo/eval", "Speed up archive creation.")],
    }
    labels = {"training": "feature", "calibration": "research", "evaluation": "performance"}
    for partition, candidates in rows.items():
        reviews = [_review(str(candidates[0]["candidate_id"]), labels[partition])]
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


def test_balanced_augmentation_counts_rows_and_preserves_evaluation(tmp_path) -> None:
    base = tmp_path / "base"
    output = tmp_path / "output"
    base.mkdir()
    _base_split(base)
    candidates = tmp_path / "new.jsonl"
    reviews = tmp_path / "new_reviews.jsonl"
    candidate_rows = [
        _candidate("new-a", "repo/new", "Research intermittent cache misses."),
        _candidate("new-b", "repo/new", "Trace cache eviction behavior."),
        _candidate("forced", "repo/train", "Research export compatibility."),
    ]
    candidates.write_bytes(_payload(candidate_rows))
    reviews.write_bytes(_payload([
        _review("new-a", "research", model=True),
        _review("new-b", "research", model=True),
        _review("forced", "research", model=True),
    ]))

    manifest_path = augment_balanced_split(
        base,
        output,
        candidate_paths=[candidates],
        model_review_path=reviews,
        calibration_targets={"research": 2},
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["new_rows"] == {"training": 1, "calibration": 2}
    assert manifest["actual_new_calibration_labels"]["research"] == 2
    assert manifest["forced_training_rows"] == 1
    assert manifest["family_leakage"] == 0
    assert (output / "evaluation_candidates.jsonl").read_bytes() == (
        base / "evaluation_candidates.jsonl"
    ).read_bytes()
    assert (output / "evaluation_reviews.jsonl").read_bytes() == (
        base / "evaluation_reviews.jsonl"
    ).read_bytes()


@pytest.mark.parametrize("repo", ["repo/cal", "repo/eval"])
def test_balanced_augmentation_rejects_new_family_touching_holdout(tmp_path, repo) -> None:
    base = tmp_path / "base"
    base.mkdir()
    _base_split(base)
    candidates = tmp_path / "new.jsonl"
    reviews = tmp_path / "new_reviews.jsonl"
    candidates.write_bytes(_payload([
        _candidate("new", repo, "Research another failure in this component."),
    ]))
    reviews.write_bytes(_payload([_review("new", "research", model=True)]))

    with pytest.raises(ValueError, match="leaks"):
        augment_balanced_split(
            base,
            tmp_path / "output",
            candidate_paths=[candidates],
            model_review_path=reviews,
            calibration_targets={"research": 1},
        )
