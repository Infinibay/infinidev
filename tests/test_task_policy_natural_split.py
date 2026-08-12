from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.task_policy_natural_split import (
    SPLIT_NAMES,
    load_reviewed_rows,
    split_reviewed_families,
    write_natural_splits,
)


LABELS = ("bugfix", "feature", "performance", "refactor", "research", "review")


def _candidate(candidate_id: str, conversation: str, text: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "issue_text": text,
        "source": {
            "conversation_id": conversation,
            "dataset": "fixture/natural",
            "repo": f"fixture/{conversation}",
            "programming_language": "Python",
        },
    }


def _review(candidate_id: str, policies: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "include": True,
        "policies": policies,
        "notes": "Manually fixed fixture decision.",
    }
    if not policies:
        row["uncategorized_reason"] = "answer_only"
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_split_keeps_source_families_atomic_and_each_label_supported() -> None:
    candidates = []
    decisions = {}
    for label_index, label in enumerate(LABELS):
        for example_index in range(6):
            candidate_id = f"{label}-{example_index}"
            conversation = f"conversation-{label_index}-{example_index}"
            candidates.append(_candidate(
                candidate_id,
                conversation,
                f"opaque-{label_index}-{example_index}",
            ))
            decisions[candidate_id] = _review(candidate_id, [label])
    candidates.extend([
        _candidate("paired-a", "paired-conversation", "Explain this parser result."),
        _candidate("paired-b", "paired-conversation", "Review this parser result."),
    ])
    decisions["paired-a"] = _review("paired-a", [])
    decisions["paired-b"] = _review("paired-b", ["review"])

    splits, report = split_reviewed_families(
        candidates,
        decisions,
        trials=64,
        minimum_positive_support=1,
    )

    locations = {
        str(row["candidate_id"]): split_index
        for split_index, rows in enumerate(splits)
        for row in rows
    }
    assert locations["paired-a"] == locations["paired-b"]
    assert sum(len(rows) for rows in splits) == len(candidates)
    for split_name in SPLIT_NAMES:
        assert all(
            support >= 1
            for support in report["splits"][split_name]["labels"].values()
        )


def test_load_reviewed_rows_rejects_unknown_zero_label_reason(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    review_path = tmp_path / "reviews.jsonl"
    _write_jsonl(candidate_path, [_candidate("one", "conversation-one", "A request")])
    review = _review("one", [])
    review["uncategorized_reason"] = "made_up"
    _write_jsonl(review_path, [review])

    with pytest.raises(ValueError, match="invalid zero-label reason"):
        load_reviewed_rows([candidate_path], [review_path])


def test_write_natural_splits_writes_matching_candidate_review_pairs(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    review_path = tmp_path / "reviews.jsonl"
    candidates = []
    reviews = []
    for label_index, label in enumerate(LABELS):
        for example_index in range(6):
            candidate_id = f"{label}-{example_index}"
            candidates.append(_candidate(
                candidate_id,
                f"conversation-{label_index}-{example_index}",
                f"fixture-{label_index}-{example_index}",
            ))
            reviews.append(_review(candidate_id, [label]))
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(review_path, reviews)

    manifest_path = write_natural_splits(
        tmp_path / "splits",
        candidate_paths=[candidate_path],
        review_paths=[review_path],
        trials=64,
        minimum_positive_support=1,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rows"] == len(candidates)
    for split_name in SPLIT_NAMES:
        candidate_ids = {
            json.loads(line)["candidate_id"]
            for line in (tmp_path / "splits" / f"{split_name}_candidates.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        }
        review_ids = {
            json.loads(line)["candidate_id"]
            for line in (tmp_path / "splits" / f"{split_name}_reviews.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        }
        assert candidate_ids == review_ids
