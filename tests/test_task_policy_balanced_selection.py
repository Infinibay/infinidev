"""Tests for diverse task-policy positive selection."""

from __future__ import annotations

from bench.task_policy_balanced_selection import select_balanced


def _candidate(index: int, *, repo: str, language: str, words: int) -> dict[str, object]:
    return {
        "candidate_id": f"candidate-{index}",
        "source": {"repo": repo, "programming_language": language},
        "issue_text": " ".join(f"word{index}_{item}" for item in range(words)),
    }


def _proposal(
    index: int, policies: list[str], *, confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "candidate_id": f"candidate-{index}",
        "proposal_status": "model_reviewed",
        "policies": policies,
        "confidence": confidence,
    }


def test_balanced_selection_meets_targets_with_distinct_real_rows() -> None:
    candidates = [
        _candidate(1, repo="a/one", language="Python", words=30),
        _candidate(2, repo="b/two", language="Rust", words=120),
        _candidate(3, repo="c/three", language="Go", words=350),
        _candidate(4, repo="d/four", language="TypeScript", words=90),
    ]
    proposals = [
        _proposal(1, ["research"]),
        _proposal(2, ["research", "review"]),
        _proposal(3, ["review"]),
        _proposal(4, ["feature"]),
    ]

    selected, selected_proposals, report = select_balanced(
        candidates,
        proposals,
        targets={"research": 2, "review": 2},
        minimum_confidence=0.7,
        seed=11,
    )

    assert len(selected) == len(selected_proposals) == 3
    assert report["selected_labels"] == {"research": 2, "review": 2}
    assert report["shortfalls"] == {}
    assert report["repositories"] == 3
    assert report["length_buckets"] == {"long": 1, "medium": 1, "short": 1}
    assert report["exact_normalized_text_duplicates"] == 0


def test_balanced_selection_reports_shortfalls_and_filters_low_confidence() -> None:
    candidates = [
        _candidate(1, repo="a/one", language="Python", words=30),
        _candidate(2, repo="b/two", language="Rust", words=120),
    ]
    proposals = [
        _proposal(1, ["refactor"], confidence=0.95),
        _proposal(2, ["refactor"], confidence=0.5),
    ]

    selected, _, report = select_balanced(
        candidates,
        proposals,
        targets={"refactor": 3},
        minimum_confidence=0.7,
    )

    assert [row["candidate_id"] for row in selected] == ["candidate-1"]
    assert report["selected_labels"] == {"refactor": 1}
    assert report["shortfalls"] == {"refactor": 2}
    assert report["eligible_rows"] == 1


def test_balanced_selection_rejects_unknown_proposal_candidates() -> None:
    candidates = [_candidate(1, repo="a/one", language="Python", words=30)]
    proposals = [_proposal(2, ["review"])]

    try:
        select_balanced(candidates, proposals, targets={"review": 1})
    except ValueError as exc:
        assert "unknown candidates" in str(exc)
    else:
        raise AssertionError("unknown proposal candidate must fail")


def test_balanced_selection_never_fills_a_target_with_duplicate_texts() -> None:
    first = _candidate(1, repo="a/one", language="Python", words=30)
    duplicate = _candidate(2, repo="b/two", language="Rust", words=30)
    duplicate["issue_text"] = first["issue_text"]

    selected, _, report = select_balanced(
        [first, duplicate],
        [_proposal(1, ["research"]), _proposal(2, ["research"])],
        targets={"research": 2},
    )

    assert len(selected) == 1
    assert report["shortfalls"] == {"research": 1}
    assert report["exact_normalized_text_duplicates"] == 0


def test_balanced_selection_excludes_entire_holdout_source_family() -> None:
    same_repo = _candidate(1, repo="holdout/shared", language="Python", words=30)
    independent = _candidate(2, repo="train/other", language="Rust", words=120)
    holdout = {
        "candidate_id": "holdout-1",
        "source": {"repo": "holdout/shared"},
        "issue_text": "A different request from the held-out repository.",
    }

    selected, _, report = select_balanced(
        [same_repo, independent],
        [_proposal(1, ["research"]), _proposal(2, ["research"])],
        targets={"research": 2},
        excluded_family_candidates=[holdout],
    )

    assert [row["candidate_id"] for row in selected] == ["candidate-2"]
    assert report["family_excluded_rows"] == 1
    assert report["shortfalls"] == {"research": 1}
