"""Tests for auditable real-issue intent adaptation."""

from __future__ import annotations

import json

from bench.task_policy_intent_adaptation import (
    ADAPTATION_VERSION,
    _used_parent_ids,
    adapt_candidates,
)


def _candidate(
    index: int,
    *,
    hint: str,
    repo: str,
    language: str = "Python",
    words: int = 60,
) -> dict[str, object]:
    return {
        "candidate_id": f"github-issue:{repo}:{index}",
        "source": {
            "repo": repo,
            "programming_language": language,
            "selection_query_hints": [f"is:issue label:{hint}"],
        },
        "issue_text": " ".join(f"concrete{index}_{item}" for item in range(words)),
    }


def test_adaptation_preserves_distinct_real_context_and_marks_intent_as_unreviewed() -> None:
    candidates = [
        _candidate(1, hint="question", repo="a/one", language="Python", words=30),
        _candidate(2, hint="investigation", repo="b/two", language="Rust", words=120),
    ]

    adapted, report = adapt_candidates(candidates, targets={"research": 2})

    assert len(adapted) == 2
    assert report["exact_normalized_text_duplicates"] == 0
    assert report["targets"]["research"]["repositories"] == 2
    for source, result in zip(candidates, adapted, strict=True):
        assert source["issue_text"] in result["issue_text"]
        assert result["manual_review"]["status"] == "unreviewed"
        assert result["source"]["intent_adaptation_version"] == ADAPTATION_VERSION
        assert result["source"]["intended_workflow"] == "research"


def test_adaptation_spreads_repositories_and_reports_honest_shortfall() -> None:
    candidates = [
        _candidate(1, hint="refactor", repo="a/shared"),
        _candidate(2, hint="refactor", repo="a/shared"),
        _candidate(3, hint="refactor", repo="b/other"),
    ]

    adapted, report = adapt_candidates(
        candidates,
        targets={"refactor": 3},
        max_per_repo=1,
    )

    assert len(adapted) == 2
    assert report["targets"]["refactor"]["shortfall"] == 1
    assert report["targets"]["refactor"]["repositories"] == 2


def test_adaptation_excludes_parents_already_positive_for_a_target() -> None:
    candidates = [
        _candidate(1, hint="performance", repo="a/one"),
        _candidate(2, hint="optimization", repo="b/two"),
    ]

    adapted, report = adapt_candidates(
        candidates,
        targets={"performance": 2},
        excluded_parent_ids=frozenset({"github-issue:a/one:1"}),
    )

    assert len(adapted) == 1
    assert adapted[0]["source"]["parent_candidate_id"] == "github-issue:b/two:2"
    assert report["targets"]["performance"]["shortfall"] == 1


def test_one_parent_is_never_reused_for_two_intended_workflows() -> None:
    shared = _candidate(1, hint="performance question", repo="a/one")
    other = _candidate(2, hint="question", repo="b/two")

    adapted, _ = adapt_candidates(
        [shared, other],
        targets={"performance": 1, "research": 1},
    )

    parents = [row["source"]["parent_candidate_id"] for row in adapted]
    assert len(parents) == len(set(parents)) == 2


def test_used_parent_ids_reads_adapted_and_raw_candidates(tmp_path) -> None:
    path = tmp_path / "used.jsonl"
    rows = [
        _candidate(1, hint="question", repo="a/one"),
        {
            **_candidate(2, hint="question", repo="b/two"),
            "source": {
                **_candidate(2, hint="question", repo="b/two")["source"],
                "parent_candidate_id": "github-issue:original/repo:20",
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _used_parent_ids([path]) == frozenset({
        "github-issue:a/one:1",
        "github-issue:original/repo:20",
    })
