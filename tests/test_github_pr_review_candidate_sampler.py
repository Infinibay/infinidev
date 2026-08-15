"""Tests for real pull-request review candidate acquisition."""

from __future__ import annotations

from bench.github_pr_review_candidate_sampler import (
    normalize_pull_request,
    select_candidates,
)


def _pull(
    number: int,
    *,
    repo: str = "org/project",
    language: str = "Python",
    body: str = "This changes the parser boundary and adds regression coverage for malformed input.",
) -> dict[str, object]:
    return {
        "__typename": "PullRequest",
        "id": f"PR_{number}",
        "number": number,
        "title": f"Parser boundary change {number}",
        "body": body,
        "url": f"https://github.com/{repo}/pull/{number}",
        "state": "MERGED",
        "isDraft": False,
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-02T00:00:00Z",
        "closedAt": "2025-01-02T00:00:00Z",
        "mergedAt": "2025-01-02T00:00:00Z",
        "author": {"__typename": "User", "login": f"author{number}"},
        "labels": {"nodes": [{"name": "parser", "description": ""}]},
        "repository": {
            "nameWithOwner": repo,
            "url": f"https://github.com/{repo}",
            "isPrivate": False,
            "isArchived": False,
            "isFork": False,
            "stargazerCount": 500,
            "primaryLanguage": {"name": language},
            "licenseInfo": {"spdxId": "MIT", "name": "MIT License"},
        },
    }


def test_normalize_pull_request_preserves_real_source_and_adds_review_intent() -> None:
    candidate = normalize_pull_request(_pull(7), query_hint="language:Python")

    assert candidate is not None
    assert candidate["candidate_id"] == "github-pr-review:org/project:7"
    assert "Parser boundary change 7" in candidate["issue_text"]
    assert "regression coverage" in candidate["issue_text"]
    assert candidate["source"]["programming_language"] == "Python"
    assert candidate["source"]["task_transform"] == "explicit-read-only-review-wrapper-v1"
    assert candidate["manual_review"]["status"] == "unreviewed"


def test_normalize_pull_request_rejects_drafts_bots_and_unlicensed_repositories() -> None:
    draft = _pull(1)
    draft["isDraft"] = True
    bot = _pull(2)
    bot["author"] = {"__typename": "Bot", "login": "release[bot]"}
    unlicensed = _pull(3)
    unlicensed["repository"]["licenseInfo"] = {"spdxId": "NOASSERTION"}

    assert normalize_pull_request(draft, query_hint="x") is None
    assert normalize_pull_request(bot, query_hint="x") is None
    assert normalize_pull_request(unlicensed, query_hint="x") is None


def test_selection_limits_repositories_and_retains_language_diversity() -> None:
    nodes = [
        _pull(1, repo="org/python", language="Python"),
        _pull(2, repo="org/python", language="Python"),
        _pull(3, repo="org/rust", language="Rust"),
        _pull(4, repo="org/go", language="Go"),
    ]

    selected, report = select_candidates(
        [("all", nodes)], limit=4, max_per_repo=1, min_repo_stars=20, seed=9,
    )

    assert len(selected) == 3
    assert report["selected_repositories"] == 3
    assert report["languages"] == {"Go": 1, "Python": 1, "Rust": 1}
