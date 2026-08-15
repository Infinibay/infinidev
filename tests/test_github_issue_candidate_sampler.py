from __future__ import annotations

from bench.github_issue_candidate_sampler import normalize_issue, select_candidates


def _node(number: int, *, repo: str = "owner/project", label: str = "bug") -> dict:
    return {
        "__typename": "Issue",
        "id": f"I_{number}",
        "number": number,
        "title": "Existing command returns a stale cached result",
        "body": "When valid input is submitted, the documented result remains stale after refresh.",
        "url": f"https://github.com/{repo}/issues/{number}",
        "state": "OPEN",
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-02T00:00:00Z",
        "closedAt": None,
        "author": {"__typename": "User", "login": "maintainer"},
        "labels": {"nodes": [{"name": label, "description": "selection hint"}]},
        "repository": {
            "nameWithOwner": repo,
            "url": f"https://github.com/{repo}",
            "isPrivate": False,
            "isArchived": False,
            "isFork": False,
            "stargazerCount": 500,
            "primaryLanguage": {"name": "Python"},
            "licenseInfo": {"spdxId": "Apache-2.0", "name": "Apache License 2.0"},
        },
    }


def test_normalize_issue_preserves_provenance_without_assigning_policy() -> None:
    candidate = normalize_issue(_node(7), query_hint="label:bug")

    assert candidate is not None
    assert candidate["candidate_id"] == "github-issue:owner/project:7"
    assert candidate["source"]["repo_license_spdx"] == "Apache-2.0"
    assert candidate["manual_review"]["policies"] is None
    assert "policies" not in candidate["source"]


def test_normalize_issue_rejects_pull_requests_bots_secrets_and_unknown_licenses() -> None:
    pull_request = _node(1)
    pull_request["__typename"] = "PullRequest"
    bot = _node(2)
    bot["author"] = {"__typename": "Bot", "login": "triage[bot]"}
    secret = _node(3)
    secret["body"] += " ghp_123456789012345678901234567890123456"
    unknown_license = _node(4)
    unknown_license["repository"]["licenseInfo"] = None

    assert normalize_issue(pull_request, query_hint="bug") is None
    assert normalize_issue(bot, query_hint="bug") is None
    assert normalize_issue(secret, query_hint="bug") is None
    assert normalize_issue(unknown_license, query_hint="bug") is None


def test_selection_balances_queries_and_limits_repositories() -> None:
    nodes = [
        ("bug", [_node(1), _node(2), _node(3, repo="two/project")]),
        ("enhancement", [_node(4, repo="three/project", label="enhancement")]),
    ]

    selected, report = select_candidates(
        nodes,
        limit=3,
        max_per_repo=1,
        min_repo_stars=50,
        seed=17,
    )

    assert len(selected) == 3
    assert len({row["source"]["repo"] for row in selected}) == 3
    assert report["selected_repositories"] == 3
