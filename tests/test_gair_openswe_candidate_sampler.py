from __future__ import annotations

from bench.gair_openswe_candidate_sampler import candidate_from_row, select_candidates


def _row(instance_id: str, *, repo: str = "owner/project", text: str | None = None) -> dict:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "problem_statement": text or (
            "The documented command crashes on valid input after the latest release. "
            "Restore the previous behavior and preserve compatibility."
        ),
        "base_commit": "abc123",
        "created_at": "2025-01-01",
        "license": "mit",
        "license_name": "MIT License",
        "patch": "must not be copied",
        "Dockerfile": "must not be copied",
    }


def test_candidate_copies_only_request_and_provenance() -> None:
    candidate = candidate_from_row(
        _row("owner__project-1"),
        config="openswe_oss",
        filtered_ids=frozenset({"owner__project-1"}),
    )

    assert candidate is not None
    assert candidate["candidate_id"] == "gair-openswe:owner__project-1"
    assert candidate["manual_review"]["policies"] is None
    assert "patch" not in candidate and "Dockerfile" not in candidate
    assert candidate["source"]["difficulty_filtered"] is True


def test_candidate_rejects_unfiltered_or_secret_bearing_rows() -> None:
    assert candidate_from_row(
        _row("not-filtered"), config="openswe_oss", filtered_ids=frozenset()
    ) is None
    assert candidate_from_row(
        _row("secret", text="Please diagnose this leaked token ghp_123456789012345678901234567890123456"),
        config="openswe_oss",
        filtered_ids=frozenset({"secret"}),
    ) is None


def test_selection_is_repository_diverse_and_stratified() -> None:
    rows = [
        _row("one__repo-1", repo="one/repo"),
        _row("one__repo-2", repo="one/repo"),
        _row("two__repo-1", repo="two/repo", text=(
            "Improve request latency and add a representative benchmark for this workload. "
            "The current result is correct but takes several seconds."
        )),
        _row("three__repo-1", repo="three/repo", text=(
            "Refactor the parser into independent components while preserving all public outputs "
            "and compatibility with existing callers."
        )),
    ]
    filtered = frozenset(row["instance_id"] for row in rows)

    selected, report = select_candidates(
        [("openswe_oss", rows)],
        filtered_ids=filtered,
        limit=3,
        max_per_repo=1,
        seed=5,
    )

    assert len(selected) == 3
    assert len({row["source"]["repo"] for row in selected}) == 3
    assert report["selected_repositories"] == 3
