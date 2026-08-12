"""Tests for Open-SWE manual-review candidate extraction."""

from __future__ import annotations

import hashlib
import json

import pytest

from bench.open_swe_candidate_sampler import (
    SOURCE_REVISION,
    candidate_from_row,
    exclusion_sets,
    initial_user_message,
    issue_description,
    select_candidates,
    write_candidate_queue,
)


def _row(instance: str, repo: str, category: str = "bug-fix") -> dict[str, object]:
    return {
        "instance_id": instance,
        "trajectory_id": f"trajectory-{instance}",
        "repo": repo,
        "license": "MIT",
        "language": "python",
        "resolved": 1,
        "metadata": {"category": category},
        "trajectory": [
            {"role": "system", "content": "wrapper"},
            {
                "role": "user",
                "content": (
                    "I've uploaded a repository. <issue_description>"
                    f"Issue {instance} has enough distinct words for careful manual review and labeling."
                    "</issue_description> synthetic suffix"
                ),
            },
        ],
    }


def test_issue_description_removes_benchmark_wrapper_and_suffix() -> None:
    message = "prefix <issue_description>Real issue text.\nMore evidence.</issue_description> suffix"

    assert issue_description(message) == "Real issue text.\nMore evidence."


def test_issue_description_rejects_unknown_or_incomplete_wrapper() -> None:
    with pytest.raises(ValueError, match="complete issue_description"):
        issue_description("No tagged issue here")


def test_initial_user_message_skips_other_roles_and_empty_messages() -> None:
    trajectory = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": " "},
        {"role": "assistant", "content": "assistant"},
        {"role": "user", "content": "actual request"},
    ]

    assert initial_user_message(trajectory) == "actual request"


def test_candidate_keeps_upstream_category_as_hint_without_policy_mapping() -> None:
    candidate = candidate_from_row(_row("one", "owner/repo", "feature-request"))

    assert candidate["issue_text"].startswith("Issue one")
    assert candidate["source"]["upstream_category_hint"] == "feature-request"
    assert candidate["source"]["dataset_revision"] == SOURCE_REVISION
    assert candidate["source"]["dataset_license"] == "CC-BY-4.0"
    assert candidate["manual_review"]["policies"] is None
    assert candidate["manual_review"]["status"] == "unreviewed"


def test_selection_is_deterministic_balanced_and_repo_limited() -> None:
    rows = [
        _row(f"bug-{index}", "owner/shared" if index < 3 else f"owner/bug-{index}")
        for index in range(5)
    ] + [
        _row(f"feature-{index}", f"owner/feature-{index}", "feature-request")
        for index in range(5)
    ]

    first, report = select_candidates(rows, limit=6, max_per_repo=1, seed=7)
    second, _ = select_candidates(rows, limit=6, max_per_repo=1, seed=7)

    assert first == second
    assert len(first) == 6
    assert report["selected_category_hints"] == {"bug-fix": 3, "feature-request": 3}
    assert len({candidate["source"]["repo"] for candidate in first}) == 6


def test_selection_excludes_prior_candidates_and_repositories(tmp_path) -> None:
    prior = tmp_path / "prior.jsonl"
    prior.write_text(
        json.dumps(candidate_from_row(_row("old", "owner/prior"))) + "\n",
        encoding="utf-8",
    )
    candidate_ids, repositories = exclusion_sets([prior])
    selected, report = select_candidates(
        [
            _row("old", "owner/prior"),
            _row("same-repo", "owner/prior"),
            _row("fresh", "owner/fresh"),
        ],
        limit=3,
        max_per_repo=3,
        seed=7,
        excluded_candidate_ids=candidate_ids,
        excluded_repositories=repositories,
    )

    assert [item["candidate_id"] for item in selected] == ["open-swe:fresh"]
    assert report["rejected"] == {
        "excluded_candidate_id": 1,
        "excluded_repository": 1,
    }


def test_write_candidate_queue_records_hash_and_provenance(tmp_path) -> None:
    candidate = candidate_from_row(_row("one", "owner/repo"))
    output = tmp_path / "candidates.jsonl"

    manifest_path = write_candidate_queue(
        output,
        [candidate],
        {"selected": 1},
        scan_limit=10,
        selection_limit=1,
        max_per_repo=1,
        seed=7,
    )

    payload = output.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["source"]["revision"] == SOURCE_REVISION
    assert manifest["source"]["license"] == "CC-BY-4.0"
    assert manifest["review_contract"]["individual_manual_review_required"] is True
