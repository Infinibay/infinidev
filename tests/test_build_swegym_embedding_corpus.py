from __future__ import annotations

from bench.build_swegym_embedding_corpus import _split, record_from_row


def _row(repository: str = "pydantic/pydantic") -> dict[str, str]:
    return {
        "instance_id": "pydantic__pydantic-123",
        "problem_statement": "Validation fails when an alias is nested inside a model.",
        "patch": "@@ -1,2 +1,2 @@\n-old()\n+new()",
        "repo": repository,
        "base_commit": "a" * 40,
    }


def test_verified_record_retains_repository_license_and_group_split() -> None:
    record = record_from_row(
        _row(), max_problem_chars=3_000, max_patch_chars=6_000
    )
    assert record is not None
    assert record["licenses"] == ["MIT"]
    assert record["kind"] == "issue_to_patch"
    assert record["split"] == _split("pydantic/pydantic")
    assert record["programming_language"] == "python"


def test_unknown_repository_is_rejected() -> None:
    assert record_from_row(
        _row("unknown/repo"), max_problem_chars=3_000, max_patch_chars=6_000
    ) is None


def test_non_patch_is_rejected() -> None:
    row = _row()
    row["patch"] = "plain text"
    assert record_from_row(
        row, max_problem_chars=3_000, max_patch_chars=6_000
    ) is None
