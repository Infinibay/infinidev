"""Tests for repository-disjoint Open-SWE candidate partitions."""

from __future__ import annotations

import json

from bench.open_swe_candidate_partition import partition_by_repository, write_partitions


def _candidate(candidate_id: str, repository: str) -> dict[str, object]:
    return {"candidate_id": candidate_id, "source": {"repo": repository}}


def test_partition_is_deterministic_balanced_and_repository_disjoint() -> None:
    rows = [
        _candidate("a-1", "owner/a"),
        _candidate("a-2", "owner/a"),
        _candidate("b-1", "owner/b"),
        _candidate("c-1", "owner/c"),
        _candidate("d-1", "owner/d"),
        _candidate("e-1", "owner/e"),
    ]

    first = partition_by_repository(rows, partition_count=2, seed=9)
    second = partition_by_repository(rows, partition_count=2, seed=9)

    assert first == second
    assert sorted(map(len, first)) == [3, 3]
    repository_sets = [
        {str(row["source"]["repo"]) for row in partition}  # type: ignore[index]
        for partition in first
    ]
    assert repository_sets[0].isdisjoint(repository_sets[1])


def test_write_partitions_preserves_unicode_line_separator_inside_json(tmp_path) -> None:
    source = tmp_path / "candidates.jsonl"
    row = _candidate("one", "owner/repo")
    row["issue_text"] = "Explain this Python code.\u2028Then review its edge cases."
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    outputs = write_partitions(source, tmp_path / "part", partition_count=2, seed=7)

    loaded = [
        json.loads(line)
        for path in outputs
        for line in path.read_text(encoding="utf-8").split("\n")
        if line
    ]
    assert loaded == [row]
