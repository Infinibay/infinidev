"""Partition an external candidate queue into repository-disjoint review shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def partition_by_repository(
    rows: list[dict[str, Any]],
    *,
    partition_count: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Return deterministic, nearly balanced shards without splitting repositories."""
    if partition_count < 2:
        raise ValueError("partition_count must be at least two")
    by_repository: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        source = row.get("source")
        repository = str(source.get("repo", "")) if isinstance(source, dict) else ""
        if not candidate_id or not repository:
            raise ValueError("every candidate needs candidate_id and source.repo")
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        by_repository[repository].append(row)

    repository_groups = sorted(
        by_repository.items(),
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).digest(),
    )
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(partition_count)]
    partition_repositories: list[set[str]] = [set() for _ in range(partition_count)]
    for repository, group in repository_groups:
        index = min(range(partition_count), key=lambda value: (len(partitions[value]), value))
        partitions[index].extend(group)
        partition_repositories[index].add(repository)

    for partition in partitions:
        partition.sort(key=lambda row: str(row["candidate_id"]))
    if sum(map(len, partitions)) != len(rows):
        raise AssertionError("partitioning lost candidates")
    if sum(map(len, partition_repositories)) != len(by_repository):
        raise AssertionError("partitioning reused a repository")
    return partitions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def write_partitions(
    source: Path,
    output_prefix: Path,
    *,
    partition_count: int,
    seed: int,
) -> list[Path]:
    """Write ignored candidate shards and return their paths."""
    partitions = partition_by_repository(
        _read_jsonl(source), partition_count=partition_count, seed=seed
    )
    outputs = []
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for index, rows in enumerate(partitions):
        path = output_prefix.with_name(f"{output_prefix.name}_{index}.jsonl")
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        path.write_text(payload, encoding="utf-8")
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=313)
    args = parser.parse_args()
    outputs = write_partitions(
        args.source,
        args.output_prefix,
        partition_count=args.partitions,
        seed=args.seed,
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["partition_by_repository", "write_partitions"]
