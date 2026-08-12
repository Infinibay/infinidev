"""Create near-duplicate-family-disjoint splits for external request data.

This module never assigns task-policy labels. It groups source candidates by
conversation/repository identity and conservative lexical overlap so prompt
variants cannot leak between development and a sealed reserve.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


_TOKEN = re.compile(r"https?://\S+|[\w]+", flags=re.UNICODE | re.IGNORECASE)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read LF-delimited JSON without splitting Unicode separators in strings."""
    rows: list[dict[str, Any]] = []
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


def _normalized_tokens(text: str) -> tuple[str, ...]:
    tokens = []
    for match in _TOKEN.finditer(text.casefold()):
        token = match.group(0)
        if token.startswith(("http://", "https://")):
            tokens.append("<url>")
        elif token.isdecimal():
            tokens.append("<number>")
        else:
            tokens.append(token)
    return tuple(tokens)


def _shingles(tokens: tuple[str, ...]) -> frozenset[bytes]:
    if not tokens:
        return frozenset()
    width = 3 if len(tokens) <= 20 else 5
    if len(tokens) < width:
        value = "\x1f".join(tokens).encode()
        return frozenset({hashlib.blake2b(value, digest_size=8).digest()})
    return frozenset(
        hashlib.blake2b("\x1f".join(tokens[index:index + width]).encode(), digest_size=8).digest()
        for index in range(len(tokens) - width + 1)
    )


def candidate_families(
    rows: list[dict[str, Any]],
    *,
    min_containment: float = 0.55,
    max_shingle_frequency: int = 80,
) -> list[list[dict[str, Any]]]:
    """Group exact source families and conservative near-duplicate requests."""
    if not 0 < min_containment <= 1:
        raise ValueError("min_containment must be in (0, 1]")
    if max_shingle_frequency < 2:
        raise ValueError("max_shingle_frequency must be at least two")
    seen_ids: set[str] = set()
    sources: dict[str, int] = {}
    shingle_sets: list[frozenset[bytes]] = []
    token_counts: list[int] = []
    disjoint = _DisjointSet(len(rows))
    for index, row in enumerate(rows):
        candidate_id = str(row.get("candidate_id", ""))
        source = row.get("source")
        text = str(row.get("issue_text", ""))
        if not candidate_id or not isinstance(source, dict) or not text.strip():
            raise ValueError("every candidate needs candidate_id, source, and issue_text")
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        source_group = str(
            source.get("conversation_id")
            or source.get("repo")
            or candidate_id
        )
        previous = sources.setdefault(source_group, index)
        disjoint.union(previous, index)
        tokens = _normalized_tokens(text)
        token_counts.append(len(tokens))
        shingle_sets.append(_shingles(tokens))

    postings: dict[bytes, list[int]] = defaultdict(list)
    for index, fingerprints in enumerate(shingle_sets):
        for fingerprint in fingerprints:
            postings[fingerprint].append(index)

    pair_hits: dict[tuple[int, int], int] = defaultdict(int)
    for indexes in postings.values():
        if len(indexes) < 2 or len(indexes) > max_shingle_frequency:
            continue
        for left_position, left in enumerate(indexes[:-1]):
            for right in indexes[left_position + 1:]:
                pair_hits[(left, right)] += 1

    for (left, right), shared_hint in pair_hits.items():
        minimum_size = min(len(shingle_sets[left]), len(shingle_sets[right]))
        if not minimum_size:
            continue
        required_hits = 2 if min(token_counts[left], token_counts[right]) <= 20 else 4
        if shared_hint < required_hits:
            continue
        intersection = len(shingle_sets[left] & shingle_sets[right])
        containment = intersection / minimum_size
        threshold = 0.4 if min(token_counts[left], token_counts[right]) <= 20 else min_containment
        if containment >= threshold:
            disjoint.union(left, right)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[disjoint.find(index)].append(row)
    families = list(groups.values())
    for family in families:
        family.sort(key=lambda row: str(row["candidate_id"]))
    families.sort(key=lambda family: str(family[0]["candidate_id"]))
    return families


def reviewed_candidate_ids(paths: Iterable[Path]) -> frozenset[str]:
    """Return IDs already exposed to human decisions."""
    identifiers: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id:
                raise ValueError(f"{path}: review row is missing candidate_id")
            identifiers.add(candidate_id)
    return frozenset(identifiers)


def _family_rank(family: list[dict[str, Any]], seed: int) -> bytes:
    identifiers = ":".join(str(row["candidate_id"]) for row in family)
    return hashlib.sha256(f"{seed}:{identifiers}".encode()).digest()


def split_development_queue_reserve(
    rows: list[dict[str, Any]],
    *,
    reviewed_ids: frozenset[str],
    reserve_target: int,
    queue_partitions: int,
    seed: int,
    min_containment: float = 0.55,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """Split families while excluding every reviewed family from the reserve."""
    if reserve_target < 1 or queue_partitions < 1:
        raise ValueError("reserve_target and queue_partitions must be positive")
    families = candidate_families(rows, min_containment=min_containment)
    known_ids = {str(row["candidate_id"]) for row in rows}
    unknown_reviews = reviewed_ids - known_ids
    if unknown_reviews:
        raise ValueError(f"reviewed IDs not present in candidates: {sorted(unknown_reviews)[:3]}")

    development_families = []
    unseen_families = []
    for family in families:
        identifiers = {str(row["candidate_id"]) for row in family}
        target = development_families if identifiers & reviewed_ids else unseen_families
        target.append(family)
    unseen_families.sort(key=lambda family: _family_rank(family, seed))

    reserve_families: list[list[dict[str, Any]]] = []
    reserve_size = 0
    while unseen_families and reserve_size < reserve_target:
        family = unseen_families.pop()
        reserve_families.append(family)
        reserve_size += len(family)

    queue: list[list[dict[str, Any]]] = [[] for _ in range(queue_partitions)]
    for family in sorted(unseen_families, key=lambda item: (-len(item), _family_rank(item, seed))):
        index = min(range(queue_partitions), key=lambda value: (len(queue[value]), value))
        queue[index].extend(family)

    development = [row for family in development_families for row in family]
    reserve = [row for family in reserve_families for row in family]
    for collection in [development, reserve, *queue]:
        collection.sort(key=lambda row: str(row["candidate_id"]))
    assigned_ids = {
        str(row["candidate_id"])
        for row in [*development, *reserve, *(item for block in queue for item in block)]
    }
    if assigned_ids != known_ids:
        raise AssertionError("family split lost or duplicated candidates")
    family_sizes = sorted((len(family) for family in families), reverse=True)
    report = {
        "candidates": len(rows),
        "families": len(families),
        "near_duplicate_members": sum(size for size in family_sizes if size > 1),
        "multi_member_families": sum(size > 1 for size in family_sizes),
        "largest_family_sizes": family_sizes[:10],
        "reviewed_ids": len(reviewed_ids),
        "development_family_rows": len(development),
        "queue_rows": sum(map(len, queue)),
        "queue_partition_sizes": list(map(len, queue)),
        "reserve_rows": len(reserve),
        "reserve_target": reserve_target,
        "interpretation": (
            "No task labels were inferred. Every family containing an already reviewed candidate "
            "was excluded from the sealed reserve."
        ),
    }
    return development, queue, reserve, report


def _payload(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def write_family_split(
    source: Path,
    output_prefix: Path,
    *,
    review_ledgers: Iterable[Path],
    reserve_target: int,
    queue_partitions: int,
    seed: int,
    min_containment: float,
) -> Path:
    """Write ignored family-disjoint artifacts and a provenance manifest."""
    rows = read_jsonl(source)
    reviews = reviewed_candidate_ids(review_ledgers)
    development, queue, reserve, report = split_development_queue_reserve(
        rows,
        reviewed_ids=reviews,
        reserve_target=reserve_target,
        queue_partitions=queue_partitions,
        seed=seed,
        min_containment=min_containment,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    artifacts = [("development", development), ("reserve", reserve)] + [
        (f"queue_{index}", block) for index, block in enumerate(queue)
    ]
    for name, artifact_rows in artifacts:
        path = output_prefix.with_name(f"{output_prefix.name}_{name}.jsonl")
        payload = _payload(artifact_rows)
        path.write_text(payload, encoding="utf-8")
        outputs[name] = {
            "path": path.name,
            "rows": len(artifact_rows),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    manifest_path = output_prefix.with_name(f"{output_prefix.name}_manifest.json")
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": seed,
        "min_containment": min_containment,
        "outputs": outputs,
        "report": report,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--review-ledger", type=Path, action="append", default=[])
    parser.add_argument("--reserve-target", type=int, default=100)
    parser.add_argument("--queue-partitions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1223)
    parser.add_argument("--min-containment", type=float, default=0.55)
    args = parser.parse_args()
    manifest = write_family_split(
        args.source,
        args.output_prefix,
        review_ledgers=args.review_ledger,
        reserve_target=args.reserve_target,
        queue_partitions=args.queue_partitions,
        seed=args.seed,
        min_containment=args.min_containment,
    )
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()


__all__ = [
    "candidate_families",
    "read_jsonl",
    "reviewed_candidate_ids",
    "split_development_queue_reserve",
    "write_family_split",
]
