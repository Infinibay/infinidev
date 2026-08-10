"""Build a balanced, leakage-resistant multilingual distillation mixture.

Inputs retain their original provenance. Only upstream training pools are
admitted; each semantic/code group is hash-sampled independently, exact query
pairs are deduplicated globally, and repository/contributor families receive a
stable internal train/validation/test split for fitting and model selection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from bench.fit_static_qwen3_spanish import split_for_record
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    from fit_static_qwen3_spanish import split_for_record


DEFAULT_LIMITS = {
    "instruction_to_code_change": 2_500,
    "text_to_code_retrieval": 2_000,
    "issue_to_patch": 2_500,
    "instruction_to_response": 1_000,
    "technical_prose": 5_000,
}
EXCLUDED_PROGRAMMING_LANGUAGES = {"zig"}


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _source_family(row: dict[str, Any]) -> str:
    for field in ("repository", "source_group", "path", "source_instance", "id"):
        value = row.get(field)
        if value:
            return f"{field}:{value}"
    raise ValueError("record has no stable source family")


def _group(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("source_dataset", row.get("source", "unknown"))),
        str(row.get("kind", "unknown")),
        str(row.get("language", "unknown")),
        str(row.get("programming_language", row.get("parallel_language", "none"))),
        str(row.get("query_origin", "none")),
    )


def _priority(seed: int, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{identity}".encode()).digest(), "big")


def _eligible(row: dict[str, Any]) -> bool:
    text = row.get("text")
    parallel = row.get("parallel_text")
    if not isinstance(text, str) or not text.strip():
        return False
    if not isinstance(parallel, str) or not parallel.strip():
        return False
    if str(row.get("programming_language", "")).casefold() in EXCLUDED_PROGRAMMING_LANGUAGES:
        return False
    # Pre-split datasets reserve validation/test upstream. Unsplit datasets are
    # admitted and split by family below.
    return row.get("split", "train") == "train"


def select_balanced(
    rows: Iterable[dict[str, Any]],
    *,
    limits: dict[str, int],
    default_limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Apply deterministic independent reservoirs and global pair deduplication."""
    heaps: dict[tuple[str, ...], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    seen_pairs: set[str] = set()
    for source_row in rows:
        if not _eligible(source_row):
            rejected["ineligible_or_upstream_holdout"] += 1
            continue
        row = dict(source_row)
        kind = str(row.get("kind", "unknown"))
        limit = limits.get(kind, default_limit)
        if limit <= 0:
            rejected[f"disabled_kind:{kind}"] += 1
            continue
        pair_digest = hashlib.sha256(
            (" ".join(str(row["text"]).casefold().split()) + "\0" +
             " ".join(str(row["parallel_text"]).casefold().split())).encode()
        ).hexdigest()
        if pair_digest in seen_pairs:
            rejected["duplicate_pair"] += 1
            continue
        seen_pairs.add(pair_digest)
        identity = str(row.get("id", pair_digest))
        row["split_family"] = _source_family(row)
        row["split"] = split_for_record(row, seed)
        priority = _priority(seed, identity)
        entry = (-priority, identity, row)
        heap = heaps[_group(row)]
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
            rejected["reservoir_replaced"] += 1
        else:
            rejected["reservoir_not_selected"] += 1
    selected = [entry[2] for heap in heaps.values() for entry in heap]
    return sorted(selected, key=lambda row: (_group(row), str(row.get("id", "")))), rejected


def build(args: argparse.Namespace) -> dict[str, Any]:
    all_rows = (row for path in args.input for row in _rows(path))
    records, rejected = select_balanced(
        all_rows,
        limits=DEFAULT_LIMITS,
        default_limit=args.default_limit,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return {
        "output": str(args.output),
        "records": len(records),
        "output_bytes": args.output.stat().st_size,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "by_kind": dict(sorted(Counter(str(row.get("kind")) for row in records).items())),
        "by_natural_language": dict(sorted(Counter(
            str(row.get("language")) for row in records
        ).items())),
        "by_programming_language": dict(sorted(Counter(
            str(row.get("programming_language", row.get("parallel_language", "none")))
            for row in records
        ).items())),
        "by_split": dict(sorted(Counter(str(row["split"]) for row in records).items())),
        "rejections": dict(sorted(rejected.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-limit", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.default_limit <= 0:
        parser.error("default limit must be positive")
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
