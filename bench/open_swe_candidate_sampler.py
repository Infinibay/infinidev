"""Build a manual-review queue from Open-SWE-Traces issue prompts.

This utility does not create training examples and never maps upstream metadata
to Infinidev policies. It only removes benchmark scaffolding, deduplicates task
instances, and selects a repository-diverse queue for individual human review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


SOURCE_DATASET = "nvidia/Open-SWE-Traces"
SOURCE_CONFIG = "openhands"
SOURCE_SPLIT = "minimax_m25"
SOURCE_REVISION = "ad4805a5aa7de70d99cab0bb8f99b15304c76de0"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_URL = "https://huggingface.co/datasets/nvidia/Open-SWE-Traces"
DEFAULT_OUTPUT = Path(".infinidev/external-data/open-swe/candidates.jsonl")
_ISSUE_BLOCK = re.compile(
    r"<issue_description>\s*(.*?)\s*</issue_description>",
    flags=re.DOTALL | re.IGNORECASE,
)


def initial_user_message(trajectory: list[dict[str, Any]]) -> str:
    """Return the first non-empty user message from one agent trajectory."""
    for message in trajectory:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    raise ValueError("trajectory has no non-empty user message")


def issue_description(user_message: str) -> str:
    """Extract the original issue body and reject unknown prompt wrappers."""
    match = _ISSUE_BLOCK.search(user_message)
    if match is None:
        raise ValueError("user message has no complete issue_description block")
    return match.group(1).strip()


def candidate_from_row(
    row: dict[str, Any],
    *,
    revision: str = SOURCE_REVISION,
) -> dict[str, Any]:
    """Normalize one upstream row without assigning an Infinidev label."""
    trajectory = row["trajectory"]
    metadata = row["metadata"]
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    text = issue_description(initial_user_message(trajectory))
    return {
        "candidate_id": f"open-swe:{row['instance_id']}",
        "source": {
            "dataset": SOURCE_DATASET,
            "config": SOURCE_CONFIG,
            "split": SOURCE_SPLIT,
            "dataset_revision": revision,
            "dataset_license": SOURCE_LICENSE,
            "dataset_url": SOURCE_URL,
            "instance_id": row["instance_id"],
            "trajectory_id": row["trajectory_id"],
            "repo": row["repo"],
            "repo_license": row["license"],
            "programming_language": row["language"],
            "upstream_category_hint": metadata.get("category"),
            "trajectory_resolved": row["resolved"],
        },
        "issue_text": text,
        "manual_review": {
            "status": "unreviewed",
            "include": None,
            "policies": None,
            "uncategorized_reason": None,
            "notes": None,
        },
    }


def _rank(candidate: dict[str, Any], seed: int) -> bytes:
    value = f"{seed}:{candidate['candidate_id']}".encode()
    return hashlib.sha256(value).digest()


def select_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    limit: int,
    max_per_repo: int,
    seed: int,
    excluded_candidate_ids: frozenset[str] = frozenset(),
    excluded_repositories: frozenset[str] = frozenset(),
    revision: str = SOURCE_REVISION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic category-balanced, repository-diverse queue."""
    if limit < 1 or max_per_repo < 1:
        raise ValueError("limit and max_per_repo must be positive")
    by_instance: dict[str, dict[str, Any]] = {}
    rejected = Counter()
    for row in rows:
        instance_id = str(row.get("instance_id", ""))
        if not instance_id or instance_id in by_instance:
            rejected["duplicate_or_missing_instance"] += 1
            continue
        try:
            candidate = candidate_from_row(row, revision=revision)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            rejected["invalid_prompt_shape"] += 1
            continue
        if candidate["candidate_id"] in excluded_candidate_ids:
            rejected["excluded_candidate_id"] += 1
            continue
        if candidate["source"]["repo"] in excluded_repositories:
            rejected["excluded_repository"] += 1
            continue
        word_count = len(candidate["issue_text"].split())
        if not 12 <= word_count <= 1500:
            rejected["length_outside_review_budget"] += 1
            continue
        by_instance[instance_id] = candidate

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in by_instance.values():
        hint = str(candidate["source"]["upstream_category_hint"] or "unknown")
        buckets[hint].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda candidate: _rank(candidate, seed))

    selected = []
    repo_counts: Counter[str] = Counter()
    names = sorted(buckets)
    positions = Counter()
    while len(selected) < limit:
        progressed = False
        for name in names:
            bucket = buckets[name]
            while positions[name] < len(bucket):
                candidate = bucket[positions[name]]
                positions[name] += 1
                repo = str(candidate["source"]["repo"])
                if repo_counts[repo] >= max_per_repo:
                    continue
                selected.append(candidate)
                repo_counts[repo] += 1
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    selected.sort(key=lambda candidate: candidate["candidate_id"])
    return selected, {
        "source_rows": sum(rejected.values()) + len(by_instance),
        "unique_valid_instances": len(by_instance),
        "selected": len(selected),
        "selected_repositories": len(repo_counts),
        "selected_category_hints": dict(sorted(Counter(
            str(candidate["source"]["upstream_category_hint"] or "unknown")
            for candidate in selected
        ).items())),
        "rejected": dict(sorted(rejected.items())),
        "interpretation": (
            "Upstream categories are selection hints only; every Infinidev policy remains unset "
            "until individual manual review."
        ),
    }


def exclusion_sets(paths: Iterable[Path]) -> tuple[frozenset[str], frozenset[str]]:
    """Read candidate IDs and repositories that a new queue must not reuse."""
    candidate_ids: set[str] = set()
    repositories: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidate_id = str(row["candidate_id"])
                repository = str(row["source"]["repo"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid candidate row") from exc
            candidate_ids.add(candidate_id)
            repositories.add(repository)
    return frozenset(candidate_ids), frozenset(repositories)


def stream_rows(
    *,
    scan_limit: int,
    revision: str = SOURCE_REVISION,
) -> Iterable[dict[str, Any]]:
    """Stream a bounded upstream prefix without downloading full trajectories."""
    from datasets import load_dataset

    dataset = load_dataset(
        SOURCE_DATASET,
        SOURCE_CONFIG,
        split=SOURCE_SPLIT,
        revision=revision,
        streaming=True,
    )
    yield from dataset.take(scan_limit)


def write_candidate_queue(
    output: Path,
    candidates: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    scan_limit: int,
    selection_limit: int,
    max_per_repo: int,
    seed: int,
    revision: str = SOURCE_REVISION,
) -> Path:
    """Write ignored source data plus a provenance manifest beside it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n"
        for candidate in candidates
    )
    output.write_text(payload, encoding="utf-8")
    manifest_path = output.with_suffix(output.suffix + ".provenance.json")
    manifest = {
        "artifact": {
            "path": output.name,
            "rows": len(candidates),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "distribution_notice": (
                "This downloaded artifact is not part of Infinidev's MIT-licensed source. "
                "Preserve the upstream dataset and repository attribution and license metadata."
            ),
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "config": SOURCE_CONFIG,
            "split": SOURCE_SPLIT,
            "revision": revision,
            "license": SOURCE_LICENSE,
            "url": SOURCE_URL,
        },
        "selection": {
            "scan_limit": scan_limit,
            "selection_limit": selection_limit,
            "max_per_repo": max_per_repo,
            "seed": seed,
            "report": report,
        },
        "review_contract": {
            "upstream_category_is_policy_label": False,
            "individual_manual_review_required": True,
            "source_text_remains_external": True,
            "reviewed_annotations_path": "data/task-policy-reviews/open-swe",
            "reviewed_annotations_license": "CC-BY-4.0",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--max-per-repo", type=int, default=3)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--exclude-candidates", type=Path, action="append", default=[])
    parser.add_argument("--allow-excluded-repositories", action="store_true")
    args = parser.parse_args()
    excluded_ids, excluded_repositories = exclusion_sets(args.exclude_candidates)
    candidates, report = select_candidates(
        stream_rows(scan_limit=args.scan_limit, revision=args.revision),
        limit=args.limit,
        max_per_repo=args.max_per_repo,
        seed=args.seed,
        excluded_candidate_ids=excluded_ids,
        excluded_repositories=(
            frozenset() if args.allow_excluded_repositories else excluded_repositories
        ),
        revision=args.revision,
    )
    manifest_path = write_candidate_queue(
        args.output,
        candidates,
        report,
        scan_limit=args.scan_limit,
        selection_limit=args.limit,
        max_per_repo=args.max_per_repo,
        seed=args.seed,
        revision=args.revision,
    )
    report["output"] = str(args.output)
    report["provenance_manifest"] = str(manifest_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "SOURCE_CONFIG",
    "SOURCE_DATASET",
    "SOURCE_LICENSE",
    "SOURCE_REVISION",
    "SOURCE_SPLIT",
    "SOURCE_URL",
    "candidate_from_row",
    "exclusion_sets",
    "initial_user_message",
    "issue_description",
    "select_candidates",
    "write_candidate_queue",
]
