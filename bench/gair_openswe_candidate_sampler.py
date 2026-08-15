"""Extract a compact manual-review queue from the gated GAIR/OpenSWE corpus.

Only difficulty-filtered real problem statements are retained. Dataset metadata,
patches, trajectories, Dockerfiles, and evaluation scripts are not copied into
the queue. Lexical strata diversify selection and never become policy labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


SOURCE_DATASET = "GAIR/OpenSWE"
SOURCE_REVISION = "a8db93af5335df2c8baac0cd1ff367e4d475d3d7"
SOURCE_URL = "https://huggingface.co/datasets/GAIR/OpenSWE"
FILTERED_IDS_SHA256 = "77bb3910d01b6c3082956148bb22a02f053c1642a1c9996ecb60aad0b6018b27"
DEFAULT_OUTPUT = Path(".infinidev/external-data/gair-openswe/candidates.jsonl")
_SECRET = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)
_STRATA = (
    ("performance", re.compile(
        r"\b(?:performance|latency|throughput|slow|faster|memory usage|allocation|benchmark)\b",
        re.IGNORECASE,
    )),
    ("refactor", re.compile(
        r"\b(?:refactor|restructure|reorganize|clean[ -]?up|simplif(?:y|ication)|decouple)\b",
        re.IGNORECASE,
    )),
    ("bug_boundary", re.compile(
        r"\b(?:bug|regression|crash|exception|incorrect|broken|fails?|error|unexpected)\b",
        re.IGNORECASE,
    )),
    ("feature_boundary", re.compile(
        r"\b(?:add|support|implement|introduce|allow|enable|new option|feature)\b",
        re.IGNORECASE,
    )),
)


def read_filtered_ids(path: Path) -> frozenset[str]:
    """Read the official difficulty-filtered instance identifiers."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != FILTERED_IDS_SHA256:
        raise ValueError(f"unexpected filtered_ids.csv hash: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        identifiers = {str(row.get("instance_id", "")).strip() for row in rows}
    identifiers.discard("")
    if not identifiers:
        raise ValueError("filtered ID list is empty")
    return frozenset(identifiers)


def selection_stratum(text: str) -> str:
    """Return a lexical acquisition stratum, not a task-policy label."""
    for name, pattern in _STRATA:
        if pattern.search(text):
            return name
    return "other"


def candidate_from_row(
    row: dict[str, Any],
    *,
    config: str,
    filtered_ids: frozenset[str],
) -> dict[str, Any] | None:
    """Normalize one quality-filtered problem statement without assigning labels."""
    instance_id = str(row.get("instance_id", ""))
    if not instance_id or instance_id not in filtered_ids:
        return None
    repo = str(row.get("repo", "")).strip()
    text = str(row.get("problem_statement", "")).strip()
    if not repo or not text or _SECRET.search(text):
        return None
    word_count = len(text.split())
    if not 12 <= word_count <= 1500:
        return None
    stratum = selection_stratum(text)
    return {
        "candidate_id": f"gair-openswe:{instance_id}",
        "source": {
            "dataset": SOURCE_DATASET,
            "config": config,
            "dataset_revision": SOURCE_REVISION,
            "dataset_url": SOURCE_URL,
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": row.get("base_commit"),
            "created_at": row.get("created_at"),
            "repo_license": row.get("license"),
            "repo_license_name": row.get("license_name"),
            "difficulty_filtered": True,
            "selection_stratum": stratum,
            "issue_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
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


def read_source_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSONL rows without retaining patches or environments in memory."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def exclusion_sets(paths: Iterable[Path]) -> tuple[frozenset[str], frozenset[str]]:
    """Return candidate IDs and repositories already used by prior sources."""
    identifiers: set[str] = set()
    repositories: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    identifiers.add(str(row["candidate_id"]))
                    repositories.add(str(row["source"]["repo"]).casefold())
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid candidate") from exc
    return frozenset(identifiers), frozenset(repositories)


def select_candidates(
    sources: Iterable[tuple[str, Iterable[dict[str, Any]]]],
    *,
    filtered_ids: frozenset[str],
    limit: int,
    max_per_repo: int,
    seed: int,
    excluded_ids: frozenset[str] = frozenset(),
    excluded_repositories: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select deterministic, stratum-balanced candidates across repositories."""
    if limit < 1 or max_per_repo < 1:
        raise ValueError("limit and max_per_repo must be positive")
    candidates: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    source_rows = 0
    for config, rows in sources:
        for row in rows:
            source_rows += 1
            candidate = candidate_from_row(row, config=config, filtered_ids=filtered_ids)
            if candidate is None:
                rejected["not_filtered_or_invalid"] += 1
                continue
            candidate_id = str(candidate["candidate_id"])
            repo = str(candidate["source"]["repo"]).casefold()
            if candidate_id in excluded_ids:
                rejected["excluded_candidate_id"] += 1
                continue
            if repo in excluded_repositories:
                rejected["excluded_repository"] += 1
                continue
            if candidate_id in candidates:
                rejected["duplicate_candidate_id"] += 1
                continue
            candidates[candidate_id] = candidate

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates.values():
        stratum = str(candidate["source"]["selection_stratum"])
        buckets[stratum].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row['candidate_id']}".encode()
        ).digest())

    selected: list[dict[str, Any]] = []
    positions: Counter[str] = Counter()
    repo_counts: Counter[str] = Counter()
    names = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for name in names:
            bucket = buckets[name]
            while positions[name] < len(bucket):
                candidate = bucket[positions[name]]
                positions[name] += 1
                repo = str(candidate["source"]["repo"]).casefold()
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
    selected.sort(key=lambda row: str(row["candidate_id"]))
    return selected, {
        "source_rows": source_rows,
        "official_filtered_ids": len(filtered_ids),
        "normalized_unique_candidates": len(candidates),
        "selected": len(selected),
        "selected_repositories": len(repo_counts),
        "selected_strata": dict(sorted(Counter(
            str(row["source"]["selection_stratum"]) for row in selected
        ).items())),
        "rejected": dict(sorted(rejected.items())),
        "interpretation": (
            "Lexical strata only diversify acquisition and are not policy labels. Every selected "
            "request remains unreviewed until an individual decision is recorded."
        ),
    }


def write_queue(
    output: Path,
    candidates: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    source_files: dict[str, Path],
    selection_limit: int,
    max_per_repo: int,
    seed: int,
) -> Path:
    """Write the compact ignored queue and a reproducibility manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidates
    )
    output.write_text(payload, encoding="utf-8")
    manifest_path = output.with_suffix(output.suffix + ".provenance.json")
    manifest = {
        "artifact": {
            "path": output.name,
            "rows": len(candidates),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "distribution_notice": (
                "This compact queue is an ignored external artifact. It contains third-party issue "
                "text and is not part of Infinidev's MIT-licensed source distribution."
            ),
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "url": SOURCE_URL,
            "card_license": "mixed-permissive-and-cc-by-4.0",
            "readme_project_license": "AGPL-3.0",
            "license_file_state": "empty at pinned revision",
            "source_files": {
                config: {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for config, path in sorted(source_files.items())
            },
            "filtered_ids_sha256": FILTERED_IDS_SHA256,
        },
        "selection": {
            "selection_limit": selection_limit,
            "max_per_repo": max_per_repo,
            "seed": seed,
            "report": report,
        },
        "review_contract": {
            "selection_stratum_is_policy_label": False,
            "individual_manual_review_required": True,
            "source_text_remains_external": True,
            "reviewed_annotations_path": "data/task-policy-reviews/gair-openswe",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oss", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--filtered-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--max-per-repo", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--exclude-candidates", type=Path, action="append", default=[])
    args = parser.parse_args()
    filtered_ids = read_filtered_ids(args.filtered_ids)
    excluded_ids, excluded_repositories = exclusion_sets(args.exclude_candidates)
    source_files = {"openswe_oss": args.oss, "openswe_other": args.other}
    candidates, report = select_candidates(
        (
            (config, read_source_rows(path))
            for config, path in source_files.items()
        ),
        filtered_ids=filtered_ids,
        limit=args.limit,
        max_per_repo=args.max_per_repo,
        seed=args.seed,
        excluded_ids=excluded_ids,
        excluded_repositories=excluded_repositories,
    )
    manifest = write_queue(
        args.output,
        candidates,
        report,
        source_files=source_files,
        selection_limit=args.limit,
        max_per_repo=args.max_per_repo,
        seed=args.seed,
    )
    report.update({"output": str(args.output), "provenance_manifest": str(manifest)})
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "candidate_from_row",
    "read_filtered_ids",
    "selection_stratum",
    "select_candidates",
]
