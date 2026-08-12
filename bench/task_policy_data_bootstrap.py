"""Acquire natural task requests and rebuild the reviewed policy splits.

Candidate text is reproducibly downloaded from pinned public datasets. Human
review ledgers are inputs, never generated labels: pass a transferred review
root or keep the ledgers beside the downloaded candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from bench.task_policy_natural_split import write_natural_splits


OPEN_SWE_DIRECTORY = "open-swe"
WILDCHAT_DIRECTORY = "wildchat"
EXPECTED_ROWS = 2_901
EXPECTED_FAMILIES = 2_336
EXPECTED_SPLIT_ROWS = {
    "training": 1_728,
    "calibration": 587,
    "evaluation": 586,
}


@dataclass(frozen=True)
class CandidateAcquisition:
    """One deterministic candidate download and its expected digest."""

    relative_path: str
    module: str
    arguments: tuple[str, ...]
    sha256: str


def candidate_acquisitions(data_root: Path) -> tuple[CandidateAcquisition, ...]:
    """Return the ordered acquisition plan, including exclusion dependencies."""
    open_swe = data_root / OPEN_SWE_DIRECTORY
    first = open_swe / "candidates.jsonl"
    second = open_swe / "candidates_round2.jsonl"
    third = open_swe / "candidates_round3.jsonl"
    return (
        CandidateAcquisition(
            relative_path=f"{OPEN_SWE_DIRECTORY}/candidates.jsonl",
            module="bench.open_swe_candidate_sampler",
            arguments=(
                "--scan-limit", "2000", "--limit", "240",
                "--max-per-repo", "3", "--seed", "73",
            ),
            sha256="56c61e72297660fcdac16491c3577436c06811fa585d95a67da773041514d511",
        ),
        CandidateAcquisition(
            relative_path=f"{OPEN_SWE_DIRECTORY}/candidates_round2.jsonl",
            module="bench.open_swe_candidate_sampler",
            arguments=(
                "--scan-limit", "5000", "--limit", "240",
                "--max-per-repo", "2", "--seed", "191",
                "--exclude-candidates", str(first),
            ),
            sha256="a0a0e1ac02ce1140b42fc6eb0d1fb6020d7384400e400ad2f17820550147ccc9",
        ),
        CandidateAcquisition(
            relative_path=f"{OPEN_SWE_DIRECTORY}/candidates_round3.jsonl",
            module="bench.open_swe_candidate_sampler",
            arguments=(
                "--scan-limit", "10000", "--limit", "240",
                "--max-per-repo", "2", "--seed", "271",
                "--exclude-candidates", str(first),
                "--exclude-candidates", str(second),
            ),
            sha256="e0948d90cf6aacfeadaeb2119cb768ff6daaaba4c4fea9fdfc33f409543667e6",
        ),
        CandidateAcquisition(
            relative_path=f"{OPEN_SWE_DIRECTORY}/candidates_round4.jsonl",
            module="bench.open_swe_candidate_sampler",
            arguments=(
                "--scan-limit", "20000", "--limit", "600",
                "--max-per-repo", "2", "--seed", "557",
                "--exclude-candidates", str(first),
                "--exclude-candidates", str(second),
                "--exclude-candidates", str(third),
            ),
            sha256="47e058e11f4473ddf45e0ae6176df55cc5e4d7652bdc5b5498b404a58ed65fc4",
        ),
        CandidateAcquisition(
            relative_path=f"{WILDCHAT_DIRECTORY}/candidates.jsonl",
            module="bench.wildchat_candidate_sampler",
            arguments=(
                "--scan-limit", "100000", "--limit", "2000",
                "--max-per-language", "1000", "--seed", "811",
            ),
            sha256="375b79505061610d7103091cabff45cc17b326b16a40a3b73817554881a47e1d",
        ),
    )


def review_ledger_paths(review_root: Path) -> tuple[Path, ...]:
    """Return every manually authored ledger required by the 2,901-row corpus."""
    open_swe = review_root / OPEN_SWE_DIRECTORY
    wildchat = review_root / WILDCHAT_DIRECTORY
    return (
        open_swe / "manual_holdout_reviews.jsonl",
        open_swe / "manual_reserve_a_reviews.jsonl",
        open_swe / "manual_reserve_b_reviews.jsonl",
        open_swe / "manual_reviews.jsonl",
        *(open_swe / f"round2_{index}_reviews.jsonl" for index in range(4)),
        *(open_swe / f"round3_{index}_reviews.jsonl" for index in range(4)),
        *(open_swe / f"round4_{index}_reviews.jsonl" for index in range(5)),
        *(wildchat / f"round1_{index}_reviews.jsonl" for index in range(3)),
        wildchat / "family_round1_development_reviews.jsonl",
        *(wildchat / f"family_round1_queue_{index}_reviews.jsonl" for index in range(16)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_candidate(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"candidate digest mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def acquire_candidates(data_root: Path, *, force: bool = False) -> tuple[Path, ...]:
    """Download missing candidates and verify every pinned output byte-for-byte."""
    acquired = []
    for item in candidate_acquisitions(data_root):
        output = data_root / item.relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not force:
            _verify_candidate(output, item.sha256)
            acquired.append(output)
            continue
        command = [
            sys.executable,
            "-m",
            item.module,
            str(output),
            *item.arguments,
        ]
        subprocess.run(command, check=True)
        _verify_candidate(output, item.sha256)
        acquired.append(output)
    return tuple(acquired)


def _validate_review_ledgers(paths: tuple[Path, ...]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        remainder = len(missing) - min(10, len(missing))
        suffix = f"\n  - ... and {remainder} more" if remainder else ""
        raise RuntimeError(
            "manual review ledgers cannot be regenerated; transfer them and rerun:\n"
            f"{preview}{suffix}"
        )


def _validate_split_manifest(manifest: dict[str, Any]) -> None:
    if int(manifest.get("rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError(
            f"natural split has {manifest.get('rows')} rows; expected {EXPECTED_ROWS}"
        )
    if int(manifest.get("families", -1)) != EXPECTED_FAMILIES:
        raise RuntimeError(
            f"natural split has {manifest.get('families')} families; "
            f"expected {EXPECTED_FAMILIES}"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise RuntimeError("natural split manifest is missing split summaries")
    actual_rows = {
        name: int(splits.get(name, {}).get("rows", -1))
        for name in EXPECTED_SPLIT_ROWS
    }
    if actual_rows != EXPECTED_SPLIT_ROWS:
        raise RuntimeError(
            f"natural split row counts changed: expected {EXPECTED_SPLIT_ROWS}, got {actual_rows}"
        )


def build_splits(
    data_root: Path,
    review_root: Path,
    output_dir: Path,
) -> Path:
    """Join transferred human labels and rebuild the fixed family-disjoint splits."""
    candidates = tuple(data_root / item.relative_path for item in candidate_acquisitions(data_root))
    missing_candidates = [str(path) for path in candidates if not path.is_file()]
    if missing_candidates:
        raise RuntimeError(
            "candidate data is missing; run acquisition first:\n  - "
            + "\n  - ".join(missing_candidates)
        )
    for path, item in zip(candidates, candidate_acquisitions(data_root), strict=True):
        _verify_candidate(path, item.sha256)
    reviews = review_ledger_paths(review_root)
    _validate_review_ledgers(reviews)
    manifest_path = write_natural_splits(
        output_dir,
        candidate_paths=candidates,
        review_paths=reviews,
        seed=2027,
        trials=128,
        minimum_positive_support=5,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_split_manifest(manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "acquire", "build"),
        default="all",
        help="Download candidates, build splits, or do both (default: all).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".infinidev/external-data"),
        help="Destination for downloaded candidate data.",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        help="Root containing transferred open-swe/ and wildchat/ review ledgers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "tmp" / "task-policy-natural-split-v1",
        help="Destination for generated train/calibration/evaluation files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Regenerate candidate files even when their verified copies already exist.",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    review_root = (args.review_root or args.data_root).resolve()
    if args.mode in {"all", "acquire"}:
        acquired = acquire_candidates(data_root, force=args.force_download)
        print(json.dumps({"verified_candidates": [str(path) for path in acquired]}, indent=2))
    if args.mode in {"all", "build"}:
        manifest = build_splits(data_root, review_root, args.output_dir.resolve())
        print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()


__all__ = [
    "CandidateAcquisition",
    "acquire_candidates",
    "build_splits",
    "candidate_acquisitions",
    "review_ledger_paths",
]
