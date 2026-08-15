"""Add model-labeled requests to training while preserving human holdouts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from bench.external_candidate_family_split import candidate_families, read_jsonl
from bench.task_policy_external_review import SHORT_LABEL_TO_POLICY
from bench.task_policy_natural_split import load_reviewed_rows
from bench.task_policy_multilabel_head import METHOD_LABELS


AUGMENTATION_VERSION = "task-policy-training-augmentation-v1"
PARTITIONS = ("training", "calibration", "evaluation")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def _verify_base_split(base_dir: Path) -> dict[str, Any]:
    manifest_path = base_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for partition in PARTITIONS:
        artifacts = manifest["artifacts"][partition]
        for kind in ("candidates", "reviews"):
            path = base_dir / artifacts[kind]
            actual = _sha256_bytes(path.read_bytes())
            expected = artifacts[f"{kind}_sha256"]
            if actual != expected:
                raise ValueError(f"base {partition} {kind} hash mismatch")
    return manifest


def _unique(rows: Iterable[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get(field, ""))
        if not identifier or identifier in result:
            raise ValueError(f"missing or duplicate {field}: {identifier}")
        result[identifier] = row
    return result


def _positive_support(reviews: Iterable[dict[str, Any]]) -> Counter[str]:
    """Count review labels using the canonical encoder label names."""
    support: Counter[str] = Counter()
    for decision in reviews:
        support.update(
            SHORT_LABEL_TO_POLICY.get(str(label), str(label))
            for label in decision["policies"]
        )
    return support


def augment_training_split(
    base_dir: Path,
    output_dir: Path,
    *,
    candidate_paths: Iterable[Path],
    model_review_paths: Iterable[Path] = (),
    minimum_positive_support: int = 1_000,
    model_review_path: Path | None = None,
) -> Path:
    """Build an augmented split and prove model labels never enter holdouts."""
    base_manifest = _verify_base_split(base_dir)
    base_candidates = {
        partition: read_jsonl(base_dir / base_manifest["artifacts"][partition]["candidates"])
        for partition in PARTITIONS
    }
    base_reviews = {
        partition: read_jsonl(base_dir / base_manifest["artifacts"][partition]["reviews"])
        for partition in PARTITIONS
    }
    paths = list(candidate_paths)
    review_paths = list(model_review_paths)
    if model_review_path is not None:
        review_paths.append(model_review_path)
    if not review_paths:
        raise ValueError("at least one model review ledger is required")
    model_candidates, model_decisions = load_reviewed_rows(paths, review_paths)
    if len(model_candidates) != len(model_decisions):
        raise ValueError("every model candidate must have one included decision")
    for candidate_id, decision in model_decisions.items():
        annotation = decision.get("annotation")
        if not isinstance(annotation, dict) or annotation.get("kind") != "model":
            raise ValueError(f"decision {candidate_id} lacks explicit model provenance")

    partitions = {
        **base_candidates,
        "model_training": model_candidates,
    }
    id_to_partition: dict[str, str] = {}
    all_candidates: list[dict[str, Any]] = []
    for partition, rows in partitions.items():
        for candidate_id in _unique(rows, "candidate_id"):
            if candidate_id in id_to_partition:
                raise ValueError(f"candidate appears in multiple partitions: {candidate_id}")
            id_to_partition[candidate_id] = partition
        all_candidates.extend(rows)

    leaked_families: list[dict[str, Any]] = []
    for family in candidate_families(all_candidates):
        members = {
            id_to_partition[str(row["candidate_id"])]
            for row in family
        }
        if "model_training" in members and members & {"calibration", "evaluation"}:
            leaked_families.append({
                "partitions": sorted(members),
                "candidate_ids": [str(row["candidate_id"]) for row in family],
            })
    if leaked_families:
        raise ValueError(
            f"model training data leaks into {len(leaked_families)} holdout families"
        )

    training_candidates = sorted(
        [*base_candidates["training"], *model_candidates],
        key=lambda row: str(row["candidate_id"]),
    )
    training_decisions = {
        **_unique(base_reviews["training"], "candidate_id"),
        **model_decisions,
    }
    training_reviews = [
        training_decisions[str(row["candidate_id"])] for row in training_candidates
    ]
    if minimum_positive_support < 0:
        raise ValueError("minimum positive support must not be negative")
    support = _positive_support(training_reviews)
    missing = {
        label: support[label] for label in METHOD_LABELS
        if support[label] < minimum_positive_support
    }
    if missing:
        detail = ", ".join(f"{label}={count}" for label, count in missing.items())
        raise ValueError(
            f"training requires at least {minimum_positive_support} positives per category; {detail}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for partition in PARTITIONS:
        candidate_rows = (
            training_candidates if partition == "training" else base_candidates[partition]
        )
        review_rows = training_reviews if partition == "training" else base_reviews[partition]
        candidate_payload = _jsonl(candidate_rows)
        review_payload = _jsonl(review_rows)
        candidate_path = output_dir / f"{partition}_candidates.jsonl"
        review_path = output_dir / f"{partition}_reviews.jsonl"
        candidate_path.write_bytes(candidate_payload)
        review_path.write_bytes(review_payload)
        artifacts[partition] = {
            "rows": len(candidate_rows),
            "candidates": candidate_path.name,
            "candidates_sha256": _sha256_bytes(candidate_payload),
            "reviews": review_path.name,
            "reviews_sha256": _sha256_bytes(review_payload),
        }

    label_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    for decision in training_reviews:
        label_counts.update(
            SHORT_LABEL_TO_POLICY.get(str(label), str(label))
            for label in decision["policies"]
        )
        annotation = decision.get("annotation")
        provenance_counts[
            "model" if isinstance(annotation, dict) and annotation.get("kind") == "model"
            else "human"
        ] += 1
    manifest = {
        "version": AUGMENTATION_VERSION,
        "base_manifest": {
            "path": str(base_dir / "manifest.json"),
            "sha256": _sha256_bytes((base_dir / "manifest.json").read_bytes()),
        },
        "model_reviews": [
            {"path": str(path), "sha256": _sha256_bytes(path.read_bytes())}
            for path in review_paths
        ],
        "model_review_rows": len(model_candidates),
        "holdouts_preserved_from_base": True,
        "family_leakage_into_holdouts": 0,
        "artifacts": artifacts,
        "training_labels": dict(sorted(label_counts.items())),
        "training_provenance": dict(sorted(provenance_counts.items())),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--model-review", action="append", type=Path, required=True)
    parser.add_argument("--minimum-positive-support", type=int, default=1_000)
    args = parser.parse_args()
    manifest = augment_training_split(
        args.base_dir,
        args.output_dir,
        candidate_paths=args.candidate,
        model_review_paths=args.model_review,
        minimum_positive_support=args.minimum_positive_support,
    )
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()


__all__ = ["AUGMENTATION_VERSION", "augment_training_split"]
