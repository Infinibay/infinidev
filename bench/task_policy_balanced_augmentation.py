"""Add balanced model-labeled data to train and calibration without leakage.

The existing human evaluation partition is preserved byte-for-byte.  New
families that touch base training are forced into training; only entirely new
families may extend calibration.  Model provenance remains explicit in every
ledger and in the output manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from bench.external_candidate_family_split import candidate_families, read_jsonl
from bench.task_policy_natural_split import load_reviewed_rows
from bench.task_policy_teacher_proposals import POLICIES
from bench.task_policy_training_augmentation import _verify_base_split


AUGMENTATION_VERSION = "task-policy-balanced-augmentation-v1"
PARTITIONS = ("training", "calibration", "evaluation")


def _jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def _unique(rows: Iterable[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get(field, ""))
        if not identifier or identifier in result:
            raise ValueError(f"missing or duplicate {field}: {identifier}")
        result[identifier] = row
    return result


def _label_counts(
    decisions: dict[str, dict[str, Any]], family: list[dict[str, Any]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in family:
        counts.update(
            str(label) for label in decisions[str(row["candidate_id"])]["policies"]
        )
    return counts


def _partition_counts(
    candidates: list[dict[str, Any]], decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    for candidate in candidates:
        decision = decisions[str(candidate["candidate_id"])]
        labels.update(map(str, decision["policies"]))
        annotation = decision.get("annotation")
        provenance[
            "model" if isinstance(annotation, dict) and annotation.get("kind") == "model"
            else "human"
        ] += 1
    return {
        "rows": len(candidates),
        "labels": dict(sorted(labels.items())),
        "provenance": dict(sorted(provenance.items())),
    }


def augment_balanced_split(
    base_dir: Path,
    output_dir: Path,
    *,
    candidate_paths: Iterable[Path],
    model_review_path: Path,
    calibration_targets: dict[str, int],
    seed: int = 20260820,
) -> Path:
    """Extend train/calibration with new families and preserve evaluation exactly."""
    if not calibration_targets or any(label not in POLICIES for label in calibration_targets):
        raise ValueError("calibration targets must use known policy labels")
    if any(value < 1 for value in calibration_targets.values()):
        raise ValueError("calibration target counts must be positive")
    base_manifest = _verify_base_split(base_dir)
    base_candidates = {
        partition: read_jsonl(base_dir / base_manifest["artifacts"][partition]["candidates"])
        for partition in PARTITIONS
    }
    base_reviews = {
        partition: read_jsonl(base_dir / base_manifest["artifacts"][partition]["reviews"])
        for partition in PARTITIONS
    }
    candidate_paths = list(candidate_paths)
    model_candidates, model_decisions = load_reviewed_rows(candidate_paths, [model_review_path])
    if len(model_candidates) != len(model_decisions):
        raise ValueError("every new candidate must have one included model decision")
    for candidate_id, decision in model_decisions.items():
        annotation = decision.get("annotation")
        if not isinstance(annotation, dict) or annotation.get("kind") != "model":
            raise ValueError(f"decision {candidate_id} lacks explicit model provenance")

    id_partition: dict[str, str] = {}
    all_candidates: list[dict[str, Any]] = []
    for partition, rows in base_candidates.items():
        for candidate_id in _unique(rows, "candidate_id"):
            if candidate_id in id_partition:
                raise ValueError(f"base candidate appears twice: {candidate_id}")
            id_partition[candidate_id] = partition
        all_candidates.extend(rows)
    for candidate_id in _unique(model_candidates, "candidate_id"):
        if candidate_id in id_partition:
            raise ValueError(f"new candidate already exists in base: {candidate_id}")
        id_partition[candidate_id] = "new"
    all_candidates.extend(model_candidates)

    forced_training: set[str] = set()
    eligible_families: list[list[dict[str, Any]]] = []
    for family in candidate_families(all_candidates):
        members = {id_partition[str(row["candidate_id"])] for row in family}
        new_rows = [row for row in family if id_partition[str(row["candidate_id"])] == "new"]
        if not new_rows:
            continue
        if members & {"calibration", "evaluation"}:
            raise ValueError("new data leaks into an existing calibration/evaluation family")
        if "training" in members:
            forced_training.update(str(row["candidate_id"]) for row in new_rows)
        else:
            eligible_families.append(new_rows)

    calibration_ids: set[str] = set()
    calibration_counts: Counter[str] = Counter()
    chosen_families: set[int] = set()
    family_label_counts = [
        _label_counts(model_decisions, family) for family in eligible_families
    ]
    while True:
        deficits = {
            label: target - calibration_counts[label]
            for label, target in calibration_targets.items()
        }
        active = [label for label, deficit in deficits.items() if deficit > 0]
        if not active:
            break
        label = max(active, key=lambda item: (deficits[item] / calibration_targets[item], item))
        options = [
            index for index, counts in enumerate(family_label_counts)
            if index not in chosen_families and counts[label]
        ]
        if not options:
            raise ValueError(
                f"cannot meet calibration target for {label}; short by {deficits[label]}"
            )
        index = min(
            options,
            key=lambda item: hashlib.sha256(
                f"{seed}:{label}:".encode()
                + ":".join(
                    str(row["candidate_id"]) for row in eligible_families[item]
                ).encode()
            ).digest(),
        )
        chosen_families.add(index)
        family = eligible_families[index]
        calibration_ids.update(str(row["candidate_id"]) for row in family)
        calibration_counts.update(family_label_counts[index])

    new_training = [
        row for row in model_candidates if str(row["candidate_id"]) not in calibration_ids
    ]
    if forced_training & calibration_ids:
        raise AssertionError("a base-training family entered calibration")
    new_calibration = [
        row for row in model_candidates if str(row["candidate_id"]) in calibration_ids
    ]
    output_candidates = {
        "training": sorted(
            [*base_candidates["training"], *new_training],
            key=lambda row: str(row["candidate_id"]),
        ),
        "calibration": sorted(
            [*base_candidates["calibration"], *new_calibration],
            key=lambda row: str(row["candidate_id"]),
        ),
        "evaluation": base_candidates["evaluation"],
    }
    decisions = {
        partition: {
            **_unique(base_reviews[partition], "candidate_id"),
            **(model_decisions if partition != "evaluation" else {}),
        }
        for partition in PARTITIONS
    }
    output_reviews = {
        partition: [
            decisions[partition][str(row["candidate_id"])]
            for row in output_candidates[partition]
        ]
        for partition in PARTITIONS
    }

    output_partition_by_id = {
        str(row["candidate_id"]): partition
        for partition, rows in output_candidates.items()
        for row in rows
    }
    leaked = []
    for family in candidate_families([
        row for partition in PARTITIONS for row in output_candidates[partition]
    ]):
        partitions = {
            output_partition_by_id[str(row["candidate_id"])] for row in family
        }
        if len(partitions) > 1:
            leaked.append(sorted(partitions))
    if leaked:
        raise AssertionError(f"output contains {len(leaked)} cross-partition families")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    partition_report: dict[str, Any] = {}
    for partition in PARTITIONS:
        candidate_payload = _jsonl(output_candidates[partition])
        review_payload = _jsonl(output_reviews[partition])
        candidate_path = output_dir / f"{partition}_candidates.jsonl"
        review_path = output_dir / f"{partition}_reviews.jsonl"
        candidate_path.write_bytes(candidate_payload)
        review_path.write_bytes(review_payload)
        artifacts[partition] = {
            "rows": len(output_candidates[partition]),
            "candidates": candidate_path.name,
            "candidates_sha256": hashlib.sha256(candidate_payload).hexdigest(),
            "reviews": review_path.name,
            "reviews_sha256": hashlib.sha256(review_payload).hexdigest(),
        }
        partition_report[partition] = _partition_counts(
            output_candidates[partition], decisions[partition],
        )

    base_eval_artifacts = base_manifest["artifacts"]["evaluation"]
    if (
        artifacts["evaluation"]["candidates_sha256"]
        != base_eval_artifacts["candidates_sha256"]
        or artifacts["evaluation"]["reviews_sha256"]
        != base_eval_artifacts["reviews_sha256"]
    ):
        raise AssertionError("evaluation partition changed")
    manifest = {
        "version": AUGMENTATION_VERSION,
        "seed": seed,
        "base_manifest": {
            "path": str(base_dir / "manifest.json"),
            "sha256": hashlib.sha256((base_dir / "manifest.json").read_bytes()).hexdigest(),
        },
        "model_reviews": {
            "path": str(model_review_path),
            "sha256": hashlib.sha256(model_review_path.read_bytes()).hexdigest(),
            "rows": len(model_candidates),
        },
        "requested_new_calibration_labels": dict(sorted(calibration_targets.items())),
        "actual_new_calibration_labels": dict(sorted(calibration_counts.items())),
        "new_rows": {
            "training": len(new_training), "calibration": len(new_calibration),
        },
        "forced_training_rows": len(forced_training),
        "evaluation_preserved_from_base": True,
        "family_leakage": 0,
        "artifacts": artifacts,
        "partitions": partition_report,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _target(value: str) -> tuple[str, int]:
    try:
        label, raw_count = value.rsplit("=", 1)
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target must be LABEL=COUNT") from exc
    if label not in POLICIES or count < 1:
        raise argparse.ArgumentTypeError("unknown label or invalid count")
    return label, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--model-review", type=Path, required=True)
    parser.add_argument("--calibration-target", action="append", type=_target, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    targets = dict(args.calibration_target)
    if len(targets) != len(args.calibration_target):
        parser.error("each calibration target may be specified only once")
    manifest = augment_balanced_split(
        args.base_dir, args.output_dir, candidate_paths=args.candidate,
        model_review_path=args.model_review, calibration_targets=targets, seed=args.seed,
    )
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()


__all__ = ["AUGMENTATION_VERSION", "augment_balanced_split"]
