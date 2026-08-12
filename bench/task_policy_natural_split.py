"""Freeze manually reviewed natural requests into leakage-safe model splits.

The command consumes external candidate queues and their human review ledgers.
It never invents or changes a label: it groups source identities and lexical
near-duplicates, balances already-reviewed outcomes, and writes ignored local
candidate/review pairs for training, calibration, and evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from bench.external_candidate_family_split import candidate_families, read_jsonl
from bench.task_policy_external_review import (
    EXTERNAL_UNCATEGORIZED_REASONS,
    SHORT_LABEL_TO_POLICY,
)


SPLIT_VERSION = "natural-task-policy-family-split-v1"
SPLIT_NAMES = ("training", "calibration", "evaluation")
DEFAULT_FRACTIONS = (0.60, 0.20, 0.20)


def _load_unique_rows(paths: Iterable[Path], *, id_field: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            identifier = str(row.get(id_field, ""))
            if not identifier:
                raise ValueError(f"{path}: row is missing {id_field}")
            if identifier in rows:
                raise ValueError(f"duplicate {id_field}: {identifier}")
            rows[identifier] = row
    return rows


def load_reviewed_rows(
    candidate_paths: Iterable[Path],
    review_paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Join included manual decisions to source candidates without relabeling."""
    candidates = _load_unique_rows(candidate_paths, id_field="candidate_id")
    decisions = _load_unique_rows(review_paths, id_field="candidate_id")
    selected: list[dict[str, Any]] = []
    selected_decisions: dict[str, dict[str, Any]] = {}
    for candidate_id, decision in decisions.items():
        if candidate_id not in candidates:
            raise ValueError(f"review references unknown candidate: {candidate_id}")
        if decision.get("include") is not True:
            continue
        policies = decision.get("policies")
        if not isinstance(policies, list) or any(
            str(policy) not in SHORT_LABEL_TO_POLICY for policy in policies
        ):
            raise ValueError(f"review {candidate_id} has invalid policies")
        reason = decision.get("uncategorized_reason")
        if policies and reason:
            raise ValueError(f"review {candidate_id} has both policies and a zero-label reason")
        if not policies and reason not in EXTERNAL_UNCATEGORIZED_REASONS:
            raise ValueError(f"review {candidate_id} has invalid zero-label reason")
        selected.append(candidates[candidate_id])
        selected_decisions[candidate_id] = decision
    selected.sort(key=lambda row: str(row["candidate_id"]))
    return selected, selected_decisions


def _family_counts(
    family: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> Counter[str]:
    counts: Counter[str] = Counter({"rows": len(family)})
    for row in family:
        candidate_id = str(row["candidate_id"])
        decision = decisions[candidate_id]
        policies = [str(policy) for policy in decision["policies"]]
        for policy in policies:
            counts[f"label:{policy}"] += 1
        if not policies:
            counts[f"zero:{decision['uncategorized_reason']}"] += 1
        source = row.get("source", {})
        dataset = str(source.get("dataset", "unknown")) if isinstance(source, dict) else "unknown"
        counts[f"source:{dataset}"] += 1
    return counts


def _feature_weights(totals: Counter[str]) -> dict[str, float]:
    weights = {feature: 1.0 for feature in totals}
    weights["rows"] = 1.5
    for feature in totals:
        if feature.startswith("label:"):
            weights[feature] = 6.0
        elif feature.startswith("source:"):
            weights[feature] = 1.5
        elif feature.startswith("zero:"):
            weights[feature] = 2.0
    return weights


def _assignment_score(
    split_counts: list[Counter[str]],
    totals: Counter[str],
    fractions: tuple[float, float, float],
) -> tuple[float, float, float]:
    weights = _feature_weights(totals)
    label_deviations = []
    all_deviations = []
    squared = 0.0
    for split_index, fraction in enumerate(fractions):
        for feature, total in totals.items():
            target = max(float(total) * fraction, 1.0)
            deviation = abs(split_counts[split_index][feature] - target) / target
            all_deviations.append(deviation)
            if feature.startswith("label:"):
                label_deviations.append(deviation)
            squared += weights[feature] * deviation * deviation
    return max(label_deviations, default=0.0), max(all_deviations, default=0.0), squared


def split_reviewed_families(
    rows: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    *,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 2027,
    trials: int = 256,
    minimum_positive_support: int = 5,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Balance labels while keeping conversation, repository, and text families atomic."""
    if len(fractions) != len(SPLIT_NAMES) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must contain three values summing to one")
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("split fractions must be positive")
    if trials < 1 or minimum_positive_support < 1:
        raise ValueError("trials and minimum positive support must be positive")

    families = candidate_families(rows)
    family_counts = [_family_counts(family, decisions) for family in families]
    totals: Counter[str] = Counter()
    for counts in family_counts:
        totals.update(counts)
    rarity = {
        index: sum(
            count / max(1, totals[feature])
            for feature, count in counts.items()
            if feature.startswith("label:")
        )
        for index, counts in enumerate(family_counts)
    }
    best: tuple[tuple[float, float, float], list[list[int]], list[Counter[str]]] | None = None
    for trial in range(trials):
        rng = random.Random(seed + trial)
        tie_breakers = [rng.random() for _ in families]
        order = sorted(
            range(len(families)),
            key=lambda index: (
                -rarity[index],
                -family_counts[index]["rows"],
                tie_breakers[index],
            ),
        )
        assignments: list[list[int]] = [[] for _ in SPLIT_NAMES]
        split_counts: list[Counter[str]] = [Counter() for _ in SPLIT_NAMES]
        for family_index in order:
            choices = []
            for split_index in range(len(SPLIT_NAMES)):
                candidate_counts = [Counter(item) for item in split_counts]
                candidate_counts[split_index].update(family_counts[family_index])
                choices.append((
                    _assignment_score(candidate_counts, totals, fractions),
                    tie_breakers[family_index] + split_index / 10.0,
                    split_index,
                ))
            _, _, selected_split = min(choices)
            assignments[selected_split].append(family_index)
            split_counts[selected_split].update(family_counts[family_index])
        score = _assignment_score(split_counts, totals, fractions)
        candidate = (score, assignments, split_counts)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise RuntimeError("no split assignment was produced")
    score, assignments, split_counts = best
    outputs = [
        sorted(
            (row for family_index in indexes for row in families[family_index]),
            key=lambda row: str(row["candidate_id"]),
        )
        for indexes in assignments
    ]
    label_features = [f"label:{label}" for label in SHORT_LABEL_TO_POLICY]
    unsupported = {
        SPLIT_NAMES[index]: {
            feature.removeprefix("label:"): split_counts[index][feature]
            for feature in label_features
            if split_counts[index][feature] < minimum_positive_support
        }
        for index in range(len(SPLIT_NAMES))
    }
    unsupported = {name: values for name, values in unsupported.items() if values}
    if unsupported:
        raise ValueError(
            "unable to satisfy minimum positive support without splitting families: "
            + json.dumps(unsupported, sort_keys=True)
        )
    report = {
        "version": SPLIT_VERSION,
        "seed": seed,
        "trials": trials,
        "fractions": dict(zip(SPLIT_NAMES, fractions, strict=True)),
        "families": len(families),
        "rows": len(rows),
        "score": {
            "maximum_label_relative_deviation": score[0],
            "maximum_relative_deviation": score[1],
            "weighted_squared_deviation": score[2],
        },
        "splits": {
            SPLIT_NAMES[index]: {
                "rows": len(outputs[index]),
                "families": len(assignments[index]),
                "labels": {
                    label: split_counts[index][f"label:{label}"]
                    for label in SHORT_LABEL_TO_POLICY
                },
                "zero_reasons": {
                    reason: split_counts[index][f"zero:{reason}"]
                    for reason in sorted(EXTERNAL_UNCATEGORIZED_REASONS)
                },
                "sources": {
                    feature.removeprefix("source:"): count
                    for feature, count in sorted(split_counts[index].items())
                    if feature.startswith("source:")
                },
            }
            for index in range(len(SPLIT_NAMES))
        },
    }
    return outputs, report


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def write_natural_splits(
    output_dir: Path,
    *,
    candidate_paths: Iterable[Path],
    review_paths: Iterable[Path],
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 2027,
    trials: int = 256,
    minimum_positive_support: int = 5,
) -> Path:
    """Write ignored local split artifacts and a digest-bearing manifest."""
    rows, decisions = load_reviewed_rows(candidate_paths, review_paths)
    splits, report = split_reviewed_families(
        rows,
        decisions,
        fractions=fractions,
        seed=seed,
        trials=trials,
        minimum_positive_support=minimum_positive_support,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for name, split_rows in zip(SPLIT_NAMES, splits, strict=True):
        candidate_payload = _jsonl(split_rows)
        review_payload = _jsonl(decisions[str(row["candidate_id"])] for row in split_rows)
        candidate_path = output_dir / f"{name}_candidates.jsonl"
        review_path = output_dir / f"{name}_reviews.jsonl"
        candidate_path.write_text(candidate_payload, encoding="utf-8")
        review_path.write_text(review_payload, encoding="utf-8")
        artifacts[name] = {
            "candidates": candidate_path.name,
            "candidates_sha256": hashlib.sha256(candidate_payload.encode()).hexdigest(),
            "reviews": review_path.name,
            "reviews_sha256": hashlib.sha256(review_payload.encode()).hexdigest(),
        }
    report["artifacts"] = artifacts
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--review-ledger", type=Path, action="append", required=True)
    parser.add_argument("--training-fraction", type=float, default=DEFAULT_FRACTIONS[0])
    parser.add_argument("--calibration-fraction", type=float, default=DEFAULT_FRACTIONS[1])
    parser.add_argument("--evaluation-fraction", type=float, default=DEFAULT_FRACTIONS[2])
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--minimum-positive-support", type=int, default=5)
    args = parser.parse_args()
    manifest = write_natural_splits(
        args.output_dir,
        candidate_paths=args.candidate,
        review_paths=args.review_ledger,
        fractions=(
            args.training_fraction,
            args.calibration_fraction,
            args.evaluation_fraction,
        ),
        seed=args.seed,
        trials=args.trials,
        minimum_positive_support=args.minimum_positive_support,
    )
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_FRACTIONS",
    "SPLIT_NAMES",
    "SPLIT_VERSION",
    "load_reviewed_rows",
    "split_reviewed_families",
    "write_natural_splits",
]
