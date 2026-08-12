"""Audit the manually authored task-policy corpus without generating rows."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.contextual_embedding_benchmark import (
    DEFAULT_CONTEXTUAL_MODEL,
    _encode_contextual,
)
from infinidev.engine.task_policies.registry import POLICY_BY_ID


DEFAULT_DATASET = Path(__file__).with_name(
    "task_policy_manual_v1.calibration.jsonl"
)


def load_examples(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """Load JSONL examples, ordering default shards by their numeric row id."""
    paths = [path]
    if path == DEFAULT_DATASET:
        paths.extend(
            candidate
            for candidate in sorted(path.parent.glob("task_policy_manual_v1.*.jsonl"))
            if candidate != path
        )
    rows = [
        json.loads(line)
        for dataset_path in paths
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if path == DEFAULT_DATASET:
        rows.sort(key=lambda row: int(str(row["id"]).rsplit("-", 1)[1]))
    return rows


def _counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def structural_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report balance, diversity, cardinality, and exact duplication."""
    total = len(rows)
    policy_counts = Counter(
        policy for row in rows for policy in row["policies"]
    )
    single_policy_counts = Counter(
        row["policies"][0] for row in rows if len(row["policies"]) == 1
    )
    combination_counts = Counter(
        " + ".join(sorted(row["policies"]))
        for row in rows
        if len(row["policies"]) >= 2
    )
    project_counts = Counter(row["project_type"] for row in rows)
    user_counts = Counter(row["user_type"] for row in rows)
    style_counts = Counter(row["style"] for row in rows)

    def maximum_share(counts: Counter[str]) -> dict[str, int | float | None]:
        if not counts:
            return {"name": None, "count": 0, "share": 0.0}
        name, count = max(counts.items(), key=lambda item: item[1])
        return {"name": name, "count": count, "share": count / total}

    return {
        "examples": total,
        "policies": dict(sorted(policy_counts.items())),
        "single_policies": dict(sorted(single_policy_counts.items())),
        "policy_combinations": dict(sorted(combination_counts.items())),
        "cardinality": _counts([str(len(row["policies"])) for row in rows]),
        "uncategorized": sum(not row["policies"] for row in rows),
        "uncategorized_reasons": _counts([
            str(row["uncategorized_reason"])
            for row in rows
            if not row["policies"]
        ]),
        "batches": _counts([str(row["batch"]) for row in rows]),
        "languages": _counts([str(row["language"]) for row in rows]),
        "projects": len(project_counts),
        "users": len(user_counts),
        "styles": len(style_counts),
        "non_english_share": (
            sum(row["language"] != "en" for row in rows) / total
        ),
        "difficult_share": sum(
            row["difficulty"]
            in {"D2_overlap", "D3_composed", "D4_pragmatic", "D5_contextual"}
            for row in rows
        ) / total,
        "max_project": maximum_share(project_counts),
        "max_user": maximum_share(user_counts),
        "max_style": maximum_share(style_counts),
        "duplicate_ids": total - len({row["id"] for row in rows}),
        "duplicate_texts": total - len({row["text"].casefold() for row in rows}),
        "duplicate_scenarios": total - len({row["scenario_family"] for row in rows}),
    }


def semantic_contract_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate label cardinality, authority, and policy compatibility.

    This deliberately checks only machine-verifiable invariants. Whether the
    request actually describes a bug, feature, or refactor remains a human
    review decision and must not be inferred from keywords.
    """
    method_policy_ids = {
        policy_id for policy_id, policy in POLICY_BY_ID.items() if policy.operations
    }
    violations: list[dict[str, str]] = []

    def add(row: dict[str, Any], rule: str, detail: str) -> None:
        violations.append({"id": str(row.get("id", "")), "rule": rule, "detail": detail})

    for row in rows:
        policies = [str(value) for value in row.get("policies", [])]
        authority = {str(value) for value in row.get("authority", [])}
        unknown = sorted(set(policies) - method_policy_ids)
        if unknown:
            add(row, "known_policies", f"unknown policies: {', '.join(unknown)}")
        if len(policies) != len(set(policies)):
            add(row, "unique_policies", "policy labels must not repeat")
        if len(policies) > 3:
            add(row, "maximum_cardinality", "at most three task policies may be active")

        reason = row.get("uncategorized_reason")
        if policies and reason not in {None, ""}:
            add(row, "categorized_reason", "categorized rows cannot have uncategorized_reason")
        if not policies and not str(reason or "").strip():
            add(row, "uncategorized_reason", "empty policy sets need a reason")

        known = [POLICY_BY_ID[policy_id] for policy_id in policies if policy_id in POLICY_BY_ID]
        for policy in known:
            if policy.requires_modify and "modify" not in authority:
                add(row, "requires_modify", f"{policy.id} requires modify authority")
            if policy.requires_modify and "read_only" in authority:
                add(row, "read_only_conflict", f"{policy.id} conflicts with read_only authority")
            if policy.forbids_modify and "modify" in authority:
                add(row, "forbids_modify", f"{policy.id} forbids modify authority")
        for index, policy in enumerate(known):
            for other in known[index + 1:]:
                if other.id in policy.incompatible_with or policy.id in other.incompatible_with:
                    add(row, "incompatible_policies", f"{policy.id} conflicts with {other.id}")

    return {
        "examples": len(rows),
        "violation_count": len(violations),
        "violations": violations,
    }


def semantic_neighbor_report(
    rows: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_CONTEXTUAL_MODEL,
    top_k: int = 20,
    duplicate_threshold: float = 0.95,
) -> dict[str, Any]:
    """Return nearest E5 neighbors as review candidates, never auto-rejections."""
    vectors, measurements = _encode_contextual(
        [str(row["text"]) for row in rows],
        model_name=model_name,
        batch_size=32,
        max_length=128,
    )
    similarities = vectors @ vectors.T
    upper_rows, upper_columns = np.triu_indices(len(rows), k=1)
    upper_scores = similarities[upper_rows, upper_columns]
    order = np.argsort(upper_scores)[::-1][:top_k]
    pairs = []
    for position in order:
        left_index = int(upper_rows[position])
        right_index = int(upper_columns[position])
        left = rows[left_index]
        right = rows[right_index]
        pairs.append({
            "score": float(upper_scores[position]),
            "left": left["id"],
            "right": right["id"],
            "left_policies": left["policies"],
            "right_policies": right["policies"],
            "left_scenario": left["scenario_family"],
            "right_scenario": right["scenario_family"],
        })
    return {
        "model": model_name,
        "threshold": duplicate_threshold,
        "pairs_at_or_above_threshold": int(
            np.sum(upper_scores >= duplicate_threshold)
        ),
        "measurements": asdict(measurements),
        "nearest_pairs": pairs,
        "interpretation": (
            "Neighbors are human-review candidates, not automatic duplicates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    rows = load_examples(args.dataset)
    report: dict[str, Any] = {
        "structural": structural_report(rows),
        "semantic_contract": semantic_contract_report(rows),
    }
    if args.semantic:
        report["semantic"] = semantic_neighbor_report(rows, top_k=args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
