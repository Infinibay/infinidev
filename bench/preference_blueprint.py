#!/usr/bin/env python3
"""Validate preference-family authoring blueprints before generating probes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bench.model_behavior import UTILITY_AXES, Probe, load_probes
from bench.probe_dataset import load_preference_category_targets


def audit_blueprint(
    value: object,
    targets: dict[str, int],
    completed_families: set[str] | None = None,
) -> dict[str, object]:
    """Check that planned families cover category targets with real axis tension."""
    if not isinstance(value, dict) or not isinstance(value.get("families"), list):
        raise ValueError("blueprint needs a families array")
    variants = int(value.get("variants_per_family", 0))
    if variants < 2:
        raise ValueError("variants_per_family must be at least two")
    rows = value["families"]
    completed = completed_families or set()
    counts: Counter[str] = Counter()
    ids: Counter[str] = Counter()
    issues: dict[str, list[str]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            issues[str(index)] = ["not_an_object"]
            continue
        family = str(raw.get("family", ""))
        category = str(raw.get("category", ""))
        if family in completed:
            continue
        key = family or str(index)
        row_issues: list[str] = []
        ids[family] += 1
        if category not in targets:
            row_issues.append("unknown_category")
        else:
            counts[category] += variants
        for field in ("family", "tradeoff", "information_sought", "variant_axis"):
            if not str(raw.get(field, "")).strip():
                row_issues.append(f"missing_{field}")
        axes = raw.get("axes")
        if not isinstance(axes, list) or len(set(map(str, axes))) < 2:
            row_issues.append("needs_two_axes")
        elif set(map(str, axes)) - UTILITY_AXES:
            row_issues.append("unknown_axis")
        if row_issues:
            issues[key] = row_issues
    duplicates = sorted(family for family, count in ids.items() if family and count > 1)
    shortfalls = {
        category: target - counts[category]
        for category, target in targets.items()
        if counts[category] < target
    }
    return {
        "families": sum(
            isinstance(raw, dict) and str(raw.get("family", "")) not in completed
            for raw in rows
        ),
        "planned_probes": sum(
            isinstance(raw, dict) and str(raw.get("family", "")) not in completed
            for raw in rows
        ) * variants,
        "planned_by_category": dict(sorted(counts.items())),
        "shortfalls": shortfalls,
        "duplicate_families": duplicates,
        "issues": issues,
        "passes": not shortfalls and not duplicates and not issues,
    }


def remaining_targets(
    probes: dict[str, Probe], targets: dict[str, int]
) -> dict[str, int]:
    """Return preference coverage still missing from the authored dataset."""
    counts = Counter(
        probe.category
        for probe in probes.values()
        if probe.evaluation_mode == "preference" and probe.review_status != "rejected"
    )
    return {
        category: target - counts[category]
        for category, target in targets.items()
        if counts[category] < target
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blueprint", type=Path)
    parser.add_argument("taxonomy", type=Path)
    parser.add_argument("probes", type=Path)
    args = parser.parse_args()
    targets = remaining_targets(
        (probes := load_probes(args.probes)),
        load_preference_category_targets(args.taxonomy),
    )
    report = audit_blueprint(
        json.loads(args.blueprint.read_text(encoding="utf-8")),
        targets,
        {probe.group for probe in probes.values() if probe.group},
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
