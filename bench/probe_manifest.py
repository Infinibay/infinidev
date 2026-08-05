#!/usr/bin/env python3
"""Freeze a deterministic, category-stratified behavioral probe sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from bench.model_behavior import Probe, load_observations, load_probes


def file_sha256(path: Path) -> str:
    """Return the content identity used to bind a manifest to one dataset revision."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    probes: Iterable[Probe],
    *,
    dataset_sha256: str,
    seed: int,
    per_category: int,
    evaluation_mode: str,
    excluded_probe_ids: Iterable[str] = (),
    excluded_families: Iterable[str] = (),
    allow_category_shortfalls: bool = False,
) -> dict[str, object]:
    """Select the same deterministic sample regardless of source row order."""
    if per_category < 1:
        raise ValueError("per_category must be positive")
    excluded = set(excluded_probe_ids)
    excluded_groups = set(excluded_families)
    probes = list(probes)
    expected_categories = {
        probe.category for probe in probes if probe.evaluation_mode == evaluation_mode
    }
    candidates: dict[str, list[Probe]] = defaultdict(list)
    for probe in probes:
        if (
            probe.evaluation_mode == evaluation_mode
            and probe.id not in excluded
            and probe.group not in excluded_groups
        ):
            candidates[probe.category].append(probe)
    if not candidates:
        raise ValueError("no probes matched the requested manifest")

    selected: list[Probe] = []
    shortfalls: dict[str, int] = {}
    for category in sorted(expected_categories):
        ranked = sorted(
            candidates.get(category, []),
            key=lambda probe: (
                hashlib.sha256(f"{seed}:{category}:{probe.id}".encode()).hexdigest(),
                probe.id,
            ),
        )
        selected.extend(ranked[:per_category])
        if len(ranked) < per_category:
            shortfalls[category] = per_category - len(ranked)
    if shortfalls and not allow_category_shortfalls:
        raise ValueError(f"category shortfalls: {shortfalls}")

    rows = [
        {
            "probe_id": probe.id,
            "category": probe.category,
            "family": probe.group,
            "evaluation_mode": probe.evaluation_mode,
            "review_status": probe.review_status,
        }
        for probe in selected
    ]
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "selection": {
            "seed": seed,
            "per_category": per_category,
            "evaluation_mode": evaluation_mode,
            "excluded_probe_count": len(excluded),
            "excluded_family_count": len(excluded_groups),
            "category_shortfalls": shortfalls,
        },
        "probe_count": len(rows),
        "category_count": len(expected_categories),
        "probes": rows,
    }


def build_explicit_manifest(
    probes: Mapping[str, Probe],
    *,
    dataset_sha256: str,
    probe_ids: Iterable[str],
    purpose: str,
) -> dict[str, object]:
    """Freeze a predeclared targeted sample without data-dependent reordering."""
    ids = list(probe_ids)
    if not ids:
        raise ValueError("explicit manifest needs at least one probe id")
    if len(ids) != len(set(ids)):
        raise ValueError("explicit manifest contains duplicate probe ids")
    missing = [probe_id for probe_id in ids if probe_id not in probes]
    if missing:
        raise ValueError(f"explicit manifest references unknown probes: {missing}")
    rows = [
        {
            "probe_id": probes[probe_id].id,
            "category": probes[probe_id].category,
            "family": probes[probe_id].group,
            "evaluation_mode": probes[probe_id].evaluation_mode,
            "review_status": probes[probe_id].review_status,
        }
        for probe_id in ids
    ]
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "selection": {"method": "explicit", "purpose": purpose},
        "probe_count": len(rows),
        "category_count": len({row["category"] for row in rows}),
        "probes": rows,
    }


def manifest_probe_ids(
    manifest: Mapping[str, object],
    probes: Mapping[str, Probe],
    *,
    dataset_sha256: str,
) -> list[str]:
    """Validate a frozen manifest against the exact current dataset."""
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported probe manifest schema")
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("probe manifest dataset_sha256 does not match the dataset")
    raw_rows = manifest.get("probes")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("probe manifest needs a non-empty probes array")
    ids: list[str] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("probe manifest entries must be objects")
        probe_id = str(raw_row.get("probe_id", ""))
        if probe_id not in probes:
            raise ValueError(f"probe manifest references unknown probe {probe_id!r}")
        probe = probes[probe_id]
        if raw_row.get("category") != probe.category:
            raise ValueError(f"probe manifest metadata changed for {probe_id}")
        ids.append(probe_id)
    if len(ids) != len(set(ids)):
        raise ValueError("probe manifest contains duplicate probe ids")
    if manifest.get("probe_count") != len(ids):
        raise ValueError("probe manifest probe_count is inconsistent")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--evaluation-mode", choices=("normative", "preference"), default="preference"
    )
    parser.add_argument(
        "--allow-category-shortfalls",
        action="store_true",
        help="emit remaining categories and record exhausted-category shortfalls",
    )
    parser.add_argument("--per-category", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-observations", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--exclude-observed-families",
        action="store_true",
        help="also exclude every sibling of a family present in excluded observations",
    )
    parser.add_argument("--probe-id", action="append", default=[])
    parser.add_argument("--purpose", default="targeted behavioral study")
    args = parser.parse_args()

    excluded = {
        row.probe_id
        for path in args.exclude_observations
        for row in load_observations(path)
    }
    catalog = load_probes(args.probes)
    excluded_families = (
        {catalog[probe_id].group for probe_id in excluded if catalog[probe_id].group}
        if args.exclude_observed_families
        else set()
    )
    if args.probe_id:
        if args.exclude_observations:
            parser.error("--probe-id cannot be combined with --exclude-observations")
        manifest = build_explicit_manifest(
            catalog,
            dataset_sha256=file_sha256(args.probes),
            probe_ids=args.probe_id,
            purpose=args.purpose,
        )
    else:
        manifest = build_manifest(
            catalog.values(),
            dataset_sha256=file_sha256(args.probes),
            seed=args.seed,
            per_category=args.per_category,
            evaluation_mode=args.evaluation_mode,
            excluded_probe_ids=excluded,
            excluded_families=excluded_families,
            allow_category_shortfalls=args.allow_category_shortfalls,
        )
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
