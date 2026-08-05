#!/usr/bin/env python3
"""Materialize a hash-bound probe revision without mutating its evidence base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

IMMUTABLE_FIELDS = frozenset(
    {"id", "group", "evaluation_mode", "review_status", "answer", "reviewer"}
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_path(record: dict[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    if not parts or parts[0] in IMMUTABLE_FIELDS:
        raise ValueError(f"revision cannot change protected field {path}")
    target: dict[str, Any] = record
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"revision path does not address an object: {path}")
        target = child
    if parts[-1] not in target:
        raise ValueError(f"revision path does not exist: {path}")
    target[parts[-1]] = value


def materialize_revision(
    records: list[dict[str, Any]],
    spec: Mapping[str, object],
    *,
    base_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Apply explicit field replacements and return revised rows plus lineage."""
    expected_hash = str(spec.get("base_dataset_sha256", ""))
    if expected_hash != base_sha256:
        raise ValueError("revision spec base_dataset_sha256 does not match input")
    revision_id = str(spec.get("revision_id", "")).strip()
    raw_changes = spec.get("changes")
    if not revision_id or not isinstance(raw_changes, list) or not raw_changes:
        raise ValueError("revision spec needs revision_id and non-empty changes")
    by_id = {str(record.get("id")): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("base dataset contains duplicate probe IDs")
    changed_ids: list[str] = []
    rationales: list[dict[str, object]] = []
    for raw_change in raw_changes:
        if not isinstance(raw_change, dict):
            raise ValueError("revision change must be an object")
        probe_id = str(raw_change.get("probe_id", ""))
        updates = raw_change.get("updates")
        rationale = str(raw_change.get("rationale", "")).strip()
        evidence = raw_change.get("evidence")
        if probe_id not in by_id:
            raise ValueError(f"revision references unknown probe {probe_id}")
        if probe_id in changed_ids:
            raise ValueError(f"revision repeats probe {probe_id}")
        if not isinstance(updates, dict) or not updates or not rationale:
            raise ValueError(f"revision for {probe_id} needs updates and rationale")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"revision for {probe_id} needs evidence references")
        for path, value in updates.items():
            _set_path(by_id[probe_id], str(path), value)
        changed_ids.append(probe_id)
        rationales.append(
            {
                "probe_id": probe_id,
                "updated_paths": sorted(str(path) for path in updates),
                "rationale": rationale,
                "evidence": [str(item) for item in evidence],
            }
        )
    return records, {
        "schema_version": 1,
        "revision_id": revision_id,
        "base_dataset_sha256": base_sha256,
        "changed_probe_count": len(changed_ids),
        "changes": rationales,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("lineage", type=Path)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.base.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    revised, lineage = materialize_revision(
        records, spec, base_sha256=file_sha256(args.base)
    )
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in revised),
        encoding="utf-8",
    )
    lineage["revised_dataset_sha256"] = file_sha256(args.output)
    args.lineage.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
