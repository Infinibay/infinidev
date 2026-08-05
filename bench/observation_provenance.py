#!/usr/bin/env python3
"""Bind existing observation artifacts to exact dataset and manifest hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_rows(
    rows: list[dict[str, object]],
    *,
    dataset_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, object]]:
    """Add provenance only when existing values are empty or exactly matching."""
    if not dataset_sha256 or not manifest_sha256:
        raise ValueError("dataset and manifest hashes must be non-empty")
    for row in rows:
        existing_dataset = str(row.get("dataset_sha256", ""))
        existing_manifest = str(row.get("manifest_sha256", ""))
        if existing_dataset and existing_dataset != dataset_sha256:
            raise ValueError("observation already has a different dataset_sha256")
        if existing_manifest and existing_manifest != manifest_sha256:
            raise ValueError("observation already has a different manifest_sha256")
        row["dataset_sha256"] = dataset_sha256
        row["manifest_sha256"] = manifest_sha256
    return rows


def bind_artifact(
    source: Path,
    output: Path,
    *,
    dataset: Path,
    manifest: Path,
) -> Mapping[str, object]:
    """Write a bound copy and return hash lineage for the transformation."""
    source_hash = file_sha256(source)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset_hash = file_sha256(dataset)
    manifest_hash = file_sha256(manifest)
    bind_rows(
        rows,
        dataset_sha256=dataset_hash,
        manifest_sha256=manifest_hash,
    )
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "source": str(source),
        "output": str(output),
        "source_artifact_sha256": source_hash,
        "bound_artifact_sha256": file_sha256(output),
        "dataset_sha256": dataset_hash,
        "manifest_sha256": manifest_hash,
        "observation_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lineage", type=Path)
    args = parser.parse_args()
    lineage = bind_artifact(
        args.source,
        args.output,
        dataset=args.dataset,
        manifest=args.manifest,
    )
    if args.lineage:
        args.lineage.write_text(
            json.dumps(lineage, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
