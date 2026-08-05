#!/usr/bin/env python3
"""Create hash-bound whole-probe checkpoints from a frozen campaign manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_checkpoint_manifest(
    campaign: Mapping[str, object],
    *,
    campaign_sha256: str,
    start: int,
    count: int,
    purpose: str,
) -> dict[str, object]:
    """Select a contiguous predeclared range and retain parent-manifest lineage."""
    if campaign.get("schema_version") != 1:
        raise ValueError("unsupported campaign manifest schema")
    rows = campaign.get("probes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("campaign manifest needs a non-empty probes array")
    if not campaign_sha256 or start < 0 or count < 1:
        raise ValueError("campaign hash, nonnegative start, and positive count are required")
    if start >= len(rows) or start + count > len(rows):
        raise ValueError("checkpoint range exceeds campaign probes")
    selected = rows[start : start + count]
    if not all(isinstance(row, dict) for row in selected):
        raise ValueError("campaign probe entries must be objects")
    return {
        "schema_version": 1,
        "dataset_sha256": campaign.get("dataset_sha256"),
        "selection": {
            "method": "campaign_shard",
            "purpose": purpose,
            "parent_manifest_sha256": campaign_sha256,
            "parent_probe_count": len(rows),
            "start": start,
            "count": count,
            "end_exclusive": start + count,
        },
        "probe_count": len(selected),
        "category_count": len({str(row.get("category", "")) for row in selected}),
        "probes": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    checkpoint = build_checkpoint_manifest(
        campaign,
        campaign_sha256=file_sha256(args.campaign),
        start=args.start,
        count=args.count,
        purpose=args.purpose,
    )
    args.output.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
