from __future__ import annotations

import pytest

from bench.probe_campaign import build_checkpoint_manifest


def _campaign() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_sha256": "dataset",
        "probe_count": 3,
        "probes": [
            {"probe_id": "p1", "category": "a"},
            {"probe_id": "p2", "category": "b"},
            {"probe_id": "p3", "category": "c"},
        ],
    }


def test_checkpoint_preserves_parent_order_and_hash_lineage() -> None:
    value = build_checkpoint_manifest(
        _campaign(),
        campaign_sha256="campaign",
        start=1,
        count=2,
        purpose="checkpoint two",
    )
    assert [row["probe_id"] for row in value["probes"]] == ["p2", "p3"]
    assert value["dataset_sha256"] == "dataset"
    assert value["selection"] == {
        "method": "campaign_shard",
        "purpose": "checkpoint two",
        "parent_manifest_sha256": "campaign",
        "parent_probe_count": 3,
        "start": 1,
        "count": 2,
        "end_exclusive": 3,
    }


def test_checkpoint_rejects_partial_or_out_of_range_selection() -> None:
    for start, count in ((-1, 1), (0, 0), (2, 2)):
        with pytest.raises(ValueError):
            build_checkpoint_manifest(
                _campaign(),
                campaign_sha256="campaign",
                start=start,
                count=count,
                purpose="invalid",
            )
