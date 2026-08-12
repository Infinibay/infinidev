from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bench.task_policy_data_bootstrap import (
    EXPECTED_FAMILIES,
    EXPECTED_ROWS,
    EXPECTED_SPLIT_ROWS,
    _validate_review_ledgers,
    _validate_split_manifest,
    _verify_candidate,
    candidate_acquisitions,
    review_ledger_paths,
)


def test_acquisition_plan_pins_all_sources_and_exclusion_rounds(tmp_path: Path) -> None:
    plan = candidate_acquisitions(tmp_path)

    assert len(plan) == 5
    assert len({item.sha256 for item in plan}) == 5
    assert plan[-1].module == "bench.wildchat_candidate_sampler"
    assert "--exclude-candidates" not in plan[0].arguments
    assert plan[1].arguments.count("--exclude-candidates") == 1
    assert plan[2].arguments.count("--exclude-candidates") == 2
    assert plan[3].arguments.count("--exclude-candidates") == 3


def test_candidate_digest_validation_rejects_drift(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(b"one\n")

    _verify_candidate(candidate, hashlib.sha256(b"one\n").hexdigest())
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _verify_candidate(candidate, "0" * 64)


def test_review_plan_contains_every_manual_ledger_and_explains_missing(
    tmp_path: Path,
) -> None:
    paths = review_ledger_paths(tmp_path)

    assert len(paths) == 37
    assert paths[-1].name == "family_round1_queue_15_reviews.jsonl"
    with pytest.raises(RuntimeError, match="cannot be regenerated"):
        _validate_review_ledgers(paths)


def test_split_manifest_guard_accepts_only_fixed_corpus() -> None:
    manifest = {
        "rows": EXPECTED_ROWS,
        "families": EXPECTED_FAMILIES,
        "splits": {
            name: {"rows": rows}
            for name, rows in EXPECTED_SPLIT_ROWS.items()
        },
    }

    _validate_split_manifest(manifest)
    manifest["splits"]["evaluation"]["rows"] -= 1
    with pytest.raises(RuntimeError, match="row counts changed"):
        _validate_split_manifest(manifest)
