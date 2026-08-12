from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from bench.task_policy_data_bootstrap import (
    EXPECTED_FAMILIES,
    EXPECTED_ROWS,
    EXPECTED_SPLIT_ROWS,
    TRACKED_REVIEW_ROOT,
    _validate_review_ledgers,
    _validate_split_manifest,
    _verify_candidate,
    candidate_acquisitions,
    default_review_root,
    review_ledger_paths,
)


PUBLIC_REVIEW_FIELDS = {
    "candidate_id",
    "include",
    "policies",
    "uncategorized_reason",
    "notes",
}
SENSITIVE_PATTERNS = (
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"https?://"),
    re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|pypi-[A-Za-z0-9_-]{12,})"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
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


def test_default_review_root_falls_back_to_external_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked = tmp_path / "tracked"
    external = tmp_path / "external"
    monkeypatch.setattr("bench.task_policy_data_bootstrap.TRACKED_REVIEW_ROOT", tracked)

    assert default_review_root(external) == external

    for path in review_ledger_paths(tracked):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    assert default_review_root(external) == tracked


def test_tracked_review_data_is_complete_and_minimized() -> None:
    paths = review_ledger_paths(TRACKED_REVIEW_ROOT)
    rows = []
    for path in paths:
        assert path.is_file(), path
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert set(row) <= PUBLIC_REVIEW_FIELDS
            assert len(str(row["notes"])) <= 200
            rendered = json.dumps(row, ensure_ascii=False)
            assert not any(pattern.search(rendered) for pattern in SENSITIVE_PATTERNS)
            rows.append(row)

    assert len(paths) == 37
    assert len(rows) == EXPECTED_ROWS
    assert len({row["candidate_id"] for row in rows}) == EXPECTED_ROWS


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
