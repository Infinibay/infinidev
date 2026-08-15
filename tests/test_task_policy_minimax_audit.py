from __future__ import annotations

import json

import pytest

from bench.task_policy_minimax_audit import audit_proposals


def _write(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _proposal(candidate_id: str, policies: list[str], confidence: float = 0.9) -> dict:
    return {
        "candidate_id": candidate_id,
        "proposal_status": "model_reviewed",
        "reviewer_kind": "model",
        "policies": policies,
        "uncategorized_reason": None if policies else "unsupported_method",
        "confidence": confidence,
        "notes": "The requested contract determines this category.",
    }


def test_audit_reports_exact_set_and_per_label_metrics(tmp_path) -> None:
    reference = tmp_path / "reference.jsonl"
    proposals = tmp_path / "proposals.jsonl"
    _write(reference, [
        {"candidate_id": "a", "include": True, "policies": ["bugfix"], "notes": "x"},
        {"candidate_id": "b", "include": True, "policies": ["feature"], "notes": "x"},
        {
            "candidate_id": "c",
            "include": True,
            "policies": [],
            "uncategorized_reason": "unsupported_method",
            "notes": "x",
        },
    ])
    _write(proposals, [
        _proposal("a", ["bugfix"]),
        _proposal("b", ["feature", "performance"]),
    ])

    report = audit_proposals(reference, proposals)

    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["exact_match"] == 0.5
    assert report["mean_jaccard"] == 0.75
    assert report["missing_candidate_ids"] == ["c"]
    assert report["per_label"]["bugfix"]["f1"] == 1.0
    assert report["per_label"]["bugfix"]["accuracy"] == 1.0
    assert report["per_label"]["performance"]["fp"] == 1
    assert report["per_label"]["performance"]["accuracy"] == 0.5
    assert report["disagreements"][0]["candidate_id"] == "b"


def test_audit_rejects_unknown_or_non_model_rows(tmp_path) -> None:
    reference = tmp_path / "reference.jsonl"
    proposals = tmp_path / "proposals.jsonl"
    _write(reference, [
        {"candidate_id": "a", "include": True, "policies": ["bugfix"], "notes": "x"},
    ])
    invalid = _proposal("a", ["bugfix"])
    invalid["reviewer_kind"] = "human"
    _write(proposals, [invalid])

    with pytest.raises(ValueError, match="reviewer_kind"):
        audit_proposals(reference, proposals)

    _write(proposals, [_proposal("unknown", ["bugfix"])])
    with pytest.raises(ValueError, match="unknown candidate IDs"):
        audit_proposals(reference, proposals)
