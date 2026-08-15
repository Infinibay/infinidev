from __future__ import annotations

import json

import pytest

from bench.task_policy_minimax_dataset import promote_model_proposals
from bench.task_policy_minimax_proposals import BATCH_PROMPT_VERSION, REVIEWER_VERSION


def _write(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_promote_model_proposals_requires_acceptance_and_preserves_provenance(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    proposals = tmp_path / "proposals.jsonl"
    output = tmp_path / "reviews.jsonl"
    _write(candidates, [{"candidate_id": "one", "issue_text": "Fix stale state."}])
    _write(proposals, [{
        "candidate_id": "one",
        "proposal_status": "model_reviewed",
        "reviewer_kind": "model",
        "reviewer_model": "MiniMax-M3",
        "reviewer_version": REVIEWER_VERSION,
        "prompt_version": BATCH_PROMPT_VERSION,
        "policies": ["bugfix"],
        "uncategorized_reason": None,
        "confidence": 0.91,
        "notes": "It restores existing behavior.",
        "response_id": "response-one",
    }])

    with pytest.raises(ValueError, match="explicit acceptance"):
        promote_model_proposals(
            [candidates], proposals, output, accept_model_labels=False
        )
    manifest = promote_model_proposals(
        [candidates], proposals, output, accept_model_labels=True
    )
    review = json.loads(output.read_text(encoding="utf-8"))

    assert manifest["rows"] == 1
    assert manifest["annotation_kind"] == "model"
    assert review["include"] is True
    assert review["policies"] == ["bugfix"]
    assert review["annotation"] == {
        "kind": "model",
        "model": "MiniMax-M3",
        "reviewer_version": REVIEWER_VERSION,
        "prompt_version": BATCH_PROMPT_VERSION,
        "confidence": 0.91,
        "response_id": "response-one",
    }


def test_promote_model_proposals_requires_exact_candidate_coverage(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    proposals = tmp_path / "proposals.jsonl"
    _write(candidates, [{"candidate_id": "one", "issue_text": "Fix stale state."}])
    _write(proposals, [])

    with pytest.raises(ValueError, match="coverage mismatch"):
        promote_model_proposals(
            [candidates], proposals, tmp_path / "out.jsonl", accept_model_labels=True
        )


def test_promote_model_proposals_maps_fine_zero_reason_but_preserves_source(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    proposals = tmp_path / "proposals.jsonl"
    output = tmp_path / "reviews.jsonl"
    _write(candidates, [{"candidate_id": "one", "issue_text": "How does this work?"}])
    _write(proposals, [{
        "candidate_id": "one",
        "proposal_status": "model_reviewed",
        "reviewer_kind": "model",
        "reviewer_model": "MiniMax-M3",
        "reviewer_version": REVIEWER_VERSION,
        "prompt_version": BATCH_PROMPT_VERSION,
        "policies": [],
        "uncategorized_reason": "conceptual_question",
        "confidence": 0.88,
        "notes": "Only an explanation is requested.",
        "response_id": "response-one",
    }])

    manifest = promote_model_proposals(
        [candidates], proposals, output, accept_model_labels=True
    )
    review = json.loads(output.read_text(encoding="utf-8"))

    assert review["uncategorized_reason"] == "answer_only"
    assert review["annotation"]["source_uncategorized_reason"] == "conceptual_question"
    assert manifest["source_zero_reasons"] == {"conceptual_question": 1}
    assert manifest["training_zero_reasons"] == {"answer_only": 1}
