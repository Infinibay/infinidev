"""Promote audited MiniMax proposals to an explicitly model-labeled ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from bench.task_policy_minimax_proposals import (
    BATCH_PROMPT_VERSION,
    REVIEWER_KIND,
    load_candidates,
)
from bench.task_policy_teacher_proposals import (
    UNCATEGORIZED_REASONS,
    parse_teacher_decision,
)


DATASET_VERSION = "task-policy-minimax-model-labels-v1"
TRAINING_REASON_MAP = {
    "acknowledgement": "answer_only",
    "status_only": "answer_only",
    "conceptual_question": "answer_only",
    "explanation_only": "answer_only",
    "quoted_action": "answer_only",
    "hypothetical_future": "answer_only",
    "meta_method": "answer_only",
    "ambiguous_authority": "ambiguous_method",
    "out_of_domain": "out_of_domain",
    "unsupported_method": "unsupported_method",
    "ambiguous_method": "ambiguous_method",
    "continuation_without_task": "answer_only",
    "conflicting_request": "ambiguous_method",
    "insufficient_context": "ambiguous_method",
    "reported_third_party_request": "answer_only",
    "healthy_existing_plan": "answer_only",
}
if set(TRAINING_REASON_MAP) != set(UNCATEGORIZED_REASONS):
    raise RuntimeError("training reason map does not cover the teacher taxonomy")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promote_model_proposals(
    candidate_paths: Iterable[Path],
    proposals_path: Path,
    output_path: Path,
    *,
    accept_model_labels: bool,
) -> dict[str, Any]:
    """Validate and convert proposals without disguising their model provenance."""
    if not accept_model_labels:
        raise ValueError("model labels require explicit acceptance")
    paths = list(candidate_paths)
    candidates = load_candidates(paths)
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    proposals = _read_jsonl(proposals_path)
    proposal_by_id: dict[str, dict[str, Any]] = {}
    for row in proposals:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError("proposal is missing candidate_id")
        if candidate_id in proposal_by_id:
            raise ValueError(f"duplicate proposal: {candidate_id}")
        proposal_by_id[candidate_id] = row
    if set(candidate_ids) != set(proposal_by_id):
        missing = set(candidate_ids) - set(proposal_by_id)
        extra = set(proposal_by_id) - set(candidate_ids)
        raise ValueError(
            f"proposal coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    labels: Counter[str] = Counter()
    cardinality: Counter[int] = Counter()
    source_zero_reasons: Counter[str] = Counter()
    training_zero_reasons: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        proposal = proposal_by_id[candidate_id]
        if proposal.get("proposal_status") != "model_reviewed":
            raise ValueError(f"proposal {candidate_id} is not model_reviewed")
        if proposal.get("reviewer_kind") != REVIEWER_KIND:
            raise ValueError(f"proposal {candidate_id} is not model provenance")
        if proposal.get("prompt_version") != BATCH_PROMPT_VERSION:
            raise ValueError(f"proposal {candidate_id} has an unexpected prompt version")
        decision = parse_teacher_decision(json.dumps({
            "policies": proposal.get("policies"),
            "uncategorized_reason": proposal.get("uncategorized_reason"),
            "confidence": proposal.get("confidence"),
            "rationale": proposal.get("notes"),
        }))
        labels.update(decision.policies)
        cardinality[len(decision.policies)] += 1
        training_reason = (
            TRAINING_REASON_MAP[decision.uncategorized_reason]
            if decision.uncategorized_reason is not None
            else None
        )
        if decision.uncategorized_reason:
            source_zero_reasons[decision.uncategorized_reason] += 1
            training_zero_reasons[str(training_reason)] += 1
        annotation = {
            "kind": "model",
            "model": proposal.get("reviewer_model"),
            "reviewer_version": proposal.get("reviewer_version"),
            "prompt_version": proposal.get("prompt_version"),
            "confidence": decision.confidence,
            "response_id": proposal.get("response_id"),
        }
        if decision.uncategorized_reason is not None:
            annotation["source_uncategorized_reason"] = decision.uncategorized_reason
        output_rows.append({
            "candidate_id": candidate_id,
            "include": True,
            "policies": list(decision.policies),
            "uncategorized_reason": training_reason,
            "notes": decision.rationale,
            "annotation": annotation,
        })

    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in output_rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    manifest = {
        "version": DATASET_VERSION,
        "annotation_kind": "model",
        "accepted_model_labels": True,
        "rows": len(output_rows),
        "candidate_sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "proposals": {
            "path": str(proposals_path),
            "sha256": _sha256(proposals_path),
        },
        "reviews": {
            "path": str(output_path),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        },
        "labels": dict(sorted(labels.items())),
        "cardinality": {str(key): value for key, value in sorted(cardinality.items())},
        "source_zero_reasons": dict(sorted(source_zero_reasons.items())),
        "training_zero_reasons": dict(sorted(training_zero_reasons.items())),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposals", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--accept-model-labels", action="store_true")
    args = parser.parse_args()
    manifest = promote_model_proposals(
        args.candidate,
        args.proposals,
        args.output,
        accept_model_labels=args.accept_model_labels,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["DATASET_VERSION", "TRAINING_REASON_MAP", "promote_model_proposals"]
