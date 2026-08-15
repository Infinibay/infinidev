"""Audit MiniMax task-policy proposals against an independent review ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench.task_policy_teacher_proposals import POLICIES, parse_teacher_decision


def _read_unique(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError(f"{path}:{line_number}: missing candidate_id")
        if candidate_id in rows:
            raise ValueError(f"{path}: duplicate candidate_id {candidate_id}")
        rows[candidate_id] = row
    return rows


def _labels(row: dict[str, Any], *, proposal: bool) -> frozenset[str]:
    field = "policies"
    raw = row.get(field)
    if not isinstance(raw, list) or not all(isinstance(label, str) for label in raw):
        raise ValueError(f"{field} must be a string array")
    labels = frozenset(raw)
    unknown = labels - set(POLICIES)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    if len(labels) != len(raw):
        raise ValueError("policies contains duplicates")
    if proposal:
        if row.get("proposal_status") != "model_reviewed":
            raise ValueError("proposal_status must be model_reviewed")
        if row.get("reviewer_kind") != "model":
            raise ValueError("reviewer_kind must be model")
        parse_teacher_decision(json.dumps({
            "policies": raw,
            "uncategorized_reason": row.get("uncategorized_reason"),
            "confidence": row.get("confidence"),
            "rationale": row.get("notes"),
        }))
    else:
        if row.get("include") is not True:
            raise ValueError("reference rows must have include=true")
        reason = row.get("uncategorized_reason")
        if labels and reason:
            raise ValueError("labeled reference cannot have uncategorized_reason")
        if not labels and not isinstance(reason, str):
            raise ValueError("unlabeled reference requires uncategorized_reason")
    return labels


def audit_proposals(reference_path: Path, proposal_path: Path) -> dict[str, Any]:
    """Calculate coverage, set-level agreement, and independent label metrics."""
    references = _read_unique(reference_path)
    proposals = _read_unique(proposal_path)
    unknown_ids = sorted(set(proposals) - set(references))
    if unknown_ids:
        raise ValueError(f"proposals contain {len(unknown_ids)} unknown candidate IDs")

    missing_ids = sorted(set(references) - set(proposals))
    compared_ids = sorted(set(references) & set(proposals))
    per_label = {
        label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "support": 0}
        for label in POLICIES
    }
    exact = 0
    jaccard_total = 0.0
    disagreements: list[dict[str, Any]] = []
    reference_cardinality = 0
    predicted_cardinality = 0
    for candidate_id in compared_ids:
        expected = _labels(references[candidate_id], proposal=False)
        predicted = _labels(proposals[candidate_id], proposal=True)
        reference_cardinality += len(expected)
        predicted_cardinality += len(predicted)
        if expected == predicted:
            exact += 1
        else:
            disagreements.append({
                "candidate_id": candidate_id,
                "expected": sorted(expected),
                "predicted": sorted(predicted),
                "confidence": proposals[candidate_id].get("confidence"),
                "rationale": proposals[candidate_id].get("notes"),
            })
        union = expected | predicted
        jaccard_total += len(expected & predicted) / len(union) if union else 1.0
        for label, counts in per_label.items():
            expected_positive = label in expected
            predicted_positive = label in predicted
            counts["support"] += int(expected_positive)
            counts["tp"] += int(expected_positive and predicted_positive)
            counts["fp"] += int(not expected_positive and predicted_positive)
            counts["fn"] += int(expected_positive and not predicted_positive)
            counts["tn"] += int(not expected_positive and not predicted_positive)

    metrics: dict[str, dict[str, float | int]] = {}
    for label, counts in per_label.items():
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total else 0.0
        metrics[label] = {
            **counts,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    compared = len(compared_ids)
    return {
        "reference_examples": len(references),
        "proposal_examples": len(proposals),
        "compared_examples": compared,
        "coverage": compared / len(references) if references else 0.0,
        "exact_match": exact / compared if compared else 0.0,
        "mean_jaccard": jaccard_total / compared if compared else 0.0,
        "reference_cardinality": reference_cardinality / compared if compared else 0.0,
        "predicted_cardinality": predicted_cardinality / compared if compared else 0.0,
        "missing_candidate_ids": missing_ids,
        "per_label": metrics,
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("proposals", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--min-exact-match", type=float)
    args = parser.parse_args()

    report = audit_proposals(args.reference, args.proposals)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_complete and report["coverage"] != 1.0:
        raise SystemExit(2)
    if args.min_exact_match is not None and report["exact_match"] < args.min_exact_match:
        raise SystemExit(3)


if __name__ == "__main__":
    main()


__all__ = ["audit_proposals"]
