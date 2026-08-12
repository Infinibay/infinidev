"""Audit manual review consistency inside external near-duplicate families.

This module never creates or changes task-policy labels. It joins existing
human decisions to lexical families and exposes conflicts for human review.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from bench.external_candidate_family_split import candidate_families, read_jsonl


def _decision_key(row: dict[str, Any]) -> tuple[tuple[str, ...], str | None]:
    policies = row.get("policies")
    if not isinstance(policies, list):
        raise ValueError("every review needs a policies list")
    reason = row.get("uncategorized_reason")
    return tuple(sorted(str(policy) for policy in policies)), str(reason) if reason else None


def load_review_decisions(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Load unique included manual decisions from one or more ledgers."""
    decisions: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id:
                raise ValueError(f"{path}: review is missing candidate_id")
            if candidate_id in decisions:
                raise ValueError(f"duplicate review: {candidate_id}")
            if row.get("include") is True:
                _decision_key(row)
                decisions[candidate_id] = row
    return decisions


def audit_reviewed_families(
    candidates: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    *,
    min_containment: float = 0.55,
) -> dict[str, Any]:
    """Describe reviewed-family conflicts and unreviewed family members."""
    known_ids = {str(row.get("candidate_id", "")) for row in candidates}
    unknown = set(decisions) - known_ids
    if unknown:
        raise ValueError(f"reviews reference unknown candidates: {sorted(unknown)[:3]}")

    conflicts: list[dict[str, Any]] = []
    unreviewed: list[dict[str, Any]] = []
    reviewed_family_count = 0
    reviewed_family_rows = 0
    for family_index, family in enumerate(
        candidate_families(candidates, min_containment=min_containment)
    ):
        reviewed = [row for row in family if str(row["candidate_id"]) in decisions]
        if not reviewed:
            continue
        reviewed_family_count += 1
        reviewed_family_rows += len(family)
        decision_counts = Counter(
            _decision_key(decisions[str(row["candidate_id"])]) for row in reviewed
        )
        for row in family:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in decisions:
                unreviewed.append({"family_id": family_index, **row})
        if len(decision_counts) > 1:
            conflicts.append({
                "family_id": family_index,
                "family_size": len(family),
                "reviewed_size": len(reviewed),
                "decisions": [
                    {
                        "policies": list(key[0]),
                        "uncategorized_reason": key[1],
                        "count": count,
                    }
                    for key, count in sorted(decision_counts.items())
                ],
                "members": [
                    {
                        "candidate_id": str(row["candidate_id"]),
                        "issue_text": str(row["issue_text"]),
                        "review": decisions.get(str(row["candidate_id"])),
                    }
                    for row in family
                ],
            })
    unreviewed.sort(key=lambda row: (int(row["family_id"]), str(row["candidate_id"])))
    return {
        "candidate_rows": len(candidates),
        "reviewed_rows": len(decisions),
        "reviewed_families": reviewed_family_count,
        "reviewed_family_rows": reviewed_family_rows,
        "unreviewed_reviewed_family_rows": len(unreviewed),
        "conflicting_reviewed_families": len(conflicts),
        "conflicts": conflicts,
        "unreviewed": unreviewed,
    }


def _json_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--review-ledger", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--unreviewed", type=Path, required=True)
    parser.add_argument("--min-containment", type=float, default=0.55)
    args = parser.parse_args()

    report = audit_reviewed_families(
        read_jsonl(args.candidates),
        load_review_decisions(args.review_ledger),
        min_containment=args.min_containment,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _json_payload({key: value for key, value in report.items() if key != "unreviewed"}),
        encoding="utf-8",
    )
    args.unreviewed.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in report["unreviewed"]
        ),
        encoding="utf-8",
    )
    print(_json_payload({key: value for key, value in report.items() if key not in {"conflicts", "unreviewed"}}))


if __name__ == "__main__":
    main()
