#!/usr/bin/env python3
"""Export blind probe packets and gate approval on independent review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from bench.model_behavior import Probe, load_probes, read_jsonl


@dataclass(frozen=True)
class ProbeReview:
    """One independent review decision for a blinded probe."""

    probe_id: str
    reviewer: str
    dataset_sha256: str
    verdict: str
    evaluation_mode: str
    answer: str | None
    rationale: str
    effects_valid: bool | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProbeReview:
        verdict = str(value.get("verdict", ""))
        mode = str(value.get("evaluation_mode", ""))
        raw_answer = value.get("answer")
        answer = str(raw_answer).upper() if raw_answer not in {None, ""} else None
        if verdict not in {"accept", "revise", "reject"}:
            raise ValueError("review verdict must be accept, revise, or reject")
        if mode not in {"normative", "preference"}:
            raise ValueError("review evaluation_mode must be normative or preference")
        if mode == "normative" and answer not in {"A", "B", "C", "D"}:
            raise ValueError("normative review needs an A-D answer")
        if mode == "preference" and answer is not None:
            raise ValueError("preference review must not define an answer")
        reviewer = str(value.get("reviewer", "")).strip()
        rationale = str(value.get("rationale", "")).strip()
        dataset_sha256 = str(value.get("dataset_sha256", "")).strip()
        if not reviewer or not rationale or not dataset_sha256:
            raise ValueError("review needs reviewer, rationale, and dataset_sha256")
        raw_effects = value.get("effects_valid")
        effects_valid = raw_effects if isinstance(raw_effects, bool) else None
        if mode == "preference" and effects_valid is None:
            raise ValueError("preference review needs effects_valid")
        return cls(
            probe_id=str(value.get("probe_id", "")),
            reviewer=reviewer,
            dataset_sha256=dataset_sha256,
            verdict=verdict,
            evaluation_mode=mode,
            answer=answer,
            rationale=rationale,
            effects_valid=effects_valid,
        )


def blind_packet(
    probes: Mapping[str, Probe], *, dataset_sha256: str
) -> dict[str, object]:
    """Remove author labels and rationales while retaining fields a reviewer must assess."""
    items: list[dict[str, object]] = []
    for probe in sorted(probes.values(), key=lambda item: (item.category, item.group or "", item.id)):
        item: dict[str, object] = {
            "probe_id": probe.id,
            "family": probe.group,
            "category": probe.category,
            "scenario": probe.scenario or probe.prompt,
            "user_request": probe.user_request or "",
            "choices": probe.choices,
            "evaluation_mode": probe.evaluation_mode,
            "tags": list(probe.tags),
        }
        if probe.evaluation_mode == "preference":
            item["choice_effects"] = probe.choice_effects
        items.append(item)
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "review_instructions": {
            "normative": (
                "Independently select the single best action. Mark revise or reject if more than "
                "one action is defensible or the scenario lacks decisive evidence."
            ),
            "preference": (
                "Confirm that no universal answer exists, every option is safe and competent, and "
                "the declared effects match the actual trade-off."
            ),
        },
        "items": items,
    }


def load_reviews(path: Path) -> list[ProbeReview]:
    """Load independent review decisions from JSONL."""
    return [ProbeReview.from_dict(value) for value in read_jsonl(path)]


def review_report(
    probes: Mapping[str, Probe],
    reviews: Iterable[ProbeReview],
    *,
    dataset_sha256: str,
    min_reviews: int = 1,
) -> dict[str, object]:
    """Return probe and whole-family approval decisions with explicit disagreement reasons."""
    if min_reviews < 1:
        raise ValueError("min_reviews must be positive")
    indexed: dict[str, list[ProbeReview]] = defaultdict(list)
    duplicate_reviewers: list[str] = []
    seen: set[tuple[str, str]] = set()
    for review in reviews:
        if review.probe_id not in probes:
            raise ValueError(f"review references unknown probe: {review.probe_id}")
        if review.dataset_sha256 != dataset_sha256:
            raise ValueError("review dataset hash does not match current probes")
        key = (review.probe_id, review.reviewer)
        if key in seen:
            duplicate_reviewers.append(f"{review.probe_id}:{review.reviewer}")
            continue
        seen.add(key)
        indexed[review.probe_id].append(review)

    probe_decisions: dict[str, dict[str, object]] = {}
    for probe in probes.values():
        rows = indexed.get(probe.id, [])
        reasons: list[str] = []
        independent = [row for row in rows if row.reviewer != probe.generator]
        if len(independent) != len(rows):
            reasons.append("author cannot review their own probe")
        if len({row.reviewer for row in independent}) < min_reviews:
            reasons.append("insufficient independent reviews")
        if any(row.verdict != "accept" for row in independent):
            reasons.append("reviewer requested revision or rejection")
        if any(row.evaluation_mode != probe.evaluation_mode for row in independent):
            reasons.append("evaluation mode disagreement")
        if probe.evaluation_mode == "normative":
            if any(row.answer != probe.answer for row in independent):
                reasons.append("gold answer disagreement")
        elif any(row.effects_valid is not True for row in independent):
            reasons.append("preference effects not accepted")
        probe_decisions[probe.id] = {
            "approved": not reasons,
            "reviewers": sorted({row.reviewer for row in independent}),
            "reasons": reasons,
            "rationales": [row.rationale for row in independent],
        }

    groups: dict[str, list[str]] = defaultdict(list)
    for probe in probes.values():
        groups[probe.group or probe.id].append(probe.id)
    approved_families = sorted(
        group
        for group, probe_ids in groups.items()
        if all(probe_decisions[probe_id]["approved"] for probe_id in probe_ids)
    )
    approved_family_set = set(approved_families)
    approved_probes = sorted(
        probe_id
        for group, probe_ids in groups.items()
        if group in approved_family_set
        for probe_id in probe_ids
    )
    return {
        "dataset_sha256": dataset_sha256,
        "min_reviews": min_reviews,
        "review_rows": sum(len(rows) for rows in indexed.values()),
        "duplicate_reviewer_rows": sorted(duplicate_reviewers),
        "approved_probes": approved_probes,
        "approved_families": approved_families,
        "probe_decisions": probe_decisions,
    }


def apply_review_report(
    probes: Mapping[str, Probe], report: Mapping[str, object]
) -> list[dict[str, object]]:
    """Return a dataset copy with only whole independently reviewed families approved."""
    approved = set(str(item) for item in report.get("approved_probes", []))
    decisions = report.get("probe_decisions")
    if not isinstance(decisions, dict):
        raise ValueError("review report needs probe_decisions")
    output: list[dict[str, object]] = []
    for probe in probes.values():
        value = asdict(probe)
        if probe.id in approved:
            decision = decisions.get(probe.id)
            if not isinstance(decision, dict):
                raise ValueError(f"missing review decision for {probe.id}")
            reviewers = decision.get("reviewers")
            if not isinstance(reviewers, list) or not reviewers:
                raise ValueError(f"approved probe lacks reviewers: {probe.id}")
            value["review_status"] = "approved"
            value["reviewer"] = ",".join(sorted(str(item) for item in reviewers))
        output.append(value)
    return output


def _dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("probes", type=Path)
    export.add_argument("output", type=Path)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("probes", type=Path)
    report_parser.add_argument("reviews", type=Path)
    report_parser.add_argument("output", type=Path)
    report_parser.add_argument("--min-reviews", type=int, default=1)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("probes", type=Path)
    apply_parser.add_argument("report", type=Path)
    apply_parser.add_argument("output", type=Path)

    args = parser.parse_args()
    probes = load_probes(args.probes)
    dataset_sha256 = _dataset_sha256(args.probes)
    if args.command == "export":
        value = blind_packet(probes, dataset_sha256=dataset_sha256)
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return
    if args.command == "report":
        value = review_report(
            probes,
            load_reviews(args.reviews),
            dataset_sha256=dataset_sha256,
            min_reviews=args.min_reviews,
        )
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return
    value = json.loads(args.report.read_text(encoding="utf-8"))
    if value.get("dataset_sha256") != dataset_sha256:
        parser.error("report dataset hash does not match current probes")
    rows = apply_review_report(probes, value)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
