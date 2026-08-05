#!/usr/bin/env python3
"""Blind semantic review and adjudication for prompt-comprehension families."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

if __package__:
    from bench.prompt_comprehension import COMPREHENSION_FIELDS, ComprehensionCase, load_cases
else:
    from prompt_comprehension import COMPREHENSION_FIELDS, ComprehensionCase, load_cases


REVIEW_CHECKS = (
    "equivalents_preserve_meaning",
    "contrast_changes_only_intended_variable",
    "wording_is_natural",
    "semantic_completeness",
    "execution_sufficiency",
    "authorization_is_unambiguous",
    "no_split_leakage_detected",
)
LEGACY_REVIEW_CHECKS = (
    "equivalents_preserve_meaning",
    "contrast_changes_only_intended_variable",
    "wording_is_natural",
    "requests_are_self_contained",
    "authorization_is_unambiguous",
    "no_split_leakage_detected",
)
GATING_REVIEW_CHECKS = tuple(
    check for check in REVIEW_CHECKS if check != "execution_sufficiency"
)
PILOT_STUDY_KINDS = ("linguistic", "execution")
PILOT_SELECTION_VERSION = 1
CHECK_RESULTS = ("pass", "fail", "not_applicable_by_design")


def _check_passes(value: object) -> bool:
    return value is True or value in {"pass", "not_applicable_by_design"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        rows.append(value)
    return rows


def blind_packet(cases: Iterable[ComprehensionCase], *, dataset_sha256: str) -> dict[str, object]:
    """Export requests and relations while withholding every authored interpretation key."""
    families: dict[str, list[ComprehensionCase]] = defaultdict(list)
    for case in cases:
        families[case.family_id].append(case)
    items = []
    for family_id in sorted(families):
        variants = sorted(families[family_id], key=lambda case: case.id)
        anchors = [case for case in variants if case.intended_relation == "anchor"]
        if len(anchors) != 1:
            raise ValueError(f"blind family needs exactly one anchor: {family_id}")
        anchor = anchors[0]
        items.append(
            {
                "family_id": family_id,
                "category": variants[0].category,
                "research_question_id": variants[0].research_question_id,
                "relation_map": [
                    {
                        "left_case_id": anchor.id,
                        "right_case_id": case.id,
                        "expected_relation": {
                            "equivalent": "meaning_preserving",
                            "contrast": "meaning_changing",
                            "adversarial": "deliberate_stressor",
                        }[case.intended_relation],
                    }
                    for case in variants
                    if case.id != anchor.id
                ],
                "variants": [
                    {
                        "case_id": case.id,
                        "variant_id": case.variant_id,
                        "intended_relation": case.intended_relation,
                        "request": case.request,
                        "split": case.split,
                    }
                    for case in variants
                ],
            }
        )
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "instructions": (
            "Review each family without access to the authored expected keys. Reconstruct every "
            "variant using all comprehension fields, assess the family checks, and record any "
            "template or diversity concern. This phase cannot approve the dataset; a separate "
            "adjudicator must compare reconstructions with the hidden authored keys."
        ),
        "required_interpretation_fields": list(COMPREHENSION_FIELDS),
        "required_family_checks": list(REVIEW_CHECKS),
        "review_row_contract": {
            "family_id": "family identifier from this packet",
            "reviewer": "stable identity distinct from the question author",
            "dataset_sha256": dataset_sha256,
            "verdict": "accept | revise | reject",
            "rationale": "why the family is or is not a valid controlled instrument",
            "checks": {check: "pass | fail | not_applicable_by_design" for check in REVIEW_CHECKS},
            "reconstructions": {
                "CASE_ID": {field: "independent reconstruction" for field in COMPREHENSION_FIELDS}
            },
            "diversity_concern": "optional scenario or template-dependence concern",
        },
        "families": items,
    }


def select_pilot_cases(
    cases: Iterable[ComprehensionCase], *, seed: str = "semantic-pilot-v1"
) -> tuple[list[ComprehensionCase], dict[str, object]]:
    """Select one linguistic and one execution family per domain without cherry-picking."""
    rows = list(cases)
    families: dict[str, list[ComprehensionCase]] = defaultdict(list)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case in rows:
        families[case.family_id].append(case)
    for family_id, variants in families.items():
        profile = variants[0].stimulus_profile or {}
        domain = profile.get("domain", variants[0].category)
        study_kind = profile.get("study_kind", "")
        if study_kind in PILOT_STUDY_KINDS:
            strata[(domain, study_kind)].append(family_id)

    domains = sorted({domain for domain, _ in strata})
    selected: list[str] = []
    used_dimensions: set[str] = set()

    def research_dimension(case: ComprehensionCase) -> str:
        profile = case.stimulus_profile or {}
        if profile.get("study_kind") == "execution":
            return f"execution:{profile.get('execution_dimension', case.research_question_id)}"
        return f"linguistic:{profile.get('phenomenon', case.research_question_id)}"

    for domain in domains:
        for study_kind in PILOT_STUDY_KINDS:
            candidates = strata.get((domain, study_kind), [])
            if not candidates:
                raise ValueError(f"pilot stratum has no families: {domain}/{study_kind}")

            def rank(family_id: str) -> tuple[bool, str]:
                dimension = research_dimension(families[family_id][0])
                digest = hashlib.sha256(f"{seed}|{domain}|{study_kind}|{family_id}".encode()).hexdigest()
                return dimension in used_dimensions, digest

            family_id = min(candidates, key=rank)
            selected.append(family_id)
            used_dimensions.add(research_dimension(families[family_id][0]))

    selected_set = set(selected)
    selected_cases = sorted(
        (case for case in rows if case.family_id in selected_set),
        key=lambda case: (case.category, case.family_id, case.id),
    )
    return selected_cases, {
        "selection_version": PILOT_SELECTION_VERSION,
        "seed": seed,
        "domains": domains,
        "study_kinds": list(PILOT_STUDY_KINDS),
        "selected_family_ids": selected,
        "selected_question_ids": sorted(
            {families[family_id][0].research_question_id for family_id in selected}
        ),
        "selected_research_dimensions": sorted(used_dimensions),
    }


def review_template(packet: Mapping[str, object]) -> list[dict[str, object]]:
    """Build syntactically complete fail-closed review rows for a blinded packet."""
    families = packet.get("families")
    if not isinstance(families, list):
        raise ValueError("review packet has no families")
    dataset_sha256 = str(packet.get("dataset_sha256", ""))
    rows = []
    for family in families:
        if not isinstance(family, dict) or not isinstance(family.get("variants"), list):
            raise ValueError("review packet contains a malformed family")
        reconstructions = {}
        for variant in family["variants"]:
            if not isinstance(variant, dict):
                raise ValueError("review packet contains a malformed variant")
            reconstruction = {field: [] for field in COMPREHENSION_FIELDS}
            reconstruction["objective"] = "TODO"
            reconstruction["priority_resolution"] = ""
            reconstructions[str(variant.get("case_id", ""))] = reconstruction
        rows.append(
            {
                "family_id": family.get("family_id"),
                "reviewer": "TODO-independent-reviewer",
                "dataset_sha256": dataset_sha256,
                "verdict": "revise",
                "rationale": "TODO",
                "checks": {check: "fail" for check in REVIEW_CHECKS},
                "reconstructions": reconstructions,
                "diversity_concern": "",
            }
        )
    return rows


def shard_packets(packet: Mapping[str, object], *, shard_count: int) -> list[dict[str, object]]:
    """Split a blind packet deterministically while preserving its review contract."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    families = packet.get("families")
    if not isinstance(families, list) or shard_count > len(families):
        raise ValueError("shard_count cannot exceed the number of packet families")
    ordered = sorted(
        families,
        key=lambda row: (
            str(row.get("category", "")) if isinstance(row, dict) else "",
            str(row.get("family_id", "")) if isinstance(row, dict) else "",
        ),
    )
    buckets: list[list[object]] = [[] for _ in range(shard_count)]
    for index, family in enumerate(ordered):
        buckets[index % shard_count].append(family)
    shards = []
    for index, bucket in enumerate(buckets, 1):
        value = dict(packet)
        value["shard"] = {"index": index, "count": shard_count}
        value["families"] = bucket
        shards.append(value)
    return shards


def review_progress(
    packet: Mapping[str, object], review_rows: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    """Validate completed blind reviews without exposing or judging authored keys."""
    families = packet.get("families")
    if not isinstance(families, list):
        raise ValueError("review packet has no families")
    expected = {
        str(family.get("family_id", "")): {
            str(variant.get("case_id", ""))
            for variant in family.get("variants", [])
            if isinstance(variant, dict)
        }
        for family in families
        if isinstance(family, dict)
    }
    digest = str(packet.get("dataset_sha256", ""))
    seen: set[tuple[str, str]] = set()
    complete: set[str] = set()
    issues: list[dict[str, str]] = []
    for row_number, row in enumerate(review_rows, 1):
        family_id = str(row.get("family_id", ""))
        reviewer = str(row.get("reviewer", "")).strip()
        prefix = f"row {row_number}"
        if family_id not in expected:
            issues.append({"row": prefix, "issue": f"unknown family: {family_id}"})
            continue
        identity = (family_id, reviewer)
        if identity in seen:
            issues.append({"row": prefix, "issue": "duplicate family/reviewer"})
            continue
        seen.add(identity)
        try:
            review = FamilyReview.from_dict(row)
        except ValueError as error:
            issues.append({"row": prefix, "issue": str(error)})
            continue
        row_issues = []
        if review.dataset_sha256 != digest:
            row_issues.append("dataset hash mismatch")
        if reviewer.startswith("TODO"):
            row_issues.append("reviewer is still a placeholder")
        if review.rationale == "TODO":
            row_issues.append("rationale is still a placeholder")
        if set(review.reconstructions) != expected[family_id]:
            row_issues.append("reconstructions do not match packet variants")
        if any(
            reconstruction.get("objective") == "TODO"
            for reconstruction in review.reconstructions.values()
        ):
            row_issues.append("a reconstruction is still a placeholder")
        if row_issues:
            issues.extend({"row": prefix, "issue": issue} for issue in row_issues)
        else:
            complete.add(family_id)
    missing = sorted(set(expected) - complete)
    return {
        "schema_version": 1,
        "dataset_sha256": digest,
        "all_complete": not issues and not missing,
        "counts": {
            "expected_families": len(expected),
            "completed_families": len(complete),
            "missing_families": len(missing),
            "issues": len(issues),
        },
        "completed_family_ids": sorted(complete),
        "missing_family_ids": missing,
        "issues": issues,
        "boundary": (
            "Completeness validates review evidence shape and provenance only. It does not compare "
            "against authored keys or approve any family."
        ),
    }


def pilot_manifest(
    selected_cases: Iterable[ComprehensionCase],
    selection: Mapping[str, object],
    *,
    source_dataset_sha256: str,
) -> dict[str, object]:
    """Describe the deterministic pilot selection and its actual balance."""
    rows = list(selected_cases)
    family_ids = {case.family_id for case in rows}
    domains = Counter((case.stimulus_profile or {}).get("domain", case.category) for case in rows)
    kinds = Counter((case.stimulus_profile or {}).get("study_kind", "") for case in rows)
    relations = Counter(case.intended_relation for case in rows)
    return {
        "schema_version": 1,
        "source_dataset_sha256": source_dataset_sha256,
        "selection": dict(selection),
        "counts": {
            "families": len(family_ids),
            "cases": len(rows),
            "domains": dict(sorted(domains.items())),
            "study_kinds": dict(sorted(kinds.items())),
            "relations": dict(sorted(relations.items())),
        },
        "purpose": (
            "Detect semantic-key, naturalness, isolation, leakage, and template-dependence problems "
            "before reviewing all 224 families. This pilot is not a model-execution manifest."
        ),
    }


@dataclass(frozen=True)
class FamilyReview:
    """One independent, key-blind review of a complete controlled family."""

    family_id: str
    reviewer: str
    dataset_sha256: str
    verdict: str
    rationale: str
    checks: dict[str, str]
    reconstructions: dict[str, dict[str, object]]
    diversity_concern: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FamilyReview:
        verdict = str(value.get("verdict", ""))
        if verdict not in {"accept", "revise", "reject"}:
            raise ValueError("review verdict must be accept, revise, or reject")
        checks = value.get("checks")
        reconstructions = value.get("reconstructions")
        check_keys = frozenset(checks) if isinstance(checks, dict) else frozenset()
        if not isinstance(checks, dict) or check_keys not in {
            frozenset(REVIEW_CHECKS),
            frozenset(LEGACY_REVIEW_CHECKS),
        }:
            raise ValueError("review checks do not match the required semantic checks")
        if check_keys == frozenset(LEGACY_REVIEW_CHECKS):
            checks = dict(checks)
            legacy_value = checks.pop("requests_are_self_contained")
            checks["semantic_completeness"] = legacy_value
            checks["execution_sufficiency"] = legacy_value
        normalized_checks = {
            str(key): ("pass" if item is True else "fail" if item is False else str(item))
            for key, item in checks.items()
        }
        if any(item not in CHECK_RESULTS for item in normalized_checks.values()):
            raise ValueError("every semantic check must use a supported check result")
        if not isinstance(reconstructions, dict):
            raise ValueError("review needs per-case reconstructions")
        normalized: dict[str, dict[str, object]] = {}
        for case_id, reconstruction in reconstructions.items():
            if not isinstance(reconstruction, dict) or set(reconstruction) != set(COMPREHENSION_FIELDS):
                raise ValueError(f"reconstruction has wrong fields: {case_id}")
            normalized[str(case_id)] = dict(reconstruction)
        reviewer = str(value.get("reviewer", "")).strip()
        rationale = str(value.get("rationale", "")).strip()
        dataset_sha256 = str(value.get("dataset_sha256", "")).strip()
        family_id = str(value.get("family_id", "")).strip()
        if not all((reviewer, rationale, dataset_sha256, family_id)):
            raise ValueError("review needs family, reviewer, hash, and rationale")
        return cls(
            family_id,
            reviewer,
            dataset_sha256,
            verdict,
            rationale,
            normalized_checks,
            normalized,
            str(value.get("diversity_concern", "")).strip(),
        )


def build_dossier(
    cases: Iterable[ComprehensionCase],
    reviews: Iterable[FamilyReview],
    *,
    dataset_sha256: str,
    min_reviews: int = 1,
) -> dict[str, object]:
    """Reveal keys only in an adjudication dossier; never auto-approve families."""
    if min_reviews < 1:
        raise ValueError("min_reviews must be positive")
    families: dict[str, list[ComprehensionCase]] = defaultdict(list)
    for case in cases:
        families[case.family_id].append(case)
    indexed: dict[str, list[FamilyReview]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for review in reviews:
        if review.family_id not in families:
            raise ValueError(f"review references unknown family: {review.family_id}")
        if review.dataset_sha256 != dataset_sha256:
            raise ValueError("review dataset hash does not match current cases")
        identity = (review.family_id, review.reviewer)
        if identity in seen:
            raise ValueError(f"duplicate family/reviewer row: {review.family_id}:{review.reviewer}")
        seen.add(identity)
        expected_ids = {case.id for case in families[review.family_id]}
        if set(review.reconstructions) != expected_ids:
            raise ValueError(f"review does not reconstruct every family variant: {review.family_id}")
        indexed[review.family_id].append(review)

    rows = []
    ready = []
    for family_id in sorted(families):
        family_reviews = indexed.get(family_id, [])
        is_ready = (
            len({review.reviewer for review in family_reviews}) >= min_reviews
            and all(review.verdict == "accept" for review in family_reviews)
            and all(
                all(_check_passes(review.checks[check]) for check in GATING_REVIEW_CHECKS)
                for review in family_reviews
            )
        )
        if is_ready:
            ready.append(family_id)
        rows.append(
            {
                "family_id": family_id,
                "ready_for_adjudication": is_ready,
                "authored_keys": {
                    case.id: case.expected for case in sorted(families[family_id], key=lambda item: item.id)
                },
                "blind_reviews": [asdict(review) for review in family_reviews],
            }
        )
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "min_reviews": min_reviews,
        "approval_boundary": (
            "ready_for_adjudication is not approval. A separate adjudication row must explicitly "
            "compare the authored keys and blind reconstructions."
        ),
        "ready_for_adjudication": ready,
        "families": rows,
    }


def apply_adjudications(
    cases: Iterable[ComprehensionCase],
    dossier: Mapping[str, object],
    adjudications: Iterable[Mapping[str, object]],
    *,
    dataset_sha256: str,
) -> list[dict[str, object]]:
    """Approve or reject whole families only after explicit independent adjudication."""
    if dossier.get("dataset_sha256") != dataset_sha256:
        raise ValueError("dossier dataset hash does not match current cases")
    ready = {str(item) for item in dossier.get("ready_for_adjudication", [])}
    decisions: dict[str, tuple[str, str]] = {}
    for row in adjudications:
        family_id = str(row.get("family_id", ""))
        decision = str(row.get("decision", ""))
        adjudicator = str(row.get("adjudicator", "")).strip()
        rationale = str(row.get("rationale", "")).strip()
        if row.get("dataset_sha256") != dataset_sha256:
            raise ValueError("adjudication dataset hash does not match current cases")
        if decision not in {"approve", "reject"} or not adjudicator or not rationale:
            raise ValueError("adjudication needs decision, adjudicator, and rationale")
        if family_id in decisions:
            raise ValueError(f"duplicate adjudication: {family_id}")
        if decision == "approve" and family_id not in ready:
            raise ValueError(f"family is not ready for adjudication: {family_id}")
        decisions[family_id] = (decision, adjudicator)

    output = []
    for case in cases:
        value = asdict(case)
        decision = decisions.get(case.family_id)
        if decision:
            value["review_status"] = "approved" if decision[0] == "approve" else "rejected"
            value["reviewer"] = decision[1]
        output.append(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("cases", type=Path)
    export.add_argument("output", type=Path)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("cases", type=Path)
    pilot.add_argument("packet", type=Path)
    pilot.add_argument("reviews_template", type=Path)
    pilot.add_argument("manifest", type=Path)
    pilot.add_argument("--seed", default="semantic-pilot-v1")
    shard = commands.add_parser("shard")
    shard.add_argument("packet", type=Path)
    shard.add_argument("output_dir", type=Path)
    shard.add_argument("--shards", type=int, default=4)
    check = commands.add_parser("check")
    check.add_argument("packet", type=Path)
    check.add_argument("reviews", type=Path)
    check.add_argument("output", type=Path)
    dossier = commands.add_parser("dossier")
    dossier.add_argument("cases", type=Path)
    dossier.add_argument("reviews", type=Path)
    dossier.add_argument("output", type=Path)
    dossier.add_argument("--min-reviews", type=int, default=1)
    apply = commands.add_parser("apply")
    apply.add_argument("cases", type=Path)
    apply.add_argument("dossier", type=Path)
    apply.add_argument("adjudications", type=Path)
    apply.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "shard":
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shards = shard_packets(packet, shard_count=args.shards)
        assignments = []
        for index, shard_packet in enumerate(shards, 1):
            packet_path = args.output_dir / f"shard-{index:02d}.review-packet.json"
            template_path = args.output_dir / f"shard-{index:02d}.reviews.template.jsonl"
            packet_path.write_text(
                json.dumps(shard_packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            template_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in review_template(shard_packet)
                ),
                encoding="utf-8",
            )
            assignments.append(
                {
                    "shard": index,
                    "packet": packet_path.name,
                    "reviews_template": template_path.name,
                    "family_ids": [row["family_id"] for row in shard_packet["families"]],
                }
            )
        (args.output_dir / "assignments.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_sha256": packet.get("dataset_sha256"),
                    "assignments": assignments,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    if args.command == "check":
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        value = review_progress(packet, _read_jsonl(args.reviews))
        args.output.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not value["all_complete"]:
            raise SystemExit(1)
        return
    cases = load_cases(args.cases)
    digest = _sha256(args.cases)
    if args.command == "export":
        value = blind_packet(cases, dataset_sha256=digest)
        args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "pilot":
        selected, selection = select_pilot_cases(cases, seed=args.seed)
        packet = blind_packet(selected, dataset_sha256=digest)
        packet["pilot_selection"] = selection
        args.packet.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.reviews_template.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in review_template(packet)
            ),
            encoding="utf-8",
        )
        manifest = pilot_manifest(
            selected, selection, source_dataset_sha256=digest
        )
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif args.command == "dossier":
        reviews = [FamilyReview.from_dict(row) for row in _read_jsonl(args.reviews)]
        value = build_dossier(cases, reviews, dataset_sha256=digest, min_reviews=args.min_reviews)
        args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        value = json.loads(args.dossier.read_text(encoding="utf-8"))
        rows = apply_adjudications(cases, value, _read_jsonl(args.adjudications), dataset_sha256=digest)
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
