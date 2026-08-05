#!/usr/bin/env python3
"""Audit the complete prompt-comprehension battery before any provider run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from bench.generate_prompt_comprehension_battery import (
        BUILDERS,
        DOMAINS,
        EXECUTION_DIMENSIONS,
        _question_catalog,
    )
    from bench.prompt_comprehension import COMPREHENSION_FIELDS, load_cases
else:
    from generate_prompt_comprehension_battery import (
        BUILDERS,
        DOMAINS,
        EXECUTION_DIMENSIONS,
        _question_catalog,
    )
    from prompt_comprehension import COMPREHENSION_FIELDS, load_cases


STIMULUS_FIELDS = {
    "study_kind",
    "phenomenon",
    "domain",
    "register",
    "structure",
    "modality",
    "ambiguity",
    "conflict",
    "noise",
    "example_role",
    "language",
    "instruction_position",
}


def audit_battery(cases_path: Path, registry_path: Path) -> dict[str, object]:
    cases = load_cases(cases_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_rows = registry.get("families") if isinstance(registry, dict) else None
    question_rows = registry.get("questions") if isinstance(registry, dict) else None
    if not isinstance(registry_rows, list):
        raise ValueError("family registry has no families")
    if not isinstance(question_rows, list):
        raise ValueError("family registry has no questions")
    families: dict[str, list[object]] = defaultdict(list)
    normalized_requests: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        families[case.family_id].append(case)
        normalized_requests[" ".join(case.request.lower().split())].append(case.id)
    registry_ids = [str(row.get("family_id", "")) for row in registry_rows if isinstance(row, dict)]
    expected_questions = _question_catalog()
    expected_question_ids = [row["question_id"] for row in expected_questions]
    question_ids = [
        str(row.get("question_id", "")) for row in question_rows if isinstance(row, dict)
    ]
    duplicate_question_ids = sorted(
        question_id for question_id, count in Counter(question_ids).items() if count > 1
    )
    catalog_question_ids = set(question_ids)
    referenced_question_ids: set[str] = set()
    families_without_questions: list[str] = []
    invalid_family_question_links: list[str] = []
    unknown_question_references: dict[str, list[str]] = {}
    for row in registry_rows:
        if not isinstance(row, dict):
            continue
        family_id = str(row.get("family_id", ""))
        family_question_ids = row.get("question_ids")
        if not isinstance(family_question_ids, list) or not family_question_ids:
            families_without_questions.append(family_id)
            continue
        normalized_ids = [str(question_id) for question_id in family_question_ids]
        referenced_question_ids.update(normalized_ids)
        unknown_ids = [
            question_id for question_id in normalized_ids if question_id not in catalog_question_ids
        ]
        if unknown_ids:
            unknown_question_references[family_id] = unknown_ids
        phenomenon = str(row.get("phenomenon", ""))
        domain = str(row.get("domain", ""))
        execution_dimension = str(row.get("execution_dimension", ""))
        expected_links = (
            [f"execution--{domain}--{execution_dimension}"]
            if execution_dimension
            else [f"behavior--{phenomenon}"]
        )
        if normalized_ids != expected_links:
            invalid_family_question_links.append(family_id)
    orphan_question_ids = sorted(catalog_question_ids - referenced_question_ids)
    phenomena = Counter(
        str((case.stimulus_profile or {}).get("phenomenon", "")) for case in cases
    )
    domains = Counter(case.category for case in cases)
    relations = Counter(case.intended_relation for case in cases)
    splits = Counter(case.split for case in cases)
    family_splits = {name: {case.split for case in rows} for name, rows in families.items()}
    study_kinds = Counter(
        str((case.stimulus_profile or {}).get("study_kind", "")) for case in cases
    )
    execution_dimensions = Counter(
        str((case.stimulus_profile or {}).get("execution_dimension", ""))
        for case in cases
        if (case.stimulus_profile or {}).get("study_kind") == "execution"
    )
    question_links_match_cases = all(
        {
            case.research_question_id
            for case in rows
        }
        == set(
            str(question_id)
            for question_id in next(
                row["question_ids"]
                for row in registry_rows
                if isinstance(row, dict) and row.get("family_id") == family_id
            )
        )
        for family_id, rows in families.items()
    )
    family_semantics_match_question = all(
        all(
            (
                case.research_question_id
                == f"execution--{case.category}--{(case.stimulus_profile or {}).get('execution_dimension')}"
                and (case.stimulus_profile or {}).get("execution_dimension")
                == row.get("execution_dimension")
                and row.get("phenomenon") == "execution_policy_comprehension"
            )
            if row.get("execution_dimension")
            else (
                case.research_question_id == f"behavior--{row.get('phenomenon')}"
                and (case.stimulus_profile or {}).get("phenomenon") == row.get("phenomenon")
                and (case.stimulus_profile or {}).get("study_kind") == "linguistic"
            )
            for case in families[str(row.get("family_id"))]
        )
        for row in registry_rows
        if isinstance(row, dict) and str(row.get("family_id")) in families
    )
    malformed_families: list[str] = []
    equivalent_key_mismatches: list[str] = []
    unchanged_semantic_contrasts: list[str] = []
    for family_id, rows in families.items():
        anchors = [case for case in rows if case.intended_relation == "anchor"]
        if len(rows) != 3 or len(anchors) != 1 or len({case.variant_id for case in rows}) != 3:
            malformed_families.append(family_id)
            continue
        anchor_key = anchors[0].expected
        if any(
            case.intended_relation == "equivalent" and case.expected != anchor_key
            for case in rows
        ):
            equivalent_key_mismatches.append(family_id)
        if any(
            case.intended_relation in {"contrast", "adversarial"}
            and case.expected == anchor_key
            for case in rows
        ):
            unchanged_semantic_contrasts.append(family_id)
    registry_complete = all(
        isinstance(row, dict)
        and all(
            row.get(field)
            for field in (
                "family_id",
                "problem_id",
                "research_question",
                "product_utility",
                "information_needed_about_model",
                "competing_hypotheses",
                "evidence_fields",
                "possible_interventions",
                "held_out_confirmation",
            )
        )
        for row in registry_rows
    )
    checks = {
        "exactly_672_cases": len(cases) == 672,
        "exactly_224_families": len(families) == 224,
        "three_variants_per_family_with_one_anchor": not malformed_families,
        "eighteen_linguistic_phenomena_plus_execution": (
            set(phenomena) == {*BUILDERS, "execution_policy_comprehension"}
            and all(phenomena[name] == 24 for name in BUILDERS)
            and phenomena["execution_policy_comprehension"] == 240
        ),
        "eight_domains": {domain.id for domain in DOMAINS} == set(domains)
        and all(count == 84 for count in domains.values()),
        "study_kind_counts": study_kinds == {"linguistic": 432, "execution": 240},
        "ten_execution_dimensions_materialized": (
            set(execution_dimensions) == {dimension[0] for dimension in EXECUTION_DIMENSIONS}
            and all(count == 24 for count in execution_dimensions.values())
        ),
        "balanced_splits": splits == {"calibration": 336, "validation": 336},
        "families_are_split_atomic": all(len(value) == 1 for value in family_splits.values()),
        "every_phenomenon_and_domain_has_both_splits": all(
            {case.split for case in cases if (case.stimulus_profile or {}).get("phenomenon") == phenomenon}
            == {"calibration", "validation"}
            for phenomenon in {*BUILDERS, "execution_policy_comprehension"}
        )
        and all(
            {case.split for case in cases if case.category == domain.id}
            == {"calibration", "validation"}
            for domain in DOMAINS
        ),
        "all_requests_unique": all(len(ids) == 1 for ids in normalized_requests.values()),
        "equivalent_variants_preserve_keys": not equivalent_key_mismatches,
        "semantic_contrasts_change_keys": not unchanged_semantic_contrasts,
        "all_expected_fields_present": all(
            set(COMPREHENSION_FIELDS) <= set(case.expected) for case in cases
        ),
        "all_stimulus_dimensions_recorded": all(
            STIMULUS_FIELDS <= set(case.stimulus_profile or {}) for case in cases
        ),
        "all_cases_trace_to_problem": all(case.problem_id for case in cases),
        "registry_matches_families": len(registry_ids) == len(set(registry_ids))
        and set(registry_ids) == set(families),
        "registry_has_research_chain": registry_complete,
        "question_catalog_matches_taxonomy": question_rows == expected_questions,
        "question_ids_are_unique": len(question_ids) == len(set(question_ids)),
        "every_family_has_questions": not families_without_questions,
        "all_question_references_exist": not unknown_question_references,
        "no_orphan_questions": not orphan_question_ids,
        "family_question_links_match_taxonomy": not invalid_family_question_links,
        "case_question_links_match_family": question_links_match_cases,
        "family_semantics_match_question": family_semantics_match_question,
        "all_cases_remain_drafts": all(case.review_status == "draft" for case in cases),
    }
    return {
        "schema_version": 1,
        "all_passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "cases": len(cases),
            "families": len(families),
            "phenomena": dict(sorted(phenomena.items())),
            "domains": dict(sorted(domains.items())),
            "relations": dict(sorted(relations.items())),
            "splits": dict(sorted(splits.items())),
            "study_kinds": dict(sorted(study_kinds.items())),
            "execution_dimensions": dict(sorted(execution_dimensions.items())),
        },
        "failures": {
            "malformed_families": malformed_families,
            "equivalent_key_mismatches": equivalent_key_mismatches,
            "unchanged_semantic_contrasts": unchanged_semantic_contrasts,
            "duplicate_request_groups": [ids for ids in normalized_requests.values() if len(ids) > 1],
        },
        "review_boundary": (
            "Passing proves structure, traceability, controlled-key relationships, and coverage. "
            "Cases remain drafts until human semantic review; this audit cannot establish that a "
            "reviewed interpretation key is substantively correct."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit_battery(args.cases, args.registry)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
