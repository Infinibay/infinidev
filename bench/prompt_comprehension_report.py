#!/usr/bin/env python3
"""Render raw, category-separated prompt-comprehension evidence without fake optima."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Mapping

from bench.prompt_comprehension import ComprehensionObservation, load_cases


PARSE_ERROR_MESSAGES = (
    "response did not contain a JSON object",
    "response fields do not match comprehension contract",
    "understanding reconstruction is empty",
    "objective is empty",
    "comprehension field must be a list of strings:",
    "priority_resolution must be text",
    "confidence must be between 0 and 1",
)
INSUFFICIENT_EVIDENCE = "evidencia insuficiente"


def load_observations(path: Path) -> list[ComprehensionObservation]:
    """Load legacy observation rows or the durable runner ledger format."""
    observations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("comprehension observation row must be an object")
        value = row.get("observation", row)
        if not isinstance(value, dict):
            raise ValueError("comprehension ledger row has no observation object")
        observations.append(ComprehensionObservation(**value))
    return observations


def load_registry(path: Path | None) -> dict[str, dict[str, object]]:
    """Load family research metadata, keyed by immutable family ID."""
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("families") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError("comprehension registry has no families")
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("family_id", "")).strip():
            raise ValueError("comprehension registry contains an invalid family")
        family_id = str(row["family_id"])
        if family_id in indexed:
            raise ValueError(f"duplicate comprehension registry family: {family_id}")
        indexed[family_id] = dict(row)
    return indexed


def _error_type(error: str) -> str | None:
    """Classify only errors emitted by the known response parser as parse failures."""
    if not error:
        return None
    message = error.partition(": ")[2] or error
    if any(message == known or message.startswith(known) for known in PARSE_ERROR_MESSAGES):
        return "parse_error"
    return "provider_error"


def _record(case: object, observation: ComprehensionObservation) -> dict[str, object]:
    error_type = _error_type(observation.error)
    return {
        "case": asdict(case),
        "observation": asdict(observation),
        "outcome": {
            "status": "success" if error_type is None else "failure",
            "error_type": error_type,
        },
    }


def _record_key(record: Mapping[str, object]) -> tuple[str, str, str]:
    case = record.get("case")
    observation = record.get("observation")
    if not isinstance(case, dict) or not isinstance(observation, dict):
        return ("", "", "")
    return (
        str(case.get("variant_id", "")),
        str(case.get("id", "")),
        str(observation.get("condition", "")),
    )


def _registry_metadata(records: list[dict[str, object]]) -> dict[str, list[str]]:
    """Preserve reviewed case-registry coordinates without deriving a score."""
    cases = [record.get("case") for record in records]
    case_rows = [case for case in cases if isinstance(case, dict)]

    def values(field: str) -> list[str]:
        return sorted({str(case[field]) for case in case_rows if case.get(field)})

    return {
        "case_ids": values("id"),
        "categories": values("category"),
        "problem_ids": values("problem_id"),
        "family_ids": values("family_id"),
        "variant_ids": values("variant_id"),
        "intended_relations": values("intended_relation"),
    }


def _analysis_group(records: list[dict[str, object]], purpose: str) -> dict[str, object]:
    ordered = sorted(records, key=_record_key)
    has_failure = any(
        isinstance(record.get("outcome"), dict)
        and record["outcome"].get("status") == "failure"
        for record in ordered
    )
    return {
        "purpose": purpose,
        "evidence_status": INSUFFICIENT_EVIDENCE if has_failure else "observed",
        "registry": _registry_metadata(ordered),
        "records": ordered,
    }


def build_report(
    cases_path: Path,
    observations: list[ComprehensionObservation],
    *,
    registry_path: Path | None = None,
) -> dict[str, object]:
    """Build a lossless report; numeric summaries describe collection, not quality."""
    cases = {case.id: case for case in load_cases(cases_path)}
    registry = load_registry(registry_path)
    if not observations:
        raise ValueError("comprehension report needs observations")
    ordered_observations = sorted(
        observations,
        key=lambda row: (row.case_id, row.condition, row.condition_sha256),
    )
    dataset_sha = ordered_observations[0].dataset_sha256
    identity = ordered_observations[0].model_identity
    if any(row.dataset_sha256 != dataset_sha for row in ordered_observations):
        raise ValueError("comprehension observations mix dataset versions")
    if any(row.model_identity != identity for row in ordered_observations):
        raise ValueError("comprehension report requires one immutable model route")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    families: dict[str, list[dict[str, object]]] = defaultdict(list)
    scenarios: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in ordered_observations:
        case = cases.get(row.case_id)
        if case is None:
            raise ValueError(f"unknown comprehension case: {row.case_id}")
        key = (row.case_id, row.condition, row.condition_sha256)
        if key in seen:
            raise ValueError(f"duplicate comprehension observation: {key}")
        seen.add(key)
        record = _record(case, row)
        grouped[case.category].append(record)
        families[case.family_id].append(record)
        scenarios[case.problem_id or case.id].append(record)
    by_condition: dict[str, list[ComprehensionObservation]] = defaultdict(list)
    for row in ordered_observations:
        by_condition[row.condition].append(row)
    summaries = {}
    for condition, rows in sorted(by_condition.items()):
        error_types = [_error_type(row.error) for row in rows]
        summaries[condition] = {
            "calls": len(rows),
            "successes": sum(error_type is None for error_type in error_types),
            "provider_errors": error_types.count("provider_error"),
            "parse_errors": error_types.count("parse_error"),
            "evidence_status": (
                INSUFFICIENT_EVIDENCE if any(error_types) else "observed"
            ),
            "mean_latency_seconds": mean(row.latency_seconds for row in rows),
            "input_tokens": sum(row.input_tokens or 0 for row in rows),
            "output_tokens": sum(row.output_tokens or 0 for row in rows),
        }
    family_purpose = (
        "Compare equivalent variants for semantic stability and contrast or adversarial variants "
        "for sensitivity to the changed meaning. Inspect concrete reconstructions; this is not a "
        "claim about hidden reasoning."
    )
    scenario_purpose = (
        "Inspect all variants belonging to the same reviewed scenario without combining them into "
        "a global score."
    )
    missing_registry = set(families) - set(registry) if registry_path is not None else set()
    if missing_registry:
        raise ValueError(
            "registry lacks observed families: "
            f"missing={sorted(missing_registry)[:3]}"
        )
    return {
        "schema_version": 2,
        "interpretation_boundary": (
            "There is no globally optimal comprehension style. Numeric summaries describe call "
            "health and cost only. Interpret the model's concrete reconstruction and structured "
            "fields against each user's desired outcome; do not replace answers with axis scores."
        ),
        "dataset_sha256": dataset_sha,
        "model_identity": identity,
        "condition_summaries": summaries,
        "family_analysis": {
            family_id: {
                **_analysis_group(records, family_purpose),
                "research": registry.get(family_id),
                "variants": sorted(records, key=_record_key),
            }
            for family_id, records in sorted(families.items())
        },
        "scenario_analysis": {
            scenario_id: _analysis_group(records, scenario_purpose)
            for scenario_id, records in sorted(scenarios.items())
        },
        "categories": {
            category: sorted(records, key=_record_key)
            for category, records in sorted(grouped.items())
        },
    }


def canonical_json_bytes(report: Mapping[str, object]) -> bytes:
    """Serialize a derived report canonically for byte-identical reproduction."""
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Prompt comprehension report",
        "",
        str(report.get("interpretation_boundary")),
        "",
        f"Model identity: `{report.get('model_identity')}`.",
        "",
        "## Collection summary",
        "",
        f"`{json.dumps(report.get('condition_summaries'), sort_keys=True)}`",
        "",
        "## Controlled families",
        "",
        "Equivalent variants should preserve meaning; contrast and adversarial variants should "
        "change only the intended semantic consequence. Compare the actual answers, not merely a score.",
    ]
    families = report.get("family_analysis")
    if isinstance(families, dict):
        for family_id in sorted(families):
            family = families[family_id]
            if not isinstance(family, dict):
                continue
            lines.extend(
                [
                    "",
                    f"### `{family_id}`",
                    "",
                    str(family.get("purpose", "")),
                    "",
                    f"Evidence status: **{family.get('evidence_status', '')}**.",
                ]
            )
            research = family.get("research")
            if isinstance(research, dict):
                lines.extend(
                    [
                        "",
                        f"Research question: {research.get('research_question', '')}",
                        "",
                        f"Product utility: {research.get('product_utility', '')}",
                        "",
                        "Information sought: "
                        f"{research.get('information_needed_about_model', '')}",
                        "",
                        f"Possible interventions: {research.get('possible_interventions', [])}",
                    ]
                )
            variants = family.get("variants")
            if not isinstance(variants, list):
                continue
            for record in variants:
                if not isinstance(record, dict):
                    continue
                case = record.get("case")
                observation = record.get("observation")
                outcome = record.get("outcome")
                if not isinstance(case, dict) or not isinstance(observation, dict):
                    continue
                parsed = observation.get("parsed")
                understanding = parsed.get("understanding", "") if isinstance(parsed, dict) else ""
                error_type = outcome.get("error_type") if isinstance(outcome, dict) else None
                detail = understanding or str(observation.get("error", ""))
                lines.extend(
                    [
                        "",
                        f"- `{case.get('variant_id')}` ({case.get('intended_relation')}, "
                        f"{error_type or 'success'}): {detail}",
                    ]
                )
    scenarios = report.get("scenario_analysis")
    if isinstance(scenarios, dict):
        lines.extend(["", "## Reviewed scenarios"])
        for scenario_id in sorted(scenarios):
            scenario = scenarios[scenario_id]
            if not isinstance(scenario, dict):
                continue
            lines.extend(
                [
                    "",
                    f"### `{scenario_id}`",
                    "",
                    str(scenario.get("purpose", "")),
                    "",
                    f"Evidence status: **{scenario.get('evidence_status', '')}**.",
                ]
            )
    categories = report.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("comprehension report has no categories")
    for category in sorted(categories):
        records = categories[category]
        lines.extend(["", f"## {category}"])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            case = record.get("case")
            observation = record.get("observation")
            if not isinstance(case, dict) or not isinstance(observation, dict):
                continue
            parsed = observation.get("parsed")
            lines.extend(
                [
                    "",
                    f"### `{case.get('id')}` — {observation.get('condition')}",
                    "",
                    f"Request under test: {case.get('request')}",
                    "",
                    f"Collection error: `{observation.get('error')}`",
                    "",
                    "#### Free reconstruction",
                    "",
                    str(parsed.get("understanding", "")) if isinstance(parsed, dict) else "",
                    "",
                    "#### Structured understanding",
                    "",
                    "```json",
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "#### Reviewed interpretation key",
                    "",
                    "```json",
                    json.dumps(case.get("expected"), ensure_ascii=False, indent=2),
                    "```",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    report = build_report(
        args.cases,
        load_observations(args.observations),
        registry_path=args.registry,
    )
    args.output_json.write_bytes(canonical_json_bytes(report))
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
