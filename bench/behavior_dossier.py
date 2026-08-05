#!/usr/bin/env python3
"""Build a qualitative, evidence-linked dossier from behavioral probe responses."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from bench.model_behavior import Observation, Probe, load_observations, load_probes


def build_behavior_dossier(
    probes: dict[str, Probe],
    observations: Iterable[Observation],
    *,
    response_chars: int = 1200,
) -> dict[str, object]:
    """Describe observable decisions and expressed criteria without inferring hidden thought."""
    if response_chars < 0:
        raise ValueError("response_chars must be non-negative")
    rows = list(observations)
    unknown = sorted({row.probe_id for row in rows} - set(probes))
    if unknown:
        raise ValueError(f"unknown probe ids in observations: {', '.join(unknown)}")

    by_condition: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        by_condition[row.condition].append(row)

    conditions: dict[str, object] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        categories: dict[str, object] = {}
        category_rows: dict[str, list[Observation]] = defaultdict(list)
        for row in condition_rows:
            category_rows[probes[row.probe_id].category].append(row)
        for category, selected_rows in sorted(category_rows.items()):
            categories[category] = _category_dossier(
                probes, selected_rows, response_chars=response_chars
            )
        conditions[condition] = {
            "attempted": len(condition_rows),
            "errors": sum(row.error is not None for row in condition_rows),
            "categories": categories,
        }
    return {
        "interpretation_boundary": (
            "This dossier reports selected actions, expressed decision criteria, stated missing "
            "context, and perturbation behavior. It does not claim access to private reasoning or "
            "treat model explanations as faithful chain-of-thought."
        ),
        "prompt_authoring_rule": (
            "Write candidate guidance from repeated evidence-linked patterns, preserve observed "
            "strengths, and validate every candidate on held-out probes. Numeric metrics are gates "
            "and comparison aids, not substitutes for the response evidence below."
        ),
        "conditions": conditions,
    }


def _category_dossier(
    probes: dict[str, Probe],
    rows: list[Observation],
    *,
    response_chars: int,
) -> dict[str, object]:
    decisions = [_decision_record(probes[row.probe_id], row, response_chars) for row in rows]
    successful = [record for record in decisions if record["error"] is None]
    normative = [record for record in successful if record["evaluation_mode"] == "normative"]
    preferences = [
        record for record in successful if record["evaluation_mode"] == "preference"
    ]
    failures = [record for record in normative if record["correct"] is False]
    strengths = [record for record in normative if record["correct"] is True]
    criteria = _statements(rows, "decision_criterion")
    missing_context = _statements(rows, "missing_context")
    return {
        "attempted": len(rows),
        "errors": sum(row.error is not None for row in rows),
        "normative_decisions": len(normative),
        "preference_decisions": len(preferences),
        "expressed_decision_criteria": criteria,
        "stated_missing_context": missing_context,
        "normative_failures": failures,
        "normative_strength_examples": strengths,
        "preference_choice_examples": preferences,
        "perturbation_families": _family_records(probes, rows),
        "prompt_authoring_evidence": _prompt_implications(probes, rows),
    }


def _decision_record(
    probe: Probe, row: Observation, response_chars: int
) -> dict[str, object]:
    selected_action = probe.choices.get(row.answer)
    expected_action = probe.choices.get(probe.answer) if probe.answer else None
    response = row.response_text
    truncated = response_chars > 0 and len(response) > response_chars
    if response_chars == 0:
        response_excerpt = ""
        truncated = bool(response)
    else:
        response_excerpt = response[:response_chars]
    return {
        "probe_id": probe.id,
        "family": probe.group,
        "evaluation_mode": probe.evaluation_mode,
        "elicitation_protocol": row.elicitation_protocol,
        "scenario": probe.scenario or probe.prompt,
        "user_request": probe.user_request or "",
        "selected_key": row.answer,
        "selected_action": selected_action,
        "expected_key": probe.answer,
        "expected_action": expected_action,
        "correct": row.answer == probe.answer if probe.answer else None,
        "confidence": row.confidence,
        "expressed_decision_criterion": row.decision_criterion,
        "stated_missing_context": row.missing_context,
        "response_excerpt": response_excerpt,
        "response_truncated": truncated,
        "error": row.error,
        "tags": list(probe.tags),
        "probe_hypothesis": str(probe.analysis.get("hypothesis", "")),
        "expected_failure_signal": str(probe.analysis.get("failure_signal", "")),
    }


def _statements(rows: list[Observation], field: str) -> list[dict[str, object]]:
    values = [str(getattr(row, field)).strip() for row in rows]
    counts = Counter(value for value in values if value)
    return [
        {"statement": statement, "count": count}
        for statement, count in counts.most_common()
    ]


def _family_records(
    probes: dict[str, Probe], rows: list[Observation]
) -> list[dict[str, object]]:
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        group = probes[row.probe_id].group
        if group and row.error is None:
            grouped[group].append(row)
    records: list[dict[str, object]] = []
    for group, family_rows in sorted(grouped.items()):
        if len(family_rows) < 2:
            continue
        selected = [
            {
                "probe_id": row.probe_id,
                "selected_key": row.answer,
                "selected_action": probes[row.probe_id].choices.get(row.answer),
            }
            for row in family_rows
        ]
        modes = {probes[row.probe_id].evaluation_mode for row in family_rows}
        all_normative_correct = (
            all(row.answer == probes[row.probe_id].answer for row in family_rows)
            if modes == {"normative"}
            else None
        )
        records.append(
            {
                "family": group,
                "evaluation_modes": sorted(modes),
                "all_normative_variants_correct": all_normative_correct,
                "selected_actions_by_variant": selected,
            }
        )
    return records


def _prompt_implications(
    probes: dict[str, Probe], rows: list[Observation]
) -> list[dict[str, object]]:
    evidence: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        probe = probes[row.probe_id]
        if row.error is not None or probe.evaluation_mode != "normative":
            continue
        if row.answer == probe.answer:
            continue
        implication = str(probe.analysis.get("calibration_use", "")).strip()
        failure = str(probe.analysis.get("failure_signal", "")).strip()
        if implication:
            evidence[(implication, failure)].append(probe.id)
    return [
        {
            "candidate_guidance_hypothesis": implication,
            "observed_failure_pattern": failure,
            "evidence_probe_ids": sorted(probe_ids),
            "evidence_count": len(probe_ids),
            "status": (
                "repeated_pattern_ready_for_candidate_generation"
                if len(probe_ids) >= 2
                else "single_observation_needs_replication"
            ),
        }
        for (implication, failure), probe_ids in sorted(evidence.items())
    ]


def render_markdown(dossier: dict[str, object]) -> str:
    """Render the dossier as a category-oriented report for prompt authors."""
    lines = [
        "# Model behavior dossier",
        "",
        str(dossier["interpretation_boundary"]),
        "",
        str(dossier["prompt_authoring_rule"]),
    ]
    conditions = dossier.get("conditions", {})
    if not isinstance(conditions, dict):
        return "\n".join(lines) + "\n"
    for condition, raw_condition in conditions.items():
        if not isinstance(raw_condition, dict):
            continue
        lines.extend(["", f"## Condition: {condition}"])
        categories = raw_condition.get("categories", {})
        if not isinstance(categories, dict):
            continue
        for category, raw_category in categories.items():
            if not isinstance(raw_category, dict):
                continue
            lines.extend(["", f"### {category}"])
            lines.append(
                f"Attempts: {raw_category.get('attempted', 0)}; "
                f"normative failures: {len(raw_category.get('normative_failures', []))}; "
                f"preference decisions: {raw_category.get('preference_decisions', 0)}."
            )
            _render_statements(
                lines,
                "Recurring expressed decision criteria",
                raw_category.get("expressed_decision_criteria"),
            )
            _render_statements(
                lines,
                "Recurring stated missing context",
                raw_category.get("stated_missing_context"),
            )
            _render_decisions(lines, "Observed normative failures", raw_category.get("normative_failures"))
            _render_decisions(
                lines,
                "Observed normative strengths",
                raw_category.get("normative_strength_examples"),
            )
            _render_decisions(lines, "Observed preference choices", raw_category.get("preference_choice_examples"))
            implications = raw_category.get("prompt_authoring_evidence")
            if isinstance(implications, list) and implications:
                lines.extend(["", "#### Candidate prompt hypotheses"])
                for item in implications:
                    if isinstance(item, dict):
                        lines.append(
                            f"- {item.get('candidate_guidance_hypothesis')} "
                            f"(evidence: {', '.join(item.get('evidence_probe_ids', []))})"
                        )
    return "\n".join(lines) + "\n"


def _render_decisions(lines: list[str], title: str, value: object) -> None:
    if not isinstance(value, list) or not value:
        return
    lines.extend(["", f"#### {title}"])
    for item in value:
        if not isinstance(item, dict):
            continue
        lines.extend(
            [
                "",
                f"- `{item.get('probe_id')}` selected **{item.get('selected_key')}**: "
                f"{item.get('selected_action')}",
                f"  - Expressed criterion: {item.get('expressed_decision_criterion') or '(none recorded)'}",
                f"  - Stated missing context: {item.get('stated_missing_context') or '(none recorded)'}",
            ]
        )
        if item.get("expected_action"):
            lines.append(f"  - Normative expected action: {item.get('expected_action')}")


def _render_statements(lines: list[str], title: str, value: object) -> None:
    if not isinstance(value, list) or not value:
        return
    lines.extend(["", f"#### {title}"])
    for item in value:
        if isinstance(item, dict):
            lines.append(f"- {item.get('statement')} (observed {item.get('count')} times)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--response-chars", type=int, default=1200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dossier = build_behavior_dossier(
        load_probes(args.probes),
        load_observations(args.observations),
        response_chars=args.response_chars,
    )
    rendered = (
        json.dumps(dossier, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(dossier)
    )
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
