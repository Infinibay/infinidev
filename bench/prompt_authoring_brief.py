#!/usr/bin/env python3
"""Turn a behavior dossier into an evidence-first brief for prompt candidate authors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_authoring_brief(
    dossier: dict[str, object], *, condition: str
) -> dict[str, object]:
    """Collect raw behavioral evidence that candidate guidance must explain."""
    conditions = dossier.get("conditions")
    if not isinstance(conditions, dict) or condition not in conditions:
        raise ValueError(f"condition is missing from behavior dossier: {condition}")
    raw_condition = conditions[condition]
    if not isinstance(raw_condition, dict):
        raise ValueError("condition dossier must be an object")
    categories = raw_condition.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("condition dossier needs categories")

    category_briefs: dict[str, object] = {}
    for category, raw_category in sorted(categories.items()):
        if not isinstance(raw_category, dict):
            continue
        failures = _records(raw_category.get("normative_failures"))
        strengths = _records(raw_category.get("normative_strength_examples"))
        preferences = _records(raw_category.get("preference_choice_examples"))
        unstable = [
            family
            for family in _records(raw_category.get("perturbation_families"))
            if family.get("all_normative_variants_correct") is False
        ]
        if not any((failures, strengths, preferences, unstable)):
            continue
        category_briefs[str(category)] = {
            "failures_to_address": failures,
            "strengths_to_preserve": strengths,
            "preference_behavior_to_condition_not_universalize": preferences,
            "unstable_perturbation_families": unstable,
            "candidate_hypotheses": _records(
                raw_category.get("prompt_authoring_evidence")
            ),
            "expressed_decision_criteria": _records(
                raw_category.get("expressed_decision_criteria")
            ),
            "stated_missing_context": _records(
                raw_category.get("stated_missing_context")
            ),
        }
    return {
        "source_condition": condition,
        "authoring_contract": [
            "Base each candidate fragment on cited response records and probe IDs, not aggregate "
            "scores alone.",
            "Describe externally observable decision policy; do not claim access to private "
            "chain-of-thought.",
            "Preserve strengths already demonstrated by the model and avoid restating them unless "
            "a candidate needs them to prevent regression.",
            "Treat preference choices as profile-conditioned behavior, never as universal defects.",
            "Prefer the smallest guidance that addresses a repeated failure pattern; a single "
            "failure remains a hypothesis requiring more evidence.",
            "Every proposed fragment remains a candidate until held-out validation beats the "
            "unchanged baseline without normative regressions.",
        ],
        "categories": category_briefs,
    }


def _records(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def render_markdown(brief: dict[str, object]) -> str:
    """Render the evidence brief without replacing records with a numeric summary."""
    lines = [
        "# Evidence-first prompt authoring brief",
        "",
        f"Source condition: `{brief.get('source_condition', '')}`",
        "",
        "## Authoring contract",
    ]
    contract = brief.get("authoring_contract", [])
    if isinstance(contract, list):
        lines.extend(f"- {item}" for item in contract)
    categories = brief.get("categories", {})
    if not isinstance(categories, dict):
        return "\n".join(lines) + "\n"
    for category, raw_category in categories.items():
        if not isinstance(raw_category, dict):
            continue
        lines.extend(["", f"## {category}"])
        _render_records(lines, "Failures to address", raw_category.get("failures_to_address"))
        _render_records(lines, "Strengths to preserve", raw_category.get("strengths_to_preserve"))
        _render_records(
            lines,
            "Preference behavior to condition, not universalize",
            raw_category.get("preference_behavior_to_condition_not_universalize"),
        )
        _render_records(
            lines,
            "Unstable perturbation families",
            raw_category.get("unstable_perturbation_families"),
        )
        _render_records(lines, "Candidate hypotheses", raw_category.get("candidate_hypotheses"))
    return "\n".join(lines) + "\n"


def _render_records(lines: list[str], title: str, value: object) -> None:
    records = _records(value)
    if not records:
        return
    lines.extend(["", f"### {title}"])
    for record in records:
        probe_id = record.get("probe_id")
        if probe_id:
            lines.append(
                f"- `{probe_id}` selected {record.get('selected_key')}: "
                f"{record.get('selected_action')}"
            )
            criterion = record.get("expressed_decision_criterion")
            if criterion:
                lines.append(f"  - Expressed criterion: {criterion}")
            expected = record.get("expected_action")
            if expected:
                lines.append(f"  - Expected action: {expected}")
            continue
        lines.append(f"- `{json.dumps(record, ensure_ascii=False, sort_keys=True)}`")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dossier", type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = json.loads(args.dossier.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        parser.error("dossier must contain a JSON object")
    brief = build_authoring_brief(value, condition=args.condition)
    rendered = (
        json.dumps(brief, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(brief)
    )
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
