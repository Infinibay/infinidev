#!/usr/bin/env python3
"""Report how isolated self-report elicitation changes model choices and feedback."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping

from bench.compare_elicitation import compare_protocols
from bench.model_behavior import Probe, load_observations, load_probes


def build_elicitation_report(
    probes: Mapping[str, Probe],
    protocol_pairs: Mapping[str, tuple[Path, Path]],
) -> dict[str, object]:
    """Enrich protocol comparisons with the concrete actions behind selected keys."""
    if not protocol_pairs:
        raise ValueError("at least one model protocol pair is required")
    models: dict[str, object] = {}
    expected_probe_ids: set[str] | None = None
    for model, (choice_path, report_path) in protocol_pairs.items():
        comparison = compare_protocols(
            load_observations(choice_path), load_observations(report_path)
        )
        records: list[dict[str, object]] = []
        for raw_record in comparison["records"]:
            if not isinstance(raw_record, dict):
                continue
            probe_id = str(raw_record["probe_id"])
            if probe_id not in probes:
                raise ValueError(f"{model} contains unknown probe {probe_id}")
            probe = probes[probe_id]
            choice_key = str(raw_record["choice_only_answer"])
            report_key = str(raw_record["self_report_answer"])
            records.append(
                {
                    **raw_record,
                    "category": probe.category,
                    "scenario": probe.scenario or probe.prompt,
                    "user_request": probe.user_request or "",
                    "choice_only_action": probe.choices.get(choice_key),
                    "self_report_action": probe.choices.get(report_key),
                    "review_status": probe.review_status,
                }
            )
        current_ids = {str(row["probe_id"]) for row in records}
        if expected_probe_ids is None:
            expected_probe_ids = current_ids
        elif current_ids != expected_probe_ids:
            raise ValueError(f"{model} does not use the same probe set")
        successful = [
            row
            for row in records
            if row["choice_only_error"] is None and row["self_report_error"] is None
        ]
        confidence = [
            float(row["verbal_confidence"])
            for row in successful
            if row["verbal_confidence"] is not None
        ]
        models[model] = {
            "paired": len(records),
            "successful_pairs": len(successful),
            "unchanged_choices": sum(bool(row["answer_agrees"]) for row in successful),
            "changed_choices": sum(not bool(row["answer_agrees"]) for row in successful),
            "median_verbal_confidence": statistics.median(confidence) if confidence else None,
            "records": records,
        }
    return {
        "interpretation_boundary": (
            "Decision criteria and confidence are model self-reports, not privileged access to "
            "private reasoning. Because elicitation itself can change a choice, choice-only and "
            "self-report observations remain separate evidence conditions."
        ),
        "models": models,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    """Render criteria and concrete actions as the primary evidence."""
    lines = [
        "# Choice-only versus self-report elicitation",
        "",
        str(report["interpretation_boundary"]),
        "",
        "All calls were isolated, had no system message, and used no user preference profile.",
        "",
        "## Summary",
    ]
    models = report.get("models", {})
    if not isinstance(models, dict):
        return "\n".join(lines) + "\n"
    for model, raw_model in models.items():
        if not isinstance(raw_model, dict):
            continue
        confidence = raw_model.get("median_verbal_confidence")
        confidence_text = (
            f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
        )
        lines.append(
            f"- **{model}**: {raw_model.get('changed_choices')}/"
            f"{raw_model.get('successful_pairs')} choices changed; median self-reported "
            f"confidence {confidence_text}."
        )

    categories: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for model, raw_model in models.items():
        if not isinstance(raw_model, dict):
            continue
        records = raw_model.get("records", [])
        if not isinstance(records, list):
            continue
        for row in records:
            if isinstance(row, dict):
                categories.setdefault(str(row.get("category", "unknown")), []).append(
                    (str(model), row)
                )
    for category, entries in categories.items():
        lines.extend(["", f"## Category: {category}"])
        by_probe: dict[str, list[tuple[str, dict[str, object]]]] = {}
        for model, row in entries:
            by_probe.setdefault(str(row["probe_id"]), []).append((model, row))
        for probe_id, rows in by_probe.items():
            exemplar = rows[0][1]
            lines.extend(["", f"### `{probe_id}`", "", f"Scenario: {exemplar['scenario']}"])
            if exemplar.get("user_request"):
                lines.append(f"User request: {exemplar['user_request']}")
            for model, row in rows:
                status = "changed" if not row.get("answer_agrees") else "unchanged"
                lines.extend(
                    [
                        "",
                        f"- **{model}** ({status})",
                        f"  - Choice-only: **{row.get('choice_only_answer')}** — "
                        f"{row.get('choice_only_action')}",
                        f"  - Self-report: **{row.get('self_report_answer')}** — "
                        f"{row.get('self_report_action')}",
                        f"  - Expressed criterion: "
                        f"{row.get('expressed_decision_criterion') or '(none)' }",
                        f"  - Stated missing context: "
                        f"{row.get('stated_missing_context') or '(none)' }",
                        f"  - Verbal confidence: {row.get('verbal_confidence')}",
                        f"  - Raw self-report: `{row.get('self_report_response')}`",
                    ]
                )
    return "\n".join(lines) + "\n"


def _parse_pair(value: str) -> tuple[str, tuple[Path, Path]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model pair must be LABEL=CHOICE_PATH,REPORT_PATH")
    label, raw_paths = value.split("=", 1)
    paths = raw_paths.split(",", 1)
    if not label.strip() or len(paths) != 2 or not all(path.strip() for path in paths):
        raise argparse.ArgumentTypeError("model pair must be LABEL=CHOICE_PATH,REPORT_PATH")
    return label.strip(), (Path(paths[0]), Path(paths[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--model", action="append", type=_parse_pair, required=True)
    args = parser.parse_args()
    pairs = dict(args.model)
    if len(pairs) != len(args.model):
        parser.error("model labels must be unique")
    report = build_elicitation_report(load_probes(args.probes), pairs)
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
