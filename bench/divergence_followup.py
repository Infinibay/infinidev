#!/usr/bin/env python3
"""Build a dataset-bound counterbalanced follow-up from cross-model divergences."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from bench.model_behavior import Probe, load_probes
from bench.probe_manifest import build_explicit_manifest, file_sha256


def build_followup(
    probes: Mapping[str, Probe],
    comparison: Mapping[str, object],
    *,
    dataset_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Select every divergent question and retain concrete response evidence."""
    raw_questions = comparison.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError("comparison needs a questions array")

    divergent: list[dict[str, object]] = []
    for raw in raw_questions:
        if not isinstance(raw, dict) or raw.get("unanimous") is not False:
            continue
        probe_id = str(raw.get("probe_id", ""))
        if probe_id not in probes:
            raise ValueError(f"comparison references unknown probe {probe_id!r}")
        model_rows = raw.get("models")
        if not isinstance(model_rows, dict) or len(model_rows) < 2:
            raise ValueError(f"divergent probe {probe_id} needs at least two model rows")
        actions: dict[str, dict[str, object]] = {}
        for model, model_row in model_rows.items():
            if not isinstance(model_row, dict):
                raise ValueError(f"invalid model row for {probe_id}/{model}")
            actions[str(model)] = {
                "selected_key": model_row.get("selected_key"),
                "selected_action": model_row.get("selected_action"),
                "raw_response": model_row.get("raw_response"),
            }
        probe = probes[probe_id]
        divergent.append(
            {
                "probe_id": probe_id,
                "category": probe.category,
                "family": probe.group,
                "evaluation_mode": probe.evaluation_mode,
                "choice_count": len(probe.choices),
                "actions": actions,
            }
        )

    if not divergent:
        raise ValueError("comparison contains no divergent questions")
    probe_ids = [str(row["probe_id"]) for row in divergent]
    if len(probe_ids) != len(set(probe_ids)):
        raise ValueError("comparison contains duplicate divergent probe ids")

    manifest = build_explicit_manifest(
        probes,
        dataset_sha256=dataset_sha256,
        probe_ids=probe_ids,
        purpose=(
            "Complete option-position rotations for every cross-model divergence in the "
            "single-presentation exhaustive raw baseline"
        ),
    )
    repetitions = math.lcm(*(int(row["choice_count"]) for row in divergent))
    family_members: dict[str, list[str]] = defaultdict(list)
    for probe in probes.values():
        if probe.group:
            family_members[probe.group].append(probe.id)
    divergent_families = Counter(str(row["family"] or row["probe_id"]) for row in divergent)
    report = {
        "interpretation_boundary": (
            "Selection is data-dependent and intended only to resolve position sensitivity and "
            "replication. It is not an unbiased estimate of population behavior and does not "
            "authorize prompt changes by itself."
        ),
        "dataset_sha256": dataset_sha256,
        "probe_count": len(divergent),
        "normative_count": sum(
            row["evaluation_mode"] == "normative" for row in divergent
        ),
        "preference_count": sum(
            row["evaluation_mode"] == "preference" for row in divergent
        ),
        "category_counts": dict(Counter(str(row["category"]) for row in divergent)),
        "family_count": len(divergent_families),
        "complete_rotation_repetitions": repetitions,
        "calls_per_model": len(divergent) * repetitions,
        "divergences": [
            {
                **row,
                "family_members": sorted(
                    family_members.get(str(row["family"]), [str(row["probe_id"])])
                ),
                "divergent_siblings_in_baseline": divergent_families[
                    str(row["family"] or row["probe_id"])
                ],
            }
            for row in divergent
        ],
    }
    return manifest, report


def render_markdown(report: Mapping[str, object], *, model_count: int) -> str:
    """Render actions first, with the experiment budget and limits visible."""
    calls_per_model = int(report["calls_per_model"])
    lines = [
        "# Counterbalanced divergence follow-up",
        "",
        str(report["interpretation_boundary"]),
        "",
        "## Frozen scope",
        "",
        f"- Dataset SHA-256: `{report['dataset_sha256']}`",
        f"- Divergent probes: {report['probe_count']}",
        f"- Normative: {report['normative_count']}; preference: {report['preference_count']}",
        f"- Distinct families: {report['family_count']}",
        f"- Complete rotations per probe: {report['complete_rotation_repetitions']}",
        f"- Calls: {calls_per_model} per model; {calls_per_model * model_count} total for "
        f"{model_count} models",
        "",
        "The normative probes require human preflight before provider calls. A draft key miss can "
        "reflect an ambiguous option rather than a model defect.",
    ]
    raw_rows = report.get("divergences", [])
    if not isinstance(raw_rows, list):
        return "\n".join(lines) + "\n"
    by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in raw_rows:
        if isinstance(row, dict):
            by_category[str(row["category"])].append(row)
    for category, rows in sorted(by_category.items()):
        lines.extend(["", f"## {category}"])
        for row in rows:
            lines.extend(
                [
                    "",
                    f"### `{row['probe_id']}`",
                    f"Mode: `{row['evaluation_mode']}`; family: `{row['family']}`; "
                    f"family members: `{', '.join(row['family_members'])}`.",
                    "",
                    "Observed single-presentation actions:",
                ]
            )
            actions = row.get("actions", {})
            if isinstance(actions, dict):
                for model, action in actions.items():
                    if isinstance(action, dict):
                        lines.append(
                            f"- **{model}**: {action.get('selected_action')} "
                            f"(key `{action.get('selected_key')}`, raw "
                            f"`{action.get('raw_response')}`)"
                        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    probes = load_probes(args.probes)
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    manifest, report = build_followup(
        probes,
        comparison,
        dataset_sha256=file_sha256(args.probes),
    )
    model_count = len(comparison.get("models", {}))
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_report.write_text(
        render_markdown(report, model_count=model_count), encoding="utf-8"
    )
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
