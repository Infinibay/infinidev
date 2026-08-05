#!/usr/bin/env python3
"""Report repeated isolated choices without collapsing unstable behavior into one score."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from bench.model_behavior import Observation, Probe, load_observations, load_probes


def build_stability_report(
    probes: Mapping[str, Probe],
    observations_by_model: Mapping[str, Iterable[Observation]],
) -> dict[str, object]:
    """Retain all repetitions and identify exact within-model choice stability."""
    if len(observations_by_model) < 2:
        raise ValueError("stability comparison needs at least two models")
    model_records: dict[str, object] = {}
    expected_probe_ids: set[str] | None = None
    for model, raw_rows in observations_by_model.items():
        rows = list(raw_rows)
        grouped: dict[str, list[Observation]] = defaultdict(list)
        for row in rows:
            if row.probe_id not in probes:
                raise ValueError(f"{model} contains unknown probe {row.probe_id}")
            grouped[row.probe_id].append(row)
        probe_ids = set(grouped)
        if expected_probe_ids is None:
            expected_probe_ids = probe_ids
        elif probe_ids != expected_probe_ids:
            raise ValueError(f"{model} does not use the same probe set")
        records: list[dict[str, object]] = []
        for probe_id, probe_rows in grouped.items():
            probe = probes[probe_id]
            ordered = sorted(probe_rows, key=lambda row: row.repetition)
            repetitions = [row.repetition for row in ordered]
            if len(repetitions) != len(set(repetitions)):
                raise ValueError(f"{model}/{probe_id} has duplicate repetitions")
            successful = [row for row in ordered if not row.error]
            counts = Counter(row.answer for row in successful)
            modal_count = max(counts.values()) if counts else 0
            modal_keys = sorted(key for key, count in counts.items() if count == modal_count)
            records.append(
                {
                    "probe_id": probe_id,
                    "category": probe.category,
                    "scenario": probe.scenario or probe.prompt,
                    "user_request": probe.user_request or "",
                    "attempts": len(ordered),
                    "errors": len(ordered) - len(successful),
                    "stable": len(counts) == 1 and len(successful) == len(ordered),
                    "answer_counts": dict(sorted(counts.items())),
                    "modal_keys": modal_keys,
                    "modal_actions": [probe.choices.get(key) for key in modal_keys],
                    "modal_share": modal_count / len(successful) if successful else None,
                    "repetitions": [
                        {
                            "repetition": row.repetition,
                            "selected_key": row.answer,
                            "selected_action": probe.choices.get(row.answer),
                            "provider_answer": row.provider_answer or row.answer,
                            "choice_mapping": row.choice_mapping,
                            "option_order_protocol": row.option_order_protocol,
                            "raw_response": row.response_text,
                            "error": row.error,
                            "latency_seconds": row.latency_seconds,
                            "output_tokens": row.output_tokens,
                        }
                        for row in ordered
                    ],
                }
            )
        stable = sum(bool(record["stable"]) for record in records)
        model_records[model] = {
            "probe_count": len(records),
            "stable_probes": stable,
            "unstable_probes": len(records) - stable,
            "records": records,
        }

    probe_comparisons: list[dict[str, object]] = []
    for probe_id in sorted(expected_probe_ids or set()):
        modes: dict[str, list[str]] = {}
        stable: dict[str, bool] = {}
        for model, raw_model in model_records.items():
            assert isinstance(raw_model, dict)
            record = next(
                row for row in raw_model["records"] if row["probe_id"] == probe_id
            )
            modes[model] = list(record["modal_keys"])
            stable[model] = bool(record["stable"])
        unique_single_modes = {
            values[0] for values in modes.values() if len(values) == 1
        }
        comparable = all(len(values) == 1 for values in modes.values())
        probe_comparisons.append(
            {
                "probe_id": probe_id,
                "modal_keys_by_model": modes,
                "stable_by_model": stable,
                "modal_agreement": comparable and len(unique_single_modes) == 1,
            }
        )
    return {
        "interpretation_boundary": (
            "Repeated isolated choices can reveal obvious instability but cannot estimate a "
            "population-level preference reliably without a justified sampling model. Every "
            "selected action remains primary evidence; modal choices and shares are compact "
            "summaries only."
        ),
        "models": model_records,
        "probe_comparisons": probe_comparisons,
        "cross_model_modal_agreements": sum(
            bool(row["modal_agreement"]) for row in probe_comparisons
        ),
    }


def render_markdown(report: Mapping[str, object]) -> str:
    """Render stable and unstable repeated actions model by model."""
    lines = [
        "# Repeated choice stability",
        "",
        str(report["interpretation_boundary"]),
        "",
        "Each repetition used a fresh conversation, no system message, no preference profile, "
        "and choice-only elicitation.",
        "",
        "## Summary",
    ]
    models = report.get("models", {})
    if not isinstance(models, dict):
        return "\n".join(lines) + "\n"
    for model, raw_model in models.items():
        if isinstance(raw_model, dict):
            lines.append(
                f"- **{model}**: {raw_model.get('stable_probes')}/"
                f"{raw_model.get('probe_count')} probes exactly stable across repetitions."
            )
    comparisons = report.get("probe_comparisons", [])
    lines.append(
        f"- Cross-model modal agreement: {report.get('cross_model_modal_agreements', 0)}/"
        f"{len(comparisons) if isinstance(comparisons, list) else 0} probes."
    )
    for model, raw_model in models.items():
        if not isinstance(raw_model, dict):
            continue
        lines.extend(["", f"## Model: {model}"])
        records = raw_model.get("records", [])
        if not isinstance(records, list):
            continue
        for row in records:
            if not isinstance(row, dict):
                continue
            state = "stable" if row.get("stable") else "unstable"
            lines.extend(
                [
                    "",
                    f"### `{row.get('probe_id')}` — {state}",
                    "",
                    f"Scenario: {row.get('scenario')}",
                    f"Observed counts: `{json.dumps(row.get('answer_counts'), sort_keys=True)}`.",
                ]
            )
            repetitions = row.get("repetitions", [])
            if isinstance(repetitions, list):
                for repetition in repetitions:
                    if isinstance(repetition, dict):
                        lines.append(
                            f"- Repetition {repetition.get('repetition')}: "
                            f"**{repetition.get('selected_key')}** — "
                            f"{repetition.get('selected_action')}; raw: "
                            f"`{repetition.get('raw_response')}`; provider letter: "
                            f"`{repetition.get('provider_answer')}`; canonical mapping: "
                            f"`{json.dumps(repetition.get('choice_mapping'), sort_keys=True)}`"
                        )
    return "\n".join(lines) + "\n"


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model input must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("model input must be LABEL=PATH")
    return label.strip(), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    args = parser.parse_args()
    model_paths = dict(args.model)
    if len(model_paths) != len(args.model):
        parser.error("model labels must be unique")
    report = build_stability_report(
        load_probes(args.probes),
        {model: load_observations(path) for model, path in model_paths.items()},
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
