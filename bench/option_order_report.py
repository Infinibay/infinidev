#!/usr/bin/env python3
"""Compare fixed-order and counterbalanced MCQ action observations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from bench.model_behavior import Observation, Probe, load_observations, load_probes


def _summary(probe: Probe, rows: Iterable[Observation]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: row.repetition)
    if any(row.error for row in ordered):
        raise ValueError(f"option-order report requires successful rows for {probe.id}")
    action_counts = Counter(row.answer for row in ordered)
    provider_counts = Counter((row.provider_answer or row.answer) for row in ordered)
    modal_count = max(action_counts.values())
    modes = sorted(key for key, count in action_counts.items() if count == modal_count)
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "provider_letter_counts": dict(sorted(provider_counts.items())),
        "modal_keys": modes,
        "modal_actions": [probe.choices[key] for key in modes],
        "stable": len(action_counts) == 1,
        "repetitions": [
            {
                "repetition": row.repetition,
                "canonical_key": row.answer,
                "canonical_action": probe.choices[row.answer],
                "provider_answer": row.provider_answer or row.answer,
                "choice_mapping": row.choice_mapping,
                "presentation_id": row.presentation_id or "fixed",
                "raw_response": row.response_text,
            }
            for row in ordered
        ],
    }


def _relation(fixed_modes: list[str], balanced_modes: list[str]) -> str:
    if len(fixed_modes) == 1 and len(balanced_modes) == 1:
        return "same_unique" if fixed_modes == balanced_modes else "changed_unique"
    if len(fixed_modes) == 1 and fixed_modes[0] in balanced_modes:
        return "balanced_tie_contains_fixed"
    return "incomparable_or_excluding_tie"


def build_option_order_report(
    probes: Mapping[str, Probe],
    observations_by_model: Mapping[str, Mapping[str, Iterable[Observation]]],
) -> dict[str, object]:
    """Align canonical actions while retaining displayed provider letters and mappings."""
    models: dict[str, object] = {}
    expected_probe_ids: set[str] | None = None
    for model, conditions in observations_by_model.items():
        if set(conditions) != {"fixed", "balanced"}:
            raise ValueError(f"{model} needs fixed and balanced observations")
        grouped: dict[str, dict[str, list[Observation]]] = {}
        identities: set[str] = set()
        for condition, rows_iter in conditions.items():
            by_probe: dict[str, list[Observation]] = defaultdict(list)
            for row in rows_iter:
                if row.probe_id not in probes:
                    raise ValueError(f"{model} contains unknown probe {row.probe_id}")
                expected_protocol = "fixed" if condition == "fixed" else "balanced_rotation"
                if row.option_order_protocol != expected_protocol:
                    raise ValueError(
                        f"{model}/{condition} has option protocol {row.option_order_protocol}"
                    )
                by_probe[row.probe_id].append(row)
                identities.add(row.model_identity)
            grouped[condition] = by_probe
        fixed_ids = set(grouped["fixed"])
        balanced_ids = set(grouped["balanced"])
        if fixed_ids != balanced_ids:
            raise ValueError(f"{model} fixed and balanced probe sets differ")
        if expected_probe_ids is None:
            expected_probe_ids = fixed_ids
        elif fixed_ids != expected_probe_ids:
            raise ValueError("models do not use the same option-order probe set")
        if len(identities) != 1:
            raise ValueError(f"{model} mixes model identities")

        records: list[dict[str, object]] = []
        for probe_id in sorted(fixed_ids):
            probe = probes[probe_id]
            fixed = _summary(probe, grouped["fixed"][probe_id])
            balanced = _summary(probe, grouped["balanced"][probe_id])
            relation = _relation(
                list(fixed["modal_keys"]), list(balanced["modal_keys"])
            )
            records.append(
                {
                    "probe_id": probe.id,
                    "category": probe.category,
                    "scenario": probe.scenario or probe.prompt,
                    "fixed": fixed,
                    "balanced": balanced,
                    "modal_relation": relation,
                }
            )
        relation_counts = Counter(str(record["modal_relation"]) for record in records)
        provider_letters = Counter(
            (row.provider_answer or row.answer)
            for rows in grouped["balanced"].values()
            for row in rows
        )
        models[model] = {
            "model_identity": next(iter(identities)),
            "probe_count": len(records),
            "modal_relation_counts": dict(sorted(relation_counts.items())),
            "balanced_provider_letter_counts": dict(sorted(provider_letters.items())),
            "records": records,
        }
    return {
        "interpretation_boundary": (
            "Fixed-order and balanced runs differ in both presentation and sample count. A changed "
            "or tied canonical mode reveals sensitivity but does not identify a single causal token "
            "bias. Provider letters, mappings, and canonical actions are retained separately."
        ),
        "models": models,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    """Render concrete canonical actions and displayed-letter evidence."""
    lines = [
        "# Fixed-order versus counterbalanced MCQ report",
        "",
        str(report["interpretation_boundary"]),
    ]
    models = report.get("models", {})
    if not isinstance(models, dict):
        return "\n".join(lines) + "\n"
    for model, raw_model in models.items():
        if not isinstance(raw_model, dict):
            continue
        lines.extend(
            [
                "",
                f"## Model: {model}",
                "",
                f"Modal relations: `{json.dumps(raw_model.get('modal_relation_counts'), sort_keys=True)}`.",
                f"Displayed provider-letter counts in the balanced run: "
                f"`{json.dumps(raw_model.get('balanced_provider_letter_counts'), sort_keys=True)}`.",
            ]
        )
        records = raw_model.get("records", [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            lines.extend(
                [
                    "",
                    f"### `{record.get('probe_id')}` — {record.get('modal_relation')}",
                    "",
                    f"Scenario: {record.get('scenario')}",
                ]
            )
            for condition in ("fixed", "balanced"):
                summary = record.get(condition, {})
                if not isinstance(summary, dict):
                    continue
                lines.append(
                    f"- **{condition}** canonical counts: "
                    f"`{json.dumps(summary.get('action_counts'), sort_keys=True)}`; modal actions: "
                    f"{summary.get('modal_actions')}; exact stability: {summary.get('stable')}."
                )
                repetitions = summary.get("repetitions", [])
                if isinstance(repetitions, list):
                    for row in repetitions:
                        if not isinstance(row, dict):
                            continue
                        lines.append(
                            f"  - r{row.get('repetition')}: provider "
                            f"**{row.get('provider_answer')}** -> canonical "
                            f"**{row.get('canonical_key')}** — {row.get('canonical_action')}; "
                            f"mapping `{json.dumps(row.get('choice_mapping'), sort_keys=True)}`."
                        )
    return "\n".join(lines) + "\n"


def _parse_model(value: str) -> tuple[str, tuple[Path, Path]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model input must be LABEL=FIXED,BALANCED")
    label, raw_paths = value.split("=", 1)
    paths = tuple(Path(item) for item in raw_paths.split(","))
    if not label.strip() or len(paths) != 2:
        raise argparse.ArgumentTypeError("model input must be LABEL=FIXED,BALANCED")
    return label.strip(), (paths[0], paths[1])


def _load_manifest_probe_ids(path: Path) -> set[str]:
    """Load the exact probe selection frozen by a behavior-run manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_probes = payload.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError("manifest must contain a non-empty probes list")
    probe_ids = {
        str(row.get("probe_id", "")).strip()
        for row in raw_probes
        if isinstance(row, dict)
    }
    if "" in probe_ids or len(probe_ids) != len(raw_probes):
        raise ValueError("manifest probe ids must be non-empty and unique")
    return probe_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="limit both fixed and balanced observations to frozen manifest probe ids",
    )
    args = parser.parse_args()
    model_paths = dict(args.model)
    selected_probe_ids = _load_manifest_probe_ids(args.manifest) if args.manifest else None

    def selected(path: Path) -> list[Observation]:
        rows = load_observations(path)
        if selected_probe_ids is None:
            return rows
        return [row for row in rows if row.probe_id in selected_probe_ids]

    report = build_option_order_report(
        load_probes(args.probes),
        {
            model: {
                "fixed": selected(paths[0]),
                "balanced": selected(paths[1]),
            }
            for model, paths in model_paths.items()
        },
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
