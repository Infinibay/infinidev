#!/usr/bin/env python3
"""Compare repeated raw and user-profile-conditioned behavioral choices."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from bench.model_behavior import Observation, Probe, load_observations, load_probes


CONDITIONS = ("raw", "fast_autonomy", "quality_control")
EXPECTED_PROFILE_NAMES = {
    "raw": "",
    "fast_autonomy": "fast-autonomy",
    "quality_control": "quality-control",
}


def _choice_summary(probe: Probe, rows: Iterable[Observation]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: row.repetition)
    if any(row.error for row in ordered):
        raise ValueError(f"profile report requires successful rows for {probe.id}")
    repetitions = [row.repetition for row in ordered]
    if len(repetitions) != len(set(repetitions)):
        raise ValueError(f"duplicate repetitions for {probe.id}")
    counts = Counter(row.answer for row in ordered)
    modal_count = max(counts.values())
    modal_keys = sorted(key for key, count in counts.items() if count == modal_count)
    return {
        "stable": len(counts) == 1,
        "answer_counts": dict(sorted(counts.items())),
        "modal_keys": modal_keys,
        "modal_actions": [probe.choices[key] for key in modal_keys],
        "repetitions": [
            {
                "repetition": row.repetition,
                "selected_key": row.answer,
                "selected_action": probe.choices[row.answer],
                "raw_response": row.response_text,
                "provider_answer": row.provider_answer or row.answer,
                "choice_mapping": row.choice_mapping,
                "option_order_protocol": row.option_order_protocol,
            }
            for row in ordered
        ],
    }


def _unique_mode(summary: Mapping[str, object]) -> str | None:
    modes = summary.get("modal_keys")
    return str(modes[0]) if isinstance(modes, list) and len(modes) == 1 else None


def build_profile_adaptation_report(
    probes: Mapping[str, Probe],
    observations_by_model: Mapping[str, Mapping[str, Iterable[Observation]]],
) -> dict[str, object]:
    """Preserve concrete actions while locating replicated profile-sensitive changes."""
    report_models: dict[str, object] = {}
    expected_probe_ids: set[str] | None = None
    for model, condition_rows in observations_by_model.items():
        if set(condition_rows) != set(CONDITIONS):
            raise ValueError(f"{model} needs raw, fast_autonomy, and quality_control rows")
        grouped: dict[str, dict[str, list[Observation]]] = {}
        identities: set[str] = set()
        option_protocols: set[str] = set()
        profile_hashes: dict[str, set[str]] = defaultdict(set)
        condition_probe_ids: dict[str, set[str]] = {}
        for condition, rows_iter in condition_rows.items():
            by_probe: dict[str, list[Observation]] = defaultdict(list)
            for row in rows_iter:
                if row.probe_id not in probes:
                    raise ValueError(f"{model} contains unknown probe {row.probe_id}")
                by_probe[row.probe_id].append(row)
                identities.add(row.model_identity)
                option_protocols.add(row.option_order_protocol)
                if row.utility_profile != EXPECTED_PROFILE_NAMES[condition]:
                    raise ValueError(
                        f"{model}/{condition} has unexpected utility profile "
                        f"{row.utility_profile!r}"
                    )
                profile_hashes[condition].add(row.utility_profile_sha256)
            grouped[condition] = by_probe
            condition_probe_ids[condition] = set(by_probe)
        probe_sets = list(condition_probe_ids.values())
        if any(probe_ids != probe_sets[0] for probe_ids in probe_sets[1:]):
            raise ValueError(f"{model} profile conditions do not use the same probe set")
        if expected_probe_ids is None:
            expected_probe_ids = probe_sets[0]
        elif probe_sets[0] != expected_probe_ids:
            raise ValueError("models do not use the same profile probe set")
        if len(identities) != 1:
            raise ValueError(f"{model} profile conditions mix model identities")
        if len(option_protocols) != 1:
            raise ValueError(f"{model} profile conditions mix option-order protocols")
        if profile_hashes["raw"] != {""}:
            raise ValueError(f"{model} raw rows unexpectedly have a profile hash")
        for condition in ("fast_autonomy", "quality_control"):
            if len(profile_hashes[condition]) != 1 or "" in profile_hashes[condition]:
                raise ValueError(f"{model}/{condition} needs one non-empty profile hash")

        records: list[dict[str, object]] = []
        for probe_id in sorted(probe_sets[0]):
            probe = probes[probe_id]
            repetition_sets = [
                {row.repetition for row in grouped[condition][probe_id]}
                for condition in CONDITIONS
            ]
            if any(values != repetition_sets[0] for values in repetition_sets[1:]):
                raise ValueError(f"{model}/{probe_id} profile repetitions do not match")
            summaries = {
                condition: _choice_summary(probe, grouped[condition][probe_id])
                for condition in CONDITIONS
            }
            raw_mode = _unique_mode(summaries["raw"])
            fast_mode = _unique_mode(summaries["fast_autonomy"])
            quality_mode = _unique_mode(summaries["quality_control"])
            records.append(
                {
                    "probe_id": probe.id,
                    "category": probe.category,
                    "scenario": probe.scenario or probe.prompt,
                    "user_request": probe.user_request or "",
                    "conditions": summaries,
                    "fast_quality_modal_change": (
                        fast_mode is not None
                        and quality_mode is not None
                        and fast_mode != quality_mode
                    ),
                    "raw_fast_modal_change": (
                        raw_mode is not None and fast_mode is not None and raw_mode != fast_mode
                    ),
                    "raw_quality_modal_change": (
                        raw_mode is not None
                        and quality_mode is not None
                        and raw_mode != quality_mode
                    ),
                }
            )
        report_models[model] = {
            "model_identity": next(iter(identities)),
            "option_order_protocol": next(iter(option_protocols)),
            "probe_count": len(records),
            "fast_quality_modal_changes": sum(
                bool(row["fast_quality_modal_change"]) for row in records
            ),
            "raw_fast_modal_changes": sum(bool(row["raw_fast_modal_change"]) for row in records),
            "raw_quality_modal_changes": sum(
                bool(row["raw_quality_modal_change"]) for row in records
            ),
            "records": records,
        }
    return {
        "interpretation_boundary": (
            "Profile changes describe externally observable repeated choices, not private reasoning. "
            "There is no universally optimal preference answer; inspect the concrete actions against "
            "the active user's priorities. A modal change is reported only when both conditions have "
            "a unique mode across the recorded repetitions."
        ),
        "models": report_models,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    """Render each repeated action before aggregate profile-change counts."""
    lines = [
        "# Repeated user-profile adaptation report",
        "",
        str(report["interpretation_boundary"]),
        "",
        "Every repetition used a fresh conversation, no system message, and choice-only elicitation.",
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
                f"Option-order protocol: `{raw_model.get('option_order_protocol')}`.",
                "",
                f"Unique modal action changed between fast/autonomy and quality/control on "
                f"{raw_model.get('fast_quality_modal_changes')}/{raw_model.get('probe_count')} "
                "probes. This count is an index; the actions below are the evidence.",
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
                    f"### `{record.get('probe_id')}` — {record.get('category')}",
                    "",
                    f"Scenario: {record.get('scenario')}",
                ]
            )
            conditions = record.get("conditions", {})
            if not isinstance(conditions, dict):
                continue
            for condition in CONDITIONS:
                summary = conditions.get(condition, {})
                if not isinstance(summary, dict):
                    continue
                lines.append(
                    f"- **{condition}**: counts `{json.dumps(summary.get('answer_counts'), sort_keys=True)}`; "
                    f"modal action(s): {summary.get('modal_actions')}; exact stability: "
                    f"{summary.get('stable')}."
                )
                repetitions = summary.get("repetitions", [])
                if isinstance(repetitions, list):
                    rendered = "; ".join(
                        f"r{item.get('repetition')}={item.get('selected_key')} — "
                        f"{item.get('selected_action')}"
                        for item in repetitions
                        if isinstance(item, dict)
                    )
                    lines.append(f"  Observed repetitions: {rendered}.")
            lines.append(
                "- Profile separation: "
                + (
                    "fast/autonomy and quality/control have different unique modes."
                    if record.get("fast_quality_modal_change")
                    else "no different unique mode was established."
                )
            )
    return "\n".join(lines) + "\n"


def _parse_model(value: str) -> tuple[str, tuple[Path, Path, Path]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model input must be LABEL=RAW,FAST,QUALITY")
    label, raw_paths = value.split("=", 1)
    paths = tuple(Path(item) for item in raw_paths.split(","))
    if not label.strip() or len(paths) != 3:
        raise argparse.ArgumentTypeError("model input must be LABEL=RAW,FAST,QUALITY")
    return label.strip(), (paths[0], paths[1], paths[2])


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
    report = build_profile_adaptation_report(
        load_probes(args.probes),
        {
            model: {
                condition: load_observations(path)
                for condition, path in zip(CONDITIONS, paths, strict=True)
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
