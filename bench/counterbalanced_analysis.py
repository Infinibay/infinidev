#!/usr/bin/env python3
"""Synthesize counterbalanced behavior evidence without hiding concrete actions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from bench.model_behavior import Probe, load_probes


def classify_probe(records: Mapping[str, Mapping[str, object]]) -> str:
    """Classify agreement while keeping ties and instability explicit."""
    modes = [list(record.get("modal_keys", [])) for record in records.values()]
    stable = [bool(record.get("stable")) for record in records.values()]
    if all(stable) and all(len(mode) == 1 for mode in modes):
        return "stable_shared" if len({mode[0] for mode in modes}) == 1 else "stable_divergence"
    if all(len(mode) == 1 for mode in modes):
        return "shared_modal" if len({mode[0] for mode in modes}) == 1 else "divergent_modal"
    return "modal_tie"


def build_analysis(
    probes: Mapping[str, Probe],
    stability: Mapping[str, object],
    option_order: Mapping[str, object],
) -> dict[str, object]:
    """Join stability, baseline relation, and concrete selected actions per probe."""
    stability_models = stability.get("models", {})
    order_models = option_order.get("models", {})
    if not isinstance(stability_models, dict) or not isinstance(order_models, dict):
        raise ValueError("reports must contain model mappings")
    if set(stability_models) != set(order_models):
        raise ValueError("stability and option-order reports use different models")

    stable_index: dict[str, dict[str, Mapping[str, object]]] = {}
    order_index: dict[str, dict[str, Mapping[str, object]]] = {}
    for model in stability_models:
        stable_payload = stability_models[model]
        order_payload = order_models[model]
        if not isinstance(stable_payload, dict) or not isinstance(order_payload, dict):
            raise ValueError(f"invalid report payload for {model}")
        stable_index[model] = {
            str(row["probe_id"]): row
            for row in stable_payload.get("records", [])
            if isinstance(row, dict)
        }
        order_index[model] = {
            str(row["probe_id"]): row
            for row in order_payload.get("records", [])
            if isinstance(row, dict)
        }
    probe_ids = set.intersection(*(set(rows) for rows in stable_index.values()))
    if any(set(rows) != probe_ids for rows in stable_index.values()):
        raise ValueError("models do not share the same stability probe set")

    records: list[dict[str, object]] = []
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    for probe_id in sorted(probe_ids, key=lambda item: (probes[item].category, item)):
        probe = probes[probe_id]
        model_records = {model: stable_index[model][probe_id] for model in stable_index}
        classification = classify_probe(model_records)
        categories[probe.category][classification] += 1
        models: dict[str, object] = {}
        for model, stable_record in model_records.items():
            order_record = order_index[model][probe_id]
            fixed = order_record["fixed"]
            balanced = order_record["balanced"]
            assert isinstance(fixed, dict) and isinstance(balanced, dict)
            models[model] = {
                "fixed_key": list(fixed.get("modal_keys", [])),
                "fixed_actions": list(fixed.get("modal_actions", [])),
                "balanced_counts": dict(stable_record.get("answer_counts", {})),
                "balanced_modal_keys": list(stable_record.get("modal_keys", [])),
                "balanced_modal_actions": list(stable_record.get("modal_actions", [])),
                "exactly_stable": bool(stable_record.get("stable")),
                "fixed_to_balanced_relation": order_record.get("modal_relation"),
                "repetitions": stable_record.get("repetitions", []),
            }
        records.append(
            {
                "probe_id": probe_id,
                "category": probe.category,
                "family": probe.group,
                "evaluation_mode": probe.evaluation_mode,
                "scenario": probe.scenario or probe.prompt,
                "user_request": probe.user_request or "",
                "choices": probe.choices,
                "draft_normative_key": probe.answer,
                "classification": classification,
                "models": models,
            }
        )

    classifications = Counter(str(row["classification"]) for row in records)
    model_summaries: dict[str, object] = {}
    for model, raw_stability in stability_models.items():
        assert isinstance(raw_stability, dict)
        raw_order = order_models[model]
        assert isinstance(raw_order, dict)
        letters = dict(raw_order.get("balanced_provider_letter_counts", {}))
        total_letters = sum(int(value) for value in letters.values())
        model_summaries[model] = {
            "stable_probes": raw_stability.get("stable_probes"),
            "unstable_probes": raw_stability.get("unstable_probes"),
            "fixed_to_balanced_relations": raw_order.get("modal_relation_counts", {}),
            "displayed_letter_counts": letters,
            "displayed_a_share": (int(letters.get("A", 0)) / total_letters if total_letters else None),
        }
    return {
        "interpretation_boundary": (
            "These are externally observable choices, not private reasoning traces. Preference "
            "probes have no universal optimum. Numeric summaries locate patterns; the concrete "
            "actions and all four selections remain the evidence used for prompt design."
        ),
        "selection_boundary": (
            "The 78 probes were selected because the initial fixed-order model responses diverged, "
            "so this follow-up must not be generalized to the other 606 probes or to population rates."
        ),
        "probe_count": len(records),
        "classification_counts": dict(sorted(classifications.items())),
        "cross_model_modal_agreements": stability.get("cross_model_modal_agreements"),
        "models": model_summaries,
        "categories": {
            category: dict(sorted(counts.items())) for category, counts in sorted(categories.items())
        },
        "records": records,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    """Render an action-first report suitable for model-specific prompt decisions."""
    lines = [
        "# Counterbalanced divergence analysis",
        "",
        str(report["interpretation_boundary"]),
        "",
        str(report["selection_boundary"]),
        "",
        "## What changed after counterbalancing",
        "",
        f"Classification counts: `{json.dumps(report.get('classification_counts'), sort_keys=True)}`.",
        f"Cross-model unique modal agreement: {report.get('cross_model_modal_agreements')}/"
        f"{report.get('probe_count')} probes.",
    ]
    models = report.get("models", {})
    if isinstance(models, dict):
        for model, raw in models.items():
            if not isinstance(raw, dict):
                continue
            a_share = raw.get("displayed_a_share")
            a_text = f"{100 * a_share:.1f}%" if isinstance(a_share, (int, float)) else "n/a"
            lines.append(
                f"- **{model}**: {raw.get('stable_probes')} exactly stable; "
                f"{raw.get('unstable_probes')} unstable; fixed→balanced relations "
                f"`{json.dumps(raw.get('fixed_to_balanced_relations'), sort_keys=True)}`; "
                f"displayed A selected {a_text}."
            )
    lines.extend(
        [
            "",
            "## Consequences for Infinidev prompt calibration",
            "",
            "1. Do not calibrate from one fixed-order answer. A unique balanced mode changed from "
            "the original answer for many probes, and most probes were not exactly stable.",
            "2. Treat model answers as raw behavioral priors. Preference choices must be resolved "
            "against an explicit user objective such as autonomy, control, speed, quality, or cost.",
            "3. Retain concrete actions, not only axis scores. A prompt candidate should state the "
            "behavior it is trying to encourage or counteract and link back to these selections.",
            "4. Use repeated balanced presentations for MCQ discovery. The displayed-letter skew is "
            "evidence that canonical remapping alone does not make a one-shot answer robust.",
            "5. Promote shared stable actions only to candidates for outcome evaluation, never "
            "directly to universal system rules. Model-specific stable divergences are candidates "
            "for per-model guidance, again conditioned on the user's objective.",
            "",
            "## Complete action-level evidence",
        ]
    )
    current_category = ""
    records = report.get("records", [])
    if isinstance(records, list):
        for row in records:
            if not isinstance(row, dict):
                continue
            category = str(row.get("category"))
            if category != current_category:
                lines.extend(["", f"### Category: {category}"])
                current_category = category
            lines.extend(
                [
                    "",
                    f"#### `{row.get('probe_id')}` — {row.get('classification')}",
                    "",
                    f"Scenario: {row.get('scenario')}",
                ]
            )
            if row.get("user_request"):
                lines.append(f"User request: {row.get('user_request')}")
            if row.get("evaluation_mode") == "preference":
                lines.append("Interpretation: raw preference; no universally correct action.")
            else:
                lines.append(f"Draft normative key: `{row.get('draft_normative_key')}`.")
            raw_models = row.get("models", {})
            if isinstance(raw_models, dict):
                for model, model_row in raw_models.items():
                    if not isinstance(model_row, dict):
                        continue
                    lines.append(
                        f"- **{model}**: fixed {model_row.get('fixed_key')} → balanced counts "
                        f"`{json.dumps(model_row.get('balanced_counts'), sort_keys=True)}`; modal "
                        f"action(s): {model_row.get('balanced_modal_actions')}; exactly stable: "
                        f"{model_row.get('exactly_stable')}."
                    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("stability_json", type=Path)
    parser.add_argument("option_order_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    report = build_analysis(
        load_probes(args.probes),
        json.loads(args.stability_json.read_text(encoding="utf-8")),
        json.loads(args.option_order_json.read_text(encoding="utf-8")),
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
