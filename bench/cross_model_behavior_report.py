#!/usr/bin/env python3
"""Render an evidence-first comparison of isolated model probe responses."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from bench.model_behavior import Observation, Probe, load_observations, load_probes


def build_comparison(
    probes: Mapping[str, Probe],
    observations_by_model: Mapping[str, Iterable[Observation]],
) -> dict[str, object]:
    """Align model observations and retain the concrete action behind every key."""
    if len(observations_by_model) < 2:
        raise ValueError("comparison needs observations from at least two models")
    indexed: dict[str, dict[tuple[str, str, int], Observation]] = {}
    order: list[tuple[str, str, int]] = []
    for model, raw_rows in observations_by_model.items():
        rows = list(raw_rows)
        current: dict[tuple[str, str, int], Observation] = {}
        for row in rows:
            if row.probe_id not in probes:
                raise ValueError(f"{model} contains unknown probe {row.probe_id}")
            key = (row.probe_id, row.condition, row.repetition)
            if key in current:
                raise ValueError(f"{model} contains duplicate observation {key}")
            current[key] = row
        indexed[model] = current
        if not order:
            order = list(current)
    expected = set(order)
    for model, rows in indexed.items():
        missing = expected - set(rows)
        extra = set(rows) - expected
        if missing or extra:
            raise ValueError(
                f"{model} does not match comparison set: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

    questions: list[dict[str, object]] = []
    for probe_id, condition, repetition in order:
        probe = probes[probe_id]
        model_rows: dict[str, object] = {}
        successful_answers: list[str] = []
        for model, rows in indexed.items():
            row = rows[(probe_id, condition, repetition)]
            if not row.error:
                successful_answers.append(row.answer)
            model_rows[model] = {
                "selected_key": row.answer,
                "selected_action": probe.choices.get(row.answer),
                "provider_answer": row.provider_answer or row.answer,
                "choice_mapping": row.choice_mapping,
                "option_order_protocol": row.option_order_protocol,
                "correct": row.answer == probe.answer if probe.answer else None,
                "raw_response": row.response_text,
                "error": row.error,
                "latency_seconds": row.latency_seconds,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "model_identity": row.model_identity,
                "utility_profile": row.utility_profile,
                "utility_profile_sha256": row.utility_profile_sha256,
            }
        questions.append(
            {
                "probe_id": probe.id,
                "category": probe.category,
                "family": probe.group,
                "evaluation_mode": probe.evaluation_mode,
                "review_status": probe.review_status,
                "scenario": probe.scenario or probe.prompt,
                "user_request": probe.user_request or "",
                "choices": probe.choices,
                "expected_key": probe.answer,
                "expected_action": probe.choices.get(probe.answer or ""),
                "unanimous": bool(successful_answers)
                and len(set(successful_answers)) == 1
                and len(successful_answers) == len(indexed),
                "models": model_rows,
            }
        )

    return {
        "interpretation_boundary": (
            "This report compares externally observable selections and raw replies. "
            "It does not infer private chain-of-thought. Correctness uses draft normative "
            "keys and is exploratory until independent review approves the probes."
        ),
        "protocol": (
            "Each call used a fresh conversation, no system message, choice-only elicitation, "
            "one active request at a time, and no automatic retries."
        ),
        "models": {
            model: _model_summary(rows.values(), probes)
            for model, rows in indexed.items()
        },
        "questions": questions,
        "unanimous_questions": sum(bool(row["unanimous"]) for row in questions),
        "divergent_questions": sum(not bool(row["unanimous"]) for row in questions),
    }


def _model_summary(
    rows: Iterable[Observation], probes: Mapping[str, Probe]
) -> dict[str, object]:
    selected = list(rows)
    successful = [row for row in selected if not row.error]
    normative = [row for row in successful if probes[row.probe_id].answer]
    latencies = [row.latency_seconds for row in successful if row.latency_seconds is not None]
    return {
        "attempted": len(selected),
        "errors": len(selected) - len(successful),
        "normative_matches": sum(
            row.answer == probes[row.probe_id].answer for row in normative
        ),
        "normative_total": len(normative),
        "preference_total": sum(
            probes[row.probe_id].evaluation_mode == "preference" for row in successful
        ),
        "utility_profiles": sorted({row.utility_profile for row in successful if row.utility_profile}),
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "input_tokens": sum(row.input_tokens or 0 for row in successful),
        "output_tokens": sum(row.output_tokens or 0 for row in successful),
    }


def render_markdown(report: Mapping[str, object]) -> str:
    """Render model selections first and aggregate numbers second."""
    questions = report.get("questions", [])
    lines = [
        "# Comparative raw-behavior report",
        "",
        str(report["interpretation_boundary"]),
        "",
        str(report["protocol"]),
        "",
        "## Run summary",
        "",
        f"Unanimous questions: {report.get('unanimous_questions', 0)}; "
        f"divergent questions: {report.get('divergent_questions', 0)}.",
    ]
    models = report.get("models", {})
    if isinstance(models, dict):
        for model, raw_summary in models.items():
            if not isinstance(raw_summary, dict):
                continue
            latency = raw_summary.get("median_latency_seconds")
            latency_text = f"{latency:.3f}s" if isinstance(latency, (int, float)) else "n/a"
            normative_total = int(raw_summary.get("normative_total", 0))
            preference_total = int(raw_summary.get("preference_total", 0))
            evidence_parts: list[str] = []
            if normative_total:
                evidence_parts.append(
                    f"{raw_summary.get('normative_matches')}/{normative_total} "
                    "draft-key matches"
                )
            if preference_total:
                evidence_parts.append(f"{preference_total} preference choices")
            lines.append(
                f"- **{model}**: {', '.join(evidence_parts)}; "
                f"{raw_summary.get('errors')} errors; median latency {latency_text}."
            )

    by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    if isinstance(questions, list):
        for row in questions:
            if isinstance(row, dict):
                by_category[str(row.get("category", "unknown"))].append(row)
    for category, category_rows in by_category.items():
        lines.extend(["", f"## Category: {category}"])
        for row in category_rows:
            lines.extend(["", f"### `{row.get('probe_id')}`"])
            lines.append(f"Scenario: {row.get('scenario')}")
            if row.get("user_request"):
                lines.append(f"User request: {row.get('user_request')}")
            lines.extend(["", "Offered actions:"])
            choices = row.get("choices", {})
            if isinstance(choices, dict):
                for key, action in choices.items():
                    lines.append(f"- **{key}** — {action}")
            lines.append("")
            if row.get("evaluation_mode") == "normative":
                lines.append(
                    f"Draft normative key: **{row.get('expected_key')}** — "
                    f"{row.get('expected_action')}"
                )
            else:
                lines.append(
                    "Preference probe: there is no universal correct action; interpret each "
                    "selection against its stated user profile."
                )
            lines.extend(["", "Observed responses:"])
            model_rows = row.get("models", {})
            if isinstance(model_rows, dict):
                for model, raw_model_row in model_rows.items():
                    if not isinstance(raw_model_row, dict):
                        continue
                    lines.append(
                        f"- **{model}** selected **{raw_model_row.get('selected_key')}** — "
                        f"{raw_model_row.get('selected_action')}; "
                        f"profile: `{raw_model_row.get('utility_profile') or 'none'}`; raw: "
                        f"`{raw_model_row.get('raw_response')}`; provider letter: "
                        f"`{raw_model_row.get('provider_answer')}`; canonical mapping: "
                        f"`{json.dumps(raw_model_row.get('choice_mapping'), sort_keys=True)}`"
                    )
            lines.append(
                "Observed pattern: "
                + ("unanimous selection." if row.get("unanimous") else "models diverged.")
            )
    return "\n".join(lines) + "\n"


def _parse_model_input(value: str) -> tuple[str, Path]:
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
    parser.add_argument("--model", action="append", type=_parse_model_input, required=True)
    args = parser.parse_args()
    model_inputs = dict(args.model)
    if len(model_inputs) != len(args.model):
        parser.error("model labels must be unique")
    report = build_comparison(
        load_probes(args.probes),
        {label: load_observations(path) for label, path in model_inputs.items()},
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
