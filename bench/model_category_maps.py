#!/usr/bin/env python3
"""Render one exhaustive category-by-category decision map per model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping


def evidence_label(model_record: Mapping[str, object]) -> str:
    """Describe how strongly the four balanced selections support a policy."""
    if model_record.get("exactly_stable") is True:
        return "stable_prior_4_of_4"
    modes = model_record.get("balanced_modal_keys")
    if isinstance(modes, list) and len(modes) == 1:
        return "position_sensitive_unique_mode"
    return "unresolved_modal_tie"


def build_category_maps(analysis: Mapping[str, object]) -> dict[str, object]:
    """Pivot the action-complete analysis from probe-first to model/category-first."""
    raw_models = analysis.get("models")
    records = analysis.get("records")
    if not isinstance(raw_models, dict) or not isinstance(records, list):
        raise ValueError("analysis needs models and records")
    result_models: dict[str, object] = {}
    for model, model_summary in raw_models.items():
        categories: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            if not isinstance(record, dict):
                continue
            record_models = record.get("models")
            model_record = record_models.get(model) if isinstance(record_models, dict) else None
            if not isinstance(model_record, dict):
                raise ValueError(f"record is missing model evidence: {record.get('probe_id')}/{model}")
            categories[str(record.get("category", "unknown"))].append(
                {
                    "probe_id": record.get("probe_id"),
                    "family": record.get("family"),
                    "evaluation_mode": record.get("evaluation_mode"),
                    "scenario": record.get("scenario"),
                    "user_request": record.get("user_request"),
                    "evidence_label": evidence_label(model_record),
                    "fixed_actions": model_record.get("fixed_actions"),
                    "balanced_counts": model_record.get("balanced_counts"),
                    "observed_policy": model_record.get("balanced_modal_actions"),
                    "fixed_to_balanced_relation": model_record.get("fixed_to_balanced_relation"),
                    "cross_model_classification": record.get("classification"),
                }
            )
        rendered_categories: dict[str, object] = {}
        for category, category_records in sorted(categories.items()):
            labels = Counter(str(row["evidence_label"]) for row in category_records)
            rendered_categories[category] = {
                "probe_count": len(category_records),
                "evidence_counts": dict(sorted(labels.items())),
                "stable_policies": [
                    {
                        "probe_id": row["probe_id"],
                        "policy": row["observed_policy"],
                    }
                    for row in category_records
                    if row["evidence_label"] == "stable_prior_4_of_4"
                ],
                "records": category_records,
            }
        result_models[str(model)] = {
            "summary": model_summary,
            "category_count": len(rendered_categories),
            "categories": rendered_categories,
        }
    return {
        "interpretation_boundary": (
            "This is a map of externally observable action selection, not private chain-of-thought. "
            "A stable prior means the same canonical action survived all four displayed positions. "
            "A unique mode or tie remains position-sensitive evidence, and preference actions are "
            "not universal optima."
        ),
        "selection_boundary": analysis.get("selection_boundary"),
        "models": result_models,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    """Render model first, category second, and every concrete policy third."""
    lines = [
        "# Cómo decide cada modelo, separado por categorías",
        "",
        str(report.get("interpretation_boundary")),
        "",
        str(report.get("selection_boundary")),
    ]
    models = report.get("models")
    if not isinstance(models, dict):
        return "\n".join(lines) + "\n"
    for model, model_payload in models.items():
        if not isinstance(model_payload, dict):
            continue
        summary = model_payload.get("summary", {})
        lines.extend(["", f"## Modelo: {model}", ""])
        if isinstance(summary, dict):
            lines.append(
                f"Estabilidad global del subconjunto: {summary.get('stable_probes')} estables, "
                f"{summary.get('unstable_probes')} sensibles o empatados. Selecciones por letra "
                f"mostrada: `{json.dumps(summary.get('displayed_letter_counts'), sort_keys=True)}`."
            )
        categories = model_payload.get("categories")
        if not isinstance(categories, dict):
            continue
        for category, category_payload in categories.items():
            if not isinstance(category_payload, dict):
                continue
            lines.extend(["", f"### Categoría: {category}", ""])
            lines.append(
                f"Casos: {category_payload.get('probe_count')}; fuerza de evidencia: "
                f"`{json.dumps(category_payload.get('evidence_counts'), sort_keys=True)}`."
            )
            stable_policies = category_payload.get("stable_policies")
            if isinstance(stable_policies, list) and stable_policies:
                lines.extend(["", "Políticas que sobrevivieron las cuatro posiciones:"])
                for item in stable_policies:
                    if isinstance(item, dict):
                        lines.append(f"- `{item.get('probe_id')}` → {item.get('policy')}")
            records = category_payload.get("records")
            if not isinstance(records, list):
                continue
            lines.extend(["", "Mapa completo de decisiones:"])
            for record in records:
                if not isinstance(record, dict):
                    continue
                lines.extend(
                    [
                        "",
                        f"#### `{record.get('probe_id')}` — {record.get('evidence_label')}",
                        "",
                        f"Situación: {record.get('scenario')}",
                    ]
                )
                if record.get("user_request"):
                    lines.append(f"Pedido: {record.get('user_request')}")
                lines.append(
                    f"- Política observada: {record.get('observed_policy')}"
                )
                lines.append(
                    f"- Conteo en cuatro posiciones: "
                    f"`{json.dumps(record.get('balanced_counts'), sort_keys=True)}`."
                )
                lines.append(
                    f"- Respuesta fija original: {record.get('fixed_actions')}; relación: "
                    f"`{record.get('fixed_to_balanced_relation')}`."
                )
                boundary = (
                    "preferencia cruda, dependiente del objetivo del usuario"
                    if record.get("evaluation_mode") == "preference"
                    else "decisión normativa draft, pendiente de revisión independiente"
                )
                lines.append(f"- Límite: {boundary}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    report = build_category_maps(json.loads(args.analysis.read_text(encoding="utf-8")))
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
