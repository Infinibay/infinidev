"""Evaluate task-policy predictions against operational usability gates.

The classifier benchmarks optimise aggregate scores. This module answers a
different question: whether their out-of-fold predictions are safe enough to
change prompts automatically, and which kinds of request still fail.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Callable

from bench.task_policy_manual_audit import load_examples
from bench.task_policy_multilabel_head import METHOD_LABELS


USABILITY_EVAL_VERSION = "task-policy-usability-v1"


def _prediction_map(
    report: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, tuple[str, ...]]:
    """Reconstruct full out-of-fold predictions from a benchmark report."""
    evaluated = int(report.get("evaluated_examples", report["examples"]))
    if evaluated != len(rows):
        raise ValueError(
            "usability evaluation requires a complete out-of-fold report: "
            f"report has {evaluated} predictions for {len(rows)} rows"
        )
    predictions = {
        str(row["id"]): tuple(str(policy) for policy in row["policies"])
        for row in rows
    }
    known_ids = set(predictions)
    for error in report["errors"]:
        row_id = str(error["id"])
        if row_id not in known_ids:
            raise ValueError(f"report error references unknown row: {row_id}")
        predictions[row_id] = tuple(str(policy) for policy in error["predicted"])
    return predictions


def _metrics(
    rows: list[dict[str, Any]], predictions: dict[str, tuple[str, ...]]
) -> dict[str, Any]:
    """Compute exactness, activation safety, and multi-label error direction."""
    exact = false_activations = covered_tasks = 0
    over = under = mixed = 0
    true_positives = Counter({label: 0 for label in METHOD_LABELS})
    false_positives = Counter({label: 0 for label in METHOD_LABELS})
    false_negatives = Counter({label: 0 for label in METHOD_LABELS})
    negatives = sum(not row["policies"] for row in rows)
    tasks = len(rows) - negatives
    for row in rows:
        expected = set(str(policy) for policy in row["policies"])
        predicted = set(predictions[str(row["id"])])
        if expected == predicted:
            exact += 1
        elif predicted > expected:
            over += 1
        elif predicted < expected:
            under += 1
        else:
            mixed += 1
        if not expected and predicted:
            false_activations += 1
        if expected and predicted:
            covered_tasks += 1
        for label in METHOD_LABELS:
            if label in expected and label in predicted:
                true_positives[label] += 1
            elif label not in expected and label in predicted:
                false_positives[label] += 1
            elif label in expected and label not in predicted:
                false_negatives[label] += 1
    per_label = {}
    f1_values = []
    for label in METHOD_LABELS:
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return {
        "examples": len(rows),
        "exact_match": exact / len(rows) if rows else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "false_activations": false_activations,
        "negative_examples": negatives,
        "false_activation_rate": false_activations / negatives if negatives else 0.0,
        "task_coverage": covered_tasks / tasks if tasks else 1.0,
        "error_direction": {"over": over, "under": under, "mixed": mixed},
        "per_label": per_label,
    }


def _grouped_metrics(
    rows: list[dict[str, Any]],
    predictions: dict[str, tuple[str, ...]],
    key: Callable[[dict[str, Any]], str],
    *,
    minimum_support: int = 1,
) -> dict[str, dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {
        name: _metrics(group, predictions)
        for name, group in sorted(groups.items())
        if len(group) >= minimum_support
    }


def evaluate_report(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    minimum_slice_support: int = 5,
) -> dict[str, Any]:
    """Return slice diagnostics and conservative deployment decisions."""
    predictions = _prediction_map(report, rows)
    overall = _metrics(rows, predictions)
    cardinality = _grouped_metrics(
        rows, predictions, lambda row: str(len(row["policies"]))
    )
    languages = _grouped_metrics(
        rows,
        predictions,
        lambda row: str(row["language"]),
        minimum_support=minimum_slice_support,
    )
    difficulties = _grouped_metrics(
        rows,
        predictions,
        lambda row: str(row["difficulty"]),
        minimum_support=minimum_slice_support,
    )
    batches = _grouped_metrics(
        rows,
        predictions,
        lambda row: str(row["batch"]),
        minimum_support=minimum_slice_support,
    )
    negative_reasons = _grouped_metrics(
        [row for row in rows if not row["policies"]],
        predictions,
        lambda row: str(row["uncategorized_reason"]),
    )
    core_cardinalities = [value for value in cardinality.values() if value["examples"] >= 10]
    label_precisions = [
        value["precision"]
        for value in overall["per_label"].values()
        if value["support"]
    ]
    automatic_checks = {
        "overall_exact_at_least_95pct": overall["exact_match"] >= 0.95,
        "zero_false_activations": overall["false_activations"] == 0,
        "every_core_cardinality_at_least_90pct": all(
            value["exact_match"] >= 0.90 for value in core_cardinalities
        ),
        "every_label_precision_at_least_95pct": all(
            precision >= 0.95 for precision in label_precisions
        ),
    }
    advisory_checks = {
        "macro_f1_at_least_80pct": overall["macro_f1"] >= 0.80,
        "task_coverage_at_least_90pct": overall["task_coverage"] >= 0.90,
        "false_activation_rate_at_most_10pct": (
            overall["false_activation_rate"] <= 0.10
        ),
    }
    return {
        "version": USABILITY_EVAL_VERSION,
        "model": report.get("model"),
        "head": report.get("head", "fine_tuned_encoder"),
        "overall": overall,
        "slices": {
            "cardinality": cardinality,
            "language": languages,
            "difficulty": difficulties,
            "batch": batches,
            "negative_reason": negative_reasons,
        },
        "deployment": {
            "automatic_prompt_injection": {
                "usable": all(automatic_checks.values()),
                "checks": automatic_checks,
            },
            "shadow_or_advisory": {
                "usable": all(advisory_checks.values()),
                "checks": advisory_checks,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--minimum-slice-support", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_examples()
    evaluations = [
        evaluate_report(
            json.loads(path.read_text(encoding="utf-8")),
            rows,
            minimum_slice_support=args.minimum_slice_support,
        )
        for path in args.reports
    ]
    rendered = json.dumps({"evaluations": evaluations}, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "reports": len(evaluations)}))


if __name__ == "__main__":
    main()


__all__ = ["USABILITY_EVAL_VERSION", "evaluate_report"]
