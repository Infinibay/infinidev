"""Evaluate overlapping-window inference for a frozen task-policy encoder.

Long GitHub issues are often truncated by the deployed encoder. This benchmark
scores every overlapping token window once, selects an aggregation strategy on
human calibration data, and reports the result on the development evaluation
split. It does not modify the checkpoint or production runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from bench.task_policy_encoder_finetune import QUERY_TOKENS, _load_frozen
from bench.task_policy_gliclass_finetune import _binary_metrics, _load_partition, _targets
from bench.task_policy_multilabel_head import METHOD_LABELS
from bench.task_policy_pairwise_finetune import _select_threshold


ACCURACY_TARGET = 0.95
RECALL_TARGET = 0.95


def _aggregate(values: np.ndarray, owners: np.ndarray, examples: int, mode: str) -> np.ndarray:
    """Aggregate window scores back to one row per original example."""
    rows = []
    for owner in range(examples):
        selected = values[owners == owner]
        if not len(selected):
            raise ValueError(f"example {owner} has no token window")
        if mode == "first":
            aggregate = selected[0]
        elif mode == "max":
            aggregate = selected.max(axis=0)
        elif mode == "mean":
            aggregate = selected.mean(axis=0)
        elif mode == "top2_mean":
            count = min(2, len(selected))
            aggregate = np.sort(selected, axis=0)[-count:].mean(axis=0)
        else:
            raise ValueError(f"unknown aggregation mode: {mode}")
        rows.append(aggregate)
    return np.stack(rows)


def _metric_key(metrics: dict[str, float | int]) -> tuple[float, ...]:
    accuracy = float(metrics["accuracy"])
    recall = float(metrics["recall"])
    return (
        float(accuracy >= ACCURACY_TARGET and recall >= RECALL_TARGET),
        min(accuracy / ACCURACY_TARGET, recall / RECALL_TARGET),
        min(accuracy, recall),
        float(metrics["f1"]),
        float(metrics["precision"]),
    )


def _score_windows(
    model: object,
    tokenizer: object,
    texts: list[str],
    *,
    max_length: int,
    stride: int,
    batch_size: int,
    device: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Score all overlapping token windows and retain their source row."""
    import torch

    architecture = model.architecture  # type: ignore[attr-defined]
    query_count = len(QUERY_TOKENS) if architecture == "query_tokens" else 0
    content_length = max_length - query_count
    if content_length <= stride:
        raise ValueError("max_length must exceed query tokens and stride")

    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    owners: list[int] = []
    truncated_examples = 0
    for owner, text in enumerate(texts):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=content_length,
            stride=stride,
            return_overflowing_tokens=True,
        )
        windows = encoded["input_ids"]
        masks = encoded["attention_mask"]
        if windows and isinstance(windows[0], int):
            windows = [windows]
            masks = [masks]
        if len(windows) > 1:
            truncated_examples += 1
        input_ids.extend(windows)
        attention_masks.extend(masks)
        owners.extend([owner] * len(windows))

    method_chunks = []
    task_chunks = []
    query_ids = tokenizer.convert_tokens_to_ids(list(QUERY_TOKENS)) if query_count else []
    model.eval()  # type: ignore[attr-defined]
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for start in range(0, len(input_ids), batch_size):
            batch = tokenizer.pad(
                {
                    "input_ids": input_ids[start:start + batch_size],
                    "attention_mask": attention_masks[start:start + batch_size],
                },
                padding=True,
                return_tensors="pt",
            )
            if query_count:
                queries = torch.tensor(query_ids, dtype=torch.long)[None, :].expand(
                    len(batch["input_ids"]), -1,
                )
                batch["input_ids"] = torch.cat((batch["input_ids"], queries), dim=1)
                batch["attention_mask"] = torch.cat((
                    batch["attention_mask"], torch.ones_like(queries),
                ), dim=1)
            batch = {name: value.to(device) for name, value in batch.items()}
            method_logits, task_logits = model(**batch)  # type: ignore[operator]
            method_chunks.append(method_logits.sigmoid().float().cpu().numpy())
            task_chunks.append(task_logits.sigmoid().float().cpu().numpy())
    elapsed = time.perf_counter() - started
    return (
        np.concatenate(method_chunks),
        np.concatenate(task_chunks),
        np.asarray(owners),
        {
            "examples": len(texts),
            "windows": len(input_ids),
            "truncated_examples": truncated_examples,
            "inference_ms": round(elapsed * 1000, 3),
        },
    )


def _candidate_scores(
    method_windows: np.ndarray,
    task_windows: np.ndarray,
    owners: np.ndarray,
    examples: int,
) -> dict[str, np.ndarray]:
    candidates = {}
    for method_mode in ("first", "max", "mean", "top2_mean"):
        methods = _aggregate(method_windows, owners, examples, method_mode)
        for task_mode in ("max", "mean"):
            tasks = _aggregate(task_windows[:, None], owners, examples, task_mode)[:, 0]
            for task_threshold in (0.0, 0.15, 0.25, 0.35, 0.45):
                gated = methods.copy()
                gated[tasks < task_threshold] = 0.0
                name = f"{method_mode}-task-{task_mode}-{task_threshold:.2f}"
                candidates[name] = gated
    return candidates


def _select(
    candidates: dict[str, np.ndarray],
    expected: np.ndarray,
) -> tuple[tuple[str, ...], tuple[float, ...], dict[str, Any]]:
    names = []
    thresholds = []
    diagnostics = {}
    for index, label in enumerate(METHOD_LABELS):
        best: tuple[tuple[float, ...], str, float, dict[str, float | int]] | None = None
        for name, scores in candidates.items():
            threshold = _select_threshold(
                scores[:, index],
                expected[:, index],
                accuracy_target=ACCURACY_TARGET,
                recall_target=RECALL_TARGET,
            )
            metrics = _binary_metrics(scores[:, index], expected[:, index], threshold)
            candidate = (_metric_key(metrics), name, threshold, metrics)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            raise RuntimeError(f"no window strategy selected for {label}")
        _, name, threshold, metrics = best
        names.append(name)
        thresholds.append(threshold)
        diagnostics[label] = {"strategy": name, "threshold": threshold, "metrics": metrics}
    return tuple(names), tuple(thresholds), diagnostics


def _selected(candidates: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.stack([candidates[name][:, index] for index, name in enumerate(names)], axis=1)


def _report(
    scores: np.ndarray,
    expected: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    predicted = scores >= np.asarray(thresholds)[None, :]
    per_label = {
        label: _binary_metrics(scores[:, index], expected[:, index], thresholds[index])
        for index, label in enumerate(METHOD_LABELS)
    }
    return {
        "examples": len(expected),
        "exact_match": float(np.mean(np.all(predicted == expected, axis=1))),
        "micro_accuracy": float(np.mean(predicted == expected)),
        "per_label": per_label,
        "gate": {
            "accuracy_target": ACCURACY_TARGET,
            "recall_target": RECALL_TARGET,
            "all_labels_pass": all(_metric_key(metrics)[0] for metrics in per_label.values()),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    import torch

    args = _parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    model, tokenizer, metadata = _load_frozen(args.checkpoint, device)
    max_length = int(metadata["parameters"]["max_length"])
    calibration = _load_partition(args.data_root, "calibration")
    evaluation = _load_partition(args.data_root, "evaluation")

    cal_windows, cal_tasks, cal_owners, cal_stats = _score_windows(
        model,
        tokenizer,
        [row.text for row in calibration],
        max_length=max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
    )
    cal_candidates = _candidate_scores(cal_windows, cal_tasks, cal_owners, len(calibration))
    names, thresholds, selection = _select(cal_candidates, _targets(calibration))

    eval_windows, eval_tasks, eval_owners, eval_stats = _score_windows(
        model,
        tokenizer,
        [row.text for row in evaluation],
        max_length=max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
    )
    eval_candidates = _candidate_scores(
        eval_windows, eval_tasks, eval_owners, len(evaluation),
    )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "stride": args.stride,
        "selection": selection,
        "calibration_windows": cal_stats,
        "evaluation_windows": eval_stats,
        "calibration": _report(
            _selected(cal_candidates, names), _targets(calibration), thresholds,
        ),
        "evaluation": _report(
            _selected(eval_candidates, names), _targets(evaluation), thresholds,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "exact_match": report["evaluation"]["exact_match"],
        "gate": report["evaluation"]["gate"],
        "windows": eval_stats,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
