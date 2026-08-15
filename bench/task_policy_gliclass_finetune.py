"""Fine-tune and evaluate GLiClass on the natural task-policy corpus.

Checkpoint selection, label wording, and thresholds use only the calibration
partition. The evaluation partition is loaded once after those choices have
been frozen and saved with the checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable

import numpy as np

from bench.task_policy_external_review import ExternalReview, load_external_reviews
from bench.task_policy_multilabel_head import METHOD_LABELS


MODEL_NAME = "knowledgator/gliclass-base-v3.0"
RUN_VERSION = "task-policy-gliclass-natural-v1"
ACCURACY_TARGET = 0.95

LABEL_STYLES: dict[str, dict[str, str]] = {
    "concise": {
        "bugfix.root_cause": "Software bug fix",
        "feature.contract_first": "New software feature",
        "refactor.preserve_behavior": "Code refactoring",
        "research.evidence_first": "Technical research",
        "review.read_only": "Code review",
        "performance.measure_first": "Software performance optimization",
    },
    "descriptive": {
        "bugfix.root_cause": "Fix incorrect behavior in existing software",
        "feature.contract_first": "Implement behavior that the software does not have yet",
        "refactor.preserve_behavior": "Reorganize existing code while preserving behavior",
        "research.evidence_first": "Investigate a technical question and report evidence",
        "review.read_only": "Review existing code without changing it",
        "performance.measure_first": "Improve software performance using measurements",
    },
}


@dataclass(frozen=True)
class FinetuneParameters:
    """Training parameters fixed before evaluation is opened."""

    model_name: str = MODEL_NAME
    label_style: str = "auto"
    max_length: int = 512
    batch_size: int = 8
    evaluation_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    positive_weight_power: float = 0.5
    warmup_ratio: float = 0.06
    minimum_precision: float = 0.85
    minimum_recall: float = 0.50
    accuracy_target: float = ACCURACY_TARGET
    seed: int = 41
    bf16: bool = True


def _label_texts(style: str) -> tuple[str, ...]:
    try:
        mapping = LABEL_STYLES[style]
    except KeyError as exc:
        raise ValueError(f"unknown label style: {style}") from exc
    return tuple(mapping[label] for label in METHOD_LABELS)


def _training_rows(
    reviews: Iterable[ExternalReview],
    *,
    label_style: str,
) -> list[dict[str, object]]:
    """Convert reviewed rows to GLiClass multi-label examples."""
    mapping = LABEL_STYLES[label_style]
    all_labels = list(_label_texts(label_style))
    return [
        {
            "text": review.text,
            "all_labels": all_labels.copy(),
            "true_labels": [mapping[policy] for policy in review.policies],
        }
        for review in reviews
    ]


def _targets(reviews: Iterable[ExternalReview]) -> np.ndarray:
    rows = list(reviews)
    positions = {label: index for index, label in enumerate(METHOD_LABELS)}
    result = np.zeros((len(rows), len(METHOD_LABELS)), dtype=np.bool_)
    for row_index, review in enumerate(rows):
        for policy in review.policies:
            result[row_index, positions[policy]] = True
    return result


def _binary_metrics(
    scores: np.ndarray,
    expected: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & expected))
    false_positive = int(np.sum(predicted & ~expected))
    false_negative = int(np.sum(~predicted & expected))
    true_negative = int(np.sum(~predicted & ~expected))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (true_positive + true_negative) / len(expected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(np.sum(expected)),
        "predicted_positive": int(np.sum(predicted)),
    }


def _select_threshold(
    scores: np.ndarray,
    expected: np.ndarray,
    *,
    accuracy_target: float,
    minimum_precision: float,
    minimum_recall: float,
) -> float:
    """Select a supported threshold without rewarding all-negative collapse."""
    candidates = sorted({
        0.0,
        1.0,
        *(float(value) for value in np.arange(0.05, 0.951, 0.025)),
        *(float(value) for value in scores),
        *(float(np.nextafter(value, np.inf)) for value in scores),
    })
    best: tuple[tuple[float, ...], float] | None = None
    for threshold in candidates:
        metrics = _binary_metrics(scores, expected, threshold)
        accuracy = float(metrics["accuracy"])
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        f1 = float(metrics["f1"])
        quality = (
            accuracy >= accuracy_target
            and precision >= minimum_precision
            and recall > 0.0
        )
        key = (
            float(quality),
            float(recall > 0.0),
            accuracy,
            f1,
            precision,
            float(recall >= minimum_recall),
            recall,
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold)
    if best is None:
        raise RuntimeError("threshold calibration produced no candidate")
    return best[1]


def _calibrate_thresholds(
    scores: np.ndarray,
    expected: np.ndarray,
    parameters: FinetuneParameters,
) -> tuple[float, ...]:
    return tuple(
        _select_threshold(
            scores[:, index],
            expected[:, index],
            accuracy_target=parameters.accuracy_target,
            minimum_precision=parameters.minimum_precision,
            minimum_recall=parameters.minimum_recall,
        )
        for index in range(len(METHOD_LABELS))
    )


def _prediction_report(
    scores: np.ndarray,
    expected: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    threshold_array = np.asarray(thresholds)[None, :]
    predicted = scores >= threshold_array
    per_label = {
        label: _binary_metrics(scores[:, index], expected[:, index], thresholds[index])
        for index, label in enumerate(METHOD_LABELS)
    }
    return {
        "examples": len(expected),
        "exact_match": float(np.mean(np.all(predicted == expected, axis=1))),
        "micro_accuracy": float(np.mean(predicted == expected)),
        "predicted_cardinality": float(np.mean(predicted.sum(axis=1))),
        "expected_cardinality": float(np.mean(expected.sum(axis=1))),
        "per_label": per_label,
        "gate": {
            "accuracy_target": ACCURACY_TARGET,
            "all_labels_pass": all(
                float(metrics["accuracy"]) >= ACCURACY_TARGET
                and float(metrics["recall"]) > 0.0
                for metrics in per_label.values()
            ),
        },
    }


def _checkpoint_key(report: dict[str, Any], *, accuracy_target: float) -> tuple[float, ...]:
    metrics = [report["per_label"][label] for label in METHOD_LABELS]
    accuracies = [float(item["accuracy"]) for item in metrics]
    recalls = [float(item["recall"]) for item in metrics]
    precisions = [float(item["precision"]) for item in metrics]
    passing = sum(
        accuracy >= accuracy_target and recall > 0.0
        for accuracy, recall in zip(accuracies, recalls, strict=True)
    )
    return (
        float(passing == len(METHOD_LABELS)),
        passing / len(METHOD_LABELS),
        sum(recall > 0.0 for recall in recalls) / len(METHOD_LABELS),
        min(accuracies),
        float(np.mean(accuracies)),
        min(recalls),
        float(np.mean(recalls)),
        float(np.mean(precisions)),
        float(report["exact_match"]),
    )


def _positive_weights(targets: np.ndarray, power: float) -> np.ndarray:
    if not 0.0 <= power <= 1.0:
        raise ValueError("positive_weight_power must be between 0 and 1")
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    return np.power(negatives / np.maximum(positives, 1), power).astype(np.float32)


def _validate_partitions(
    training: list[ExternalReview],
    calibration: list[ExternalReview],
) -> None:
    training_ids = {row.candidate_id for row in training}
    calibration_ids = {row.candidate_id for row in calibration}
    overlap = sorted(training_ids & calibration_ids)
    if overlap:
        raise ValueError(f"training and calibration overlap: {overlap[:3]}")
    training_repos = {row.repo for row in training}
    calibration_repos = {row.repo for row in calibration}
    repo_overlap = sorted(training_repos & calibration_repos)
    if repo_overlap:
        raise ValueError(
            "training and calibration repositories overlap: " + ", ".join(repo_overlap)
        )


def _load_partition(root: Path, name: str) -> list[ExternalReview]:
    return load_external_reviews(
        root / f"{name}_candidates.jsonl",
        root / f"{name}_reviews.jsonl",
    )


def _configure_multilabel(model: object) -> None:
    """Repair the published checkpoint defaults for six independent logits."""
    for config in (model.config, model.model.config):  # type: ignore[attr-defined]
        config.problem_type = "multi_label_classification"
        config.focal_loss_reduction = "none"


def _dataset(
    reviews: list[ExternalReview],
    *,
    tokenizer: object,
    model: object,
    label_style: str,
    max_length: int,
) -> object:
    from gliclass.data_processing import AugmentationConfig, GLiClassDataset

    return GLiClassDataset(
        _training_rows(reviews, label_style=label_style),
        tokenizer,
        AugmentationConfig(enabled=False),
        max_length=max_length,
        problem_type="multi_label_classification",
        architecture_type=model.config.architecture_type,  # type: ignore[attr-defined]
        add_description=True,
        prompt_first=model.config.prompt_first,  # type: ignore[attr-defined]
        shuffle_labels=False,
    )


def _loader(
    dataset: object,
    *,
    model: object,
    batch_size: int,
    shuffle: bool,
) -> object:
    from torch.utils.data import DataLoader
    from gliclass.data_processing import DataCollatorWithPadding

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(config=model.config),  # type: ignore[attr-defined]
    )


def _move_batch(batch: dict[str, object], device: object) -> dict[str, object]:
    import torch

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _score_reviews(
    model: object,
    tokenizer: object,
    reviews: list[ExternalReview],
    *,
    label_style: str,
    max_length: int,
    batch_size: int,
    device: object,
) -> np.ndarray:
    import torch

    dataset = _dataset(
        reviews,
        tokenizer=tokenizer,
        model=model,
        label_style=label_style,
        max_length=max_length,
    )
    loader = _loader(dataset, model=model, batch_size=batch_size, shuffle=False)
    chunks = []
    model.eval()  # type: ignore[attr-defined]
    with torch.inference_mode():
        for raw_batch in loader:  # type: ignore[union-attr]
            batch = _move_batch(raw_batch, device)
            outputs = model(**batch)  # type: ignore[operator]
            chunks.append(outputs.logits.sigmoid().float().cpu().numpy())
    return np.concatenate(chunks)


def _select_label_style(
    model: object,
    tokenizer: object,
    calibration: list[ExternalReview],
    parameters: FinetuneParameters,
    device: object,
) -> tuple[str, dict[str, Any]]:
    styles = tuple(LABEL_STYLES) if parameters.label_style == "auto" else (parameters.label_style,)
    expected = _targets(calibration)
    diagnostics: dict[str, Any] = {}
    best: tuple[tuple[float, ...], str] | None = None
    for style in styles:
        scores = _score_reviews(
            model,
            tokenizer,
            calibration,
            label_style=style,
            max_length=parameters.max_length,
            batch_size=parameters.evaluation_batch_size,
            device=device,
        )
        thresholds = _calibrate_thresholds(scores, expected, parameters)
        report = _prediction_report(scores, expected, thresholds)
        diagnostics[style] = {"thresholds": thresholds, "report": report}
        key = _checkpoint_key(report, accuracy_target=parameters.accuracy_target)
        if best is None or key > best[0]:
            best = (key, style)
    if best is None:
        raise RuntimeError("label-style selection produced no candidate")
    return best[1], diagnostics


def _save_checkpoint(
    model: object,
    tokenizer: object,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)  # type: ignore[attr-defined]
    tokenizer.save_pretrained(output_dir)  # type: ignore[attr-defined]
    (output_dir / "task_policy_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _train(
    model: object,
    tokenizer: object,
    training: list[ExternalReview],
    calibration: list[ExternalReview],
    *,
    label_style: str,
    parameters: FinetuneParameters,
    output_dir: Path,
    device: object,
) -> tuple[tuple[float, ...], list[dict[str, Any]], int]:
    import torch
    from torch.nn import functional as functional
    from transformers import get_cosine_schedule_with_warmup

    train_dataset = _dataset(
        training,
        tokenizer=tokenizer,
        model=model,
        label_style=label_style,
        max_length=parameters.max_length,
    )
    train_loader = _loader(
        train_dataset,
        model=model,
        batch_size=parameters.batch_size,
        shuffle=True,
    )
    encoder_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():  # type: ignore[attr-defined]
        if not parameter.requires_grad:
            continue
        destination = encoder_parameters if "encoder_model" in name else head_parameters
        destination.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": parameters.encoder_learning_rate},
            {"params": head_parameters, "lr": parameters.head_learning_rate},
        ],
        weight_decay=parameters.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / parameters.gradient_accumulation_steps  # type: ignore[arg-type]
    )
    total_updates = parameters.epochs * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * parameters.warmup_ratio),
        num_training_steps=total_updates,
    )
    targets = _targets(training)
    positive_weights = torch.tensor(
        _positive_weights(targets, parameters.positive_weight_power),
        device=device,
    )
    use_bf16 = parameters.bf16 and getattr(device, "type", str(device)) == "cuda"
    history = []
    best_key: tuple[float, ...] | None = None
    best_thresholds = (0.5,) * len(METHOD_LABELS)
    best_epoch = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, parameters.epochs + 1):
        model.train()  # type: ignore[attr-defined]
        losses = []
        for batch_index, raw_batch in enumerate(train_loader, 1):  # type: ignore[union-attr]
            batch = _move_batch(raw_batch, device)
            target = batch.pop("labels")
            with torch.autocast(
                device_type=getattr(device, "type", str(device)),
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                outputs = model(**batch)  # type: ignore[operator]
                loss = functional.binary_cross_entropy_with_logits(
                    outputs.logits.float(),
                    target.float(),  # type: ignore[union-attr]
                    pos_weight=positive_weights,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}, batch {batch_index}: {loss}"
                )
            losses.append(float(loss.detach()))
            (loss / parameters.gradient_accumulation_steps).backward()
            if (
                batch_index % parameters.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)  # type: ignore[arg-type]
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore[attr-defined]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        calibration_scores = _score_reviews(
            model,
            tokenizer,
            calibration,
            label_style=label_style,
            max_length=parameters.max_length,
            batch_size=parameters.evaluation_batch_size,
            device=device,
        )
        thresholds = _calibrate_thresholds(
            calibration_scores,
            _targets(calibration),
            parameters,
        )
        report = _prediction_report(
            calibration_scores,
            _targets(calibration),
            thresholds,
        )
        key = _checkpoint_key(report, accuracy_target=parameters.accuracy_target)
        epoch_record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "thresholds": thresholds,
            "calibration": report,
        }
        history.append(epoch_record)
        print(json.dumps({
            "epoch": epoch,
            "training_loss": epoch_record["training_loss"],
            "calibration_exact_match": report["exact_match"],
            "calibration_gate": report["gate"],
        }, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_thresholds = thresholds
            best_epoch = epoch
            _save_checkpoint(
                model,
                tokenizer,
                output_dir,
                {
                    "run_version": RUN_VERSION,
                    "parameters": asdict(parameters),
                    "label_style": label_style,
                    "label_texts": dict(zip(METHOD_LABELS, _label_texts(label_style), strict=True)),
                    "thresholds": dict(zip(METHOD_LABELS, thresholds, strict=True)),
                    "selected_epoch": epoch,
                    "calibration": report,
                },
            )
    return best_thresholds, history, best_epoch


def _latency_report(
    model: object,
    tokenizer: object,
    reviews: list[ExternalReview],
    *,
    label_style: str,
    max_length: int,
    device: object,
    samples: int,
) -> dict[str, float | int]:
    import torch

    if samples <= 0:
        return {"samples": 0}
    timings = []
    selected = [reviews[index % len(reviews)] for index in range(samples + 10)]
    for index, review in enumerate(selected):
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        _score_reviews(
            model,
            tokenizer,
            [review],
            label_style=label_style,
            max_length=max_length,
            batch_size=1,
            device=device,
        )
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if index >= 10:
            timings.append(elapsed_ms)
    return {
        "samples": samples,
        "p50_ms": statistics.median(timings),
        "p95_ms": float(np.percentile(timings, 95)),
        "max_ms": max(timings),
    }


def _parse_args() -> argparse.Namespace:
    default_root = Path.home() / "tmp" / "task-policy-natural-split-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "tmp" / "task-policy-gliclass-base-v3-natural-v1",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--label-style", choices=("auto", *LABEL_STYLES), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--latency-samples", type=int, default=100)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("epochs, batch size, and gradient accumulation must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; pass --overwrite-output-dir"
        )

    import torch
    from transformers import AutoTokenizer
    from gliclass import GLiClassModel

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(41)
    requested_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested_device == "auto":
        requested_device = "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    parameters = FinetuneParameters(
        model_name=args.model,
        label_style=args.label_style,
        max_length=args.max_length,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        bf16=not args.no_bf16,
    )
    training = _load_partition(args.data_root, "training")
    calibration = _load_partition(args.data_root, "calibration")
    _validate_partitions(training, calibration)
    print(json.dumps({
        "device": str(device),
        "training_examples": len(training),
        "calibration_examples": len(calibration),
        "evaluation_opened": False,
    }, sort_keys=True), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(parameters.model_name)
    model = GLiClassModel.from_pretrained(parameters.model_name)
    _configure_multilabel(model)
    model.to(device)
    label_style, style_diagnostics = _select_label_style(
        model,
        tokenizer,
        calibration,
        parameters,
        device,
    )
    print(json.dumps({
        "selected_label_style": label_style,
        "style_exact_match": {
            style: details["report"]["exact_match"]
            for style, details in style_diagnostics.items()
        },
    }, sort_keys=True), flush=True)

    thresholds, history, best_epoch = _train(
        model,
        tokenizer,
        training,
        calibration,
        label_style=label_style,
        parameters=parameters,
        output_dir=args.output_dir,
        device=device,
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = GLiClassModel.from_pretrained(args.output_dir)
    _configure_multilabel(model)
    model.to(device)
    frozen_config = json.loads(
        (args.output_dir / "task_policy_config.json").read_text(encoding="utf-8")
    )
    thresholds = tuple(
        float(frozen_config["thresholds"][label]) for label in METHOD_LABELS
    )

    evaluation = _load_partition(args.data_root, "evaluation")
    all_prior_ids = {row.candidate_id for row in training + calibration}
    overlap = sorted(all_prior_ids & {row.candidate_id for row in evaluation})
    if overlap:
        raise ValueError(f"evaluation overlaps earlier partitions: {overlap[:3]}")
    prior_repos = {row.repo for row in training + calibration}
    evaluation_repos = {row.repo for row in evaluation}
    repo_overlap = sorted(prior_repos & evaluation_repos)
    if repo_overlap:
        raise ValueError(
            "evaluation repositories overlap earlier partitions: " + ", ".join(repo_overlap)
        )
    evaluation_scores = _score_reviews(
        model,
        tokenizer,
        evaluation,
        label_style=label_style,
        max_length=parameters.max_length,
        batch_size=parameters.evaluation_batch_size,
        device=device,
    )
    evaluation_report = _prediction_report(
        evaluation_scores,
        _targets(evaluation),
        thresholds,
    )
    latency = _latency_report(
        model,
        tokenizer,
        evaluation,
        label_style=label_style,
        max_length=parameters.max_length,
        device=device,
        samples=args.latency_samples,
    )
    report = {
        "run_version": RUN_VERSION,
        "parameters": asdict(parameters),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "counts": {
            "training": len(training),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
        },
        "selected_label_style": label_style,
        "label_texts": dict(zip(METHOD_LABELS, _label_texts(label_style), strict=True)),
        "style_diagnostics": style_diagnostics,
        "selected_epoch": best_epoch,
        "thresholds": dict(zip(METHOD_LABELS, thresholds, strict=True)),
        "history": history,
        "evaluation": evaluation_report,
        "latency": latency,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(args.output_dir),
        "report": str(report_path),
        "selected_epoch": best_epoch,
        "evaluation_exact_match": evaluation_report["exact_match"],
        "evaluation_gate": evaluation_report["gate"],
        "latency": latency,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
