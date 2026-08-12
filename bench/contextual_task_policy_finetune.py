"""Fine-tune the final layers of a contextual encoder for task policies.

The experiment writes no model artifact by default. It exists to determine
whether a contextual encoder can satisfy the quality gate before any runtime
dependency or packaging decision is made.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from bench.contextual_embedding_benchmark import (
    DEFAULT_CONTEXTUAL_MODEL,
    MAX_POLICY_CARDINALITY,
    _cardinality_class_weights,
    _cardinality_targets,
    _prediction_metrics,
)
from bench.task_policy_hierarchical_head import METHOD_LABELS
from bench.task_policy_multilabel_head import _targets, build_multilabel_corpus


FINETUNE_VERSION = "contextual-task-policy-finetune-v1"


@dataclass(frozen=True)
class FinetuneParameters:
    """Small CPU-friendly fine-tuning configuration."""

    max_length: int = 128
    batch_size: int = 32
    epochs: int = 5
    unfrozen_layers: int = 2
    encoder_learning_rate: float = 0.00002
    head_learning_rate: float = 0.001
    weight_decay: float = 0.001
    cardinality_loss_weight: float = 0.7
    seed: int = 23


def _set_trainable_layers(encoder: object, unfrozen_layers: int) -> tuple[int, int]:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    layers = list(encoder.encoder.layer)
    if not 0 <= unfrozen_layers <= len(layers):
        raise ValueError(
            f"unfrozen_layers must be between 0 and {len(layers)}, got {unfrozen_layers}"
        )
    for layer in layers[len(layers) - unfrozen_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    trainable = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in encoder.parameters())
    return trainable, total


def _predict(
    model: object,
    encoded: dict[str, object],
    *,
    batch_size: int,
) -> list[tuple[str, ...]]:
    import torch

    model.eval()
    method_rows = []
    cardinality_rows = []
    with torch.inference_mode():
        for start in range(0, len(encoded["input_ids"]), batch_size):
            batch = {
                name: value[start:start + batch_size]
                for name, value in encoded.items()
            }
            methods, cardinality = model(**batch)
            method_rows.append(methods)
            cardinality_rows.append(cardinality)
    method_logits = torch.cat(method_rows)
    cardinalities = torch.cat(cardinality_rows).argmax(dim=1)
    predictions = []
    for row, cardinality in zip(method_logits, cardinalities.tolist(), strict=True):
        if cardinality == 0:
            predictions.append(())
            continue
        indices = row.topk(k=min(int(cardinality), len(METHOD_LABELS))).indices.tolist()
        predictions.append(tuple(METHOD_LABELS[index] for index in indices))
    return predictions


def run_finetune(
    *,
    model_name: str = DEFAULT_CONTEXTUAL_MODEL,
    parameters: FinetuneParameters | None = None,
) -> dict[str, object]:
    """Fine-tune selected encoder layers and open holdout only at the end."""
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    parameters = parameters or FinetuneParameters()
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    calibration = build_multilabel_corpus("calibration")
    validation = build_multilabel_corpus("validation")
    holdout = build_multilabel_corpus("holdout")
    examples = calibration + validation + holdout
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_all = tokenizer(
        [f"query: {item.text}" for item in examples],
        max_length=parameters.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    calibration_end = len(calibration)
    validation_end = calibration_end + len(validation)
    encoded_calibration = {
        name: value[:calibration_end] for name, value in encoded_all.items()
    }
    encoded_validation = {
        name: value[calibration_end:validation_end]
        for name, value in encoded_all.items()
    }
    encoded_holdout = {
        name: value[validation_end:] for name, value in encoded_all.items()
    }

    class ContextualTaskPolicyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(0.1)
            self.methods = nn.Linear(hidden_size, len(METHOD_LABELS))
            self.cardinality = nn.Linear(hidden_size, MAX_POLICY_CARDINALITY + 1)

        def forward(self, **encoded: object) -> tuple[object, object]:
            output = self.encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"][..., None].bool()
            pooled = output.masked_fill(~mask, 0.0).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp(min=1)
            pooled = self.dropout(pooled)
            return self.methods(pooled), self.cardinality(pooled)

    model = ContextualTaskPolicyModel()
    trainable_encoder, total_encoder = _set_trainable_layers(
        model.encoder, parameters.unfrozen_layers
    )
    head_parameters = list(model.methods.parameters()) + list(model.cardinality.parameters())
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": parameters.encoder_learning_rate},
            {"params": head_parameters, "lr": parameters.head_learning_rate},
        ],
        weight_decay=parameters.weight_decay,
    )
    calibration_y = torch.from_numpy(_targets(calibration).astype(np.float32))
    calibration_cardinality = torch.tensor(
        _cardinality_targets(calibration), dtype=torch.long
    )
    positive_counts = calibration_y.sum(dim=0)
    method_loss = nn.BCEWithLogitsLoss(
        pos_weight=(len(calibration) - positive_counts) / positive_counts.clamp(min=1)
    )
    cardinality_loss = nn.CrossEntropyLoss(
        weight=torch.from_numpy(
            _cardinality_class_weights(
                calibration_cardinality.numpy(),
                power=parameters.cardinality_balance_power,
            )
        ).to(dtype=torch.float32)
    )
    best_key = (-1.0, -1.0, -1.0)
    best_state = None
    epoch_reports = []
    training_started = time.perf_counter()
    for epoch in range(1, parameters.epochs + 1):
        model.train()
        order = torch.randperm(len(calibration))
        running_loss = 0.0
        for start in range(0, len(calibration), parameters.batch_size):
            indices = order[start:start + parameters.batch_size]
            batch = {
                name: value[indices] for name, value in encoded_calibration.items()
            }
            method_logits, cardinality_logits = model(**batch)
            loss = method_loss(method_logits, calibration_y[indices])
            loss = loss + parameters.cardinality_loss_weight * cardinality_loss(
                cardinality_logits, calibration_cardinality[indices]
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                encoder_parameters + head_parameters, max_norm=1.0
            )
            optimizer.step()
            running_loss += float(loss.detach()) * len(indices)
        validation_predictions = _predict(
            model, encoded_validation, batch_size=parameters.batch_size
        )
        validation_report = _prediction_metrics(validation, validation_predictions)
        key = (
            float(validation_report["exact_match"]),
            float(validation_report["micro_precision"]),
            float(validation_report["micro_recall"]),
        )
        epoch_reports.append({
            "epoch": epoch,
            "loss": running_loss / len(calibration),
            "exact_match": validation_report["exact_match"],
            "macro_f1": validation_report["macro_f1"],
            "micro_precision": validation_report["micro_precision"],
            "micro_recall": validation_report["micro_recall"],
            "false_activations": validation_report["false_activations"],
        })
        if key > best_key:
            best_key = key
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("fine-tuning produced no checkpoint")
    model.load_state_dict(best_state)
    validation_report = _prediction_metrics(
        validation,
        _predict(model, encoded_validation, batch_size=parameters.batch_size),
    )
    holdout_report = _prediction_metrics(
        holdout,
        _predict(model, encoded_holdout, batch_size=parameters.batch_size),
    )
    passed = (
        holdout_report["exact_match"] > 0.95
        and holdout_report["macro_f1"] > 0.95
        and holdout_report["micro_precision"] > 0.95
        and holdout_report["micro_recall"] > 0.95
        and holdout_report["false_activations"] == 0
    )
    return {
        "version": FINETUNE_VERSION,
        "model": model_name,
        "parameters": asdict(parameters),
        "encoder_parameters": {
            "trainable": trainable_encoder,
            "total": total_encoder,
            "trainable_fraction": trainable_encoder / total_encoder,
        },
        "examples": {
            "calibration": len(calibration),
            "validation": len(validation),
            "holdout": len(holdout),
        },
        "training_seconds": training_seconds,
        "epochs": epoch_reports,
        "validation": validation_report,
        "holdout": holdout_report,
        "quality_gate": {"passed": passed},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_CONTEXTUAL_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--unfrozen-layers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_finetune(
        model_name=args.model,
        parameters=FinetuneParameters(
            epochs=args.epochs,
            unfrozen_layers=args.unfrozen_layers,
        ),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()


__all__ = ["FINETUNE_VERSION", "FinetuneParameters", "run_finetune"]
