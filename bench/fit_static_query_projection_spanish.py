"""Fit a regularized Spanish query-projection residual in the v2 space.

Unlike token-row adaptation, this experiment learns shared directions from
Spanish queries while keeping every passage vector and stored index unchanged.
The dense residual starts at zero, is norm-bounded after every update, and is
selected only by source-family validation retrieval.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterator, Sequence

import numpy as np
from tokenizers import Tokenizer

try:
    from bench.fit_static_openai_contrastive_spanish import grouped_retrieval
    from bench.fit_static_qwen3_spanish import (
        _load_artifact,
        _quantize_rows,
        _sha256,
        _static_vectors,
        _token_rows,
        load_corpus,
    )
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    from fit_static_openai_contrastive_spanish import grouped_retrieval
    from fit_static_qwen3_spanish import (
        _load_artifact,
        _quantize_rows,
        _sha256,
        _static_vectors,
        _token_rows,
        load_corpus,
    )


def _batches(
    records: Sequence[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
) -> Iterator[list[int]]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if record["split"] == "train":
            groups.setdefault(str(record["source"]), []).append(index)
    batches: list[list[int]] = []
    for indices in groups.values():
        rng.shuffle(indices)
        batches.extend(
            indices[start:start + batch_size]
            for start in range(0, len(indices), batch_size)
            if len(indices[start:start + batch_size]) >= 2
        )
    rng.shuffle(batches)
    yield from batches


def clamp_projection_delta(delta: Any, base: Any, maximum_ratio: float) -> float:
    """Project a residual into a Frobenius-norm ball around the base matrix."""
    if maximum_ratio <= 0:
        raise ValueError("maximum ratio must be positive")
    import torch

    with torch.no_grad():
        norm = torch.linalg.vector_norm(delta)
        limit = torch.linalg.vector_norm(base) * maximum_ratio
        if norm > limit:
            delta.mul_(limit / torch.clamp(norm, min=1e-12))
        return float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(base))


def fit_projection(
    artifact: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    preservation_weight: float,
    weight_decay: float,
    maximum_delta_ratio: float,
    temperature: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and validation-select a bounded dense query projection."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("projection fitting requires torch") from exc
    if epochs <= 0 or batch_size < 2 or temperature <= 0:
        raise ValueError("invalid training hyperparameters")

    torch.manual_seed(seed)
    rng = random.Random(seed)
    tokenizer = Tokenizer.from_str(bytes(artifact["tokenizer"]).decode("utf-8"))
    query_rows = _token_rows(
        tokenizer, artifact["lut"], [str(record["text"]) for record in records]
    )
    passage_rows = _token_rows(
        tokenizer,
        artifact["lut"],
        [str(record["parallel_text"]) for record in records],
    )
    query_latents, base_queries = _static_vectors(
        query_rows, artifact["A_float"], artifact["B"]
    )
    _, base_passages = _static_vectors(
        passage_rows, artifact["A_float"], artifact["B"]
    )

    base_projection = torch.from_numpy(artifact["B"].copy())
    delta = torch.nn.Parameter(torch.zeros_like(base_projection))
    optimizer = torch.optim.AdamW(
        [delta], learning_rate, weight_decay=weight_decay
    )
    query_latents_t = torch.from_numpy(query_latents)
    base_queries_t = torch.from_numpy(base_queries)
    passages_t = torch.from_numpy(base_passages)

    baseline_validation = grouped_retrieval(
        records, base_queries, base_passages, "validation"
    )
    best_score = baseline_validation["macro_mrr"]
    best_epoch = 0
    best_delta = np.zeros_like(artifact["B"], dtype=np.float32)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        delta_ratio = 0.0
        for batch in _batches(records, batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            predicted = query_latents_t[batch] @ (base_projection + delta)
            predicted = torch.nn.functional.normalize(predicted, dim=1)
            logits = predicted @ passages_t[batch].T / temperature
            retrieval_loss = torch.nn.functional.cross_entropy(
                logits, torch.arange(len(batch))
            )
            preservation = 1.0 - torch.sum(
                predicted * base_queries_t[batch], dim=1
            ).mean()
            loss = retrieval_loss + preservation_weight * preservation
            loss.backward()
            optimizer.step()
            delta_ratio = clamp_projection_delta(
                delta, base_projection, maximum_delta_ratio
            )
            losses.append(float(loss.detach()))

        with torch.no_grad():
            candidate_queries = torch.nn.functional.normalize(
                query_latents_t @ (base_projection + delta), dim=1
            ).numpy()
        validation = grouped_retrieval(
            records, candidate_queries, base_passages, "validation"
        )
        report = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "delta_ratio": delta_ratio,
            "validation_macro_mrr": validation["macro_mrr"],
            "validation_macro_recall@1": validation["macro_recall@1"],
        }
        history.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
        if validation["macro_mrr"] > best_score:
            best_score = validation["macro_mrr"]
            best_epoch = epoch
            with torch.no_grad():
                best_delta = delta.numpy().copy()

    final_projection = artifact["B"] + best_delta
    _, final_queries = _static_vectors(
        query_rows, artifact["A_float"], final_projection
    )
    return final_projection, {
        "selected_epoch": best_epoch,
        "baseline_validation": baseline_validation,
        "selected_validation": grouped_retrieval(
            records, final_queries, base_passages, "validation"
        ),
        "baseline_heldout_test": grouped_retrieval(
            records, base_queries, base_passages, "test"
        ),
        "heldout_test": grouped_retrieval(
            records, final_queries, base_passages, "test"
        ),
        "history": history,
        "records_by_split": dict(Counter(str(row["split"]) for row in records)),
        "objective": {
            "preservation_weight": preservation_weight,
            "weight_decay": weight_decay,
            "maximum_delta_ratio": maximum_delta_ratio,
            "temperature": temperature,
        },
    }


def write_candidate(
    output: Path,
    *,
    source_path: Path,
    corpus_path: Path,
    artifact: dict[str, Any],
    projection: np.ndarray,
    report: dict[str, Any],
) -> None:
    """Write a complete experimental artifact for query-only evaluation."""
    quantized, scales = _quantize_rows(artifact["A_float"])
    meta = dict(artifact["meta_json"])
    meta.update({
        "name": "ken/static-qwen3-r512-v2-es-query-projection-experimental",
        "parent": meta.get("name"),
        "parent_sha256": _sha256(source_path),
        "spanish_corpus_sha256": _sha256(corpus_path),
        "calibration": {
            "method": "bounded dense query-projection residual",
            "report": report,
        },
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        lut=artifact["lut"].astype(np.int32),
        A=quantized,
        A_scale=scales,
        B=projection.astype(np.float32),
        tokenizer=artifact["tokenizer"],
        meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", action="append")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--preservation-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--maximum-delta-ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    records = [
        record for record in load_corpus(args.corpus, seed=args.seed)
        if record.get("language") == "es" and record.get("parallel_text")
    ]
    if args.kind:
        kinds = set(args.kind)
        records = [record for record in records if record.get("kind") in kinds]
    if not records:
        raise SystemExit("corpus filters selected no Spanish pairs")
    artifact = _load_artifact(args.artifact)
    projection, report = fit_projection(
        artifact,
        records,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        preservation_weight=args.preservation_weight,
        weight_decay=args.weight_decay,
        maximum_delta_ratio=args.maximum_delta_ratio,
        temperature=args.temperature,
        seed=args.seed,
    )
    write_candidate(
        args.output,
        source_path=args.artifact,
        corpus_path=args.corpus,
        artifact=artifact,
        projection=projection,
        report=report,
    )
    print(json.dumps({"output": str(args.output), **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
