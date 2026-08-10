"""Fit a query-only Spanish adapter with contrastive OpenAI distillation.

The student keeps Ken's existing passage vectors.  Only collision-safe token
rows touched by Spanish queries receive sparse residuals.  Training combines a
paired retrieval loss with the OpenAI teacher's in-batch similarity
distribution; validation chooses the epoch by macro MRR across programming
languages, and the held-out test split is evaluated exactly once afterwards.

This script performs no API calls.  It consumes the SQLite teacher cache made
by ``collect_openai_embedding_teacher.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Iterator, Sequence

import numpy as np
from tokenizers import Tokenizer

try:
    from bench.fit_static_openai_teacher import load_cached_examples
    from bench.fit_static_qwen3_spanish import (
        _load_artifact,
        _static_vectors,
        _token_rows,
        fit as _unused_fit,
        load_corpus,
        write_candidate,
    )
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    from fit_static_openai_teacher import load_cached_examples
    from fit_static_qwen3_spanish import (
        _load_artifact,
        _static_vectors,
        _token_rows,
        fit as _unused_fit,
        load_corpus,
        write_candidate,
    )

del _unused_fit


def _ranks(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    order = np.argsort(-(queries @ passages.T), axis=1)
    return np.asarray([
        int(np.flatnonzero(order[index] == index)[0]) + 1
        for index in range(len(order))
    ])


def grouped_retrieval(
    records: Sequence[dict[str, Any]],
    queries: np.ndarray,
    passages: np.ndarray,
    split: str,
) -> dict[str, Any]:
    """Evaluate paired retrieval separately per programming-language source."""
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if record["split"] == split:
            groups.setdefault(str(record["source"]), []).append(index)
    report: dict[str, Any] = {"groups": {}}
    mrr_values: list[float] = []
    recall1_values: list[float] = []
    for source, indices in sorted(groups.items()):
        selected = np.asarray(indices)
        ranks = _ranks(queries[selected], passages[selected])
        metrics = {
            "records": len(indices),
            "recall@1": float(np.mean(ranks <= 1)),
            "recall@5": float(np.mean(ranks <= 5)),
            "recall@10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1.0 / ranks)),
            "median_rank": float(np.median(ranks)),
        }
        report["groups"][source] = metrics
        mrr_values.append(metrics["mrr"])
        recall1_values.append(metrics["recall@1"])
    report["macro_mrr"] = float(np.mean(mrr_values)) if mrr_values else 0.0
    report["macro_recall@1"] = (
        float(np.mean(recall1_values)) if recall1_values else 0.0
    )
    return report


def _batches_by_source(
    records: Sequence[dict[str, Any]], batch_size: int, rng: random.Random
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


def fit_contrastive(
    artifact: dict[str, Any],
    records: Sequence[dict[str, Any]],
    query_teacher: np.ndarray,
    passage_teacher: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    shrinkage: float,
    maximum_row_delta: float,
    teacher_temperature: float,
    student_temperature: float,
    distill_weight: float,
    positive_weight: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit sparse query rows with paired and relational retrieval losses."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("contrastive fitting requires torch") from exc
    if epochs <= 0 or batch_size < 2:
        raise ValueError("epochs must be positive and batch_size at least two")
    if teacher_temperature <= 0 or student_temperature <= 0:
        raise ValueError("temperatures must be positive")

    torch.manual_seed(seed)
    rng = random.Random(seed)
    tokenizer = Tokenizer.from_str(bytes(artifact["tokenizer"]).decode("utf-8"))
    table = artifact["A_float"]
    projection = artifact["B"]
    lut = artifact["lut"]
    query_rows = _token_rows(
        tokenizer, lut, [str(record["text"]) for record in records]
    )
    passage_rows = _token_rows(
        tokenizer, lut, [str(record["parallel_text"]) for record in records]
    )
    query_latent, base_queries = _static_vectors(query_rows, table, projection)
    _, base_passages = _static_vectors(passage_rows, table, projection)

    train_indices = [
        index for index, record in enumerate(records) if record["split"] == "train"
    ]
    collision_count = np.bincount(lut, minlength=table.shape[0])
    touched = np.unique(np.concatenate([query_rows[index] for index in train_indices]))
    adaptable = touched[collision_count[touched] == 1]
    global_to_local = np.full(table.shape[0], len(adaptable), dtype=np.int64)
    global_to_local[adaptable] = np.arange(len(adaptable), dtype=np.int64)

    delta = torch.nn.EmbeddingBag(
        len(adaptable) + 1,
        table.shape[1],
        mode="sum",
        sparse=True,
        include_last_offset=True,
        padding_idx=len(adaptable),
    )
    with torch.no_grad():
        delta.weight.zero_()
    optimizer = torch.optim.SparseAdam(delta.parameters(), lr=learning_rate)
    query_latent_t = torch.from_numpy(query_latent)
    passage_t = torch.from_numpy(base_passages)
    query_teacher_t = torch.from_numpy(query_teacher.astype(np.float32))
    passage_teacher_t = torch.from_numpy(passage_teacher.astype(np.float32))
    projection_t = torch.from_numpy(projection)

    def delta_sum(indices: Sequence[int]) -> Any:
        local = [global_to_local[query_rows[index]] for index in indices]
        lengths = np.asarray([len(rows) for rows in local], dtype=np.int64)
        flat = np.concatenate(local) if local else np.empty(0, dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        return delta(torch.from_numpy(flat), torch.from_numpy(offsets))

    baseline_validation = grouped_retrieval(
        records, base_queries, base_passages, "validation"
    )
    best_score = baseline_validation["macro_mrr"]
    best_epoch = 0
    best_delta = np.zeros((len(adaptable), table.shape[1]), dtype=np.float32)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for batch in _batches_by_source(records, batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            predicted = (
                query_latent_t[batch] + delta_sum(batch)
            ) @ projection_t
            predicted = torch.nn.functional.normalize(predicted, dim=1)
            student_logits = predicted @ passage_t[batch].T / student_temperature
            labels = torch.arange(len(batch))
            paired_loss = torch.nn.functional.cross_entropy(student_logits, labels)

            with torch.no_grad():
                teacher_logits = (
                    query_teacher_t[batch] @ passage_teacher_t[batch].T
                    / teacher_temperature
                )
                teacher_probabilities = torch.softmax(teacher_logits, dim=1)
            relational_loss = torch.nn.functional.kl_div(
                torch.log_softmax(student_logits, dim=1),
                teacher_probabilities,
                reduction="batchmean",
            )
            positive_loss = 1.0 - torch.sum(
                predicted * passage_t[batch], dim=1
            ).mean()
            loss = (
                paired_loss
                + distill_weight * relational_loss
                + positive_weight * positive_loss
            )
            loss.backward()
            optimizer.step()

            selected = np.unique(np.concatenate([
                global_to_local[query_rows[index]] for index in batch
            ]))
            selected = selected[selected < len(adaptable)]
            if len(selected):
                with torch.no_grad():
                    selected_t = torch.from_numpy(selected)
                    weights = delta.weight[selected_t]
                    weights.mul_(1.0 / (1.0 + learning_rate * shrinkage))
                    base_norm = torch.from_numpy(
                        np.linalg.norm(table[adaptable[selected]], axis=1)
                    )
                    limit = torch.clamp(base_norm * maximum_row_delta, min=1e-4)
                    norm = torch.linalg.vector_norm(weights, dim=1)
                    factor = torch.minimum(
                        torch.ones_like(norm), limit / torch.clamp(norm, min=1e-12)
                    )
                    delta.weight[selected_t] = weights * factor[:, None]
            losses.append(float(loss.detach()))

        candidate_table = table.copy()
        with torch.no_grad():
            candidate_delta = delta.weight[:-1].cpu().numpy().copy()
        candidate_table[adaptable] += candidate_delta
        _, candidate_queries = _static_vectors(
            query_rows, candidate_table, projection
        )
        validation = grouped_retrieval(
            records, candidate_queries, base_passages, "validation"
        )
        epoch_report = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation": validation,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        if validation["macro_mrr"] > best_score:
            best_score = validation["macro_mrr"]
            best_epoch = epoch
            best_delta = candidate_delta

    final_table = table.copy()
    final_table[adaptable] += best_delta
    _, final_queries = _static_vectors(query_rows, final_table, projection)
    return final_table, {
        "selected_epoch": best_epoch,
        "adaptable_rows": int(len(adaptable)),
        "ambiguous_touched_rows_frozen": int(len(touched) - len(adaptable)),
        "records_by_split": dict(Counter(str(row["split"]) for row in records)),
        "baseline_validation": baseline_validation,
        "selected_validation": grouped_retrieval(
            records, final_queries, base_passages, "validation"
        ),
        "heldout_test": grouped_retrieval(
            records, final_queries, base_passages, "test"
        ),
        "baseline_heldout_test": grouped_retrieval(
            records, base_queries, base_passages, "test"
        ),
        "history": history,
        "objective": {
            "teacher_temperature": teacher_temperature,
            "student_temperature": student_temperature,
            "distill_weight": distill_weight,
            "positive_weight": positive_weight,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--maximum-row-delta", type=float, default=0.5)
    parser.add_argument("--teacher-temperature", type=float, default=0.05)
    parser.add_argument("--student-temperature", type=float, default=0.05)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--positive-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    records = [
        record for record in load_corpus(args.corpus, seed=args.seed)
        if record.get("language") == "es" and record.get("parallel_text")
    ]
    artifact = _load_artifact(args.artifact)
    examples = load_cached_examples(
        records,
        args.teacher_cache,
        fields=("text", "parallel_text"),
        model="text-embedding-3-large",
        dimensions=1024,
    )
    vectors = {
        (example.record_id, example.field): example.target for example in examples
    }
    query_teacher = np.asarray([
        vectors[(str(record["id"]), "text")] for record in records
    ])
    passage_teacher = np.asarray([
        vectors[(str(record["id"]), "parallel_text")] for record in records
    ])
    table, report = fit_contrastive(
        artifact,
        records,
        query_teacher,
        passage_teacher,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        shrinkage=args.shrinkage,
        maximum_row_delta=args.maximum_row_delta,
        teacher_temperature=args.teacher_temperature,
        student_temperature=args.student_temperature,
        distill_weight=args.distill_weight,
        positive_weight=args.positive_weight,
        seed=args.seed,
    )
    report["target_mode"] = "paired-contrastive-openai-relational"
    write_candidate(
        args.output,
        source_path=args.artifact,
        artifact=artifact,
        table=table,
        report=report,
        corpus_path=args.corpus,
    )
    print(json.dumps({"output": str(args.output), **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
