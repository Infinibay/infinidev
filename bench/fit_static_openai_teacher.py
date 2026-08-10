"""Distil a cached OpenAI embedding teacher into Ken's static architecture.

The first-stage student is deliberately mathematical and auditable: keep the
well-covered Qwen tokenizer table as a 512-dimensional text feature map, then
fit a new ridge projection into the OpenAI teacher space.  Training uses only
source-family train rows.  Validation and test rows never influence the fitted
projection and include bilingual paired-retrieval gates.

This script performs no API calls.  It consumes the resumable SQLite cache made
by ``collect_openai_embedding_teacher.py``, so repeated fitting is free.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from typing import Any, Sequence

import numpy as np
from tokenizers import Tokenizer

try:
    from bench.collect_openai_embedding_teacher import (
        DEFAULT_DIMENSIONS,
        DEFAULT_MODEL,
        _digest,
    )
    from bench.fit_static_qwen3_spanish import (
        _load_artifact,
        _quantize_rows,
        _sha256,
        _static_vectors,
        _token_rows,
        load_corpus,
    )
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    from collect_openai_embedding_teacher import (
        DEFAULT_DIMENSIONS,
        DEFAULT_MODEL,
        _digest,
    )
    from fit_static_qwen3_spanish import (
        _load_artifact,
        _quantize_rows,
        _sha256,
        _static_vectors,
        _token_rows,
        load_corpus,
    )


@dataclass(frozen=True)
class CachedExample:
    """A split-aware text paired with one cached teacher vector."""

    record_id: str
    field: str
    text: str
    split: str
    target: np.ndarray


def load_cached_examples(
    records: Sequence[dict[str, Any]],
    cache_path: Path,
    *,
    fields: tuple[str, ...],
    model: str,
    dimensions: int,
) -> list[CachedExample]:
    """Join corpus rows to a cache and reject partial or wrong-space data."""
    connection = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        expected = {"model": model, "dimensions": str(dimensions)}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"teacher cache identity mismatch: expected {expected}, got {metadata}"
            )
        examples: list[CachedExample] = []
        missing: list[str] = []
        for index, record in enumerate(records):
            record_id = str(record.get("id", index))
            for field in fields:
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                text = value.strip()
                digest = _digest(text, model=model, dimensions=dimensions)
                row = connection.execute(
                    "SELECT vector FROM embeddings WHERE digest = ?", (digest,)
                ).fetchone()
                if row is None:
                    missing.append(f"{record_id}:{field}")
                    continue
                vector = np.frombuffer(row[0], dtype="<f4").copy()
                if vector.shape != (dimensions,) or not np.all(np.isfinite(vector)):
                    raise ValueError(
                        f"invalid cached vector for {record_id}:{field}: {vector.shape}"
                    )
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-12:
                    raise ValueError(f"zero cached vector for {record_id}:{field}")
                examples.append(CachedExample(
                    record_id=record_id,
                    field=field,
                    text=text,
                    split=str(record["split"]),
                    target=(vector / norm).astype(np.float32),
                ))
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(
                f"teacher cache is missing {len(missing)} requested texts ({preview})"
            )
        return examples
    finally:
        connection.close()


def fit_ridge_projection(
    latents: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    *,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a reduced-rank projection to centered teacher vectors."""
    if penalty <= 0:
        raise ValueError("ridge penalty must be positive")
    train_latents = np.asarray(latents[train_indices], dtype=np.float64)
    train_targets = np.asarray(targets[train_indices], dtype=np.float64)
    target_center = train_targets.mean(axis=0)
    centered = train_targets - target_center
    gram = train_latents.T @ train_latents
    cross = train_latents.T @ centered
    projection = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0], dtype=np.float64),
        cross,
    )
    return projection.astype(np.float32), target_center.astype(np.float32)


def ridge_sufficient_statistics(
    latents: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    *,
    chunk_size: int = 8_192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ridge Gram/cross matrices once with bounded working memory."""
    if not len(train_indices) or chunk_size <= 0:
        raise ValueError("training indices and chunk size must be non-empty")
    target_center = np.zeros(targets.shape[1], dtype=np.float64)
    for start in range(0, len(train_indices), chunk_size):
        indices = train_indices[start:start + chunk_size]
        target_center += np.asarray(targets[indices], dtype=np.float64).sum(axis=0)
    target_center /= len(train_indices)

    gram = np.zeros((latents.shape[1], latents.shape[1]), dtype=np.float64)
    cross = np.zeros((latents.shape[1], targets.shape[1]), dtype=np.float64)
    for start in range(0, len(train_indices), chunk_size):
        indices = train_indices[start:start + chunk_size]
        selected_latents = np.asarray(latents[indices], dtype=np.float64)
        selected_targets = (
            np.asarray(targets[indices], dtype=np.float64) - target_center
        )
        gram += selected_latents.T @ selected_latents
        cross += selected_latents.T @ selected_targets
    return gram, cross, target_center.astype(np.float32)


def projections_from_statistics(
    gram: np.ndarray,
    cross: np.ndarray,
    penalties: Sequence[float],
) -> list[np.ndarray]:
    """Solve a family of ridge projections from shared sufficient statistics."""
    if any(penalty <= 0 for penalty in penalties):
        raise ValueError("ridge penalties must be positive")
    identity = np.eye(gram.shape[0], dtype=np.float64)
    return [
        np.linalg.solve(gram + penalty * identity, cross).astype(np.float32)
        for penalty in penalties
    ]


def _centered_targets(targets: np.ndarray, center: np.ndarray) -> np.ndarray:
    centered = targets - center
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(left * right, axis=1)


def _paired_ranks(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    order = np.argsort(-(queries @ passages.T), axis=1)
    return np.asarray([
        int(np.flatnonzero(order[index] == index)[0]) + 1
        for index in range(len(order))
    ])


def _retrieval_metrics(ranks: np.ndarray) -> dict[str, float]:
    if not len(ranks):
        return {}
    return {
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def evaluate_candidate(
    examples: Sequence[CachedExample],
    student_vectors: np.ndarray,
    targets: np.ndarray,
    target_center: np.ndarray,
    *,
    maximum_retrieval_pairs: int = 2_048,
) -> dict[str, Any]:
    """Measure teacher imitation and paired bilingual retrieval by split."""
    centered_targets = _centered_targets(targets, target_center)
    report: dict[str, Any] = {"splits": {}}
    for split in ("train", "validation", "test"):
        indices = np.asarray([
            index for index, example in enumerate(examples) if example.split == split
        ])
        if not len(indices):
            continue
        report["splits"][split] = {
            "examples": int(len(indices)),
            "teacher_cosine_mean": float(np.mean(_cosine(
                student_vectors[indices], centered_targets[indices]
            ))),
            "teacher_cosine_p10": float(np.quantile(_cosine(
                student_vectors[indices], centered_targets[indices]
            ), 0.10)),
        }

        by_record: dict[str, dict[str, int]] = {}
        for index in indices:
            example = examples[int(index)]
            by_record.setdefault(example.record_id, {})[example.field] = int(index)
        available_pairs = [
            fields for fields in by_record.values()
            if "text" in fields and "parallel_text" in fields
        ]
        pairs = sorted(
            available_pairs,
            key=lambda fields: hashlib.sha256(
                examples[fields["text"]].record_id.encode()
            ).digest(),
        )[:maximum_retrieval_pairs]
        if pairs:
            query_indices = np.asarray([pair["text"] for pair in pairs])
            passage_indices = np.asarray([pair["parallel_text"] for pair in pairs])
            report["splits"][split]["bilingual_pairs"] = len(pairs)
            report["splits"][split]["bilingual_pairs_available"] = len(
                available_pairs
            )
            report["splits"][split]["student_bilingual_retrieval"] = _retrieval_metrics(
                _paired_ranks(
                    student_vectors[query_indices], student_vectors[passage_indices]
                )
            )
            report["splits"][split]["teacher_bilingual_retrieval"] = _retrieval_metrics(
                _paired_ranks(targets[query_indices], targets[passage_indices])
            )
    return report


def fit_candidate(
    artifact: dict[str, Any],
    examples: Sequence[CachedExample],
    *,
    ridge_penalties: tuple[float, ...],
    refine_epochs: int,
    refine_batch_size: int,
    refine_learning_rate: float,
    maximum_row_delta: float,
    seed: int,
    maximum_selection_examples: int = 8_192,
    maximum_retrieval_pairs: int = 2_048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Select ridge strength on validation and evaluate once on test."""
    tokenizer = Tokenizer.from_str(bytes(artifact["tokenizer"]).decode("utf-8"))
    rows = _token_rows(
        tokenizer, artifact["lut"], [example.text for example in examples]
    )
    latents, _ = _static_vectors(rows, artifact["A_float"], artifact["B"])
    targets = np.asarray([example.target for example in examples], dtype=np.float32)
    print(json.dumps({
        "stage": "static_features",
        "examples": len(examples),
        "latent_dimensions": int(latents.shape[1]),
    }), flush=True)
    train_indices = np.asarray([
        index for index, example in enumerate(examples) if example.split == "train"
    ])
    validation_indices = np.asarray([
        index for index, example in enumerate(examples) if example.split == "validation"
    ])
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("training and validation splits must both be non-empty")
    if maximum_selection_examples <= 0 or maximum_retrieval_pairs <= 0:
        raise ValueError("evaluation limits must be positive")

    selection_rng = np.random.default_rng(seed)
    if len(validation_indices) > maximum_selection_examples:
        validation_selection = np.sort(selection_rng.choice(
            validation_indices,
            size=maximum_selection_examples,
            replace=False,
        ))
    else:
        validation_selection = validation_indices
    gram, cross, shared_center = ridge_sufficient_statistics(
        latents, targets, train_indices
    )
    print(json.dumps({
        "stage": "ridge_statistics",
        "training_examples": int(len(train_indices)),
    }), flush=True)
    projections = projections_from_statistics(gram, cross, ridge_penalties)
    centered_validation = _centered_targets(
        targets[validation_selection], shared_center
    )

    candidates: list[dict[str, float]] = []
    best_score = -float("inf")
    best_projection: np.ndarray | None = None
    best_center: np.ndarray | None = None
    best_penalty = 0.0
    for penalty, projection in zip(ridge_penalties, projections, strict=True):
        center = shared_center
        output = latents[validation_selection] @ projection
        output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
        validation_score = float(np.mean(_cosine(
            output, centered_validation
        )))
        candidates.append({
            "ridge_penalty": penalty,
            "validation_teacher_cosine": validation_score,
        })
        print(json.dumps({
            "stage": "ridge_candidate",
            "penalty": penalty,
            "validation_teacher_cosine": validation_score,
        }), flush=True)
        if validation_score > best_score:
            best_score = validation_score
            best_projection = projection
            best_center = center
            best_penalty = penalty
    assert best_projection is not None and best_center is not None
    student_vectors = latents @ best_projection
    student_vectors /= np.maximum(
        np.linalg.norm(student_vectors, axis=1, keepdims=True), 1e-12
    )
    print(json.dumps({
        "stage": "selected_projection",
        "penalty": best_penalty,
        "evaluated_pairs_per_split": maximum_retrieval_pairs,
    }), flush=True)
    report = evaluate_candidate(
        examples,
        student_vectors,
        targets,
        best_center,
        maximum_retrieval_pairs=maximum_retrieval_pairs,
    )
    report["selection"] = {
        "ridge_penalty": best_penalty,
        "validation_teacher_cosine": best_score,
        "candidates": candidates,
        "validation_examples": int(len(validation_selection)),
    }
    report["examples_by_split"] = dict(Counter(example.split for example in examples))
    report["examples_by_field"] = dict(Counter(example.field for example in examples))
    table = artifact["A_float"]
    if refine_epochs:
        table, best_projection, refinement = refine_token_rows(
            table=table,
            projection=best_projection,
            target_center=best_center,
            rows=rows,
            targets=targets,
            examples=examples,
            ridge_penalty=best_penalty,
            epochs=refine_epochs,
            batch_size=refine_batch_size,
            learning_rate=refine_learning_rate,
            maximum_row_delta=maximum_row_delta,
            lut=artifact["lut"],
            seed=seed,
        )
        report["refinement"] = refinement
        final_latents, final_vectors = _static_vectors(
            rows, table, best_projection
        )
        del final_latents
        report["final"] = evaluate_candidate(
            examples,
            final_vectors,
            targets,
            best_center,
            maximum_retrieval_pairs=maximum_retrieval_pairs,
        )
    return table, best_projection, best_center, report


def refine_token_rows(
    *,
    table: np.ndarray,
    projection: np.ndarray,
    target_center: np.ndarray,
    rows: Sequence[np.ndarray],
    targets: np.ndarray,
    examples: Sequence[CachedExample],
    ridge_penalty: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    maximum_row_delta: float,
    lut: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Alternately refine collision-safe token rows and the dense projection."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("token-row refinement requires torch") from exc

    if epochs < 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("invalid refinement hyperparameters")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_indices = [
        index for index, example in enumerate(examples) if example.split == "train"
    ]
    validation_indices = np.asarray([
        index for index, example in enumerate(examples)
        if example.split == "validation"
    ])
    collision_count = np.bincount(lut, minlength=table.shape[0])
    touched = np.unique(np.concatenate([rows[index] for index in train_indices]))
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
    base_latents, _ = _static_vectors(rows, table, projection)
    base_latents_t = torch.from_numpy(base_latents)
    centered_targets = _centered_targets(targets, target_center).astype(np.float32)
    targets_t = torch.from_numpy(centered_targets)
    projection_t = torch.from_numpy(projection.copy())

    def delta_sum(indices: Sequence[int]) -> torch.Tensor:
        local_rows = [global_to_local[rows[index]] for index in indices]
        lengths = np.asarray([len(item) for item in local_rows], dtype=np.int64)
        flat = (
            np.concatenate(local_rows)
            if local_rows else np.empty(0, dtype=np.int64)
        )
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        return delta(torch.from_numpy(flat), torch.from_numpy(offsets))

    initial_vectors = base_latents @ projection
    initial_vectors /= np.maximum(
        np.linalg.norm(initial_vectors, axis=1, keepdims=True), 1e-12
    )
    initial_validation = float(np.mean(_cosine(
        initial_vectors[validation_indices], centered_targets[validation_indices]
    )))
    best_score = initial_validation
    best_epoch = 0
    best_table = table.copy()
    best_projection = projection.copy()
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        order = train_indices.copy()
        rng.shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            predicted = (
                base_latents_t[batch] + delta_sum(batch)
            ) @ projection_t
            loss = 1.0 - torch.nn.functional.cosine_similarity(
                predicted, targets_t[batch], dim=1
            ).mean()
            loss.backward()
            optimizer.step()

            selected = np.unique(np.concatenate([
                global_to_local[rows[index]] for index in batch
            ]))
            selected = selected[selected < len(adaptable)]
            if len(selected):
                with torch.no_grad():
                    selected_t = torch.from_numpy(selected)
                    weights = delta.weight[selected_t]
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
            candidate_table[adaptable] += delta.weight[:-1].cpu().numpy()
        candidate_latents, _ = _static_vectors(
            rows, candidate_table, projection_t.numpy()
        )
        candidate_projection, _ = fit_ridge_projection(
            candidate_latents,
            targets,
            np.asarray(train_indices),
            penalty=ridge_penalty,
        )
        projection_t = torch.from_numpy(candidate_projection)
        candidate_vectors = candidate_latents @ candidate_projection
        candidate_vectors /= np.maximum(
            np.linalg.norm(candidate_vectors, axis=1, keepdims=True), 1e-12
        )
        validation_score = float(np.mean(_cosine(
            candidate_vectors[validation_indices], centered_targets[validation_indices]
        )))
        epoch_report = evaluate_candidate(
            examples, candidate_vectors, targets, target_center
        )["splits"]["validation"]
        epoch_report.update({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation_teacher_cosine": validation_score,
        })
        history.append(epoch_report)
        print(json.dumps({"refinement": epoch_report}, sort_keys=True), flush=True)
        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_table = candidate_table
            best_projection = candidate_projection

    return best_table, best_projection, {
        "adaptable_rows": int(len(adaptable)),
        "ambiguous_touched_rows_frozen": int(len(touched) - len(adaptable)),
        "initial_validation_teacher_cosine": initial_validation,
        "selected_epoch": best_epoch,
        "selected_validation_teacher_cosine": best_score,
        "history": history,
    }


def write_candidate(
    output: Path,
    *,
    source_path: Path,
    artifact: dict[str, Any],
    table: np.ndarray,
    projection: np.ndarray,
    target_center: np.ndarray,
    teacher_cache: Path,
    corpus_path: Path,
    model: str,
    dimensions: int,
    report: dict[str, Any],
) -> None:
    """Write an experimental static artifact with complete provenance."""
    quantized, scales = _quantize_rows(table)
    meta = {
        "name": "ken/static-openai-te3-large-r512-v1-experimental",
        "dim": dimensions,
        "rank": int(projection.shape[0]),
        "rows": int(quantized.shape[0]),
        "teacher": model,
        "objective": "ridge projection from frozen Ken token-sum features",
        "parent": artifact["meta_json"].get("name"),
        "parent_sha256": _sha256(source_path),
        "teacher_cache_sha256": _sha256(teacher_cache),
        "corpus_sha256": _sha256(corpus_path),
        "target_center_sha256": hashlib.sha256(target_center.tobytes()).hexdigest(),
        "report": report,
    }
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
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fields", nargs="+", default=["text", "parallel_text"])
    parser.add_argument("--teacher", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--ridge-penalties",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    )
    parser.add_argument("--refine-epochs", type=int, default=8)
    parser.add_argument("--refine-batch-size", type=int, default=256)
    parser.add_argument("--refine-learning-rate", type=float, default=0.02)
    parser.add_argument("--maximum-row-delta", type=float, default=0.75)
    parser.add_argument("--maximum-selection-examples", type=int, default=8_192)
    parser.add_argument("--maximum-retrieval-pairs", type=int, default=2_048)
    args = parser.parse_args()

    artifact = _load_artifact(args.artifact)
    records = load_corpus(args.corpus, maximum=args.max_records, seed=args.seed)
    examples = load_cached_examples(
        records,
        args.teacher_cache,
        fields=tuple(args.fields),
        model=args.teacher,
        dimensions=args.dimensions,
    )
    table, projection, center, report = fit_candidate(
        artifact,
        examples,
        ridge_penalties=tuple(args.ridge_penalties),
        refine_epochs=args.refine_epochs,
        refine_batch_size=args.refine_batch_size,
        refine_learning_rate=args.refine_learning_rate,
        maximum_row_delta=args.maximum_row_delta,
        seed=args.seed,
        maximum_selection_examples=args.maximum_selection_examples,
        maximum_retrieval_pairs=args.maximum_retrieval_pairs,
    )
    write_candidate(
        args.output,
        source_path=args.artifact,
        artifact=artifact,
        table=table,
        projection=projection,
        target_center=center,
        teacher_cache=args.teacher_cache,
        corpus_path=args.corpus,
        model=args.teacher,
        dimensions=args.dimensions,
        report=report,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
