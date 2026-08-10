"""Distil OpenAI relational signal into a static Spanish query adapter.

OpenAI and Ken use unrelated coordinate systems, so matching coordinates
directly is the wrong objective.  Instead, the OpenAI teacher retrieves nearby
code for each Spanish query.  A weighted centroid of those passages in Ken's
existing space is blended with the exact English translation target.  The
existing sparse stochastic fitter then learns Spanish token residuals while
the production passage space remains unchanged.

This script performs no API calls.  It consumes a complete SQLite cache made by
``collect_openai_embedding_teacher.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tokenizers import Tokenizer

try:
    from bench.fit_static_openai_teacher import load_cached_examples
    from bench.fit_static_qwen3_spanish import (
        _codesearchnet_pairs,
        _load_artifact,
        _static_vectors,
        _token_rows,
        fit,
        load_corpus,
        parallel_static_targets,
        select_codesearchnet_gate,
        select_codesearchnet_replay,
        write_candidate,
    )
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    from fit_static_openai_teacher import load_cached_examples
    from fit_static_qwen3_spanish import (
        _codesearchnet_pairs,
        _load_artifact,
        _static_vectors,
        _token_rows,
        fit,
        load_corpus,
        parallel_static_targets,
        select_codesearchnet_gate,
        select_codesearchnet_replay,
        write_candidate,
    )


def relational_targets(
    query_teacher: np.ndarray,
    code_teacher: np.ndarray,
    code_static: np.ndarray,
    translation_static: np.ndarray,
    *,
    top_k: int,
    temperature: float,
    code_weight: float,
    chunk_size: int = 256,
) -> tuple[np.ndarray, dict[str, float]]:
    """Blend exact translations with teacher-selected static code centroids."""
    if top_k <= 0 or top_k > len(code_teacher):
        raise ValueError("top_k must be between one and the code-pool size")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= code_weight <= 1.0:
        raise ValueError("code_weight must be between zero and one")
    if query_teacher.shape[1] != code_teacher.shape[1]:
        raise ValueError("teacher query and code dimensions differ")
    if code_static.shape[1] != translation_static.shape[1]:
        raise ValueError("static code and translation dimensions differ")

    result = np.empty_like(translation_static, dtype=np.float32)
    selected_scores: list[np.ndarray] = []
    for start in range(0, len(query_teacher), chunk_size):
        stop = min(start + chunk_size, len(query_teacher))
        similarities = query_teacher[start:stop] @ code_teacher.T
        indices = np.argpartition(similarities, -top_k, axis=1)[:, -top_k:]
        scores = np.take_along_axis(similarities, indices, axis=1)
        order = np.argsort(-scores, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        scores = np.take_along_axis(scores, order, axis=1)
        weights = np.exp((scores - scores[:, :1]) / temperature)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        centroids = np.einsum("bk,bkd->bd", weights, code_static[indices])
        centroids /= np.maximum(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12
        )
        blended = (
            (1.0 - code_weight) * translation_static[start:stop]
            + code_weight * centroids
        )
        result[start:stop] = blended / np.maximum(
            np.linalg.norm(blended, axis=1, keepdims=True), 1e-12
        )
        selected_scores.append(scores)
    scores = np.concatenate(selected_scores, axis=0)
    return result, {
        "top1_teacher_cosine_mean": float(np.mean(scores[:, 0])),
        "top1_teacher_cosine_p10": float(np.quantile(scores[:, 0], 0.10)),
        "topk_teacher_cosine_mean": float(np.mean(scores)),
    }


def cached_vectors_by_key(
    records: Sequence[dict[str, Any]], cache_path: Path
) -> dict[tuple[str, str], np.ndarray]:
    """Load cached teacher vectors keyed by corpus record and field."""
    examples = load_cached_examples(
        records,
        cache_path,
        fields=("text", "parallel_text"),
        model="text-embedding-3-large",
        dimensions=1024,
    )
    return {
        (example.record_id, example.field): example.target
        for example in examples
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replay-codesearchnet", type=Path, action="append", default=[]
    )
    parser.add_argument("--replay-records", type=int, default=1_000)
    parser.add_argument("--retrieval-gate-records", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--code-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--maximum-row-delta", type=float, default=0.35)
    parser.add_argument("--anchor-weight", type=float, default=3.0)
    parser.add_argument("--anchor-ratio", type=float, default=1.0)
    parser.add_argument("--minimum-replay-cosine", type=float, default=0.995)
    parser.add_argument("--minimum-recall5-delta", type=float, default=-0.003)
    parser.add_argument("--minimum-mrr-delta", type=float, default=-0.002)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    records = load_corpus(args.corpus, seed=args.seed)
    spanish = [
        record for record in records
        if record.get("language") == "es" and record.get("parallel_text")
    ]
    code = [
        record for record in records
        if str(record.get("source", "")).startswith("codesearchnet_")
        and record["split"] == "train"
        and record.get("parallel_text")
    ]
    if not spanish or not code:
        raise SystemExit("corpus must contain aligned Spanish and training code records")

    artifact = _load_artifact(args.artifact)
    cache = cached_vectors_by_key(records, args.teacher_cache)
    query_teacher = np.asarray([
        cache[(str(record["id"]), "text")] for record in spanish
    ])
    code_teacher = np.asarray([
        cache[(str(record["id"]), "parallel_text")] for record in code
    ])
    tokenizer = Tokenizer.from_str(bytes(artifact["tokenizer"]).decode("utf-8"))
    code_rows = _token_rows(
        tokenizer,
        artifact["lut"],
        [str(record["parallel_text"]) for record in code],
    )
    _, code_static = _static_vectors(
        code_rows, artifact["A_float"], artifact["B"]
    )
    translations = parallel_static_targets(spanish, artifact)
    targets, relational_report = relational_targets(
        query_teacher,
        code_teacher,
        code_static,
        translations,
        top_k=args.top_k,
        temperature=args.temperature,
        code_weight=args.code_weight,
    )

    replay: list[dict[str, str]] = []
    retrieval_gate: list[dict[str, str]] = []
    for path in args.replay_codesearchnet:
        pairs = _codesearchnet_pairs(path)
        replay.extend(select_codesearchnet_replay(
            pairs, args.replay_records, args.seed
        ))
        retrieval_gate.extend(select_codesearchnet_gate(
            pairs, args.retrieval_gate_records, args.seed
        ))
    table, report = fit(
        artifact=artifact,
        records=spanish,
        targets=targets,
        replay=replay,
        retrieval_gate=retrieval_gate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        shrinkage=args.shrinkage,
        maximum_row_delta=args.maximum_row_delta,
        anchor_weight=args.anchor_weight,
        anchor_ratio=args.anchor_ratio,
        minimum_replay_cosine=args.minimum_replay_cosine,
        minimum_recall5_delta=args.minimum_recall5_delta,
        minimum_mrr_delta=args.minimum_mrr_delta,
        seed=args.seed,
    )
    report["target_mode"] = "openai-relational-code-centroid"
    report["relational"] = {
        **relational_report,
        "code_pool_records": len(code),
        "spanish_records": len(spanish),
        "top_k": args.top_k,
        "temperature": args.temperature,
        "code_weight": args.code_weight,
    }
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
