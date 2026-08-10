"""Build a Spanish-query adapter for the static Qwen3 embedding table.

The base table remains untouched for passages and non-Spanish queries.  A
Bernoulli token-language model selects Spanish queries, then a sparse residual
learned offline is added before the fixed projection.  This avoids changing
stored code vectors while improving cross-language query-to-code retrieval.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tokenizers import Tokenizer

from fit_static_qwen3_spanish import _load_artifact, _quantize_rows, load_corpus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_frequency(
    tokenizer: Tokenizer, texts: Sequence[str], vocab_size: int
) -> np.ndarray:
    frequency = np.zeros(vocab_size, dtype=np.int64)
    for encoding in tokenizer.encode_batch(list(texts), add_special_tokens=False):
        if encoding.ids:
            frequency[np.unique(np.asarray(encoding.ids, dtype=np.int64))] += 1
    return frequency


def fit_log_odds(
    tokenizer: Tokenizer,
    spanish: Sequence[str],
    english: Sequence[str],
    *,
    vocab_size: int,
    alpha: float = 1.0,
    clip: float = 4.0,
) -> np.ndarray:
    """Fit clipped Bernoulli NB log odds for Spanish versus English."""
    es_df = _document_frequency(tokenizer, spanish, vocab_size)
    en_df = _document_frequency(tokenizer, english, vocab_size)
    es_logit = np.log((es_df + alpha) / (len(spanish) - es_df + alpha))
    en_logit = np.log((en_df + alpha) / (len(english) - en_df + alpha))
    return np.clip(es_logit - en_logit, -clip, clip).astype(np.float32)


def language_scores(
    tokenizer: Tokenizer, texts: Sequence[str], log_odds: np.ndarray
) -> np.ndarray:
    """Score texts with length-stabilized unique-token evidence."""
    scores = []
    for encoding in tokenizer.encode_batch(list(texts), add_special_tokens=False):
        ids = np.unique(np.asarray(encoding.ids, dtype=np.int64))
        scores.append(
            float(log_odds[ids].sum() / np.sqrt(max(len(ids), 1)))
            if len(ids) else -float("inf")
        )
    return np.asarray(scores, dtype=np.float32)


def conservative_threshold(english_scores: np.ndarray, maximum_fpr: float) -> float:
    """Choose the smallest threshold whose empirical English FPR is bounded."""
    if not 0.0 <= maximum_fpr < 1.0:
        raise ValueError("maximum_fpr must be in [0, 1)")
    quantile = np.quantile(english_scores, 1.0 - maximum_fpr, method="higher")
    return float(np.nextafter(np.float32(quantile), np.float32(np.inf)))


def _metrics(
    spanish_scores: np.ndarray, english_scores: np.ndarray, threshold: float
) -> dict[str, float]:
    return {
        "spanish_recall": float(np.mean(spanish_scores >= threshold)),
        "english_false_positive_rate": float(np.mean(english_scores >= threshold)),
        "balanced_accuracy": float(
            0.5 * (
                np.mean(spanish_scores >= threshold)
                + np.mean(english_scores < threshold)
            )
        ),
        "spanish_score_median": float(np.median(spanish_scores)),
        "english_score_median": float(np.median(english_scores)),
    }


def _mconala_intents(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        str(row["rewritten_intent"])
        for row in rows
        if row.get("rewritten_intent") and row.get("snippet")
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    base = _load_artifact(args.artifact)
    candidate = _load_artifact(args.candidate)
    if not np.array_equal(base["lut"], candidate["lut"]):
        raise ValueError("candidate LUT differs from its base artifact")
    if not np.array_equal(base["B"], candidate["B"]):
        raise ValueError("candidate projection differs from its base artifact")

    tokenizer = Tokenizer.from_str(bytes(base["tokenizer"]).decode("utf-8"))
    records = [
        record for record in load_corpus(args.corpus, seed=args.seed)
        if record.get("parallel_text")
    ]
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }
    train = by_split["train"]
    log_odds = fit_log_odds(
        tokenizer,
        [str(record["text"]) for record in train],
        [str(record["parallel_text"]) for record in train],
        vocab_size=len(base["lut"]),
        alpha=args.alpha,
        clip=args.clip,
    )

    split_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, split_records in by_split.items():
        split_scores[split] = (
            language_scores(tokenizer, [str(row["text"]) for row in split_records], log_odds),
            language_scores(
                tokenizer,
                [str(row["parallel_text"]) for row in split_records],
                log_odds,
            ),
        )
    threshold = conservative_threshold(
        split_scores["validation"][1], args.maximum_english_fpr
    )
    metrics = {
        split: _metrics(es_scores, en_scores, threshold)
        for split, (es_scores, en_scores) in split_scores.items()
    }
    if args.mconala:
        mconala_scores = language_scores(
            tokenizer, _mconala_intents(args.mconala), log_odds
        )
        metrics["mconala"] = {
            "spanish_recall": float(np.mean(mconala_scores >= threshold)),
            "score_median": float(np.median(mconala_scores)),
        }

    delta = candidate["A_float"] - base["A_float"]
    delta_norm = np.linalg.norm(delta, axis=1)
    rows = np.flatnonzero(delta_norm > args.minimum_delta_norm).astype(np.int32)
    quantized, scales = _quantize_rows(delta[rows])
    meta = {
        "name": "ken/static-qwen3-r512-v2-es-query-adapter",
        "parent_name": base["meta_json"].get("name"),
        "parent_sha256": _sha256(args.artifact),
        "candidate_sha256": _sha256(args.candidate),
        "corpus_sha256": _sha256(args.corpus),
        "selector": {
            "method": "clipped Bernoulli token log odds; unique-token sum / sqrt(n)",
            "threshold": threshold,
            "maximum_english_fpr": args.maximum_english_fpr,
            "alpha": args.alpha,
            "clip": args.clip,
            "metrics": metrics,
        },
        "residual_rows": int(len(rows)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rows=rows,
        delta=quantized,
        delta_scale=scales,
        language_log_odds=log_odds.astype(np.float16),
        language_threshold=np.asarray(threshold, dtype=np.float32),
        meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8),
    )
    return {"output": str(args.output), **meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mconala", type=Path)
    parser.add_argument("--maximum-english-fpr", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=4.0)
    parser.add_argument("--minimum-delta-norm", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
