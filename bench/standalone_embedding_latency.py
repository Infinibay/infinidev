"""Measure one embedding checkpoint in an isolated dependency environment.

This probe intentionally uses only Python, PyTorch, Transformers and NumPy so
new checkpoints can be tested without changing Infinidev's locked environment.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


def _rss_mib() -> float:
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(np.ceil(quantile * len(ordered))) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--prefix", default="query: ")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-glob")
    parser.add_argument("--output-vectors", type=Path)
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument(
        "--text",
        default=(
            "Investigá por qué el worker pierde mensajes después de reconectar, "
            "corregí la carrera y preservá el comportamiento sano."
        ),
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    dtype = getattr(torch, args.dtype)
    rss_before = _rss_mib()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        attn_implementation=args.attention,
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    rss_loaded = _rss_mib()

    def encode(texts: list[str]) -> torch.Tensor:
        encoded = tokenizer(
            [f"{args.prefix}{text}" for text in texts],
            max_length=args.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"][..., None].bool()
            pooled = output.masked_fill(~mask, 0.0).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp(min=1)
            if args.dimensions is not None:
                pooled = pooled[:, :args.dimensions]
            return functional.normalize(pooled.to(torch.float32), p=2, dim=1)

    for _ in range(args.warmups):
        encode([args.text])
    timings = []
    for _ in range(args.runs):
        started = time.perf_counter()
        vector = encode([args.text])
        timings.append((time.perf_counter() - started) * 1000.0)
    report = {
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "attention": args.attention,
        "max_length": args.max_length,
        "dimensions": int(vector.shape[1]),
        "load_seconds": load_seconds,
        "rss_loaded_mib": rss_loaded,
        "rss_delta_mib": max(0.0, rss_loaded - rss_before),
        "warm_single_p50_ms": statistics.median(timings),
        "warm_single_p95_ms": _percentile(timings, 0.95),
        "runs": args.runs,
    }
    if args.dataset_glob is not None:
        if args.output_vectors is None:
            raise ValueError("--dataset-glob requires --output-vectors")
        rows = []
        for dataset_path in sorted(glob.glob(args.dataset_glob)):
            rows.extend(
                json.loads(line)
                for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        rows.sort(key=lambda row: int(str(row["id"]).rsplit("-", 1)[1]))
        texts = []
        for row in rows:
            context = [
                str(item).strip()
                for item in row["context_before"]
                if str(item).strip()
            ]
            if context:
                previous = "\n".join(f"- {item}" for item in context)
                texts.append(
                    f"Previous context:\n{previous}\n\n"
                    f"Current user message:\n{row['text']}"
                )
            else:
                texts.append(str(row["text"]))
        corpus_started = time.perf_counter()
        vectors = torch.cat([
            encode(texts[start:start + args.batch_size])
            for start in range(0, len(texts), args.batch_size)
        ]).numpy()
        corpus_seconds = time.perf_counter() - corpus_started
        args.output_vectors.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_vectors,
            vectors=vectors,
            ids=np.asarray([row["id"] for row in rows]),
            model=np.asarray(args.model),
            prefix=np.asarray(args.prefix),
            load_seconds=np.asarray(load_seconds),
            corpus_seconds=np.asarray(corpus_seconds),
            warm_single_p50_ms=np.asarray(statistics.median(timings)),
            warm_single_p95_ms=np.asarray(_percentile(timings, 0.95)),
            rss_delta_mib=np.asarray(max(0.0, rss_loaded - rss_before)),
        )
        report["corpus"] = {
            "examples": len(rows),
            "seconds": corpus_seconds,
            "examples_per_second": len(rows) / max(corpus_seconds, 1e-9),
            "output_vectors": str(args.output_vectors),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
