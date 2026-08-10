"""Fit a Spanish residual into the bundled static Qwen3 token table.

This is an offline calibration tool, not a runtime dependency.  It freezes the
rank-512 projection, learns sparse deltas only for unambiguous token rows, and
uses English/code replay texts as zero-drift anchors.  The transformer teacher
is used only to create cached targets; Infinidev continues to run the resulting
table with NumPy and tokenizers.

The important experimental boundary is the split: every source path belongs to
exactly one of train, validation, or test, so neighboring passages from the same
manual cannot leak into both fitting and evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


logger = logging.getLogger(__name__)

DEFAULT_TEACHER = "Qwen/Qwen3-Embedding-0.6B"
SPLIT_BUCKETS = {"train": range(0, 80), "validation": range(80, 90), "test": range(90, 100)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_key(record: dict[str, Any]) -> str:
    """Return the unit that must not cross a dataset split."""
    source = str(record.get("source", "unknown"))
    explicit = record.get("split_family")
    if explicit:
        return f"{source}\0{explicit}"
    path = str(record.get("path", record.get("id", "unknown")))
    return f"{source}\0{path}"


def split_for_record(record: dict[str, Any], seed: int = 17) -> str:
    """Assign a stable source-path family split using a seeded hash."""
    payload = f"{seed}\0{_family_key(record)}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    for name, buckets in SPLIT_BUCKETS.items():
        if bucket in buckets:
            return name
    raise AssertionError(f"unassigned split bucket {bucket}")


def load_corpus(
    path: Path,
    *,
    maximum: int | None = None,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Load, validate, split, and hash-sample JSONL corpus records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: missing non-empty text")
            record = dict(record)
            record["split"] = split_for_record(record, seed)
            records.append(record)
    if maximum is not None and len(records) > maximum:
        records = sorted(
            records,
            key=lambda item: hashlib.sha256(
                f"{seed}\0{item.get('id', item['text'])}".encode()
            ).digest(),
        )[:maximum]
    return sorted(records, key=lambda item: (str(item["split"]), str(item.get("id", ""))))


def iter_code_replay(root: Path) -> Iterator[dict[str, str]]:
    """Yield compact Ken-shaped English/code anchors from one source tree."""
    suffix_language = {
        ".py": "python",
        ".rs": "rust",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
    }
    ignored = {".git", ".venv", "node_modules", "target", "dist", "build"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in ignored for part in path.parts):
            continue
        language = suffix_language.get(path.suffix.lower())
        if language is None:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        identifiers: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "async def ", "fn ", "function ")):
                head = stripped.split("(", 1)[0].split("{", 1)[0]
                identifiers.append(head.replace(":", ""))
            if len(identifiers) == 12:
                break
        stem = path.stem.replace("_", " ").replace("-", " ")
        tail = " ".join(identifiers)
        text = f"{language} {stem} — {tail}" if tail else f"{language} {stem}"
        yield {"id": relative, "text": text}


def select_replay(root: Path, maximum: int, seed: int) -> list[dict[str, str]]:
    """Choose stable replay anchors without depending on directory order."""
    rows = list(iter_code_replay(root))
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['id']}".encode()).digest(),
    )[:maximum]


def _codesearchnet_pairs(path: Path) -> list[dict[str, str]]:
    """Load normalized docstring/code pairs from one parquet split."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("CodeSearchNet replay requires pandas and pyarrow") from exc
    frame = pd.read_parquet(
        path,
        columns=["func_documentation_string", "func_code_string", "func_code_url"],
    )
    pairs: list[dict[str, str]] = []
    for item in frame.to_dict(orient="records"):
        identity = str(item["func_code_url"])
        documentation = " ".join(str(item["func_documentation_string"]).split())[:700]
        code = str(item["func_code_string"])[:700]
        if len(documentation) >= 16 and len(code) >= 24:
            pairs.append({"id": identity, "query": documentation, "passage": code})
    return pairs


def _codesearchnet_partition(identity: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{identity}".encode()).digest()[:8], "big"
    ) % 5


def select_codesearchnet_replay(
    pairs: Sequence[dict[str, str]], maximum: int, seed: int
) -> list[dict[str, str]]:
    """Select anchors from 80% of a non-test CodeSearchNet split."""
    candidates: list[dict[str, str]] = []
    for pair in pairs:
        if _codesearchnet_partition(pair["id"], seed) == 4:
            continue
        candidates.extend((
            {"id": f"doc:{pair['id']}", "text": pair["query"]},
            {"id": f"code:{pair['id']}", "text": pair["passage"]},
        ))
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['id']}".encode()).digest(),
    )[:maximum]


def select_codesearchnet_gate(
    pairs: Sequence[dict[str, str]], maximum: int, seed: int
) -> list[dict[str, str]]:
    """Select paired retrieval gates from the disjoint remaining 20%."""
    candidates = [
        pair for pair in pairs
        if _codesearchnet_partition(pair["id"], seed) == 4
    ]
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(f"gate\0{seed}\0{row['id']}".encode()).digest(),
    )[:maximum]


def _load_artifact(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name] for name in artifact.files}
    table = arrays["A"]
    if table.dtype == np.int8:
        table = table.astype(np.float32) * arrays["A_scale"].astype(np.float32)[:, None]
    arrays["A_float"] = np.ascontiguousarray(table, dtype=np.float32)
    arrays["B"] = np.ascontiguousarray(arrays["B"], dtype=np.float32)
    arrays["lut"] = np.asarray(arrays["lut"], dtype=np.int64)
    arrays["meta_json"] = json.loads(bytes(arrays["meta"]).decode("utf-8"))
    return arrays


def _token_rows(tokenizer: Any, lut: np.ndarray, texts: Sequence[str]) -> list[np.ndarray]:
    encodings = tokenizer.encode_batch(list(texts), add_special_tokens=False)
    return [lut[np.asarray(encoding.ids, dtype=np.int64)] for encoding in encodings]


def _teacher_cache_matches(
    cache: dict[str, Any],
    *,
    corpus_sha256: str,
    teacher: str,
    record_ids: Sequence[str],
) -> bool:
    try:
        meta = json.loads(str(cache["meta"].item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        meta.get("corpus_sha256") == corpus_sha256
        and meta.get("teacher") == teacher
        and list(cache["ids"].astype(str)) == list(record_ids)
    )


def teacher_targets(
    records: Sequence[dict[str, Any]],
    *,
    corpus_path: Path,
    cache_path: Path,
    teacher_name: str,
    batch_size: int,
) -> np.ndarray:
    """Load teacher targets from an identity-checked cache or compute them."""
    record_ids = [str(record.get("id", index)) for index, record in enumerate(records)]
    corpus_sha256 = _sha256(corpus_path)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cache = {name: cached[name] for name in cached.files}
        if _teacher_cache_matches(
            cache,
            corpus_sha256=corpus_sha256,
            teacher=teacher_name,
            record_ids=record_ids,
        ):
            return np.asarray(cache["vectors"], dtype=np.float32)
        raise ValueError(f"teacher cache identity does not match this run: {cache_path}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "teacher target generation requires sentence-transformers in an "
            "offline calibration environment"
        ) from exc

    model = SentenceTransformer(teacher_name, local_files_only=True, device="cpu")
    vectors = model.encode(
        [str(record["text"]) for record in records],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    meta = {
        "corpus_sha256": corpus_sha256,
        "teacher": teacher_name,
        "records": len(records),
        "contract": "symmetric, no prompt, L2-normalized",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        vectors=vectors.astype(np.float16),
        ids=np.asarray(record_ids),
        meta=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    return vectors


def _quantize_rows(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.max(np.abs(table), axis=1) / 127.0
    scales = np.maximum(scales, np.finfo(np.float32).tiny).astype(np.float32)
    quantized = np.clip(np.rint(table / scales[:, None]), -127, 127).astype(np.int8)
    return quantized, scales


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(left * right, axis=1)


def _paired_retrieval_ranks(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    order = np.argsort(-(queries @ passages.T), axis=1)
    return np.asarray([
        int(np.flatnonzero(order[index] == index)[0]) + 1
        for index in range(len(order))
    ])


def _static_vectors(
    rows: Sequence[np.ndarray], table: np.ndarray, projection: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return (
            np.empty((0, table.shape[1]), dtype=np.float32),
            np.empty((0, projection.shape[1]), dtype=np.float32),
        )
    latent = np.stack([
        table[item].sum(axis=0) if len(item) else np.zeros(table.shape[1], dtype=np.float32)
        for item in rows
    ])
    output = latent @ projection
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    return latent.astype(np.float32), output.astype(np.float32)


def parallel_static_targets(
    records: Sequence[dict[str, Any]], artifact: dict[str, Any]
) -> np.ndarray:
    """Place Spanish texts at their exact English translation in v2 space."""
    from tokenizers import Tokenizer

    missing = [str(record.get("id")) for record in records if not record.get("parallel_text")]
    if missing:
        raise ValueError(
            f"parallel-static target requested but {len(missing)} records lack parallel_text"
        )
    tokenizer = Tokenizer.from_str(bytes(artifact["tokenizer"]).decode("utf-8"))
    rows = _token_rows(
        tokenizer,
        artifact["lut"],
        [str(record["parallel_text"]) for record in records],
    )
    _, vectors = _static_vectors(rows, artifact["A_float"], artifact["B"])
    return vectors


def _batch_indices(indices: Sequence[int], batch_size: int) -> Iterator[list[int]]:
    for start in range(0, len(indices), batch_size):
        yield list(indices[start:start + batch_size])


def fit(
    *,
    artifact: dict[str, Any],
    records: Sequence[dict[str, Any]],
    targets: np.ndarray,
    replay: Sequence[dict[str, str]],
    retrieval_gate: Sequence[dict[str, str]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    shrinkage: float,
    maximum_row_delta: float,
    anchor_weight: float,
    anchor_ratio: float,
    minimum_replay_cosine: float,
    minimum_recall5_delta: float,
    minimum_mrr_delta: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit collision-safe sparse row deltas using stochastic cosine loss."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("fitting requires torch in the calibration environment") from exc
    from tokenizers import Tokenizer

    torch.manual_seed(seed)
    rng = random.Random(seed)
    table = artifact["A_float"]
    projection = artifact["B"]
    lut = artifact["lut"]
    tokenizer = Tokenizer.from_str(bytes(artifact["tokenizer"]).decode("utf-8"))
    texts = [str(record["text"]) for record in records]
    rows = _token_rows(tokenizer, lut, texts)
    replay_rows = _token_rows(tokenizer, lut, [row["text"] for row in replay])
    gate_query_rows = _token_rows(
        tokenizer, lut, [row["query"] for row in retrieval_gate]
    )
    gate_passage_rows = _token_rows(
        tokenizer, lut, [row["passage"] for row in retrieval_gate]
    )
    base_latent, base_output = _static_vectors(rows, table, projection)
    replay_latent, replay_output = _static_vectors(replay_rows, table, projection)
    gate_query_latent, gate_query_output = _static_vectors(
        gate_query_rows, table, projection
    )
    gate_passage_latent, gate_passage_output = _static_vectors(
        gate_passage_rows, table, projection
    )
    baseline_gate_ranks = (
        _paired_retrieval_ranks(gate_query_output, gate_passage_output)
        if retrieval_gate else np.empty(0, dtype=np.int64)
    )

    collision_count = np.bincount(lut, minlength=table.shape[0])
    train_indices = [index for index, row in enumerate(records) if row["split"] == "train"]
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
    projection_t = torch.from_numpy(projection)
    base_latent_t = torch.from_numpy(base_latent)
    targets_t = torch.from_numpy(targets)
    replay_latent_t = torch.from_numpy(replay_latent)
    replay_output_t = torch.from_numpy(replay_output)

    def delta_sum(selected_rows: Sequence[np.ndarray]) -> torch.Tensor:
        local = [global_to_local[item] for item in selected_rows]
        lengths = [len(item) for item in local]
        flat = np.concatenate(local) if local else np.empty(0, dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        return delta(torch.from_numpy(flat), torch.from_numpy(offsets))

    history: list[dict[str, float]] = []
    best_delta: np.ndarray | None = None
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        order = train_indices.copy()
        rng.shuffle(order)
        losses: list[float] = []
        for batch in _batch_indices(order, batch_size):
            optimizer.zero_grad(set_to_none=True)
            predicted = (base_latent_t[batch] + delta_sum([rows[i] for i in batch])) @ projection_t
            target = targets_t[batch]
            semantic_loss = 1.0 - torch.nn.functional.cosine_similarity(
                predicted, target, dim=1
            ).mean()

            if replay:
                anchor_count = min(
                    len(replay), max(1, round(len(batch) * anchor_ratio))
                )
                anchor_indices = rng.sample(range(len(replay)), anchor_count)
                anchored = (
                    replay_latent_t[anchor_indices]
                    + delta_sum([replay_rows[i] for i in anchor_indices])
                ) @ projection_t
                anchor_loss = 1.0 - torch.nn.functional.cosine_similarity(
                    anchored, replay_output_t[anchor_indices], dim=1
                ).mean()
            else:
                anchor_loss = semantic_loss.new_zeros(())
            loss = semantic_loss + anchor_weight * anchor_loss
            loss.backward()
            optimizer.step()

            local_rows = np.unique(np.concatenate([global_to_local[rows[i]] for i in batch]))
            local_rows = local_rows[local_rows < len(adaptable)]
            if len(local_rows):
                with torch.no_grad():
                    selected = torch.from_numpy(local_rows)
                    weights = delta.weight[selected]
                    weights.mul_(1.0 / (1.0 + learning_rate * shrinkage))
                    base_norm = torch.from_numpy(
                        np.linalg.norm(table[adaptable[local_rows]], axis=1)
                    )
                    limit = torch.clamp(base_norm * maximum_row_delta, min=1e-4)
                    norm = torch.linalg.vector_norm(weights, dim=1)
                    factor = torch.minimum(torch.ones_like(norm), limit / torch.clamp(norm, min=1e-12))
                    delta.weight[selected] = weights * factor[:, None]
            losses.append(float(loss.detach()))

        candidate = table.copy()
        with torch.no_grad():
            candidate[adaptable] += delta.weight[:-1].cpu().numpy()
        candidate_latent, candidate_output = _static_vectors(rows, candidate, projection)
        del candidate_latent
        epoch_metrics: dict[str, float] = {"epoch": float(epoch + 1), "loss": float(np.mean(losses))}
        for split in ("train", "validation", "test"):
            indices = [index for index, row in enumerate(records) if row["split"] == split]
            if indices:
                epoch_metrics[f"{split}_teacher_cosine"] = float(
                    np.mean(_cosine_rows(candidate_output[indices], targets[indices]))
                )
                epoch_metrics[f"{split}_baseline_cosine"] = float(
                    np.mean(_cosine_rows(base_output[indices], targets[indices]))
                )
        if replay:
            _, candidate_replay = _static_vectors(replay_rows, candidate, projection)
            epoch_metrics["replay_self_cosine"] = float(
                np.mean(_cosine_rows(candidate_replay, replay_output))
            )
        if retrieval_gate:
            _, candidate_gate_queries = _static_vectors(
                gate_query_rows, candidate, projection
            )
            _, candidate_gate_passages = _static_vectors(
                gate_passage_rows, candidate, projection
            )
            candidate_gate_ranks = _paired_retrieval_ranks(
                candidate_gate_queries, candidate_gate_passages
            )
            epoch_metrics["gate_recall5_delta"] = float(
                np.mean(candidate_gate_ranks <= 5)
                - np.mean(baseline_gate_ranks <= 5)
            )
            epoch_metrics["gate_mrr_delta"] = float(
                np.mean(1.0 / candidate_gate_ranks)
                - np.mean(1.0 / baseline_gate_ranks)
            )
        validation_score = epoch_metrics.get(
            "validation_teacher_cosine", epoch_metrics["train_teacher_cosine"]
        )
        replay_score = epoch_metrics.get("replay_self_cosine", 1.0)
        gate_passed = (
            replay_score >= minimum_replay_cosine
            and epoch_metrics.get("gate_recall5_delta", 0.0) >= minimum_recall5_delta
            and epoch_metrics.get("gate_mrr_delta", 0.0) >= minimum_mrr_delta
        )
        epoch_metrics["replay_gate_passed"] = float(gate_passed)
        if gate_passed and validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch + 1
            with torch.no_grad():
                best_delta = delta.weight[:-1].cpu().numpy().copy()
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

    final_table = table.copy()
    if best_delta is None:
        # Keep a diagnostic candidate, but label it as rejected.  This lets a
        # sweep reveal why a run failed without ever confusing it with a model
        # eligible for integration.
        with torch.no_grad():
            best_delta = delta.weight[:-1].cpu().numpy().copy()
    final_table[adaptable] += best_delta
    report = {
        "adaptable_rows": int(len(adaptable)),
        "ambiguous_touched_rows_frozen": int(len(touched) - len(adaptable)),
        "collision_histogram": {
            str(int(collision)): int(count)
            for collision, count in sorted(Counter(collision_count[touched]).items())
        },
        "history": history,
        "selected_epoch": best_epoch,
        "acceptance": {
            "passed": best_epoch > 0,
            "minimum_replay_cosine": minimum_replay_cosine,
            "minimum_recall5_delta": minimum_recall5_delta,
            "minimum_mrr_delta": minimum_mrr_delta,
            "best_validation_teacher_cosine": (
                best_score if best_epoch > 0 else None
            ),
        },
        "records_by_split": dict(Counter(str(row["split"]) for row in records)),
        "replay_records": len(replay),
        "retrieval_gate_records": len(retrieval_gate),
    }
    return final_table, report


def write_candidate(
    output: Path,
    *,
    source_path: Path,
    artifact: dict[str, Any],
    table: np.ndarray,
    report: dict[str, Any],
    corpus_path: Path,
) -> None:
    """Write a new-space experimental artifact without mutating v2 in place."""
    quantized, scales = _quantize_rows(table)
    meta = dict(artifact["meta_json"])
    meta.update({
        "name": "ken/static-qwen3-r512-v3-es-experimental",
        "parent": meta.get("name"),
        "parent_sha256": _sha256(source_path),
        "spanish_corpus_sha256": _sha256(corpus_path),
        "calibration": {
            "method": "sparse stochastic token-row residual; fixed projection",
            "report": report,
        },
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        lut=artifact["lut"].astype(np.int32),
        A=quantized,
        A_scale=scales,
        B=artifact["B"],
        tokenizer=artifact["tokenizer"],
        meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument(
        "--target-mode",
        choices=("teacher", "parallel-static", "mixed"),
        default="teacher",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=0.5,
        help="teacher fraction for --target-mode mixed",
    )
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument(
        "--replay-codesearchnet",
        type=Path,
        action="append",
        help="non-test CodeSearchNet parquet used only as zero-drift anchors",
    )
    parser.add_argument("--replay-records", type=int, default=2_000)
    parser.add_argument("--retrieval-gate-records", type=int, default=1_000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--language",
        action="append",
        help="retain only this natural language; repeatable",
    )
    parser.add_argument(
        "--kind",
        action="append",
        help="retain only this record kind; repeatable",
    )
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
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

    records = load_corpus(args.corpus, maximum=args.max_records, seed=args.seed)
    if args.language:
        languages = {language.casefold() for language in args.language}
        records = [
            record for record in records
            if str(record.get("language", "")).casefold() in languages
        ]
    if args.kind:
        kinds = set(args.kind)
        records = [record for record in records if record.get("kind") in kinds]
    if not records:
        raise SystemExit("corpus filters selected no records")
    artifact = _load_artifact(args.artifact)
    if args.target_mode in {"parallel-static", "mixed"}:
        records = [record for record in records if record.get("parallel_text")]
        if not records:
            raise SystemExit("the selected corpus contains no aligned parallel_text records")
        parallel = parallel_static_targets(records, artifact)
    else:
        parallel = None
    if args.target_mode in {"teacher", "mixed"}:
        if args.teacher_cache is None:
            parser.error(f"--target-mode {args.target_mode} requires --teacher-cache")
        targets = teacher_targets(
            records,
            corpus_path=args.corpus,
            cache_path=args.teacher_cache,
            teacher_name=args.teacher,
            batch_size=args.teacher_batch_size,
        )
    else:
        assert parallel is not None
        targets = parallel
    if args.target_mode == "mixed":
        if not 0.0 <= args.teacher_weight <= 1.0:
            parser.error("--teacher-weight must be between zero and one")
        assert parallel is not None
        targets = args.teacher_weight * targets + (1.0 - args.teacher_weight) * parallel
        targets /= np.maximum(np.linalg.norm(targets, axis=1, keepdims=True), 1e-12)
    replay: list[dict[str, str]] = []
    retrieval_gate: list[dict[str, str]] = []
    if args.replay_root:
        replay.extend(select_replay(args.replay_root, args.replay_records, args.seed))
    for replay_path in args.replay_codesearchnet or []:
        codesearchnet_pairs = _codesearchnet_pairs(replay_path)
        replay.extend(select_codesearchnet_replay(
            codesearchnet_pairs, args.replay_records, args.seed
        ))
        retrieval_gate.extend(select_codesearchnet_gate(
            codesearchnet_pairs, args.retrieval_gate_records, args.seed
        ))
    table, report = fit(
        artifact=artifact,
        records=records,
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
    report["target_mode"] = args.target_mode
    if args.target_mode == "mixed":
        report["teacher_weight"] = args.teacher_weight
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
