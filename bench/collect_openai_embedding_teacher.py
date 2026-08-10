"""Collect a resumable OpenAI embedding-teacher cache for distillation.

The collector pays for each distinct text once and stores float32 vectors in
SQLite.  Training epochs consume the local cache, never the API.  A dry run
counts tokens and cost before any network request; the live path has a hard
cost ceiling, bounded batches, and full-jitter retries for transient failures.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import sqlite3
import time
from typing import Iterable, Iterator

import httpx
import numpy as np
from dotenv import load_dotenv


DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_DIMENSIONS = 1024
DEFAULT_PRICE_PER_MILLION_TOKENS = 0.13
_API_URL = "https://api.openai.com/v1/embeddings"
_MAX_INPUT_TOKENS = 8192


@dataclass(frozen=True)
class TeacherItem:
    """One unique API input with stable corpus provenance."""

    digest: str
    source_id: str
    field: str
    text: str
    tokens: int


@dataclass(frozen=True)
class CollectionSummary:
    """Cost and size of a prospective or completed collection."""

    records: int
    unique_texts: int
    cached_texts: int
    pending_texts: int
    total_tokens: int
    pending_tokens: int
    estimated_pending_usd: float


class CachedTeacherEmbedder:
    """Embedding interface backed by a complete local teacher cache."""

    def __init__(self, path: Path) -> None:
        self._path = path
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        finally:
            connection.close()
        self.model_name = metadata["model"]
        self._dimensions = int(metadata["dimensions"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        self.space_id = f"teacher-cache:{self.model_name}:{self._dimensions}:{digest}"

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        connection = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        try:
            vectors: list[np.ndarray] = []
            for text in texts:
                normalized = text.strip()
                digest = _digest(
                    normalized,
                    model=self.model_name,
                    dimensions=self._dimensions,
                )
                row = connection.execute(
                    "SELECT vector FROM embeddings WHERE digest = ?", (digest,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"teacher cache is missing text digest {digest[:12]}")
                vectors.append(np.frombuffer(row[0], dtype="<f4").copy())
            return vectors
        finally:
            connection.close()

    def __call__(self, texts: list[str]) -> list[np.ndarray]:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> list[np.ndarray]:
        return self._embed(texts)

    def embed_passages(self, texts: list[str]) -> list[np.ndarray]:
        return self._embed(texts)


def _encoding():
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit(
            "token counting requires tiktoken (installed transitively with litellm)"
        ) from exc
    return tiktoken.get_encoding("cl100k_base")


def _digest(text: str, *, model: str, dimensions: int) -> str:
    identity = f"{model}\0{dimensions}\0{text}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def load_jsonl_items(
    path: Path,
    *,
    fields: tuple[str, ...],
    model: str,
    dimensions: int,
    maximum_records: int | None = None,
    sample_seed: int = 17,
) -> tuple[list[TeacherItem], int]:
    """Hash-sample records, then deduplicate selected string fields."""
    if maximum_records is not None and maximum_records <= 0:
        raise ValueError("maximum_records must be positive")
    encoding = _encoding()
    unique: dict[str, TeacherItem] = {}
    selected: list[tuple[int, int, dict[str, object]]] = []
    with path.open("r", encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"expected an object at {path}:{line_number}")
            identity = str(row.get("id", row.get("text", line_number)))
            priority = int.from_bytes(
                hashlib.sha256(f"{sample_seed}\0{identity}".encode()).digest(), "big"
            )
            if maximum_records is None:
                selected.append((priority, line_number, row))
            else:
                entry = (-priority, -line_number, row)
                if len(selected) < maximum_records:
                    heapq.heappush(selected, entry)
                elif priority < -selected[0][0]:
                    heapq.heapreplace(selected, entry)

    if maximum_records is not None:
        selected = [(-priority, -line_number, row) for priority, line_number, row in selected]
    selected.sort(key=lambda item: (item[0], item[1]))
    for _, line_number, row in selected:
        source_id = str(row.get("id", line_number))
        for field in fields:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            text = value.strip()
            tokens = len(encoding.encode(text))
            if tokens > _MAX_INPUT_TOKENS:
                raise SystemExit(
                    f"{path}:{line_number} field {field!r} has {tokens} tokens; "
                    f"the model limit is {_MAX_INPUT_TOKENS}"
                )
            digest = _digest(text, model=model, dimensions=dimensions)
            unique.setdefault(digest, TeacherItem(
                digest=digest,
                source_id=source_id,
                field=field,
                text=text,
                tokens=tokens,
            ))
    return list(unique.values()), len(selected)


def iter_batches(
    items: Iterable[TeacherItem],
    *,
    maximum_items: int,
    maximum_tokens: int,
) -> Iterator[list[TeacherItem]]:
    """Greedily batch without exceeding either API guardrail."""
    batch: list[TeacherItem] = []
    tokens = 0
    for item in items:
        if batch and (
            len(batch) >= maximum_items or tokens + item.tokens > maximum_tokens
        ):
            yield batch
            batch = []
            tokens = 0
        batch.append(item)
        tokens += item.tokens
    if batch:
        yield batch


def estimate_cost(tokens: int, price_per_million_tokens: float) -> float:
    """Return input-only embedding cost in US dollars."""
    return tokens * price_per_million_tokens / 1_000_000


def _connect(path: Path, *, model: str, dimensions: int) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            digest TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            field TEXT NOT NULL,
            text TEXT NOT NULL,
            tokens INTEGER NOT NULL,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    expected = {"model": model, "dimensions": str(dimensions)}
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    conflicts = {
        key: (existing[key], value)
        for key, value in expected.items()
        if key in existing and existing[key] != value
    }
    if conflicts:
        raise SystemExit(f"teacher cache identity mismatch: {conflicts}")
    connection.executemany(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
        expected.items(),
    )
    connection.commit()
    return connection


def _cached_digests(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT digest FROM embeddings")}


def summarize(
    items: list[TeacherItem],
    *,
    records: int,
    cached: set[str],
    price_per_million_tokens: float,
) -> CollectionSummary:
    """Summarize exact billable work after cache hits and text deduplication."""
    pending = [item for item in items if item.digest not in cached]
    return CollectionSummary(
        records=records,
        unique_texts=len(items),
        cached_texts=len(items) - len(pending),
        pending_texts=len(pending),
        total_tokens=sum(item.tokens for item in items),
        pending_tokens=sum(item.tokens for item in pending),
        estimated_pending_usd=estimate_cost(
            sum(item.tokens for item in pending), price_per_million_tokens
        ),
    )


def _decode_vectors(payload: dict[str, object], dimensions: int) -> list[np.ndarray]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("embedding response has no data list")
    ordered = sorted(rows, key=lambda row: int(row["index"]))
    vectors: list[np.ndarray] = []
    for row in ordered:
        encoded = row.get("embedding")
        if not isinstance(encoded, str):
            raise ValueError("base64 embedding response contains a non-string vector")
        vector = np.frombuffer(base64.b64decode(encoded), dtype="<f4").copy()
        if vector.shape != (dimensions,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid embedding vector shape or values: {vector.shape}")
        vectors.append(vector)
    return vectors


def _request_batch(
    client: httpx.Client,
    batch: list[TeacherItem],
    *,
    api_key: str,
    model: str,
    dimensions: int,
    maximum_retries: int,
    rng: random.Random,
) -> tuple[list[np.ndarray], int]:
    """Request one batch with exponential full-jitter transient retries."""
    for attempt in range(maximum_retries + 1):
        try:
            response = client.post(
                _API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "input": [item.text for item in batch],
                    "dimensions": dimensions,
                    "encoding_format": "base64",
                },
            )
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                response.raise_for_status()
                payload = response.json()
                vectors = _decode_vectors(payload, dimensions)
                if len(vectors) != len(batch):
                    raise ValueError("embedding response length does not match input")
                usage = payload.get("usage", {})
                actual_tokens = int(usage.get("prompt_tokens", sum(x.tokens for x in batch)))
                return vectors, actual_tokens
            error = f"HTTP {response.status_code}: {response.text[:500]}"
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt >= maximum_retries:
            raise RuntimeError(f"embedding batch failed after retries: {error}")
        ceiling = min(30.0, 0.5 * (2 ** attempt))
        time.sleep(rng.uniform(0.0, ceiling))
    raise AssertionError("unreachable retry loop")


def collect(
    connection: sqlite3.Connection,
    items: list[TeacherItem],
    *,
    api_key: str,
    model: str,
    dimensions: int,
    batch_items: int,
    batch_tokens: int,
    maximum_retries: int,
) -> tuple[int, int]:
    """Collect pending vectors transactionally and return texts/tokens added."""
    cached = _cached_digests(connection)
    pending = [item for item in items if item.digest not in cached]
    collected = 0
    actual_tokens = 0
    rng = random.Random(20260809)
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
        for batch in iter_batches(
            pending, maximum_items=batch_items, maximum_tokens=batch_tokens
        ):
            vectors, used_tokens = _request_batch(
                client,
                batch,
                api_key=api_key,
                model=model,
                dimensions=dimensions,
                maximum_retries=maximum_retries,
                rng=rng,
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO embeddings(
                    digest, source_id, field, text, tokens, vector
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.digest,
                        item.source_id,
                        item.field,
                        item.text,
                        item.tokens,
                        vector.astype("<f4", copy=False).tobytes(),
                    )
                    for item, vector in zip(batch, vectors, strict=True)
                ],
            )
            connection.commit()
            collected += len(batch)
            actual_tokens += used_tokens
            print(
                f"collected={collected}/{len(pending)} "
                f"actual_tokens={actual_tokens}",
                flush=True,
            )
    return collected, actual_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--fields", nargs="+", default=["text"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument(
        "--price-per-million-tokens",
        type=float,
        default=DEFAULT_PRICE_PER_MILLION_TOKENS,
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--batch-items", type=int, default=128)
    parser.add_argument("--batch-tokens", type=int, default=100_000)
    parser.add_argument("--maximum-retries", type=int, default=8)
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dimensions <= 0 or args.batch_items <= 0 or args.batch_tokens <= 0:
        raise SystemExit("dimensions and batch limits must be positive")
    items, records = load_jsonl_items(
        args.corpus,
        fields=tuple(args.fields),
        model=args.model,
        dimensions=args.dimensions,
        maximum_records=args.max_records,
        sample_seed=args.sample_seed,
    )
    connection = _connect(args.cache, model=args.model, dimensions=args.dimensions)
    try:
        summary = summarize(
            items,
            records=records,
            cached=_cached_digests(connection),
            price_per_million_tokens=args.price_per_million_tokens,
        )
        print(json.dumps(summary.__dict__, indent=2))
        if args.dry_run or not summary.pending_texts:
            return
        if summary.estimated_pending_usd > args.max_usd:
            raise SystemExit(
                f"estimated pending cost ${summary.estimated_pending_usd:.4f} "
                f"exceeds --max-usd ${args.max_usd:.4f}"
            )
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for live collection")
        collected, actual_tokens = collect(
            connection,
            items,
            api_key=api_key,
            model=args.model,
            dimensions=args.dimensions,
            batch_items=args.batch_items,
            batch_tokens=args.batch_tokens,
            maximum_retries=args.maximum_retries,
        )
        print(json.dumps({
            "collected_texts": collected,
            "actual_tokens": actual_tokens,
            "actual_cost_usd": estimate_cost(
                actual_tokens, args.price_per_million_tokens
            ),
        }, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
