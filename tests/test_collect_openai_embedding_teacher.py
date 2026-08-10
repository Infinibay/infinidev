"""Tests for the resumable OpenAI embedding-teacher collector."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
import pytest

from bench.collect_openai_embedding_teacher import (
    CachedTeacherEmbedder,
    TeacherItem,
    _digest,
    estimate_cost,
    iter_batches,
    load_jsonl_items,
    summarize,
)


def _item(name: str, tokens: int) -> TeacherItem:
    return TeacherItem(
        digest=name,
        source_id=name,
        field="text",
        text=name,
        tokens=tokens,
    )


def test_cost_uses_input_tokens_only() -> None:
    assert estimate_cost(1_000_000, 0.13) == pytest.approx(0.13)
    assert estimate_cost(22_765_556, 0.13) == pytest.approx(2.95952228)


def test_batches_respect_item_and_token_limits() -> None:
    batches = list(iter_batches(
        [_item("a", 4), _item("b", 7), _item("c", 2), _item("d", 2)],
        maximum_items=2,
        maximum_tokens=10,
    ))

    assert [[item.digest for item in batch] for batch in batches] == [
        ["a"], ["b", "c"], ["d"],
    ]


def test_jsonl_loader_deduplicates_equal_text_across_fields(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join((
        json.dumps({"id": "a", "text": "same text", "parallel_text": "otro"}),
        json.dumps({"id": "b", "text": "same text", "parallel_text": "different"}),
    )) + "\n", encoding="utf-8")

    class Encoding:
        @staticmethod
        def encode(text: str) -> list[str]:
            return text.split()

    monkeypatch.setattr(
        "bench.collect_openai_embedding_teacher._encoding", lambda: Encoding()
    )
    items, records = load_jsonl_items(
        corpus,
        fields=("text", "parallel_text"),
        model="teacher",
        dimensions=1024,
    )

    assert records == 2
    assert {item.text for item in items} == {"same text", "otro", "different"}


def test_jsonl_loader_hash_sample_matches_fitter_selection(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"id": f"row-{index}", "text": f"text {index}"}
        for index in range(30)
    ]
    corpus.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    class Encoding:
        @staticmethod
        def encode(text: str) -> list[str]:
            return text.split()

    monkeypatch.setattr(
        "bench.collect_openai_embedding_teacher._encoding", lambda: Encoding()
    )
    items, records = load_jsonl_items(
        corpus,
        fields=("text",),
        model="teacher",
        dimensions=4,
        maximum_records=7,
        sample_seed=23,
    )
    expected = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"23\0{row['id']}".encode()).digest(),
    )[:7]

    assert records == 7
    assert {item.source_id for item in items} == {row["id"] for row in expected}


def test_summary_charges_only_uncached_unique_texts() -> None:
    items = [_item("a", 10), _item("b", 20), _item("c", 30)]

    summary = summarize(
        items,
        records=4,
        cached={"a"},
        price_per_million_tokens=0.13,
    )

    assert summary.records == 4
    assert summary.unique_texts == 3
    assert summary.cached_texts == 1
    assert summary.pending_texts == 2
    assert summary.pending_tokens == 50
    assert summary.estimated_pending_usd == pytest.approx(0.0000065)


def test_cached_embedder_uses_collector_whitespace_normalization(tmp_path) -> None:
    cache = tmp_path / "cache.sqlite"
    connection = sqlite3.connect(cache)
    connection.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE embeddings(digest TEXT PRIMARY KEY, vector BLOB NOT NULL);
    """)
    connection.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        (("model", "teacher"), ("dimensions", "2")),
    )
    digest = _digest("normalized", model="teacher", dimensions=2)
    connection.execute(
        "INSERT INTO embeddings VALUES (?, ?)",
        (digest, np.asarray([1.0, 0.0], dtype="<f4").tobytes()),
    )
    connection.commit()
    connection.close()

    vectors = CachedTeacherEmbedder(cache).embed_queries(["  normalized\n"])

    np.testing.assert_array_equal(vectors[0], np.asarray([1.0, 0.0]))
