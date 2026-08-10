"""Tests for fitting a static projection to cached OpenAI embeddings."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from bench.collect_openai_embedding_teacher import _digest
from bench.fit_static_openai_teacher import (
    CachedExample,
    evaluate_candidate,
    fit_ridge_projection,
    load_cached_examples,
    projections_from_statistics,
    ridge_sufficient_statistics,
)


def test_ridge_projection_recovers_heldout_linear_teacher() -> None:
    rng = np.random.default_rng(42)
    latents = rng.normal(size=(240, 12)).astype(np.float32)
    mapping = rng.normal(size=(12, 20)).astype(np.float32)
    targets = latents @ mapping
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)

    projection, center = fit_ridge_projection(
        latents, targets, np.arange(180), penalty=0.01
    )
    predicted = latents[180:] @ projection
    expected = targets[180:] - center
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)

    assert float(np.mean(np.sum(predicted * expected, axis=1))) > 0.98


def test_shared_ridge_statistics_match_single_projection() -> None:
    rng = np.random.default_rng(9)
    latents = rng.normal(size=(80, 7)).astype(np.float32)
    targets = rng.normal(size=(80, 11)).astype(np.float32)
    train = np.arange(60)

    expected, expected_center = fit_ridge_projection(
        latents, targets, train, penalty=0.3
    )
    gram, cross, center = ridge_sufficient_statistics(
        latents, targets, train, chunk_size=13
    )
    [actual] = projections_from_statistics(gram, cross, (0.3,))

    assert np.allclose(actual, expected, atol=1e-5)
    assert np.allclose(center, expected_center, atol=1e-6)


def test_retrieval_report_caps_quadratic_pair_matrix() -> None:
    examples = []
    vectors = []
    for index in range(8):
        vector = np.zeros(8, dtype=np.float32)
        vector[index] = 1.0
        for field in ("text", "parallel_text"):
            examples.append(CachedExample(
                record_id=f"row-{index}",
                field=field,
                text=f"{field}-{index}",
                split="train",
                target=vector,
            ))
            vectors.append(vector)

    report = evaluate_candidate(
        examples,
        np.asarray(vectors),
        np.asarray(vectors),
        np.zeros(8, dtype=np.float32),
        maximum_retrieval_pairs=3,
    )

    assert report["splits"]["train"]["bilingual_pairs"] == 3
    assert report["splits"]["train"]["bilingual_pairs_available"] == 8


def _teacher_cache(path: Path, text: str, vector: np.ndarray) -> None:
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE embeddings(
            digest TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            field TEXT NOT NULL,
            text TEXT NOT NULL,
            tokens INTEGER NOT NULL,
            vector BLOB NOT NULL
        );
    """)
    connection.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        (("model", "teacher"), ("dimensions", "4")),
    )
    connection.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
        (
            _digest(text, model="teacher", dimensions=4),
            "row",
            "text",
            text,
            2,
            vector.astype("<f4").tobytes(),
        ),
    )
    connection.commit()
    connection.close()


def test_cached_examples_validate_identity_and_normalize(tmp_path: Path) -> None:
    cache = tmp_path / "teacher.sqlite"
    _teacher_cache(cache, "hola mundo", np.asarray([2.0, 0.0, 0.0, 0.0]))
    records = [{"id": "row", "text": "hola mundo", "split": "train"}]

    examples = load_cached_examples(
        records,
        cache,
        fields=("text",),
        model="teacher",
        dimensions=4,
    )

    assert len(examples) == 1
    assert np.array_equal(examples[0].target, np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_cached_examples_reject_partial_cache(tmp_path: Path) -> None:
    cache = tmp_path / "teacher.sqlite"
    _teacher_cache(cache, "present", np.ones(4, dtype=np.float32))
    records = [{"id": "row", "text": "missing", "split": "test"}]

    with pytest.raises(ValueError, match="missing 1 requested texts"):
        load_cached_examples(
            records,
            cache,
            fields=("text",),
            model="teacher",
            dimensions=4,
        )
