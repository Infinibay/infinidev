from __future__ import annotations

import numpy as np
import pytest

from infinidev.tools.base.static_qwen3_embedder import (
    STATIC_OPENAI_FAMILY,
    STATIC_QWEN3_DIM,
    STATIC_QWEN3_MODEL,
    StaticQwen3Embedder,
    get_static_qwen3_embedder,
)
from infinidev.tools.base.embeddings import embedding_is_current


def test_bundled_static_qwen3_artifact_is_loadable_and_normalized() -> None:
    embedder = get_static_qwen3_embedder()

    assert embedder is not None
    vectors = embedder([
        "Extend the callback registry with one-shot handlers",
        "Amplía el registro con callbacks de una sola ejecución",
        "",
    ])

    assert embedder.model_name == STATIC_QWEN3_MODEL
    assert embedder.dim == STATIC_QWEN3_DIM
    assert embedder.meta["teacher"] == "Qwen/Qwen3-Embedding-0.6B"
    assert len(vectors) == 3
    assert all(vector.shape == (STATIC_QWEN3_DIM,) for vector in vectors)
    assert float(np.linalg.norm(vectors[0])) == pytest.approx(1.0, abs=1e-6)
    assert float(np.linalg.norm(vectors[1])) == pytest.approx(1.0, abs=1e-6)
    assert np.count_nonzero(vectors[2]) == 0


def test_static_qwen3_embedding_is_deterministic() -> None:
    embedder = StaticQwen3Embedder()
    text = "Implement semantic request detection without another LLM call"

    first = embedder.embed_query(text)
    second = embedder.embed_passages([text])[0]

    np.testing.assert_array_equal(first, second)


def test_spanish_adapter_is_query_only_and_language_gated() -> None:
    embedder = StaticQwen3Embedder()
    spanish = "Busca la función que valida argumentos JSON antes de ejecutar la herramienta"
    english = "Find the function that validates JSON arguments before running the tool"

    spanish_query = embedder.embed_query(spanish)
    spanish_passage = embedder.embed_passages([spanish])[0]
    english_query = embedder.embed_query(english)
    english_passage = embedder.embed_passages([english])[0]

    assert embedder.spanish_adapter_meta["parent_name"] == STATIC_QWEN3_MODEL
    assert float(spanish_query @ spanish_passage) < 0.999
    np.testing.assert_array_equal(english_query, english_passage)


def test_empty_batch_members_do_not_change_neighboring_vectors() -> None:
    embedder = StaticQwen3Embedder()
    texts = ["", "primer texto técnico", "", "second technical text", ""]

    batched = embedder(texts)

    for index in (1, 3):
        np.testing.assert_allclose(
            batched[index], embedder([texts[index]])[0], rtol=0.0, atol=1e-7
        )
    assert all(np.count_nonzero(batched[index]) == 0 for index in (0, 2, 4))


def test_static_qwen3_space_id_is_exact_and_stable() -> None:
    first = StaticQwen3Embedder()
    second = StaticQwen3Embedder()

    assert first.space_id == second.space_id
    assert first.space_id.startswith(f"{STATIC_QWEN3_MODEL}:{STATIC_QWEN3_DIM}:")


def test_same_dimension_from_another_space_is_not_current() -> None:
    vector = np.zeros(STATIC_QWEN3_DIM, dtype=np.float32).tobytes()

    assert embedding_is_current(
        vector,
        "another-model:1024:deadbeef",
        live_space="current-model:1024:cafebabe",
        dim=STATIC_QWEN3_DIM,
    ) is False


def test_shared_embedding_backend_prefers_static_qwen3(monkeypatch) -> None:
    from infinidev.tools.base import dedup

    dedup._embed_fn = None
    monkeypatch.setattr(
        "infinidev.tools.base.static_qwen3_embedder._singleton",
        None,
    )

    backend = dedup._get_embed_fn()

    assert isinstance(backend, StaticQwen3Embedder)
    dedup._embed_fn = None
