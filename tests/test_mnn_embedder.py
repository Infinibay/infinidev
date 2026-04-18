"""Smoke test for the MNN-backed embedder.

Skipped unless both MNN is importable and INFINIDEV_MNN_MODEL_PATH points
at a valid `.mnn` file. Run `scripts/convert_minilm_to_mnn.py` first to
produce the model file locally.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("MNN")


@pytest.fixture
def model_path() -> str:
    path = os.environ.get("INFINIDEV_MNN_MODEL_PATH")
    if not path or not os.path.isfile(path):
        pytest.skip("INFINIDEV_MNN_MODEL_PATH not set; skipping MNN test")
    return path


def test_mnn_matches_chromadb_onnx(model_path):
    """MNN output must be cosine≈1 with ChromaDB's ONNX for stored BLOB compat."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    from infinidev.tools.base.mnn_embedder import MNNEmbedder

    mnn = MNNEmbedder(model_path)
    onnx = DefaultEmbeddingFunction()

    samples = [
        "fix authentication bug in login flow",
        "CacheWarmupScheduler periodic preloading",
        "def compute_embedding(text): return embed(text)",
        "¿cómo funciona el login?",
    ]
    mnn_out = mnn(samples)
    onnx_out = onnx(samples)

    for i, (m, o) in enumerate(zip(mnn_out, onnx_out)):
        mv = np.asarray(m, dtype=np.float32)
        ov = np.asarray(o, dtype=np.float32)
        cos = float(np.dot(mv, ov) / (np.linalg.norm(mv) * np.linalg.norm(ov)))
        assert cos > 0.99, f"drift too large on sample {i!r}: cos={cos:.4f}"


def test_dedup_picks_mnn_when_configured(model_path):
    """dedup._get_embed_fn should switch to MNN when the env var is set."""
    from infinidev.tools.base import dedup
    from infinidev.tools.base.mnn_embedder import MNNEmbedder

    # Reset the module singleton so we re-probe.
    dedup._embed_fn = None
    fn = dedup._get_embed_fn()
    assert isinstance(fn, MNNEmbedder), f"expected MNNEmbedder, got {type(fn).__name__}"
