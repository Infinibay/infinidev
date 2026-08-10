from __future__ import annotations

import numpy as np

from bench.fit_static_openai_contrastive_spanish import grouped_retrieval


def test_grouped_retrieval_is_macro_averaged_by_source() -> None:
    records = [
        {"source": "python", "split": "validation"},
        {"source": "python", "split": "validation"},
        {"source": "java", "split": "validation"},
        {"source": "java", "split": "test"},
    ]
    queries = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float32)
    passages = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float32)

    report = grouped_retrieval(records, queries, passages, "validation")

    assert report["macro_mrr"] == 1.0
    assert report["macro_recall@1"] == 1.0
    assert set(report["groups"]) == {"java", "python"}
