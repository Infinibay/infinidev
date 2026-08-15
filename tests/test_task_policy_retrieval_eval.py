from __future__ import annotations

import numpy as np

from bench.task_policy_retrieval_eval import (
    _margin_scores,
    _prototype_scores,
    _weighted_knn_scores,
)


def test_retrieval_scores_rank_matching_label_above_other_label() -> None:
    train = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    targets = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    knn = _weighted_knn_scores(
        train, targets, queries, neighbors=3, temperature=0.05,
    )
    margin = _margin_scores(train, targets, queries, neighbors=1)
    prototype = _prototype_scores(train, targets, queries)

    for scores in (knn, margin, prototype):
        assert scores[0, 0] > scores[0, 1]
        assert scores[1, 1] > scores[1, 0]
