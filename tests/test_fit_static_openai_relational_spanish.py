from __future__ import annotations

import numpy as np
import pytest

from bench.fit_static_openai_relational_spanish import relational_targets


def test_relational_targets_follow_teacher_neighbors_in_static_space() -> None:
    query_teacher = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    code_teacher = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32)
    code_static = np.asarray([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float32)
    translations = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    targets, report = relational_targets(
        query_teacher,
        code_teacher,
        code_static,
        translations,
        top_k=1,
        temperature=0.1,
        code_weight=1.0,
    )

    np.testing.assert_allclose(targets, [[0.0, 1.0], [1.0, 0.0]])
    assert report["top1_teacher_cosine_mean"] == pytest.approx(1.0)


def test_relational_targets_blend_and_normalize() -> None:
    targets, _ = relational_targets(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        top_k=1,
        temperature=0.1,
        code_weight=0.5,
    )

    np.testing.assert_allclose(targets, [[2 ** -0.5, 2 ** -0.5]], atol=1e-6)


@pytest.mark.parametrize(
    ("top_k", "temperature", "code_weight"),
    [(0, 0.1, 0.5), (1, 0.0, 0.5), (1, 0.1, -0.1), (1, 0.1, 1.1)],
)
def test_relational_targets_validate_hyperparameters(
    top_k: int, temperature: float, code_weight: float
) -> None:
    with pytest.raises(ValueError):
        relational_targets(
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([[1.0]], dtype=np.float32),
            top_k=top_k,
            temperature=temperature,
            code_weight=code_weight,
        )
