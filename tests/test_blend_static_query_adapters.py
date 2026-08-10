from __future__ import annotations

import numpy as np
import pytest

from bench.blend_static_query_adapters import blend


def _adapter(rows: list[int], values: list[list[float]], parent: str = "base"):
    return {
        "rows": np.asarray(rows, dtype=np.int32),
        "delta_float": np.asarray(values, dtype=np.float32),
        "meta_json": {"parent_sha256": parent},
    }


def test_blend_interpolates_union_of_sparse_rows() -> None:
    rows, residual = blend(
        _adapter([1, 3], [[2.0, 0.0], [0.0, 2.0]]),
        _adapter([2, 3], [[4.0, 0.0], [2.0, 2.0]]),
        second_weight=0.25,
    )

    np.testing.assert_array_equal(rows, [1, 2, 3])
    np.testing.assert_allclose(
        residual,
        [[1.5, 0.0], [1.0, 0.0], [0.5, 2.0]],
    )


def test_blend_rejects_different_parents() -> None:
    with pytest.raises(ValueError, match="same exact parent"):
        blend(
            _adapter([1], [[1.0]], parent="a"),
            _adapter([1], [[1.0]], parent="b"),
            second_weight=0.5,
        )
