from __future__ import annotations

import numpy as np
import pytest

from bench.task_policy_windowed_eval import _aggregate, _selected


def test_aggregate_windows_supports_first_max_mean_and_top_two() -> None:
    values = np.asarray([
        [0.1, 0.8],
        [0.9, 0.2],
        [0.5, 0.4],
        [0.3, 0.7],
    ])
    owners = np.asarray([0, 0, 0, 1])

    np.testing.assert_allclose(_aggregate(values, owners, 2, "first"), [[0.1, 0.8], [0.3, 0.7]])
    np.testing.assert_allclose(_aggregate(values, owners, 2, "max"), [[0.9, 0.8], [0.3, 0.7]])
    np.testing.assert_allclose(
        _aggregate(values, owners, 2, "mean"), [[0.5, 1.4 / 3], [0.3, 0.7]],
    )
    np.testing.assert_allclose(
        _aggregate(values, owners, 2, "top2_mean"), [[0.7, 0.6], [0.3, 0.7]],
    )


def test_aggregate_rejects_missing_examples_and_unknown_modes() -> None:
    values = np.asarray([[0.2]])
    owners = np.asarray([0])

    with pytest.raises(ValueError, match="has no token window"):
        _aggregate(values, owners, 2, "max")
    with pytest.raises(ValueError, match="unknown aggregation mode"):
        _aggregate(values, owners, 1, "median")


def test_selected_uses_independent_strategy_per_label() -> None:
    candidates = {
        "left": np.asarray([[0.9, 0.1], [0.8, 0.2]]),
        "right": np.asarray([[0.3, 0.7], [0.4, 0.6]]),
    }

    np.testing.assert_allclose(
        _selected(candidates, ("left", "right")),
        [[0.9, 0.7], [0.8, 0.6]],
    )
