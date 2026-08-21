from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from bench.fit_static_query_projection_spanish import clamp_projection_delta


def test_projection_delta_is_clamped_to_relative_norm() -> None:
    base = torch.ones((4, 6))
    delta = torch.full((4, 6), 20.0)

    ratio = clamp_projection_delta(delta, base, 0.05)

    assert ratio == pytest.approx(0.05, abs=1e-6)


def test_projection_delta_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="maximum ratio"):
        clamp_projection_delta(torch.ones(2, 2), torch.ones(2, 2), 0.0)
