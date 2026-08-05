from __future__ import annotations

from bench.model_behavior import Observation
from bench.select_observations import select_observations


def test_select_observations_preserves_order_and_filters_repetition() -> None:
    rows = [
        Observation("p", "raw", "A", None, repetition=2),
        Observation("p", "raw", "B", None, repetition=0),
        Observation("p", "raw", "C", None, repetition=1),
    ]

    selected = select_observations(rows, repetitions=[0, 2])

    assert [(row.repetition, row.answer) for row in selected] == [(2, "A"), (0, "B")]


def test_select_observations_filters_probe_ids_without_reordering() -> None:
    rows = [
        Observation("p2", "raw", "A", None),
        Observation("p1", "raw", "B", None),
        Observation("p3", "raw", "C", None),
    ]

    selected = select_observations(rows, probe_ids=["p1", "p2"])

    assert [row.probe_id for row in selected] == ["p2", "p1"]
