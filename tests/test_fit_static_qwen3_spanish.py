"""Tests for the offline Spanish static-head calibration pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


_SCRIPT = Path(__file__).parents[1] / "bench" / "fit_static_qwen3_spanish.py"
_SPEC = importlib.util.spec_from_file_location("fit_static_qwen3_spanish", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
fit_spanish = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fit_spanish
_SPEC.loader.exec_module(fit_spanish)

_CALIBRATION_SCRIPT = Path(__file__).parents[1] / "bench" / "static_qwen3_calibration.py"
_CALIBRATION_SPEC = importlib.util.spec_from_file_location(
    "static_qwen3_calibration", _CALIBRATION_SCRIPT
)
assert _CALIBRATION_SPEC is not None and _CALIBRATION_SPEC.loader is not None
calibration = importlib.util.module_from_spec(_CALIBRATION_SPEC)
sys.modules[_CALIBRATION_SPEC.name] = calibration
_CALIBRATION_SPEC.loader.exec_module(calibration)


def test_split_keeps_source_path_family_together() -> None:
    first = {"source": "manual", "path": "chapter.po", "id": "one"}
    neighbor = {"source": "manual", "path": "chapter.po", "id": "two"}
    other = {"source": "manual", "path": "other.po", "id": "three"}

    assert fit_spanish.split_for_record(first) == fit_spanish.split_for_record(neighbor)
    assert fit_spanish._family_key(first) != fit_spanish._family_key(other)


def test_load_corpus_sampling_is_stable_and_preserves_splits(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"c","source":"s","path":"c.po","text":"tres"}\n'
        '{"id":"a","source":"s","path":"a.po","text":"uno"}\n'
        '{"id":"b","source":"s","path":"b.po","text":"dos"}\n',
        encoding="utf-8",
    )

    first = fit_spanish.load_corpus(corpus, maximum=2, seed=9)
    second = fit_spanish.load_corpus(corpus, maximum=2, seed=9)

    assert first == second
    assert len(first) == 2
    assert all(row["split"] in {"train", "validation", "test"} for row in first)


def test_quantize_rows_round_trips_with_per_row_scale() -> None:
    table = np.asarray([[0.0, 1.0, -1.0], [0.0001, 0.0002, -0.0003]], dtype=np.float32)

    quantized, scales = fit_spanish._quantize_rows(table)
    restored = quantized.astype(np.float32) * scales[:, None]

    assert quantized.dtype == np.int8
    assert np.all(scales > 0)
    assert np.allclose(restored, table, atol=float(scales.max()))


def test_code_replay_is_compact_and_ken_shaped(tmp_path: Path) -> None:
    source = tmp_path / "worker.py"
    source.write_text(
        '"""Worker utilities."""\n\nclass Worker:\n    pass\n\ndef run_job():\n    pass\n',
        encoding="utf-8",
    )

    rows = list(fit_spanish.iter_code_replay(tmp_path))

    assert rows == [{"id": "worker.py", "text": "python worker — class Worker def run_job"}]


def test_paired_bootstrap_reports_candidate_minus_baseline() -> None:
    baseline = np.asarray([1, 2, 8, 20, 100])
    candidate = np.asarray([1, 1, 4, 10, 50])

    deltas = calibration._paired_bootstrap_deltas(
        baseline, candidate, samples=500, seed=3
    )

    assert deltas["recall@1"][0] == 0.2
    assert deltas["recall@5"][0] == 0.2
    assert deltas["mrr"][0] > 0
    assert deltas["median_rank"][0] == -4.0


def test_codesearchnet_replay_and_gate_are_disjoint() -> None:
    pairs = [
        {"id": f"row-{index}", "query": f"query {index}", "passage": f"code {index}"}
        for index in range(100)
    ]

    replay = fit_spanish.select_codesearchnet_replay(pairs, maximum=1_000, seed=17)
    gate = fit_spanish.select_codesearchnet_gate(pairs, maximum=1_000, seed=17)

    replay_ids = {row["id"].split(":", 1)[1] for row in replay}
    gate_ids = {row["id"] for row in gate}
    assert replay_ids.isdisjoint(gate_ids)
    assert replay_ids | gate_ids == {row["id"] for row in pairs}
