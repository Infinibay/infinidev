from __future__ import annotations

import pytest

from bench.observation_provenance import bind_rows


def test_bind_rows_adds_hashes_without_changing_response() -> None:
    rows = [{"probe_id": "p1", "answer": "A", "response_text": '{"answer":"A"}'}]
    bound = bind_rows(rows, dataset_sha256="dataset", manifest_sha256="manifest")
    assert bound[0]["dataset_sha256"] == "dataset"
    assert bound[0]["manifest_sha256"] == "manifest"
    assert bound[0]["response_text"] == '{"answer":"A"}'


def test_bind_rows_rejects_conflicting_existing_provenance() -> None:
    with pytest.raises(ValueError, match="different dataset"):
        bind_rows(
            [{"dataset_sha256": "other"}],
            dataset_sha256="dataset",
            manifest_sha256="manifest",
        )
