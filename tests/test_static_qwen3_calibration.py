from __future__ import annotations

from pathlib import Path

import pandas as pd

from bench.fit_static_qwen3_spanish import split_for_record
from bench.static_qwen3_calibration import _m2crb_rows, _m2crb_split


def test_m2crb_rows_filter_deduplicate_and_balance(tmp_path: Path) -> None:
    path = tmp_path / "m2crb.parquet"
    rows = []
    for natural in ("es", "fr"):
        for programming in ("python", "java"):
            for index in range(4):
                rows.append({
                    "identifier": f"{natural}-{programming}-{index}",
                    "docstring": f"Descripción extensa de función {index}",
                    "docstring_summary": f"Resumen técnico número {index}",
                    "function": f"def function_{index}():\n    return {index}",
                    "language": programming,
                    "docstring_language": natural,
                })
    pd.DataFrame(rows).to_parquet(path)

    first = _m2crb_rows(str(path), ["es"], 3)
    second = _m2crb_rows(str(path), ["ES"], 3)

    assert set(first) == {("es", "java"), ("es", "python")}
    assert all(len(group) == 3 for group in first.values())
    assert first == second

    for (natural, programming), group in first.items():
        for row in group:
            record = {
                "source": f"m2crb_{natural}_{programming}",
                "path": row["family"],
            }
            assert _m2crb_split(
                natural, programming, row["family"]
            ) == split_for_record(record)
