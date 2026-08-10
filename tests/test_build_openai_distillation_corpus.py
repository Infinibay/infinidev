from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from bench import build_openai_distillation_corpus as corpus


def test_stable_selection_is_order_independent() -> None:
    records = [{"id": str(index), "text": f"text {index}"} for index in range(20)]

    selected = corpus.select_stable(records, 7, seed=17)
    reversed_selected = corpus.select_stable(list(reversed(records)), 7, seed=17)

    assert [row["id"] for row in selected] == [row["id"] for row in reversed_selected]


def test_codesearchnet_test_split_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "codesearchnet-python-test.parquet"

    with pytest.raises(ValueError, match="refusing CodeSearchNet test split"):
        corpus.load_codesearchnet(path, language="python", max_chars=700)


def test_build_balances_sources_and_preserves_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spanish_path = tmp_path / "spanish.jsonl"
    spanish_path.write_text("\n".join(
        json.dumps({
            "id": f"es-{index}",
            "source": "python_docs_es",
            "path": f"manual/{index}.po",
            "license_class": "permissive" if index < 4 else "sharealike",
            "language": "es",
            "text": f"Texto técnico número {index}",
            "parallel_text": f"Technical text number {index}",
        })
        for index in range(5)
    ) + "\n", encoding="utf-8")

    def fake_codesearchnet(
        path: Path, *, language: str, max_chars: int
    ) -> list[dict[str, object]]:
        del path, max_chars
        return [{
            "id": f"{language}-{index}",
            "source": f"codesearchnet_{language}",
            "path": f"https://example.test/{language}/{index}",
            "language": "en",
            "text": f"Describe function number {index}",
            "parallel_text": f"def function_{index}(): return {index}",
        } for index in range(6)]

    monkeypatch.setattr(corpus, "load_codesearchnet", fake_codesearchnet)
    args = argparse.Namespace(
        spanish_jsonl=spanish_path,
        spanish_records=3,
        include_sharealike=False,
        include_monolingual_spanish=False,
        codesearchnet=["python=/data/python-validation.parquet"],
        records_per_language=4,
        max_chars=700,
        seed=17,
    )

    records = corpus.build(args)

    assert len(records) == 7
    assert sum(row["source"] == "python_docs_es" for row in records) == 3
    assert sum(row["source"] == "codesearchnet_python" for row in records) == 4
    assert {row["language"] for row in records} == {"en", "es"}
    assert all(row.get("license_class") != "sharealike" for row in records)


@pytest.mark.parametrize("value", ["python", "=path", "python="])
def test_codesearchnet_specs_require_language_and_path(value: str) -> None:
    with pytest.raises(ValueError, match="expected LANGUAGE=PATH"):
        corpus.parse_codesearchnet_specs([value])
