from __future__ import annotations

from bench.build_aya_embedding_corpus import _split, record_from_row


def _row(language: str = "spa") -> dict[str, str]:
    return {
        "inputs": "Explica cómo ordenar una lista en Python.",
        "targets": "Puedes usar sorted(lista) para obtener una lista ordenada.",
        "language": "Spanish",
        "language_code": language,
        "annotation_type": "original-annotations",
        "user_id": "contributor-17",
    }


def test_record_keeps_language_and_hides_contributor_identity() -> None:
    record = record_from_row(
        _row(), max_instruction_chars=750, max_response_chars=1_500
    )
    assert record is not None
    assert record["language"] == "es"
    assert record["parallel_language"] == "es"
    assert record["licenses"] == ["Apache-2.0"]
    assert record["source_group"] != "contributor-17"
    assert record["split"] == _split(record["source_group"])


def test_record_rejects_languages_outside_target_set() -> None:
    assert record_from_row(
        _row("zho"), max_instruction_chars=750, max_response_chars=1_500
    ) is None


def test_record_rejects_mislabeled_language() -> None:
    row = _row("eng")
    assert record_from_row(
        row, max_instruction_chars=750, max_response_chars=1_500
    ) is None


def test_record_rejects_empty_pairs() -> None:
    row = _row()
    row["targets"] = "short"
    assert record_from_row(
        row, max_instruction_chars=750, max_response_chars=1_500
    ) is None
