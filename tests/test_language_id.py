from __future__ import annotations

import pytest

from bench.language_id import detect_target_language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Remove a test due to library version dependence", "en"),
        ("Corrige el error al analizar el archivo", "es"),
        ("Adicionar suporte para carregar o arquivo", "pt"),
        ("Ajouter la prise en charge du fichier", "fr"),
        ("Aggiungi supporto per caricare il file", "it"),
    ],
)
def test_detects_target_languages_in_developer_text(text: str, expected: str) -> None:
    language, confidence = detect_target_language(text)
    assert language == expected
    assert confidence >= 0.5


@pytest.mark.parametrize(
    "text",
    [
        "Print error message on parse failure.",
        "Use utf-8 chars in example email",
        "Update mini-macro post proc macro stabilization",
    ],
)
def test_short_english_commit_messages_do_not_become_romance_languages(text: str) -> None:
    assert detect_target_language(text)[0] == "en"


def test_abstains_on_uncovered_language() -> None:
    assert detect_target_language("Fehler beim Laden der Datei beheben")[0] == "other"
