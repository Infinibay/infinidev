from __future__ import annotations

from bench.translate_embedding_queries import (
    protect_technical_text,
    select_records,
    translate_rows,
)


def _row(identity: str, programming_language: str = "python", **overrides):
    row = {
        "id": identity,
        "source": "commitpackft_python",
        "split": "train",
        "language": "en",
        "language_confidence": 0.95,
        "programming_language": programming_language,
        "text": "Improve user authentication handling",
        "parallel_text": "@@ -1 +1 @@\n-old\n+new",
    }
    row.update(overrides)
    return row


def test_protection_rebuilds_technical_spans_byte_for_byte():
    source = "Fix `user_id` in src/auth.py for --no-cache and HTTPServer"
    protected = protect_technical_text(source)

    assert protected.protected == ("`user_id`", "src/auth.py", "--no-cache", "HTTPServer")
    rebuilt = protected.rebuild(chunk.upper() for chunk in protected.chunks)
    for token in protected.protected:
        assert token in rebuilt


def test_selection_is_balanced_deterministic_and_train_only():
    rows = [_row(f"py-{index}") for index in range(20)]
    rows += [_row(f"java-{index}", "java") for index in range(20)]
    rows += [
        _row("test", split="test"),
        _row("spanish", language="es"),
        _row("uncertain", language_confidence=0.4),
        _row("zig", "zig"),
    ]

    first = select_records(rows, per_language=4, seed=7)
    second = select_records(reversed(rows), per_language=4, seed=7)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert sum(row["programming_language"] == "python" for row in first) == 4
    assert sum(row["programming_language"] == "java" for row in first) == 4


def test_selection_accepts_trusted_unsplit_parallel_corpus():
    row = _row("code-search-net")
    row.pop("split")
    row.pop("language_confidence")
    row.pop("programming_language")
    row["parallel_language"] = "python"

    selected = select_records([row], per_language=2, seed=7)

    assert selected[0]["programming_language"] == "python"


def test_translation_operates_only_on_natural_chunks_and_validates_spanish():
    row = _row("one", text="Fix `user_id` handling")
    seen = []

    def translate(chunks):
        seen.extend(chunks)
        return ["Corregir" if chunk == "Fix" else "manejo" for chunk in chunks]

    [result] = list(translate_rows([row], translate))

    assert "`user_id`" not in "".join(seen)
    assert seen == ["Fix", "handling"]
    assert result["text"] == "Corregir `user_id` manejo"
    assert result["source_text"] == "Fix `user_id` handling"
    assert result["language"] == "es"
    assert result["translation_revision"]
