from __future__ import annotations

from bench.build_commitpack_embedding_corpus import (
    _compact_diff,
    _split,
    record_from_row,
    sample_records,
)


def _row(identifier: int, *, license_name: str = "mit", subject: str | None = None):
    return {
        "commit": f"{identifier:040x}",
        "old_file": "src/main.rs",
        "new_file": "src/main.rs",
        "old_contents": "fn main() {\n    old();\n}\n",
        "new_contents": f"fn main() {{\n    new_{identifier}();\n}}\n",
        "subject": subject or f"Replace old behavior with implementation {identifier}",
        "lang": "Rust",
        "license": license_name,
        "repos": f"example/project-{identifier}",
    }


def test_compact_diff_requires_a_real_change_and_preserves_hunk() -> None:
    assert _compact_diff("same", "same", "a", "a", 500) == ""
    change = _compact_diff("old\n", "new\n", "a", "a", 500)
    assert "@@" in change
    assert "-old" in change
    assert "+new" in change


def test_record_filters_license_and_retains_provenance() -> None:
    assert record_from_row(
        _row(1, license_name="agpl-3.0"),
        "rust",
        max_instruction_chars=500,
        max_diff_chars=2_500,
    ) is None

    record = record_from_row(
        _row(2), "rust", max_instruction_chars=500, max_diff_chars=2_500
    )
    assert record is not None
    assert record["kind"] == "instruction_to_code_change"
    assert record["programming_language"] == "rust"
    assert record["repository"] == "example/project-2"
    assert record["source_url"].endswith(f"/commit/{2:040x}")
    assert record["split"] == _split("example/project-2")


def test_low_information_subject_is_rejected() -> None:
    assert record_from_row(
        _row(1, subject="Update"),
        "rust",
        max_instruction_chars=500,
        max_diff_chars=2_500,
    ) is None


def test_sampling_is_bounded_and_deterministic() -> None:
    rows = [_row(index) for index in range(20)]
    first, _ = sample_records(
        rows,
        "rust",
        limit=5,
        max_instruction_chars=500,
        max_diff_chars=2_500,
        seed=7,
    )
    second, _ = sample_records(
        reversed(rows),
        "rust",
        limit=5,
        max_instruction_chars=500,
        max_diff_chars=2_500,
        seed=7,
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 5
