from __future__ import annotations

from bench.build_stack_multilingual_corpus import (
    _assembly_dialect,
    _fallback_symbols,
    _preceding_comment,
    _provenance,
    records_from_file,
)


def _row(content: str, path: str, license_name: str = "MIT") -> dict[str, object]:
    return {
        "content": content,
        "max_stars_repo_name": "example/project",
        "max_stars_repo_path": path,
        "max_stars_repo_head_hexsha": "a" * 40,
        "max_stars_repo_licenses": [license_name],
    }


def test_provenance_requires_permissive_license() -> None:
    assert _provenance(_row("code", "src/a.rs")) is not None
    assert _provenance(_row("code", "src/a.rs", "GPL-3.0")) is None


def test_fallback_extracts_perl_and_assembly_symbols() -> None:
    perl = list(_fallback_symbols("perl", ["# docs", "sub parse_item {", "}"]))
    assembly = list(_fallback_symbols(
        "assembly", ["; docs", "entry:", "  mov eax, 1", ".Ltmp:"]
    ))

    assert perl[0][1:4] == ("parse_item", 2, 3)
    assert assembly[0][1] == "entry"
    assert len(assembly) == 1


def test_assembly_dialect_distinguishes_x86_and_arm() -> None:
    assert _assembly_dialect("mov eax, ebx\npush rax\npop rax") == "x86"
    assert _assembly_dialect("ldr x0, [x1]\nstr w0, [x2]\nbl target") == "arm"


def test_records_retain_provenance_and_documentation() -> None:
    records = list(records_from_file(
        _row("# Parse one item.\nsub parse_item {\n  return 1;\n}\n", "lib/A.pm"),
        "perl",
        max_query_chars=500,
        max_code_chars=1_400,
    ))

    assert len(records) == 1
    assert records[0]["text"] == "Parse one item."
    assert records[0]["programming_language"] == "perl"
    assert records[0]["repository"] == "example/project"
    assert records[0]["licenses"] == ["MIT"]


def test_preceding_comments_use_language_specific_syntax() -> None:
    lines = ["/* Clock source. */", "#define SOURCE 1", "#define TARGET 2"]

    assert _preceding_comment(lines, 3, 500, "c") == ""
    assert _preceding_comment(["# Parse item.", "def parse():"], 2, 500, "python") == (
        "Parse item."
    )
