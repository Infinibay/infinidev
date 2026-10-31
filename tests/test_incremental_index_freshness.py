"""Regressions for parser invalidation and bounded incremental indexing."""

from __future__ import annotations

import builtins
import hashlib
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from infinidev.code_intel import index as ci_index
from infinidev.code_intel import indexer, parsers, smart_index
from infinidev.code_intel.query import list_symbols
from infinidev.code_intel.smart_index import ensure_indexed


@pytest.mark.parametrize("stale_kind", ["parser_bump", "legacy_row"])
@pytest.mark.parametrize(
    ("suffix", "content", "expected_names"),
    [
        (".py", "def recovered():\n    return 1\n", ["recovered"]),
        (".py", "# A valid source file with no symbols.\n", []),
        (".toml", '[project]\nname = "example"\n', []),
    ],
)
def test_smart_index_refreshes_stale_parser_output(
    tmp_path, temp_db, monkeypatch, stale_kind, suffix, content, expected_names,
):
    path = tmp_path / f"module{suffix}"
    path.write_text(content)
    file_path = str(path)
    assert ensure_indexed(1, file_path) is True
    original_state = ci_index.get_file_index_state(1, file_path)
    language = parsers.detect_language(file_path)

    # Emulate an old extractor that missed symbols, without changing source bytes.
    ci_index.clear_file(1, file_path)
    if stale_kind == "parser_bump":
        monkeypatch.setitem(parsers.PARSER_VERSIONS, language, original_state[1] + 1)
    else:
        ci_index.mark_file_indexed(1, file_path, language, original_state[0], 0, 0)

    assert ensure_indexed(1, file_path) is True
    assert ci_index.get_file_index_state(1, file_path) == (
        original_state[0], parsers.get_parser_version(language),
    )
    assert [symbol.name for symbol in list_symbols(1, file_path)] == expected_names
    assert ensure_indexed(1, file_path) is False


def test_unavailable_grammar_does_not_report_a_stale_hash_as_success(
    tmp_path, temp_db, monkeypatch,
):
    path = tmp_path / "module.py"
    path.write_text("def original():\n    pass\n")
    file_path = str(path)
    assert ensure_indexed(1, file_path) is True
    original_state = ci_index.get_file_index_state(1, file_path)
    monkeypatch.setitem(parsers.PARSER_VERSIONS, "python", original_state[1] + 1)
    monkeypatch.setattr(indexer, "_get_ts_language", lambda language: None)

    assert ensure_indexed(1, file_path) is False
    assert ci_index.get_file_index_state(1, file_path) == original_state
    assert [symbol.name for symbol in list_symbols(1, file_path)] == ["original"]


@pytest.fixture
def observed_reads(monkeypatch):
    reads = []

    @contextmanager
    def observed_open(path, *args, **kwargs):
        with builtins.open(path, *args, **kwargs) as stream:
            def read(size=-1):
                data = stream.read(size)
                reads.append((os.fspath(path), len(data)))
                return data

            yield SimpleNamespace(read=read)

    monkeypatch.setattr(indexer, "open", observed_open, raising=False)
    monkeypatch.setattr(smart_index, "open", observed_open, raising=False)
    return reads


@pytest.mark.parametrize("entry_point", [ensure_indexed, indexer.index_file])
def test_oversized_source_reads_at_most_the_limit_plus_one(
    tmp_path, temp_db, observed_reads, entry_point,
):
    path = tmp_path / "generated.py"
    with path.open("wb") as stream:
        stream.truncate(8 * 1024 * 1024)
    file_path = str(path)

    assert not entry_point(1, file_path)

    assert ci_index.get_file_index_state(1, file_path) is None
    bytes_read = sum(size for observed_path, size in observed_reads if observed_path == file_path)
    assert bytes_read <= indexer.MAX_FILE_SIZE + 1


def test_unsupported_file_skips_content_and_index_lookup(tmp_path, monkeypatch, observed_reads):
    path = tmp_path / "image.bin"
    path.write_bytes(b"binary payload")

    def unexpected_lookup(*args):
        pytest.fail("unsupported file triggered an index lookup")

    monkeypatch.setattr(ci_index, "get_file_index_state", unexpected_lookup)

    assert ensure_indexed(1, str(path)) is False
    assert observed_reads == []


@pytest.mark.parametrize("entry_point", [ensure_indexed, indexer.index_file])
@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_source_size_boundary_preserves_indexability(
    tmp_path, temp_db, entry_point, extra_bytes,
):
    path = tmp_path / "boundary.py"
    prefix = b"def boundary():\n    pass\n#"
    size = indexer.MAX_FILE_SIZE + extra_bytes
    path.write_bytes(prefix + b"x" * (size - len(prefix)))
    file_path = str(path)

    result = entry_point(1, file_path)

    if extra_bytes:
        assert not result
        assert ci_index.get_file_index_state(1, file_path) is None
    else:
        assert result
        assert [symbol.name for symbol in list_symbols(1, file_path)] == ["boundary"]


@pytest.mark.parametrize("entry_point", [ensure_indexed, indexer.index_file])
def test_large_config_keeps_its_full_content_hash(tmp_path, temp_db, observed_reads, entry_point):
    path = tmp_path / "large.toml"
    content = 'key = "' + "x" * (indexer.MAX_FILE_SIZE + 1024) + '"\n'
    path.write_text(content)
    file_path = str(path)

    entry_point(1, file_path)

    stored_state = ci_index.get_file_index_state(1, file_path)
    assert stored_state is not None
    assert stored_state[0] == hashlib.sha256(content.encode()).hexdigest()[:16]
    assert list_symbols(1, file_path) == []
    assert any(size == len(content) for _, size in observed_reads)
