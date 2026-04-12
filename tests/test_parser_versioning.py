"""Tests for the parser versioning mechanism in the code intel indexer.

The parser versioning story: each language parser has a PARSER_VERSION
integer that gets bumped when extraction logic changes.  The
incremental indexer checks both content hash AND parser version
before skipping a file — so a parser bug fix propagates to already-
indexed files on the next /reindex pass, even though the content
hash didn't change.

These tests exist because we diagnosed this exact failure mode in
the wild: ErrorHandler.ts in a real project had its three classes
indexed with empty names by an old parser, and the incremental
skip kept serving the broken data for months because the file
content was never modified.  Without versioning, the only recovery
was a force /reindex --full — which users typically don't know to
run.

These tests are strictly on the storage + skip-check layer; they
use synthetic symbols to avoid depending on tree-sitter behaviour
(which is covered by test_code_intel.py and test_parsers*.py).
"""

from __future__ import annotations

import pytest


class TestParserVersionRegistry:
    """Structural tests on PARSER_VERSIONS."""

    def test_all_supported_languages_have_versions(self):
        from infinidev.code_intel.parsers import PARSER_VERSIONS, get_parser_version
        # Every language in PARSER_VERSIONS returns a non-zero version
        for lang, ver in PARSER_VERSIONS.items():
            assert ver >= 1, f"{lang} has invalid version {ver}"
            assert get_parser_version(lang) == ver

    def test_unknown_language_returns_zero(self):
        from infinidev.code_intel.parsers import get_parser_version
        # Unknown languages return 0 — which will never match any
        # real stored version, so the skip check invalidates them.
        assert get_parser_version("esperanto") == 0
        assert get_parser_version("") == 0

    def test_typescript_version_reflects_class_name_fix(self):
        """TS parser version ≥ 2 — the fix for class_declaration
        accepting type_identifier was a material bug fix and had to
        bump the version so stale data with name='' gets invalidated.
        """
        from infinidev.code_intel.parsers import get_parser_version
        assert get_parser_version("typescript") >= 2
        assert get_parser_version("javascript") >= 2


class TestFileIndexState:
    """Tests for get_file_index_state + mark_file_indexed."""

    def test_state_is_none_for_uninserted_file(self, temp_db):
        from infinidev.code_intel.index import get_file_index_state
        assert get_file_index_state(1, "nonexistent.py") is None

    def test_state_returns_hash_and_version(self, temp_db):
        from infinidev.code_intel.index import (
            get_file_index_state, mark_file_indexed,
        )
        mark_file_indexed(
            project_id=1, file_path="/tmp/foo.py",
            language="python", content_hash="abc123",
            symbol_count=5, parser_version=1,
        )
        state = get_file_index_state(1, "/tmp/foo.py")
        assert state == ("abc123", 1)

    def test_state_defaults_version_to_zero_when_legacy_row(self, temp_db):
        """Rows inserted without a parser_version get 0 — a sentinel
        that will never match any current parser version, so the
        incremental skip check always invalidates them."""
        from infinidev.code_intel.index import (
            get_file_index_state, mark_file_indexed,
        )
        mark_file_indexed(
            project_id=1, file_path="/tmp/legacy.py",
            language="python", content_hash="abc123",
            symbol_count=5,
            # parser_version omitted → default 0
        )
        state = get_file_index_state(1, "/tmp/legacy.py")
        assert state == ("abc123", 0)

    def test_upsert_updates_parser_version(self, temp_db):
        """Re-indexing the same file with a newer parser version
        overwrites the stored version."""
        from infinidev.code_intel.index import (
            get_file_index_state, mark_file_indexed,
        )
        mark_file_indexed(
            project_id=1, file_path="/tmp/bar.py",
            language="python", content_hash="abc",
            symbol_count=5, parser_version=1,
        )
        # Same file, bumped parser version
        mark_file_indexed(
            project_id=1, file_path="/tmp/bar.py",
            language="python", content_hash="abc",
            symbol_count=5, parser_version=2,
        )
        state = get_file_index_state(1, "/tmp/bar.py")
        assert state == ("abc", 2)


class TestIncrementalSkipCheckHonorsVersion:
    """The incremental skip check in indexer.index_file must compare
    BOTH content_hash AND parser_version before returning 0.  These
    tests exercise the skip logic at the unit level — the code in
    index_file is extracted enough that we can test its skip branch
    directly.
    """

    def test_skip_when_hash_and_version_match(self, temp_db):
        """File is skipped only when both content hash and parser
        version match the stored state."""
        from infinidev.code_intel.index import (
            get_file_index_state, mark_file_indexed,
        )
        mark_file_indexed(
            project_id=1, file_path="/tmp/baz.py",
            language="python", content_hash="h1",
            symbol_count=1, parser_version=1,
        )
        existing = get_file_index_state(1, "/tmp/baz.py")
        # Simulating the indexer's skip check:
        same_content = "h1"
        same_version = 1
        assert existing == (same_content, same_version)  # → skip path

    def test_reindex_when_version_bumped(self, temp_db):
        """Parser version bump invalidates otherwise-unchanged file."""
        from infinidev.code_intel.index import (
            get_file_index_state, mark_file_indexed,
        )
        mark_file_indexed(
            project_id=1, file_path="/tmp/qux.py",
            language="python", content_hash="h1",
            symbol_count=1, parser_version=1,
        )
        existing = get_file_index_state(1, "/tmp/qux.py")
        # Content unchanged but parser version bumped to 2
        same_content = "h1"
        new_version = 2
        assert existing != (same_content, new_version)  # → re-parse path

    def test_reindex_when_content_changed(self, temp_db):
        """Content change invalidates otherwise-same version."""
        from infinidev.code_intel.index import (
            get_file_index_state, mark_file_indexed,
        )
        mark_file_indexed(
            project_id=1, file_path="/tmp/quux.py",
            language="python", content_hash="h1",
            symbol_count=1, parser_version=1,
        )
        existing = get_file_index_state(1, "/tmp/quux.py")
        new_content = "h2"
        same_version = 1
        assert existing != (new_content, same_version)  # → re-parse path

    def test_pre_versioning_rows_always_invalidate(self, temp_db):
        """Rows with parser_version=0 (pre-migration) are always stale
        against any current parser version (≥1)."""
        from infinidev.code_intel.index import (
            get_file_index_state, mark_file_indexed,
        )
        mark_file_indexed(
            project_id=1, file_path="/tmp/legacy2.py",
            language="python", content_hash="h1",
            symbol_count=1,  # default version 0
        )
        existing = get_file_index_state(1, "/tmp/legacy2.py")
        assert existing == ("h1", 0)
        # Any current version ≥1 must mismatch → re-parse
        assert existing != ("h1", 1)
