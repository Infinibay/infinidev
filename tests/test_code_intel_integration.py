"""Integration tests for code intelligence: indexer + query engine."""

import os
import textwrap

import pytest

from infinidev.code_intel.indexer import index_file, index_directory, reindex_file
from infinidev.code_intel.query import (
    find_definition, find_references, list_symbols,
    search_symbols, get_signature, find_imports_of, get_index_stats,
)
from infinidev.code_intel.models import SymbolKind


@pytest.fixture
def py_project(tmp_path, temp_db):
    """Create a mini Python project for indexing."""
    # Main module
    (tmp_path / "auth.py").write_text(textwrap.dedent("""\
        import jwt
        from typing import Optional

        TOKEN_SECRET = "secret123"

        class AuthService:
            \"\"\"Handles authentication.\"\"\"

            def __init__(self, db):
                self.db = db

            def verify_token(self, token: str) -> bool:
                \"\"\"Verify a JWT token.\"\"\"
                try:
                    jwt.decode(token, TOKEN_SECRET)
                    return True
                except Exception:
                    return False

            async def refresh_token(self, token: str) -> str:
                \"\"\"Refresh an expired token.\"\"\"
                return jwt.encode({}, TOKEN_SECRET)
    """))

    # API module that uses auth
    (tmp_path / "api.py").write_text(textwrap.dedent("""\
        from auth import AuthService, TOKEN_SECRET

        def handle_login(request):
            auth = AuthService(db=None)
            if auth.verify_token(request.token):
                return {"status": "ok"}
            return {"status": "unauthorized"}

        def handle_refresh(request):
            auth = AuthService(db=None)
            new_token = auth.refresh_token(request.token)
            return {"token": new_token}
    """))

    # Tests
    (tmp_path / "test_auth.py").write_text(textwrap.dedent("""\
        from auth import AuthService

        def test_verify_valid():
            svc = AuthService(db=None)
            assert svc.verify_token("valid") == True

        def test_verify_invalid():
            svc = AuthService(db=None)
            assert svc.verify_token("invalid") == False
    """))

    return tmp_path


class TestIndexer:
    def test_index_single_file(self, py_project):
        count = index_file(1, str(py_project / "auth.py"))
        assert count > 0  # Should find class, methods, constant

    def test_index_directory(self, py_project):
        stats = index_directory(1, str(py_project))
        assert stats["files_indexed"] == 3  # auth.py, api.py, test_auth.py
        assert stats["symbols_total"] > 5

    def test_index_skips_unchanged(self, py_project):
        # First index
        stats1 = index_directory(1, str(py_project))
        # Second index — should skip all
        stats2 = index_directory(1, str(py_project))
        assert stats2["files_indexed"] == 0

    def test_reindex_forces_update(self, py_project):
        index_file(1, str(py_project / "auth.py"))
        count = reindex_file(1, str(py_project / "auth.py"))
        assert count > 0  # Should re-extract symbols

    def test_index_unsupported_file(self, py_project):
        (py_project / "readme.md").write_text("# Hello")
        count = index_file(1, str(py_project / "readme.md"))
        assert count == 0  # Markdown not supported

    def test_index_nonexistent_file(self):
        count = index_file(1, "/nonexistent/file.py")
        assert count == 0


class TestQueryFindDefinition:
    def test_find_class(self, py_project):
        index_directory(1, str(py_project))
        results = find_definition(1, "AuthService")
        assert len(results) >= 1
        assert results[0].kind == SymbolKind.class_
        assert "auth.py" in results[0].file_path

    def test_find_method(self, py_project):
        index_directory(1, str(py_project))
        results = find_definition(1, "verify_token")
        assert len(results) >= 1
        assert results[0].kind == SymbolKind.method
        assert "verify_token" in results[0].signature

    def test_find_function(self, py_project):
        index_directory(1, str(py_project))
        results = find_definition(1, "handle_login")
        assert len(results) >= 1
        assert results[0].kind == SymbolKind.function

    def test_find_constant(self, py_project):
        index_directory(1, str(py_project))
        results = find_definition(1, "TOKEN_SECRET")
        assert len(results) >= 1
        assert results[0].kind == SymbolKind.constant

    def test_find_with_kind_filter(self, py_project):
        index_directory(1, str(py_project))
        # Only classes
        results = find_definition(1, "AuthService", kind="class")
        assert len(results) >= 1
        # Only functions — should not find class
        results = find_definition(1, "AuthService", kind="function")
        assert len(results) == 0

    def test_find_nonexistent(self, py_project):
        index_directory(1, str(py_project))
        results = find_definition(1, "NonExistentSymbol")
        assert len(results) == 0


class TestQueryFindReferences:
    def test_find_all_refs(self, py_project):
        index_directory(1, str(py_project))
        refs = find_references(1, "AuthService")
        assert len(refs) >= 3  # Used in api.py and test_auth.py

    def test_find_usage_refs(self, py_project):
        index_directory(1, str(py_project))
        refs = find_references(1, "verify_token")
        assert len(refs) >= 2  # Used in api.py and test_auth.py

    def test_find_refs_in_file(self, py_project):
        index_directory(1, str(py_project))
        refs = find_references(1, "AuthService", file_path=str(py_project / "api.py"))
        assert len(refs) >= 1
        assert all("api.py" in r.file_path for r in refs)


class TestQueryListSymbols:
    def test_list_all(self, py_project):
        index_directory(1, str(py_project))
        syms = list_symbols(1, str(py_project / "auth.py"))
        names = [s.name for s in syms]
        assert "AuthService" in names
        assert "verify_token" in names
        assert "TOKEN_SECRET" in names

    def test_list_by_kind(self, py_project):
        index_directory(1, str(py_project))
        methods = list_symbols(1, str(py_project / "auth.py"), kind="method")
        assert all(s.kind == SymbolKind.method for s in methods)
        assert len(methods) >= 2


class TestQuerySearchSymbols:
    def test_fuzzy_search(self, py_project):
        index_directory(1, str(py_project))
        results = search_symbols(1, "token")
        names = [s.name for s in results]
        assert "verify_token" in names or "TOKEN_SECRET" in names

    def test_search_empty_query(self, py_project):
        index_directory(1, str(py_project))
        results = search_symbols(1, "")
        assert results == []


class TestQueryGetSignature:
    def test_get_function_signature(self, py_project):
        index_directory(1, str(py_project))
        results = get_signature(1, "verify_token")
        assert len(results) >= 1
        assert "token: str" in results[0].signature


class TestQueryFindImports:
    def test_find_imports(self, py_project):
        index_directory(1, str(py_project))
        imports = find_imports_of(1, "AuthService")
        assert len(imports) >= 2  # api.py and test_auth.py import it


class TestIndexStats:
    def test_stats(self, py_project):
        index_directory(1, str(py_project))
        stats = get_index_stats(1)
        assert stats["files"] == 3
        assert stats["symbols"] > 5
        assert stats["references"] > 10
        assert stats["imports"] > 3
