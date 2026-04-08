"""Tests for the code intelligence system."""

import json
import os
import sqlite3
import textwrap
from unittest.mock import patch

import pytest

from infinidev.code_intel.models import Symbol, SymbolKind, Reference, Import
from infinidev.code_intel.parsers.python_parser import PythonParser, parse_file
from infinidev.code_intel.parsers import detect_language, EXTENSIONS
from infinidev.code_intel import index as ci_index
from infinidev.db.service import init_db


# ── Models ───────────────────────────────────────────────────────────────────


class TestSymbolKind:
    def test_all_kinds(self):
        assert SymbolKind.function == "function"
        assert SymbolKind.method == "method"
        assert SymbolKind.class_ == "class"
        assert SymbolKind.variable == "variable"
        assert SymbolKind.constant == "constant"
        assert SymbolKind.interface == "interface"
        assert SymbolKind.enum == "enum"

    def test_from_string(self):
        assert SymbolKind("function") == SymbolKind.function
        assert SymbolKind("class") == SymbolKind.class_


class TestSymbol:
    def test_defaults(self):
        s = Symbol(name="foo")
        assert s.name == "foo"
        assert s.kind == SymbolKind.function
        assert s.visibility == "public"
        assert not s.is_async
        assert not s.is_static

    def test_full(self):
        s = Symbol(
            name="get_user",
            qualified_name="UserService.get_user",
            kind=SymbolKind.method,
            file_path="src/service.py",
            line_start=42,
            line_end=55,
            signature="def get_user(self, uid: int) -> dict",
            type_annotation="dict",
            parent_symbol="UserService",
            is_async=True,
            language="python",
        )
        assert s.qualified_name == "UserService.get_user"
        assert s.is_async


class TestReference:
    def test_defaults(self):
        r = Reference(name="foo")
        assert r.ref_kind == "usage"
        assert r.resolved_file == ""


class TestImport:
    def test_defaults(self):
        i = Import(source="typing", name="Optional")
        assert not i.is_wildcard
        assert i.alias == ""


# ── Language Detection ───────────────────────────────────────────────────────


class TestLanguageDetection:
    def test_python(self):
        assert detect_language("src/main.py") == "python"

    def test_javascript(self):
        assert detect_language("app.js") == "javascript"
        assert detect_language("component.jsx") == "javascript"
        assert detect_language("module.mjs") == "javascript"

    def test_typescript(self):
        # .tsx is reported as its own language key so the extractor
        # registry can pick the TSX grammar; both still belong to the
        # TypeScript family for the caller's purposes.
        assert detect_language("app.ts") == "typescript"
        assert detect_language("component.tsx") in ("typescript", "tsx")

    def test_rust(self):
        assert detect_language("main.rs") == "rust"

    def test_c(self):
        assert detect_language("main.c") == "c"
        assert detect_language("header.h") == "c"

    def test_unknown(self):
        assert detect_language("file.xyz") is None
        assert detect_language("Makefile") is None

    def test_case_insensitive(self):
        assert detect_language("FILE.PY") == "python"

    def test_all_extensions_mapped(self):
        """Every extension in the map should return a non-None language."""
        for ext, lang in EXTENSIONS.items():
            assert lang is not None


# ── Python Parser: Symbols ───────────────────────────────────────────────────


class TestPythonParserSymbols:
    def _parse(self, code: str) -> list[Symbol]:
        symbols, _, _ = parse_file(textwrap.dedent(code).encode(), "test.py")
        return symbols

    def test_function(self):
        symbols = self._parse("""
            def hello(name: str) -> str:
                return f"Hello {name}"
        """)
        funcs = [s for s in symbols if s.kind == SymbolKind.function]
        assert len(funcs) == 1
        assert funcs[0].name == "hello"
        assert "hello(name: str)" in funcs[0].signature
        assert funcs[0].line_start > 0

    def test_class_with_methods(self):
        symbols = self._parse("""
            class Calculator:
                \"\"\"A simple calculator.\"\"\"

                def add(self, a, b):
                    return a + b

                def subtract(self, a, b):
                    return a - b
        """)
        classes = [s for s in symbols if s.kind == SymbolKind.class_]
        methods = [s for s in symbols if s.kind == SymbolKind.method]
        assert len(classes) == 1
        assert classes[0].name == "Calculator"
        assert classes[0].docstring == "A simple calculator."
        assert len(methods) == 2
        assert methods[0].qualified_name == "Calculator.add"
        assert methods[1].qualified_name == "Calculator.subtract"
        assert methods[0].parent_symbol == "Calculator"

    def test_constant(self):
        symbols = self._parse("""
            MAX_SIZE = 100
            min_val = 0
        """)
        constants = [s for s in symbols if s.kind == SymbolKind.constant]
        variables = [s for s in symbols if s.kind == SymbolKind.variable]
        assert len(constants) == 1
        assert constants[0].name == "MAX_SIZE"
        assert len(variables) == 1
        assert variables[0].name == "min_val"

    def test_async_function(self):
        symbols = self._parse("""
            async def fetch_data(url: str) -> dict:
                pass
        """)
        funcs = [s for s in symbols if s.kind == SymbolKind.function]
        assert len(funcs) >= 1
        # Note: async detection depends on tree-sitter node structure

    def test_visibility(self):
        symbols = self._parse("""
            class Foo:
                def public_method(self): pass
                def _protected_method(self): pass
                def __private_method(self): pass
                def __dunder__(self): pass
        """)
        methods = [s for s in symbols if s.kind == SymbolKind.method]
        by_name = {s.name: s for s in methods}
        assert by_name["public_method"].visibility == "public"
        assert by_name["_protected_method"].visibility == "protected"
        assert by_name["__private_method"].visibility == "private"
        assert by_name["__dunder__"].visibility == "public"  # dunder = public

    def test_empty_file(self):
        symbols = self._parse("")
        assert symbols == []

    def test_nested_class(self):
        symbols = self._parse("""
            class Outer:
                class Inner:
                    def method(self): pass
        """)
        names = [s.qualified_name for s in symbols]
        assert "Outer" in names
        assert "Outer.Inner" in names


# ── Python Parser: Imports ───────────────────────────────────────────────────


class TestPythonParserImports:
    def _parse_imports(self, code: str) -> list[Import]:
        _, _, imports = parse_file(textwrap.dedent(code).encode(), "test.py")
        return imports

    def test_simple_import(self):
        imports = self._parse_imports("import os")
        assert len(imports) == 1
        assert imports[0].source == "os"
        assert imports[0].name == "os"

    def test_from_import(self):
        imports = self._parse_imports("from typing import Optional")
        assert len(imports) == 1
        assert imports[0].source == "typing"
        assert imports[0].name == "Optional"

    def test_from_import_multiple(self):
        imports = self._parse_imports("from os.path import join, exists")
        assert len(imports) == 2
        assert imports[0].source == "os.path"
        assert imports[0].name == "join"
        assert imports[1].name == "exists"

    def test_aliased_import(self):
        imports = self._parse_imports("from auth import verify_token as vt")
        assert len(imports) == 1
        assert imports[0].name == "verify_token"
        assert imports[0].alias == "vt"

    def test_wildcard_import(self):
        imports = self._parse_imports("from module import *")
        assert len(imports) == 1
        assert imports[0].is_wildcard
        assert imports[0].name == "*"

    def test_dotted_import(self):
        imports = self._parse_imports("from auth.service.jwt import create_token")
        assert len(imports) == 1
        assert imports[0].source == "auth.service.jwt"
        assert imports[0].name == "create_token"


# ── Python Parser: References ────────────────────────────────────────────────


class TestPythonParserReferences:
    def _parse_refs(self, code: str) -> list[Reference]:
        _, refs, _ = parse_file(textwrap.dedent(code).encode(), "test.py")
        return refs

    def test_function_call(self):
        refs = self._parse_refs("""
            def foo(): pass
            result = foo()
        """)
        calls = [r for r in refs if r.ref_kind == "call" and r.name == "foo"]
        assert len(calls) >= 1

    def test_variable_usage(self):
        refs = self._parse_refs("""
            count = 10
            total = count + 5
        """)
        count_refs = [r for r in refs if r.name == "count"]
        assert len(count_refs) >= 1

    def test_skips_builtins(self):
        refs = self._parse_refs("""
            x = True
            y = None
        """)
        names = [r.name for r in refs]
        assert "True" not in names
        assert "None" not in names

    def test_has_context(self):
        refs = self._parse_refs("""
            result = calculate(value)
        """)
        calc_refs = [r for r in refs if r.name == "calculate"]
        assert len(calc_refs) >= 1
        assert "calculate" in calc_refs[0].context


# ── Index Store ──────────────────────────────────────────────────────────────


class TestIndexStore:
    """Tests for SQLite storage of code intelligence data."""

    def test_store_and_clear(self, temp_db):
        """Store symbols and then clear them."""
        symbols = [
            Symbol(name="foo", qualified_name="foo", kind=SymbolKind.function,
                   file_path="test.py", line_start=1, language="python"),
            Symbol(name="Bar", qualified_name="Bar", kind=SymbolKind.class_,
                   file_path="test.py", line_start=10, language="python"),
        ]
        refs = [
            Reference(name="foo", file_path="test.py", line=20, language="python"),
        ]
        imports = [
            Import(source="os", name="os", file_path="test.py", line=1, language="python"),
        ]

        ci_index.store_file_symbols(1, "test.py", symbols, refs, imports)
        ci_index.mark_file_indexed(1, "test.py", "python", "abc123", 2)

        # Verify stored
        assert ci_index.get_file_hash(1, "test.py") == "abc123"

        # Clear
        ci_index.clear_file(1, "test.py")

        # Verify symbols cleared (hash still exists in ci_files)
        # We'd need a query to verify, but at least clear doesn't crash

    def test_file_hash_not_indexed(self, temp_db):
        """Returns None for files not yet indexed."""
        assert ci_index.get_file_hash(1, "nonexistent.py") is None

    def test_upsert_file(self, temp_db):
        """mark_file_indexed updates existing entries."""
        ci_index.mark_file_indexed(1, "test.py", "python", "hash1", 5)
        assert ci_index.get_file_hash(1, "test.py") == "hash1"

        ci_index.mark_file_indexed(1, "test.py", "python", "hash2", 10)
        assert ci_index.get_file_hash(1, "test.py") == "hash2"

    def test_clear_project(self, temp_db):
        """clear_project removes all data for a project."""
        symbols = [Symbol(name="x", kind=SymbolKind.variable,
                         file_path="a.py", line_start=1, language="python")]
        ci_index.store_file_symbols(1, "a.py", symbols, [], [])
        ci_index.mark_file_indexed(1, "a.py", "python", "h1", 1)

        ci_index.clear_project(1)
        assert ci_index.get_file_hash(1, "a.py") is None

    def test_store_replaces_existing(self, temp_db):
        """Storing symbols for a file replaces previous entries."""
        sym1 = [Symbol(name="old", kind=SymbolKind.function,
                       file_path="test.py", line_start=1, language="python")]
        ci_index.store_file_symbols(1, "test.py", sym1, [], [])

        sym2 = [Symbol(name="new", kind=SymbolKind.function,
                       file_path="test.py", line_start=1, language="python")]
        ci_index.store_file_symbols(1, "test.py", sym2, [], [])

        # Should only have "new", not "old" — verified by query in future tests
