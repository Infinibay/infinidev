"""Tests for the per-language symbol extractors.

Targets ``code_intel/extractors/python.py`` and
``code_intel/extractors/javascript.py``, extracted from the monolithic
``syntax_check.py`` in commit ``6964f5c``. Uses the real tree-sitter
parsers to ensure the production dispatch path is the one covered.
"""

from __future__ import annotations

import pytest

from infinidev.code_intel.extractors import (
    LANGUAGE_EXTRACTORS,
    extract_js_symbols,
    extract_python_symbols,
)
from infinidev.code_intel.extractors._common import node_name
from infinidev.code_intel.syntax_check import _load_parser


def _parse(source: str, language: str):
    """Return the tree-sitter root node + source bytes for ``source``."""
    parser = _load_parser(language)
    if parser is None:
        pytest.skip(f"tree-sitter parser for {language} not installed")
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


# ── Python extractor ─────────────────────────────────────────────────────


class TestPythonExtractor:
    def test_top_level_function(self):
        root, src = _parse("def foo():\n    pass\n", "python")
        assert extract_python_symbols(root, src) == {"foo"}

    def test_top_level_class(self):
        root, src = _parse("class Bar:\n    pass\n", "python")
        assert extract_python_symbols(root, src) == {"Bar"}

    def test_method_is_qualified_with_class(self):
        source = (
            "class Calc:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
            "    def sub(self, a, b):\n"
            "        return a - b\n"
        )
        root, src = _parse(source, "python")
        symbols = extract_python_symbols(root, src)
        assert "Calc" in symbols
        assert "Calc.add" in symbols
        assert "Calc.sub" in symbols

    def test_nested_class_method_uses_inner_class_name(self):
        source = (
            "class Outer:\n"
            "    class Inner:\n"
            "        def thing(self):\n"
            "            pass\n"
        )
        root, src = _parse(source, "python")
        symbols = extract_python_symbols(root, src)
        assert "Outer" in symbols
        assert "Outer.Inner" in symbols
        # Methods on Inner should be qualified with Inner, not Outer.
        assert "Outer.Inner.thing" in symbols

    def test_free_function_alongside_class(self):
        source = (
            "def helper():\n"
            "    pass\n"
            "\n"
            "class C:\n"
            "    def method(self):\n"
            "        pass\n"
        )
        root, src = _parse(source, "python")
        symbols = extract_python_symbols(root, src)
        assert symbols == {"helper", "C", "C.method"}

    def test_empty_source_returns_empty(self):
        root, src = _parse("", "python")
        assert extract_python_symbols(root, src) == set()


# ── JavaScript extractor ─────────────────────────────────────────────────


class TestJavaScriptExtractor:
    def test_top_level_function_declaration(self):
        root, src = _parse("function foo() { return 1; }\n", "javascript")
        assert "foo" in extract_js_symbols(root, src)

    def test_top_level_class_declaration(self):
        source = "class Widget { constructor() {} }\n"
        root, src = _parse(source, "javascript")
        symbols = extract_js_symbols(root, src)
        assert "Widget" in symbols

    def test_method_qualified_with_class(self):
        source = (
            "class Thing {\n"
            "  doStuff() { return 1; }\n"
            "  static utility() { return 2; }\n"
            "}\n"
        )
        root, src = _parse(source, "javascript")
        symbols = extract_js_symbols(root, src)
        assert "Thing" in symbols
        # methods should be qualified under the class
        qualified = {s for s in symbols if s.startswith("Thing.")}
        assert qualified, f"no methods qualified under Thing: {symbols}"


# ── Registry ─────────────────────────────────────────────────────────────


class TestLanguageExtractorsRegistry:
    def test_python_key_present(self):
        assert "python" in LANGUAGE_EXTRACTORS
        assert LANGUAGE_EXTRACTORS["python"] is extract_python_symbols

    def test_js_family_keys_present(self):
        for key in ("javascript", "typescript", "tsx", "jsx"):
            assert key in LANGUAGE_EXTRACTORS
            assert LANGUAGE_EXTRACTORS[key] is extract_js_symbols

    def test_unknown_language_not_registered(self):
        assert "brainfuck" not in LANGUAGE_EXTRACTORS


# ── node_name helper ─────────────────────────────────────────────────────


class TestNodeNameHelper:
    def test_returns_identifier_text(self):
        # Use a Python source where the function_definition node has
        # an identifier child named 'foo'.
        root, src = _parse("def foo(): pass\n", "python")
        # Descend to the first function_definition node.
        func_node = None
        def _walk(n):
            nonlocal func_node
            if n.type == "function_definition":
                func_node = n
                return
            for c in n.children:
                _walk(c)
                if func_node:
                    return
        _walk(root)
        assert func_node is not None
        assert node_name(func_node, src) == "foo"
