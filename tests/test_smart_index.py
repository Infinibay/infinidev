"""Tests for smart indexing and symbol resolution."""

import os
import tempfile
import pytest

from infinidev.code_intel.smart_index import ensure_indexed
from infinidev.code_intel.resolve import resolve_symbol, ResolveResult


@pytest.fixture
def temp_python_file(tmp_path):
    """Create a temporary Python file for testing."""
    code = '''\
class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result

    def subtract(self, a, b):
        result = a - b
        self.history.append(result)
        return result


def helper_func(x):
    return x * 2
'''
    f = tmp_path / "calc.py"
    f.write_text(code)
    return str(f)


class TestEnsureIndexed:
    def test_nonexistent_file(self):
        assert ensure_indexed(1, "/nonexistent/file.py") is False

    def test_empty_path(self):
        assert ensure_indexed(1, "") is False

    def test_indexes_python_file(self, temp_python_file):
        result = ensure_indexed(1, temp_python_file)
        assert result is True

    def test_skip_already_indexed(self, temp_python_file):
        # First call indexes
        ensure_indexed(1, temp_python_file)
        # Second call should skip (same content)
        result = ensure_indexed(1, temp_python_file)
        assert result is False

    def test_reindexes_on_change(self, temp_python_file):
        ensure_indexed(1, temp_python_file)
        # Modify file
        with open(temp_python_file, "a") as f:
            f.write("\ndef new_func():\n    pass\n")
        # Should reindex
        result = ensure_indexed(1, temp_python_file)
        assert result is True


class TestResolveSymbol:
    def test_resolve_top_level_function(self, temp_python_file):
        ensure_indexed(1, temp_python_file)
        result = resolve_symbol(1, "helper_func", temp_python_file)
        assert result.symbol is not None
        assert result.symbol.name == "helper_func"

    def test_resolve_method(self, temp_python_file):
        ensure_indexed(1, temp_python_file)
        result = resolve_symbol(1, "Calculator.add", temp_python_file)
        assert result.symbol is not None
        assert result.symbol.name == "add"
        assert result.symbol.parent_symbol == "Calculator"

    def test_resolve_class(self, temp_python_file):
        ensure_indexed(1, temp_python_file)
        result = resolve_symbol(1, "Calculator", temp_python_file)
        assert result.symbol is not None
        assert result.symbol.name == "Calculator"

    def test_resolve_nonexistent(self, temp_python_file):
        ensure_indexed(1, temp_python_file)
        result = resolve_symbol(1, "NonExistent.method", temp_python_file)
        assert result.error != ""
        assert result.symbol is None
